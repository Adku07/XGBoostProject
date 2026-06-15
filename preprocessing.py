import pandas as pd
import numpy as np
import re


# =========================
# MAP BRAND
# =========================
def map_brand(name):

    name = str(name).lower()

    if re.search(r'\begn\b', name):
        return "egn"

    elif re.search(r'\beta\b', name):
        return "eta"

    elif re.search(r'\bbiokhol\b', name):
        return "biokhol"

    return "other"


# =========================
# MAP OBJECTIVE
# =========================
def map_objective(obj):

    obj = str(obj).upper()

    if obj in [
        "OUTCOME_AWARENESS",
        "OUTCOME_ENGAGEMENT"
    ]:
        return "awareness"

    elif obj in [
        "OUTCOME_SALES",
        "OUTCOME_LEADS"
    ]:
        return "conversion"

    elif obj == "LINK_CLICKS":
        return "click"

    return "other"


# =========================
# PREPROCESS
# =========================
def preprocess_data(df, BRAND):

    # =========================
    # BASIC PREP
    # =========================
    df = df.copy()

    df["dt"] = pd.to_datetime(df["dt"])

    df["brand"] = df["ad_account_name"].apply(map_brand)

    df["objective_group"] = df["objective"].apply(map_objective)

    # remove other
    df = df[df["brand"] != "other"]

    # start date
    df = df[df["dt"] >= "2026-01-01"]

    # =========================
    # H-2
    # =========================
    cutoff = df["dt"].max() - pd.Timedelta(days=2)

    df = df[df["dt"] <= cutoff]

    # =========================
    # PIVOT SPEND
    # =========================
    pivot_spend = df.pivot_table(
        index="dt",
        columns=["brand", "objective_group"],
        values="spend",
        aggfunc="sum"
    )

    pivot_impr = df.pivot_table(
        index="dt",
        columns="brand",
        values="impressions",
        aggfunc="sum"
    )

    # =========================
    # RENAME COLS
    # =========================
    pivot_spend.columns = [
        f"spend_{brand}_{obj}"
        for brand, obj in pivot_spend.columns
    ]

    pivot_impr.columns = [
        f"impr_{brand}"
        for brand in pivot_impr.columns
    ]

    # =========================
    # COMBINE
    # =========================
    data = pd.concat(
        [pivot_spend, pivot_impr],
        axis=1
    )

    data = data.sort_index()

    # =========================
    # REMOVE LEBARAN
    # =========================
    mask_lebaran = data.index.to_series().between(
        "2026-03-15",
        "2026-04-04"
    )

    data = data[~mask_lebaran]

    # =========================
    # FILL MISSING
    # =========================
    data = data.fillna(0)

    # =========================
    # SAFE COLUMNS
    # =========================
    selected_cols = [
        f"spend_{BRAND}_awareness",
        f"spend_{BRAND}_conversion",
        f"spend_{BRAND}_click",
        f"impr_{BRAND}"
    ]

    # inject missing cols
    for col in selected_cols:

        if col not in data.columns:
            data[col] = 0

    # =========================
    # FINAL TS
    # =========================
    ts = data[selected_cols].copy()

    # =========================
    # REMOVE ZERO ROWS
    # =========================
    spend_cols = [
        f"spend_{BRAND}_awareness",
        f"spend_{BRAND}_conversion",
        f"spend_{BRAND}_click"
    ]

    ts = ts[
        (ts[spend_cols].sum(axis=1) > 0) &
        (ts[f"impr_{BRAND}"] > 0)
    ]

    # =========================
    # SORT
    # =========================
    ts.index = pd.to_datetime(ts.index)

    ts = ts.sort_index()

    # =========================
    # TOTAL SPEND
    # =========================
    ts["total_spend"] = ts[spend_cols].sum(axis=1)

    # =========================
    # OUTLIER DETECTION
    # =========================
    col = "total_spend"
    # col = f"impr_{BRAND}"

    Q1 = ts[col].quantile(0.25)

    Q3 = ts[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    # =========================
    # FLAG OUTLIER
    # =========================
    ts["is_outlier"] = (
        (ts[col] < lower_bound) |
        (ts[col] > upper_bound)
    )

    # =========================
    # OUTLIER TYPE
    # =========================
    def detect_outlier(x):

        if x < lower_bound:
            return "lower_outlier"

        elif x > upper_bound:
            return "upper_outlier"

        return "normal"

    ts["outlier_type"] = ts[col].apply(
        detect_outlier
    )

    # =========================
    # DROP OUTLIER
    # =========================
    ts = ts[~ts["is_outlier"]]

    # =========================
    # FINAL
    # =========================
    print("\nFINAL DATA COLUMNS")
    print(ts.columns.tolist())

    print("\nTOTAL DATA :", len(ts))

    lags = [1,2,3,7,14]

    # =========================
    # LAG FEATURES
    # =========================
    for lag in lags:

        # target impression
        ts[f'lag_impr_{lag}'] = \
            ts[f"impr_{BRAND}"].shift(lag)

        # awareness
        ts[f'lag_spend_awareness_{lag}'] = \
            ts[f"spend_{BRAND}_awareness"].shift(lag)

        # conversion
        ts[f'lag_spend_conversion_{lag}'] = \
            ts[f"spend_{BRAND}_conversion"].shift(lag)

        # click
        ts[f'lag_spend_click_{lag}'] = \
            ts[f"spend_{BRAND}_click"].shift(lag)

    # =========================
    # ROLLING
    # =========================
    ts['impr_roll7'] = (
        ts[f"impr_{BRAND}"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    ts['spend_awareness_roll7'] = (
        ts[f"spend_{BRAND}_awareness"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    ts['spend_conversion_roll7'] = (
        ts[f"spend_{BRAND}_conversion"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    ts['spend_click_roll7'] = (
        ts[f"spend_{BRAND}_click"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    # =========================
    # DIFF
    # =========================
    ts['impr_diff'] = (
        ts[f"impr_{BRAND}"]
        .diff()
        .shift(1)
    )

    ts['spend_awareness_diff'] = (
        ts[f"spend_{BRAND}_awareness"]
        .diff()
        .shift(1)
    )

    ts['spend_conversion_diff'] = (
        ts[f"spend_{BRAND}_conversion"]
        .diff()
        .shift(1)
    )

    ts['spend_click_diff'] = (
        ts[f"spend_{BRAND}_click"]
        .diff()
        .shift(1)
    )

    # =========================
    # TIME FEATURE
    # =========================
    ts['day_of_week'] = ts.index.dayofweek

    ts['is_weekend'] = (
        ts['day_of_week'] >= 5
    ).astype(int)

    # =========================
    # DROP NA
    # =========================
    ts = ts.dropna()

    return ts