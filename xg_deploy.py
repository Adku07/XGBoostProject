import streamlit as st
import pandas as pd
import joblib
import os
import re
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from scipy.stats import norm
from preprocessing import preprocess_data


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Ads Forecasting",
    layout="wide"
)

# =========================
# CUSTOM CSS VISUAL DASHBOARD
# =========================
st.markdown("""
<style>

/* Background utama */
.stApp {
    background: #0E1117;
    color: white;
}

/* Container spacing */
.block-container {
    padding-top: 2rem;
}

/* Card / Metric */
div[data-testid="metric-container"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #2c2f36;
}

/* Table */
[data-testid="stDataFrame"] {
    background-color: #1c1f26;
    border-radius: 10px;
}

/* Button */
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
    border: none;
}

/* Hover button biar lebih hidup */
.stButton > button:hover {
    background-color: #45a049;
}

/* Header text */
h1, h2, h3, h4 {
    color: #FFFFFF;
}

/* Divider */
hr {
    border: 1px solid #2c2f36;
}

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("# 📊 Ads Impression Forecasting Dashboard")

# =========================
# INPUT
# =========================
col1, col2 = st.columns(2)

with col1:
    brand_ui = st.selectbox(
        "🏷️ Pilih Brand",
        ["Etawaku", "Etallagen", "Biokhol"]
    )

with col2:
    steps = st.slider("📅 Forecast", 1, 14, 7)

# =========================
# CONFIG BRAND
# =========================
brand_map = {
    "Etawaku": {
        "code": "eta",
        "path": "models3/eta"
    },

    "Etallagen": {
        "code": "egn",
        "path": "models3/egn"
    },

    "Biokhol": {
        "code": "biokhol",
        "path": "models3/biokhol"
    }
}

config = brand_map[brand_ui]
BRAND = config["code"]
brand_path = config["path"]

model = joblib.load(
    os.path.join(brand_path, "model.pkl")
)

features = joblib.load(
    os.path.join(brand_path, "features.pkl")
)

residuals = joblib.load(
    os.path.join(brand_path, "residuals.pkl")
)

metrics = joblib.load(
    os.path.join(brand_path, "metrics.pkl")
)

test_pred = joblib.load(
    os.path.join(brand_path, "test_pred.pkl")
)


# =========================
# MODEL & PERFORMANCE MODEL
# =========================
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("All RMSE", f"{metrics['all_rmse']:,.0f}")
col2.metric("All MAPE", f"{metrics['all_mape']:.2f}%")
col3.metric("All WAPE", f"{metrics['all_wape']:.2f}%")

st.markdown(f""" 
### 🎯 Brand: **{brand_ui}** 
Model: XGBoost {BRAND}
""")

# =========================
# DB
# =========================
engine = create_engine(
    "mysql+pymysql://data_team_read_only:NajanTeuLeuwihTeuKurang@103.129.149.125:1453/data_team"
)

# =========================
# PREPROCESS
# =========================
# def map_brand(name):
#     name = name.lower()
#     if re.search(r'\begn\b', name): return "egn"
#     elif re.search(r'\beta\b', name): return "eta"
#     elif re.search(r'\bbiokhol\b', name): return "biokhol"
#     return "other"

def load_data():
    query = """
        SELECT 
                    dt,
                    ad_account_name,
                    objective,
                    SUM(spend) AS spend,
                    SUM(impressions) AS impressions

                FROM intern_fact_meta_insights_daily

                WHERE ad_account_id IN (
                    SELECT ad_account_id
                    FROM intern_fact_meta_insights_daily
                    WHERE dt BETWEEN '2025-12-31' AND '2026-05-30'
                    GROUP BY ad_account_id
                    HAVING SUM(spend) > 0
                )

                GROUP BY 
                    dt,
                    ad_account_name,
                    objective

                ORDER BY dt
    """
    df = pd.read_sql(query, engine)
    ts = preprocess_data(df, BRAND)
    return ts


# =========================
# FORECAST
# =========================
def forecast_n_step(model, ts, BRAND, steps):

    ts_future = ts.copy()
    predictions = []

    for i in range(steps):

        last_row = ts_future.iloc[-1:].copy()


        # =========================
        # NEXT DATE
        # =========================
        next_date = ts_future.index[-1] + pd.Timedelta(days=1)

        last_row.index = [next_date]

        # =========================
        # TIME FEATURES
        # =========================
        last_row['day_of_week'] = next_date.dayofweek

        last_row['is_weekend'] = int(
            next_date.dayofweek >= 5
        )

        # =========================
        # PREDICT
        # =========================
        X_pred = last_row.drop(
            columns=[f"impr_{BRAND}"],
            errors='ignore'
        )

        X_pred = X_pred[features].fillna(0)

        y_pred = model.predict(X_pred)[0]

        # =========================
        # APPEND PREDICTION
        # =========================
        last_row[f"impr_{BRAND}"] = y_pred

        predictions.append(
            (next_date, y_pred)
        )

        ts_future = pd.concat([
            ts_future,
            last_row
        ])

    return pd.DataFrame(
        predictions,
        columns=["date", "prediction"]
    )

# =========================
# FUNCTION CI AR(1) 
# =========================
def estimate_theta(y):
    y_t = y[1:]
    y_tm1 = y[:-1]
    theta = np.sum(y_t * y_tm1) / np.sum(y_tm1**2)
    return theta

def ar1_ci(preds, rmse, theta, alpha=0.05):
    z = norm.ppf(1 - alpha/2)

    lower = []
    upper = []

    for h, yhat in enumerate(preds, start=1):

        se_h = rmse * np.sqrt((1 - theta**(2*h)) / (1 - theta**2))

        lower.append(yhat - z * se_h) 
        upper.append(yhat + z * se_h)

    st.write("Theta:", theta)
    st.write("RMSE:", rmse)
    return np.array(lower), np.array(upper)

# =========================
# RUN
# =========================
if st.button("🚀 Run Forecast", use_container_width=True):

    ts = load_data()
    df_pred = forecast_n_step(model, ts, BRAND, steps)

    # =========================
    # CI (90%)
    # =========================
    pred_values = df_pred["prediction"].values

    # =========================
    # HITUNG THETA DARI DATA ASLI
    # =========================
    y_series = ts[f"impr_{BRAND}"].values
    theta = estimate_theta(y_series)

    # =========================
    # RMSE (pakai test)
    # =========================
    rmse = metrics["all_rmse"]
    
    # =========================
    # PREDICTION
    # =========================
    pred_values = df_pred["prediction"].values

    # =========================
    # CI AR(1)
    # =========================
    lower, upper = ar1_ci(pred_values, rmse, theta, alpha=0.1)  # 90% CI

    df_pred["lower_ci"] = lower
    df_pred["upper_ci"] = upper

    # =========================
    # KPI
    # =========================
    last_val = ts.iloc[-1][f"impr_{BRAND}"]
    next_val = df_pred.iloc[0]["prediction"]
    avg_val = df_pred["prediction"].mean()

    col1, col2, col3= st.columns(3)
    col1.metric("Last Actual", f"{int(last_val):,}")
    col2.metric("Next Day", f"{int(next_val):,}",
                f"{((next_val-last_val)/last_val)*100:.2f}%")
    col3.metric("Average Prediction", f"{int(avg_val):,}")

    # =========================
    # TABLE
    # =========================
    df_show = df_pred.copy()

    df_show.columns = ["Date", "Prediction", "Lower CI", "Upper CI"]

    df_show["Date"] = pd.to_datetime(df_show["Date"]).dt.date

    for col in ["Prediction", "Lower CI", "Upper CI"]:
        df_show[col] = df_show[col].apply(lambda x: f"{x:,.2f}")

    st.dataframe(df_show, use_container_width=True)

    # =========================
    # CHART
    # =========================
    fig, ax = plt.subplots(figsize=(14,6), facecolor="#0E1117")
    ax.set_facecolor("#151923")

    last_actual = ts.tail(9)

    ax.plot(last_actual.index, last_actual[f"impr_{BRAND}"],
            color="#00E5FF", linewidth=3, label="Actual")

    ax.plot(df_pred["date"], df_pred["prediction"],
            color="#FFB74D", linestyle="--", marker="o",
            linewidth=3, label="Forecast")
    
    test_pred.index = pd.to_datetime(test_pred.index)

    test_pred_last = test_pred.reindex(last_actual.index)

    ax.plot(
        test_pred_last.index,
        test_pred_last["prediction"],
        linestyle=":",
        marker="x",
        color="#FFA500",
        label="Test Prediction"
    )

    ax.fill_between(df_pred["date"], lower, upper,
                    color="#FFB74D", alpha=0.2,
                    label="90% CI")

    ax.axvline(x=last_actual.index[-1], linestyle="--", color="white")

    for s in ax.spines.values():
        s.set_edgecolor("#2c2f36")

    ax.set_title(f"{brand_ui} Forecast", color="white")
    ax.tick_params(colors="white")
    ax.legend()
    ax.grid(color="#2c2f36")

    st.pyplot(fig)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance[
        importance["importance"] > 0
    ]
    
    importance = importance[
        ~importance["feature"].str.contains("roll", case=False, na=False)
    ]

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    st.subheader("📈 Feature Importance")

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(
        importance["feature"][::-1],
        importance["importance"][::-1]
    )

    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")

    plt.tight_layout()

    st.pyplot(fig)