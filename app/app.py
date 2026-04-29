import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add project root to path so `src` imports work
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.model import train_model
from src.forecasting import forecast_next_days

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="VTAT Forecast Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("🚖 VTAT Forecast Dashboard")
st.caption("VTAT forecasting dashboard built with Streamlit")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Choose input method",
    ["Use local file", "Upload CSV"],
    index=0,
)

uploaded_file = None
local_path = "data/ncr_ride_bookings.csv"

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
else:
    local_path = st.sidebar.text_input(
        "Local CSV path",
        value="data/ncr_ride_bookings.csv",
        help="Path relative to the project root",
    )

st.sidebar.markdown("---")
st.sidebar.info(
    "Expected columns: Date, Time, Booking Status, Avg VTAT"
)

# New feature: filter data range and forecast horizon
forecast_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=1,
    max_value=30,
    value=7,
    help="Choose how many days to forecast beyond the available historical data."
)

# New feature: forecast adjustment channel
forecast_adjustment = st.sidebar.slider(
    "Forecast adjustment (%)",
    min_value=-50,
    max_value=50,
    value=0,
    step=1,
    help="Apply a manual adjustment to the forecast values to model expected uplift or reduction."
)

# -----------------------------
# Helper functions
# -----------------------------
def validate_columns(df: pd.DataFrame) -> None:
    required_cols = {"Date", "Time", "Booking Status", "Avg VTAT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n\n"
            f"Found columns: {list(df.columns)}"
        )


def show_kpis(raw_df: pd.DataFrame, completed_df: pd.DataFrame, daily_df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total rows", f"{len(raw_df):,}")
    c2.metric("Completed rows", f"{len(completed_df):,}")
    c3.metric("Daily rows", f"{len(daily_df):,}")
    c4.metric("Avg VTAT", f"{daily_df['avg_vtat'].mean():.2f}")


def plot_line_chart(daily_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_df["date"], daily_df["avg_vtat"])
    ax.set_title("Daily Average VTAT")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg VTAT")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)


def plot_distribution(completed_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(completed_df["Avg VTAT"].dropna(), bins=30)
    ax.set_title("VTAT Distribution")
    ax.set_xlabel("Avg VTAT")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def plot_feature_importance(model, feature_cols):
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is not available for this model.")
        return

    importance_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    st.subheader("Feature Importance")
    st.dataframe(importance_df, width="stretch")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    st.pyplot(fig)


# -----------------------------
# Main app logic
# -----------------------------
try:
    # Load data
    if data_source == "Upload CSV":
        if uploaded_file is None:
            st.warning("Upload a CSV file from the sidebar to begin.")
            st.stop()
        raw_df = load_data(uploaded_file)
    else:
        if not os.path.exists(local_path):
            st.error(f"File not found: {local_path}")
            st.stop()
        raw_df = load_data(local_path)

    # Validate raw columns
    validate_columns(raw_df)

    # Preview raw data
    with st.expander("Preview raw data", expanded=False):
        st.dataframe(raw_df.head(20), width="stretch")

    # Preprocess
    completed_df = raw_df[raw_df["Booking Status"] == "Completed"].copy()
    daily_df = preprocess(raw_df)

    if daily_df.empty:
        st.error(
            "After preprocessing, no usable daily data was left. "
            "Check whether your dataset has enough rows and valid values."
        )
        st.stop()

    if len(daily_df) < 14:
        st.warning(
            "Very small dataset detected. Forecast quality may be weak "
            "because lag and rolling features need enough history."
        )

    # Train model
    model, feature_cols = train_model(daily_df)

    # Forecast next 7 days
    forecast_values = forecast_next_days(daily_df, model, feature_cols, days=7)

    adjusted_forecast_values = [
        value * (1 + forecast_adjustment / 100.0)
        for value in forecast_values
    ]

    last_date = daily_df["date"].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq="D")
    forecast_df = pd.DataFrame(
        {
            "date": forecast_dates,
            "forecast_avg_vtat": forecast_values,
            "adjusted_forecast_avg_vtat": adjusted_forecast_values,
        }
     )

    # KPIs
    show_kpis(raw_df, completed_df, daily_df)

    st.markdown("---")

    # Tabs for cleaner UI
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Trend", "Distribution", "Forecast", "Model Details"]
    )

    with tab1:
        st.subheader("Daily VTAT Trend")
        plot_line_chart(daily_df)

        st.subheader("Daily data table")
        st.dataframe(daily_df.head(20), width="stretch")

    with tab2:
        st.subheader("VTAT Distribution")
        plot_distribution(completed_df)

    with tab3:
        st.subheader("7-Day Forecast")
        if forecast_adjustment != 0:
            st.info(f"Forecast values have been adjusted by {forecast_adjustment}%.")
        st.dataframe(forecast_df, width="stretch")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(forecast_df["date"], forecast_df["forecast_avg_vtat"], marker="o", label="Forecast")
        ax.plot(forecast_df["date"], forecast_df["adjusted_forecast_avg_vtat"], marker="o", linestyle="--", label="Adjusted Forecast")
        ax.set_title("7-Day Forecasted Avg VTAT")
        ax.set_xlabel("Date")
        ax.set_ylabel("Forecast Avg VTAT")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        csv_data = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download forecast CSV",
            data=csv_data,
            file_name="vtat_forecast.csv",
            mime="text/csv",
        )

    with tab4:
        st.subheader("Model Summary")
        st.write("Model type: **XGBoost Regressor**")
        st.write("Features used:")
        st.code("\n".join(feature_cols))

        plot_feature_importance(model, feature_cols)

    st.success("Dashboard loaded successfully.")

except Exception as e:
    st.error("Something went wrong while building the dashboard.")
    st.exception(e)