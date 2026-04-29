import pandas as pd
import numpy as np

def preprocess(df):
    # Combine date + time
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])

    # Filter completed rides
    df = df[df["Booking Status"] == "Completed"]
    df = df.dropna(subset=["Avg VTAT"])

    # Daily aggregation
    daily = (
        df.groupby(df["datetime"].dt.date)["Avg VTAT"]
        .mean()
        .reset_index()
    )

    daily.columns = ["date", "avg_vtat"]
    daily["date"] = pd.to_datetime(daily["date"])

    # Time features
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day_of_month"] = daily["date"].dt.day
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)

    # Lag features
    for lag in [1, 2, 3, 7]:
        daily[f"vtat_lag{lag}"] = daily["avg_vtat"].shift(lag)

    # Rolling features
    daily["rolling_mean_7"] = daily["avg_vtat"].rolling(7).mean()
    daily["rolling_std_7"] = daily["avg_vtat"].rolling(7).std()

    daily = daily.dropna().reset_index(drop=True)

    return daily