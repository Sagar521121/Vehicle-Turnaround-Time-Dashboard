import pandas as pd
import numpy as np

def forecast_next_days(daily, model, features, days=7):

    history = daily.copy()
    forecasts = []

    for i in range(days):
        last = history.iloc[-1]

        new_row = {
            "vtat_lag1": last["avg_vtat"],
            "vtat_lag2": history.iloc[-2]["avg_vtat"],
            "vtat_lag3": history.iloc[-3]["avg_vtat"],
            "vtat_lag7": history.iloc[-7]["avg_vtat"],

            "rolling_mean_7": history["avg_vtat"].tail(7).mean(),
            "rolling_std_7": history["avg_vtat"].tail(7).std(),

            "month": last["month"],
            "day_of_month": (last["day_of_month"] % 30) + 1,
            "day_of_week": (last["day_of_week"] + 1) % 7,
            "is_weekend": int((last["day_of_week"] + 1) % 7 in [5, 6]),
        }

        X = pd.DataFrame([new_row])[features]
        pred = model.predict(X)[0]

        forecasts.append(pred)

        new_row["avg_vtat"] = pred
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return forecasts