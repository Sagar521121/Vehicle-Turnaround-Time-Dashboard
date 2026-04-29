import xgboost as xgb

def train_model(daily):

    feature_cols = [
        col for col in daily.columns
        if "lag" in col or "rolling" in col
        or col in ["month", "day_of_month", "day_of_week", "is_weekend"]
    ]

    X = daily[feature_cols]
    y = daily["avg_vtat"]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model, feature_cols