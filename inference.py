import pandas as pd
import numpy as np
import joblib

artifacts = joblib.load("model_artifacts.pkl")

lgb_model = artifacts["lgb_model"]
logit_model = artifacts["logit_model"]
FEATURE_COLS = artifacts["FEATURE_COLS"]
LINEAR_FEATURES = artifacts["LINEAR_FEATURES"]
p95_mark_volume = artifacts["p95_mark_volume"]
p95_total_mark = artifacts["p95_total_mark"]

def predict_daily(df_daily: pd.DataFrame):
    df_daily = df_daily.copy()

    missing = set(FEATURE_COLS + LINEAR_FEATURES + ["Ticker", "Price_check1m"]) - set(df_daily.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X_tree = df_daily[FEATURE_COLS]
    df_daily["Tree_Prob"] = lgb_model.predict_proba(X_tree)[:, 1]

    df_daily["Mark_Volume"] = np.clip(df_daily["Mark_Volume"] / p95_mark_volume, 0, 1)
    df_daily["Total_Mark"] = np.clip(df_daily["Total_Mark"] / p95_total_mark, 0, 1)

    X_linear = df_daily[LINEAR_FEATURES]
    df_daily["Final_Prob"] = logit_model.predict_proba(X_linear)[:, 1]

    return (
        df_daily
        .groupby("Ticker")
        .agg(
            count_signal=("Final_Prob", "count"),
            mean_prob=("Final_Prob", "mean"),
            mean_return=("Price_check1m", "mean")
        )
        .sort_values(["count_signal", "mean_prob"], ascending=False)
    )