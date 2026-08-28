from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from feature_engineering import engineer_features
from model_architecture import MultiHorizonXGBoost
from supabase_client import fetch_features, save_model_to_supabase

HORIZONS = [24, 48, 72]  
TARGET_COL = "us_aqi"
TEST_FRACTION = 0.15  


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adds target_{h}h columns: us_aqi exactly h hours ahead, per city."""
    out_frames = []
    for city, g in df.groupby("city"):
        g = g.set_index("datetime").sort_index()
        for h in HORIZONS:
            g[f"target_{h}h"] = g[TARGET_COL].reindex(g.index + pd.Timedelta(hours=h)).values
        out_frames.append(g.reset_index())
    return pd.concat(out_frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame):
    """Enriches with physics & city-distributed features, one-hot encodes city, and prepares (X, y)."""
    df = engineer_features(df)
    df = pd.get_dummies(df, columns=["city"], prefix="city")

    target_cols = [f"target_{h}h" for h in HORIZONS]
    required_cols = ["aqi_lag_24h", "aqi_lag_48h"] + target_cols
    df_clean = df.dropna(subset=required_cols).reset_index(drop=True)

    feature_cols = [
        c for c in df_clean.columns
        if c not in ["datetime"] + target_cols
    ]

    X = df_clean[feature_cols].fillna(0.0).astype("float64")
    y = df_clean[target_cols].astype("float64")
    dt = df_clean["datetime"]
    return X, y, feature_cols, dt, df_clean


def time_based_split(X: pd.DataFrame, y: pd.DataFrame, dt: pd.Series):
    """Out-of-time train/val/test split to mirror true forward forecasting."""
    cutoff = dt.quantile(1 - TEST_FRACTION)
    train_mask = dt < cutoff
    test_mask = ~train_mask
    return (
        X[train_mask].reset_index(drop=True),
        X[test_mask].reset_index(drop=True),
        y[train_mask].reset_index(drop=True),
        y[test_mask].reset_index(drop=True),
        cutoff,
    )


def evaluate(y_true: pd.DataFrame, y_pred: np.ndarray, model_name: str = "XGBoost") -> dict:
    metrics = {"model": model_name}
    for i, h in enumerate(HORIZONS):
        col = f"target_{h}h"
        rmse = mean_squared_error(y_true[col], y_pred[:, i]) ** 0.5
        mae = mean_absolute_error(y_true[col], y_pred[:, i])
        r2 = r2_score(y_true[col], y_pred[:, i])
        metrics[f"rmse_{h}h"] = round(float(rmse), 3)
        metrics[f"mae_{h}h"] = round(float(mae), 3)
        metrics[f"r2_{h}h"] = round(float(r2), 3)
        print(f"  [{model_name}] +{h}h -> RMSE={rmse:.2f} | MAE={mae:.2f} | R2={r2:.3f}")
    metrics["avg_rmse"] = round(float(np.mean([metrics[f"rmse_{h}h"] for h in HORIZONS])), 3)
    metrics["avg_mae"] = round(float(np.mean([metrics[f"mae_{h}h"] for h in HORIZONS])), 3)
    metrics["avg_r2"] = round(float(np.mean([metrics[f"r2_{h}h"] for h in HORIZONS])), 3)
    return metrics


def evaluate_city_breakdown(X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray, model_name: str = "XGBoost"):
    print(f"\n--- {model_name.upper()} Per-City Breakdown on Test Set ---")
    for city in ["Karachi", "Lahore", "Islamabad"]:
        city_col = f"city_{city}"
        if city_col in X_test.columns:
            mask = X_test[city_col] == 1.0
            if mask.sum() > 0:
                city_y_true = y_test[mask]
                city_y_pred = y_pred[mask.values]
                rmse_24 = mean_squared_error(city_y_true["target_24h"], city_y_pred[:, 0]) ** 0.5
                mae_24 = mean_absolute_error(city_y_true["target_24h"], city_y_pred[:, 0])
                r2_24 = r2_score(city_y_true["target_24h"], city_y_pred[:, 0])
                rmse_avg = np.mean([
                    mean_squared_error(city_y_true[f"target_{h}h"], city_y_pred[:, idx]) ** 0.5
                    for idx, h in enumerate(HORIZONS)
                ])
                print(f"  {city.ljust(10)}: +24h RMSE={rmse_24:.2f}, MAE={mae_24:.2f}, R2={r2_24:.3f} | 3-Day Avg RMSE={rmse_avg:.2f}")


def train_xgboost(X_train: pd.DataFrame, y_train: pd.DataFrame):
    model = MultiHorizonXGBoost()
    model.fit(X_train, y_train)
    return model


def compute_shap_explanations(model: MultiHorizonXGBoost, X_sample: pd.DataFrame, feature_names: list[str]) -> dict:
    """Compute Tree SHAP feature importance summary."""
    try:
        import shap

        sample = X_sample.sample(min(len(X_sample), 100), random_state=42)
        base_estimator = model.estimators_[24]

        try:
            explainer = shap.TreeExplainer(base_estimator)
            shap_values = explainer.shap_values(sample)
        except Exception:
            explainer = shap.Explainer(base_estimator, sample)
            shap_values = explainer(sample).values

        vals = np.abs(shap_values).mean(axis=0)
        mean_abs_shap = {feat: float(val) for feat, val in zip(feature_names, vals)}
        sorted_shap = dict(sorted(mean_abs_shap.items(), key=lambda item: item[1], reverse=True)[:25])
        return sorted_shap
    except Exception as e:
        print(f"Notice: SHAP computation note ({e})")
        return {}


def main():
    print("Fetching continuous feature data from Supabase Data Warehouse...")
    df = fetch_features()
    if df.empty:
        raise RuntimeError("No feature records found in Supabase. Run historical_backfill.py first.")

    print(f"Loaded {len(df)} total rows from {df['datetime'].min()} to {df['datetime'].max()}.")

    print("\nEngineering atmospheric physics & city-distributed interaction features...")
    df = build_targets(df)
    X, y, feature_names, dt, clean_df = prepare_dataset(df)
    print(f"Total features: {len(feature_names)} | Total training rows: {len(X)}")

    X_train, X_test, y_train, y_test, cutoff = time_based_split(X, y, dt)
    print(f"Train Set: {len(X_train)} rows | Test Set: {len(X_test)} rows | Date Cutoff: {cutoff}")

    print("\nTraining Optimized Multi-Horizon XGBoost Regressor with Physics Features...")
    xgb = train_xgboost(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = evaluate(y_test, xgb_pred, "XGBoost")
    evaluate_city_breakdown(X_test, y_test, xgb_pred, "XGBoost")

    print(f"\n" + "=" * 70)
    print(f"[XGBOOST MODEL EVALUATION SUMMARY] (Avg RMSE: {xgb_metrics['avg_rmse']})")
    print("=" * 70)
    print(f"  - Average RMSE: {xgb_metrics['avg_rmse']} | Average MAE: {xgb_metrics['avg_mae']} | Average R2: {xgb_metrics['avg_r2']}")
    print(f"  - +24h Horizon: RMSE={xgb_metrics['rmse_24h']} | MAE={xgb_metrics['mae_24h']} | R2={xgb_metrics['r2_24h']}")
    print(f"  - +48h Horizon: RMSE={xgb_metrics['rmse_48h']} | MAE={xgb_metrics['mae_48h']} | R2={xgb_metrics['r2_48h']}")
    print(f"  - +72h Horizon: RMSE={xgb_metrics['rmse_72h']} | MAE={xgb_metrics['mae_72h']} | R2={xgb_metrics['r2_72h']}\n")

    print("Computing SHAP feature importances for XGBoost...")
    shap_summary = compute_shap_explanations(xgb, X_test, feature_names)

    print("\nRegistering 'aqi_forecast_xgboost' to Supabase Model Registry...")
    save_model_to_supabase(
        model=xgb,
        model_name="aqi_forecast_xgboost",
        metrics=xgb_metrics,
        feature_names=feature_names,
        shap_summary=shap_summary,
        extra_files=None,
    )

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()