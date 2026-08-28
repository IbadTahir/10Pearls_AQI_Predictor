from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from supabase import Client, create_client
import model_architecture

from config import (
    SUPABASE_BUCKET_MODELS,
    SUPABASE_KEY,
    SUPABASE_TABLE_FEATURES,
    SUPABASE_TABLE_MODELS,
    SUPABASE_URL,
)

_SUPABASE_CLIENT: Optional[Client] = None

SUPABASE_FEATURE_COLUMNS = [
    "city", "datetime", "hour", "day", "month", "day_of_week", "is_weekend",
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "us_aqi", "us_aqi_pm2_5", "us_aqi_pm10",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m",
    "precipitation", "cloud_cover", "dew_point_2m",
    "shortwave_radiation", "boundary_layer_height",
    "aerosol_optical_depth", "dust", "uv_index",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_rolling_mean_3h", "aqi_rolling_mean_24h", "aqi_change_rate_24h"
]


def get_supabase() -> Client:
    """Initialize and return the Supabase client."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env or environment variables."
        )

    _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _SUPABASE_CLIENT


def clean_dataframe_for_supabase(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format DataFrame for JSON serialization into Supabase PostgreSQL."""
    df_clean = df.copy()

    valid_cols = [c for c in SUPABASE_FEATURE_COLUMNS if c in df_clean.columns]
    df_clean = df_clean[valid_cols]

    if "datetime" in df_clean.columns:
        df_clean["datetime"] = pd.to_datetime(df_clean["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    df_clean = df_clean.replace({np.nan: None, np.inf: None, -np.inf: None})

    return df_clean.to_dict(orient="records")


def upsert_features(df: pd.DataFrame, batch_size: int = 500) -> int:
    """
    Batch upsert feature rows into the Supabase aqi_features table.
    Uses (city, datetime) composite primary key to update existing or insert new.
    """
    if df.empty:
        return 0

    client = get_supabase()
    records = clean_dataframe_for_supabase(df)
    total = len(records)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        response = (
            client.table(SUPABASE_TABLE_FEATURES)
            .upsert(batch, on_conflict="city,datetime")
            .execute()
        )
        inserted += len(batch)
        print(f"Upserted {inserted}/{total} rows into Supabase '{SUPABASE_TABLE_FEATURES}' table.")

    return inserted


def fetch_features(
    city: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Query feature records from Supabase table with automatic pagination
    to handle PostgREST's 1,000 row per request default limit.
    """
    client = get_supabase()
    page_size = 1000
    all_rows: List[Dict[str, Any]] = []
    offset = 0

    while True:
        query = client.table(SUPABASE_TABLE_FEATURES).select("*")

        if city:
            query = query.eq("city", city)
        if start_time:
            query = query.gte("datetime", start_time.strftime("%Y-%m-%dT%H:%M:%SZ"))
        if end_time:
            query = query.lte("datetime", end_time.strftime("%Y-%m-%dT%H:%M:%SZ"))

        query = query.order("datetime", desc=False)
        query = query.range(offset, offset + page_size - 1)

        response = query.execute()
        batch_data = response.data or []

        if not batch_data:
            break

        all_rows.extend(batch_data)
        offset += len(batch_data)

        if len(batch_data) < page_size:
            break

        if limit and len(all_rows) >= limit:
            all_rows = all_rows[:limit]
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    if "created_at" in df.columns:
        df = df.drop(columns=["created_at"])

    return df.sort_values(["city", "datetime"]).reset_index(drop=True)


def ensure_storage_bucket():
    """Ensure the model-registry storage bucket exists in Supabase."""
    client = get_supabase()
    try:
        buckets = client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        if SUPABASE_BUCKET_MODELS not in bucket_names:
            client.storage.create_bucket(SUPABASE_BUCKET_MODELS, options={"public": True})
    except Exception as e:
        print(f"Note: Storage bucket check: {e}")


def save_model_to_supabase(
    model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    feature_names: List[str],
    shap_summary: Optional[Dict[str, Any]] = None,
    extra_files: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save model artifact files to Supabase Storage and register metadata in model_registry table.
    Also creates local backup in './models/'.
    """
    client = get_supabase()
    ensure_storage_bucket()

    resp = (
        client.table(SUPABASE_TABLE_MODELS)
        .select("version")
        .eq("model_name", model_name)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    latest_version = resp.data[0]["version"] if resp.data else 0
    new_version = latest_version + 1

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    storage_folder = f"{model_name}/v{new_version}_{timestamp_str}"

    local_model_dir = Path("models") / f"{model_name}_v{new_version}"
    local_model_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts locally first
    is_keras = hasattr(model, "save") and not hasattr(model, "predict_proba")
    if extra_files and "scaler" in extra_files and extra_files["scaler"] is not None:
        joblib.dump(extra_files["scaler"], local_model_dir / "scaler.pkl")

    joblib.dump(feature_names, local_model_dir / "feature_names.pkl")

    if is_keras:
        model.save(local_model_dir / "model.keras")
    else:
        joblib.dump(model, local_model_dir / "model.pkl")

    if shap_summary:
        joblib.dump(shap_summary, local_model_dir / "shap_summary.pkl")

    uploaded_paths = []
    for file_path in local_model_dir.glob("*"):
        remote_path = f"{storage_folder}/{file_path.name}"
        try:
            with open(file_path, "rb") as f:
                client.storage.from_(SUPABASE_BUCKET_MODELS).upload(
                    path=remote_path,
                    file=f,
                    file_options={"cache-control": "3600", "upsert": "true"},
                )
            uploaded_paths.append(remote_path)
        except Exception as e:
            print(f"Warning: could not upload {file_path.name} to Supabase Storage ({e}). Local backup retained.")

    registry_entry = {
        "model_name": model_name,
        "version": new_version,
        "avg_rmse": float(metrics.get("avg_rmse", 0.0)),
        "metrics": metrics,
        "feature_names": feature_names,
        "shap_summary": shap_summary or {},
        "storage_path": storage_folder,
        "is_active": True,
    }

    client.table(SUPABASE_TABLE_MODELS).insert(registry_entry).execute()
    print(f"Registered model '{model_name}' version {new_version} in Supabase with avg_rmse={metrics.get('avg_rmse')}")
    return registry_entry


def load_latest_model(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the best active model from Supabase Storage / local cache.
    """
    client = get_supabase()

    query = client.table(SUPABASE_TABLE_MODELS).select("*").eq("is_active", True)
    if model_name:
        query = query.eq("model_name", model_name)
    query = query.order("avg_rmse", desc=False).order("version", desc=True).limit(1)

    resp = query.execute()
    if not resp.data:
        local_dirs = sorted(Path("models").glob("*_v*"))
        if not local_dirs:
            raise RuntimeError("No model found in Supabase model_registry or local models directory.")
        latest_dir = local_dirs[-1]
        return _load_from_local_dir(latest_dir, name="local_fallback")

    record = resp.data[0]
    storage_folder = record.get("storage_path")
    local_dir = Path("models") / f"{record['model_name']}_v{record['version']}"
    local_dir.mkdir(parents=True, exist_ok=True)

    if storage_folder:
        try:
            files = client.storage.from_(SUPABASE_BUCKET_MODELS).list(storage_folder)
            for file_info in files:
                file_name = file_info["name"]
                dest_path = local_dir / file_name
                if not dest_path.exists():
                    remote_file_path = f"{storage_folder}/{file_name}"
                    res = client.storage.from_(SUPABASE_BUCKET_MODELS).download(remote_file_path)
                    with open(dest_path, "wb") as f:
                        f.write(res)
        except Exception as e:
            print(f"Notice: Loading from local files if available: {e}")

    return _load_from_local_dir(local_dir, name=f"{record['model_name']} v{record['version']}", record=record)


def _load_from_local_dir(
    local_dir: Path,
    name: str = "",
    record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load model binaries and helper objects from a local directory."""
    feature_names_path = local_dir / "feature_names.pkl"
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
    elif record and "feature_names" in record:
        feature_names = record["feature_names"]
    else:
        feature_names = []

    model_pkl = local_dir / "model.pkl"
    model_keras = local_dir / "model.keras"
    scaler_path = local_dir / "scaler.pkl"
    shap_path = local_dir / "shap_summary.pkl"

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    shap_summary = joblib.load(shap_path) if shap_path.exists() else (record.get("shap_summary") if record else None)

    if model_pkl.exists():
        model = joblib.load(model_pkl)
        model_type = "sklearn"
    elif model_keras.exists():
        from tensorflow.keras.models import load_model

        model = load_model(model_keras)
        model_type = "keras"
    else:
        raise RuntimeError(f"No valid model file (model.pkl / model.keras) found in {local_dir}")

    return {
        "model": model,
        "model_type": model_type,
        "scaler": scaler,
        "feature_names": feature_names,
        "shap_summary": shap_summary,
        "model_name": name or local_dir.name,
        "metrics": record.get("metrics") if record else {},
    }
