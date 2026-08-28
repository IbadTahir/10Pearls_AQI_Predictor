from __future__ import annotations

import numpy as np
import pandas as pd

FLOAT_COLUMNS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "us_aqi", "us_aqi_pm2_5", "us_aqi_pm10",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m",
    "precipitation", "cloud_cover", "dew_point_2m",
    "shortwave_radiation", "boundary_layer_height",
    "aerosol_optical_depth", "dust", "uv_index",
]


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric columns to float64."""
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def add_time_and_cyclic_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Adds standard and cyclical sin/cos features from the datetime column."""
    dt = pd.to_datetime(df[datetime_col])
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physics-based derived features from raw weather variables."""
    eps = 1e-6  

    if "wind_direction_10m" in df.columns and "wind_speed_10m" in df.columns:
        wd_rad = np.radians(df["wind_direction_10m"].fillna(0))
        df["wind_u"] = -df["wind_speed_10m"].fillna(0) * np.sin(wd_rad)
        df["wind_v"] = -df["wind_speed_10m"].fillna(0) * np.cos(wd_rad)

    if "boundary_layer_height" in df.columns and "wind_speed_10m" in df.columns:
        blh = df["boundary_layer_height"].fillna(0)
        ws = df["wind_speed_10m"].fillna(0)
        df["ventilation_index"] = ws * blh


    if "temperature_2m" in df.columns and "dew_point_2m" in df.columns:
        df["dew_point_depression"] = df["temperature_2m"].fillna(0) - df["dew_point_2m"].fillna(0)

    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm25_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + eps)

    if "relative_humidity_2m" in df.columns:
        rh = df["relative_humidity_2m"].fillna(50).clip(0, 99.9)
        df["hygroscopic_factor"] = rh / (100.0 - rh + eps)

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    group_col: str = "city",
    datetime_col: str = "datetime",
    target_col: str = "us_aqi",
) -> pd.DataFrame:
    """
    Adds multi-lag features and rolling statistics computed PER CITY
    and strictly aligned on timestamp index.
    """
    df = df.sort_values([group_col, datetime_col]).copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    out_frames = []
    for city, g in df.groupby(group_col):
        g = g.set_index(datetime_col).sort_index()

        g["aqi_lag_1h"] = g[target_col].reindex(g.index - pd.Timedelta(hours=1)).values
        g["aqi_lag_24h"] = g[target_col].reindex(g.index - pd.Timedelta(hours=24)).values
        g["aqi_lag_48h"] = g[target_col].reindex(g.index - pd.Timedelta(hours=48)).values
        g["aqi_lag_72h"] = g[target_col].reindex(g.index - pd.Timedelta(hours=72)).values
        g["aqi_lag_168h"] = g[target_col].reindex(g.index - pd.Timedelta(hours=168)).values  # Weekly

        if "pm2_5" in g.columns:
            g["pm2_5_lag_24h"] = g["pm2_5"].reindex(g.index - pd.Timedelta(hours=24)).values
            g["pm2_5_lag_48h"] = g["pm2_5"].reindex(g.index - pd.Timedelta(hours=48)).values
        if "pm10" in g.columns:
            g["pm10_lag_24h"] = g["pm10"].reindex(g.index - pd.Timedelta(hours=24)).values

        for poll in ["nitrogen_dioxide", "sulphur_dioxide", "ozone", "carbon_monoxide"]:
            if poll in g.columns:
                g[f"{poll}_lag_24h"] = g[poll].reindex(g.index - pd.Timedelta(hours=24)).values

        g["aqi_rolling_mean_3h"] = g[target_col].rolling("3h", min_periods=1).mean()
        g["aqi_rolling_mean_6h"] = g[target_col].rolling("6h", min_periods=1).mean()
        g["aqi_rolling_mean_24h"] = g[target_col].rolling("24h", min_periods=1).mean()
        g["aqi_rolling_std_24h"] = g[target_col].rolling("24h", min_periods=1).std().fillna(0.0)
        g["aqi_rolling_min_24h"] = g[target_col].rolling("24h", min_periods=1).min()
        g["aqi_rolling_max_24h"] = g[target_col].rolling("24h", min_periods=1).max()

        if "pm2_5" in g.columns:
            g["pm2_5_rolling_mean_24h"] = g["pm2_5"].rolling("24h", min_periods=1).mean()

        if "boundary_layer_height" in g.columns:
            g["blh_rolling_mean_6h"] = g["boundary_layer_height"].rolling("6h", min_periods=1).mean()
            g["blh_rolling_min_24h"] = g["boundary_layer_height"].rolling("24h", min_periods=1).min()

        g["aqi_change_1h"] = g[target_col] - g["aqi_lag_1h"]
        g["aqi_change_rate_24h"] = g[target_col] - g["aqi_lag_24h"]

        if "surface_pressure" in g.columns:
            sp_lag_3h = g["surface_pressure"].reindex(g.index - pd.Timedelta(hours=3)).values
            g["pressure_tendency_3h"] = g["surface_pressure"] - pd.Series(sp_lag_3h, index=g.index)

        if "precipitation" in g.columns:
            g["precip_sum_6h"] = g["precipitation"].rolling("6h", min_periods=1).sum()
            g["precip_sum_24h"] = g["precipitation"].rolling("24h", min_periods=1).sum()

            g["rain_flag_6h"] = (g["precip_sum_6h"] > 0.1).astype(float)

        is_karachi = 1.0 if str(city).lower() == "karachi" else 0.0
        is_lahore = 1.0 if str(city).lower() == "lahore" else 0.0
        is_islamabad = 1.0 if str(city).lower() == "islamabad" else 0.0

        if "temperature_2m" in g.columns:
            g["temp_x_karachi"] = g["temperature_2m"] * is_karachi
            g["temp_x_lahore"] = g["temperature_2m"] * is_lahore
            g["temp_x_islamabad"] = g["temperature_2m"] * is_islamabad

        if "relative_humidity_2m" in g.columns:
            g["humidity_x_karachi"] = g["relative_humidity_2m"] * is_karachi
            g["humidity_x_lahore"] = g["relative_humidity_2m"] * is_lahore
            g["humidity_x_islamabad"] = g["relative_humidity_2m"] * is_islamabad

        if "surface_pressure" in g.columns:
            g["pressure_x_karachi"] = g["surface_pressure"] * is_karachi
            g["pressure_x_lahore"] = g["surface_pressure"] * is_lahore
            g["pressure_x_islamabad"] = g["surface_pressure"] * is_islamabad

        if "wind_speed_10m" in g.columns:
            g["wind_x_karachi"] = g["wind_speed_10m"] * is_karachi
            g["wind_x_lahore"] = g["wind_speed_10m"] * is_lahore
            g["wind_x_islamabad"] = g["wind_speed_10m"] * is_islamabad

        # City-specific boundary layer interactions
        if "boundary_layer_height" in g.columns:
            g["blh_x_karachi"] = g["boundary_layer_height"] * is_karachi
            g["blh_x_lahore"] = g["boundary_layer_height"] * is_lahore
            g["blh_x_islamabad"] = g["boundary_layer_height"] * is_islamabad

        out_frames.append(g.reset_index())

    return pd.concat(out_frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single unified feature engineering pipeline."""
    df = enforce_dtypes(df)
    df = add_time_and_cyclic_features(df)
    df = add_physics_features(df)
    df = add_lag_and_rolling_features(df)
    df = enforce_dtypes(df)
    return df