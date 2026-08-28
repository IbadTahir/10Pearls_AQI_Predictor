from __future__ import annotations

import sys
import pandas as pd
import requests

from config import (
    CITIES,
    AIR_QUALITY_URL,
    WEATHER_FORECAST_URL,
    AIR_QUALITY_HOURLY_VARS,
    WEATHER_HOURLY_VARS,
)
from feature_engineering import engineer_features
from supabase_client import upsert_features


def fetch_live_city(city: str, lat: float, lon: float, past_days: int = 4) -> pd.DataFrame:
    """Fetch recent hourly weather and air quality for a single city."""
    print(f"Fetching live data for {city}...")

    w_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "auto",
    }
    w_res = requests.get(WEATHER_FORECAST_URL, params=w_params, timeout=30)
    w_res.raise_for_status()
    w_data = w_res.json().get("hourly", {})

    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "auto",
    }
    aq_res = requests.get(AIR_QUALITY_URL, params=aq_params, timeout=30)
    aq_res.raise_for_status()
    aq_data = aq_res.json().get("hourly", {})

    w_df = pd.DataFrame(w_data)
    aq_df = pd.DataFrame(aq_data)

    if w_df.empty or aq_df.empty:
        return pd.DataFrame()

    w_df["datetime"] = pd.to_datetime(w_df["time"])
    aq_df["datetime"] = pd.to_datetime(aq_df["time"])

    w_df = w_df.drop(columns=["time"], errors="ignore")
    aq_df = aq_df.drop(columns=["time"], errors="ignore")

    merged = pd.merge(aq_df, w_df, on="datetime", how="inner")
    merged["city"] = city
    return merged


def main():
    all_frames = []
    for city, coords in CITIES.items():
        try:
            lat, lon = coords
            cdf = fetch_live_city(city, lat, lon, past_days=4)
            if not cdf.empty:
                all_frames.append(cdf)
        except Exception as e:
            print(f"Error fetching live data for {city}: {e}", file=sys.stderr)

    if not all_frames:
        raise RuntimeError("No city data fetched successfully. Check API availability/network.")

    raw = pd.concat(all_frames, ignore_index=True)
    featured = engineer_features(raw)

    valid_observed = featured.dropna(subset=["us_aqi", "aqi_lag_24h", "aqi_lag_48h"])

    recent_to_push = (
        valid_observed.sort_values("datetime")
        .groupby("city", as_index=False)
        .tail(24)
        .reset_index(drop=True)
    )

    print(f"Pushing {len(recent_to_push)} fresh hourly rows to Supabase.")
    upsert_features(recent_to_push, batch_size=100)
    print("Live feature pipeline completed successfully.")


if __name__ == "__main__":
    main()