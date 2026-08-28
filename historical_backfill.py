"""
Historical backfill pipeline.

Pulls hourly air-quality (pollutants + real US AQI) and weather data from
Open-Meteo for every city in config.CITIES, over a given date range, merges
them, engineers features, and pushes the result into Supabase Data Warehouse.

Run:
    python historical_backfill.py --start 2025-08-01 --end 2026-08-06
"""

import argparse
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    AIR_QUALITY_HOURLY_VARS,
    AIR_QUALITY_URL,
    CITIES,
    WEATHER_ARCHIVE_URL,
    WEATHER_HOURLY_VARS,
)
from feature_engineering import engineer_features
from supabase_client import upsert_features

REQUEST_TIMEOUT = (10, 90)


def build_retry_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_air_quality(city: str, lat: float, lon: float, start: str, end: str, session: requests.Session) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "start_date": start,
        "end_date": end,
        "timezone": "auto",
    }
    r = session.get(AIR_QUALITY_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def fetch_weather(city: str, lat: float, lon: float, start: str, end: str, session: requests.Session) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "start_date": start,
        "end_date": end,
        "timezone": "auto",
    }
    r = session.get(WEATHER_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def backfill_city(city: str, lat: float, lon: float, start: str, end: str, session: requests.Session) -> pd.DataFrame:
    aq = fetch_air_quality(city, lat, lon, start, end, session)
    wx = fetch_weather(city, lat, lon, start, end, session)
    merged = pd.merge(aq, wx, on=["datetime", "city"], how="inner")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Backfill historical AQI and weather features into Supabase.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Skip Supabase push, save CSV instead")
    args = parser.parse_args()

    all_frames = []
    session = build_retry_session()
    for city, (lat, lon) in CITIES.items():
        print(f"Fetching {city} ({args.start} to {args.end})...")
        try:
            df = backfill_city(city, lat, lon, args.start, args.end, session)
            all_frames.append(df)
        except requests.RequestException as e:
            print(f"  FAILED for {city}: {e}")
        time.sleep(0.5)  # respectful to free API

    if not all_frames:
        raise RuntimeError("No city data fetched successfully. Check Open-Meteo availability.")

    raw = pd.concat(all_frames, ignore_index=True)
    print(f"Raw rows collected: {len(raw)}")

    featured = engineer_features(raw)
    print(f"Featured rows ready: {len(featured)}, columns: {list(featured.columns)}")

    if args.dry_run:
        featured.to_csv("historical_features.csv", index=False)
        print("Saved to historical_features.csv (dry run, not pushed to Supabase)")
    else:
        upsert_features(featured, batch_size=500)
        print("Successfully backfilled and upserted into Supabase.")


if __name__ == "__main__":
    main()