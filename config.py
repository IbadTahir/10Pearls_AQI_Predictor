import os
from dotenv import load_dotenv

load_dotenv()

CITIES = {
    "Karachi":   (24.8607, 67.0011),
    "Lahore":    (31.5497, 74.3436),
    "Islamabad": (33.6844, 73.0479),
}

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_HOURLY_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",          
    "us_aqi_pm2_5",
    "us_aqi_pm10",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
]

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "dew_point_2m",
    "shortwave_radiation",
    "boundary_layer_height",
]

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE_FEATURES = os.getenv("SUPABASE_TABLE_FEATURES", "aqi_features")
SUPABASE_TABLE_MODELS = os.getenv("SUPABASE_TABLE_MODELS", "model_registry")
SUPABASE_BUCKET_MODELS = os.getenv("SUPABASE_BUCKET_MODELS", "model-registry")