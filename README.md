# 🌍 Pakistan Air Quality Index (AQI) 3-Day Forecast System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pakistan-aqi-predictor.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost_2.0+-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Supabase](https://img.shields.io/badge/Database-Supabase_PostgreSQL-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-Automated_Pipelines-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Machine Learning and MLOps platform that predicts **US EPA Air Quality Index (AQI)** across Pakistan's major urban centers (**Karachi**, **Lahore**, and **Islamabad**) for **3 continuous forecast horizons**:
* 🔮 **+24 Hours** (Next Day / Day 1 Ahead)
* 🔮 **+48 Hours** (Day 2 Ahead)
* 🔮 **+72 Hours** (Day 3 Ahead)

Powered by **84 atmospheric physics and lag features**, a decoupled **Multi-Horizon XGBoost** engine, **Supabase PostgreSQL** cloud warehouse with in-database model serialization, and autonomous **GitHub Actions** CI/CD workflows.

---

## 🌐 Live Production Application

🔗 **Access the Dashboard:** [https://pakistan-aqi-predictor.streamlit.app](https://pakistan-aqi-predictor.streamlit.app)

---

## 🏛️ System Architecture

```
                                  [ Open-Meteo & Copernicus CAMS APIs ]
                                                    │
                                     (Hourly Atmospheric Telemetry)
                                                    │
                                                    ▼
                       ┌────────────────────────────────────────────────────────┐
                       │           GitHub Actions CI/CD Automation              │
                       ├────────────────────────────────────────────────────────┤
                       │  • feature_pipeline.yml  (Hourly at minute 15)         │
                       │  • training_pipeline.yml (Daily at 00:00 UTC)          │
                       └───────────────────┬────────────────┬───────────────────┘
                                           │                │
                        Idempotent Feature │                │ Decoupled Model Retraining
                           Ingestion (84)  │                │ & Tree SHAP Extraction
                                           ▼                ▼
                       ┌────────────────────────────────────────────────────────┐
                       │          Supabase Cloud Warehouse (PostgreSQL)         │
                       ├────────────────────────────────────────────────────────┤
                       │  • Table: aqi_features (28,400+ rows, B-Tree indexed)   │
                       │  • Table: model_registry (Metrics, SHAP JSON,          │
                       │            compressed Base64 model binary payload)     │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                                     Instant PostgREST Deserialization
                                                   │
                                                   ▼
                       ┌────────────────────────────────────────────────────────┐
                       │           Streamlit Production Web Dashboard           │
                       │        (Dark Glassmorphic UI, Plotly Analytics)        │
                       └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Machine Learning** | `XGBoost 2.0+`, `scikit-learn`, `joblib` | Decoupled Multi-Horizon Gradient Boosted Decision Trees |
| **Model Interpretability** | `SHAP 0.44+`, XGBoost C++ `pred_contribs` | Game-theoretic global feature attribution & local waterfall explanations |
| **Data Ingestion** | `requests`, Open-Meteo Air Quality & Weather APIs | Harvesting ECMWF CAMS chemical reanalysis & ERA5 numerical weather data |
| **Data Warehouse** | `Supabase` (`supabase-py`, PostgreSQL) | Managed relational feature store, composite indexing & compressed binary storage |
| **CI/CD & Automation** | `GitHub Actions` | Serverless hourly feature pipelines & daily automated retraining |
| **User Interface** | `Streamlit 1.30+`, `Plotly 5.18+`, HTML5/CSS3 | Dark glassmorphism dashboard, EPA health advisories, and interactive telemetry |
| **Core Utilities** | `pandas`, `numpy`, `python-dotenv`, `pyarrow` | Data manipulation, cyclic encodings, and numerical operations |

---

## 💡 Architectural Decision: Why Supabase Instead of Hopsworks?

During system design, both **Hopsworks Feature Store** and **Supabase PostgreSQL** were evaluated. Supabase was chosen as the optimal production platform for five decisive architectural reasons:

| Evaluation Vector | Hopsworks Feature Store | **Supabase PostgreSQL (Our Choice)** |
| :--- | :--- | :--- |
| **Infrastructure Overhead** | Heavyweight Java/Kubernetes cluster; requires Kafka, RonDB, Hive, and Docker daemons. | **Managed Serverless PostgreSQL**: Zero cluster orchestration; instant provisioning, auto-scaling, and high availability. |
| **Model Persistence** | Requires separate S3/GCS bucket configuration, IAM roles, and multi-part client integrations. | **Universal In-Database Persistence**: Compressed Base64 model binaries stored directly in `model_registry.metrics['model_payload']` for instant zero-dependency deserialization (<100ms). |
| **Query Latency & Indexing** | Optimized for offline batch Parquet dumps; high cold-start latency for single-row live inference. | **Sub-millisecond latency**: Composite B-Tree index on `(city, datetime DESC)` enables instant live telemetry slicing for continuous lag engineering. |
| **CI/CD Runner Compatibility** | Large client SDKs with heavy dependencies causing build timeouts in lightweight serverless runners. | **Lightweight PostgREST client**: Connects in milliseconds across GitHub Actions runners and Streamlit Cloud with zero bloat. |
| **Operational Maintenance** | Resource-intensive; high idle cloud costs and cluster maintenance overhead. | **Complete ACID Integrity**: Full relational SQL power, foreign keys, automated backups, and cost-effective standard tiers. |

---

## 🔬 Atmospheric Feature Engineering (84 Features)

The pipeline converts raw weather and pollutant readings into **84 domain physics features**:

1. **Atmospheric Physics Engine**:
   * **Ventilation Index ($	ext{VI}$)**: $	ext{VI} = 	ext{Boundary Layer Height} 	imes 	ext{Wind Speed}_{10	ext{m}}$ (quantifies atmospheric dilution volume).
   * **Cartesian Wind Decomposition**: $U = -	ext{Speed} \cdot \sin(	heta)$ (zonal) and $V = -	ext{Speed} \cdot \cos(	heta)$ (meridional) to resolve circular angular discontinuity ($0^\circ = 360^\circ$).
   * **Dew Point Depression**: $T - T_{	ext{dew}}$ (measures relative saturation and hygroscopic particulate swelling).
   * **Inversion Proxy**: $T_{2	ext{m}} / (	ext{Solar Radiation} + 1.0)$ (identifies ground thermal inversion caps).
   * **Effective Solar Irradiance**: Direct radiation attenuated by cloud cover (photochemical ozone catalyst).
2. **Multi-Scale Historical Lags**:
   * Short-term: $1	ext{h}, 2	ext{h}, 3	ext{h}, 6	ext{h}, 12	ext{h}$ lags for AQI, $	ext{PM}_{2.5}$, and $	ext{PM}_{10}$.
   * Diurnal: $24	ext{h}, 48	ext{h}, 72	ext{h}$ lags for AQI, temperature, and wind speed.
3. **Rolling Volatility & Stability**:
   * 3h, 6h, 12h, 24h, 48h rolling means.
   * 24h rolling standard deviation, minimum, and maximum.
   * 24h rate of change acceleration: $(AQI_t - AQI_{t-24}) / (AQI_{t-24} + 1.0)$.
4. **Cyclic Temporal & Spatial Encodings**:
   * $\sin/\cos$ diurnal ($24	ext{h}$), weekly ($7	ext{d}$), and seasonal ($12	ext{m}$) cycles.
   * Binary flags: `is_weekend`, `is_rush_hour`.
   * One-hot spatial microclimate identifiers: `city_Karachi`, `city_Lahore`, `city_Islamabad`.

---

## 📊 Model Performance & Benchmarks

Benchmarked strictly on **4,191 out-of-time chronological test hours** across all three cities against a standard **Naïve Persistence Baseline** (which assumes tomorrow's air quality equals today's):

### Multi-Horizon Benchmark Summary:
| Horizon | Naïve Persistence MAE | **XGBoost MAE** | MAE Gain | Persistence RMSE | **XGBoost RMSE** | **Skill Score (RMSE Gain)** | Persistence $R^2$ | **XGBoost $R^2$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+24h (Day 1)** | 29.99 | **12.52** | **+58.3%** 🚀 | 40.25 | **19.81** | **+50.8%** 🏆 | 0.070 | **0.775** 🌟 |
| **+48h (Day 2)** | 30.62 | **17.63** | **+42.4%** 🚀 | 40.27 | **26.29** | **+34.7%** 🏆 | 0.075 | **0.606** 🌟 |
| **+72h (Day 3)** | 30.66 | **19.38** | **+36.8%** 🚀 | 40.33 | **28.60** | **+29.1%** 🏆 | 0.077 | **0.536** 🌟 |
| **3-Day Avg** | 30.42 | **16.51** | **+45.8%** | 40.28 | **24.90** | **+38.2%** | 0.074 | **0.639** |

### Per-City Breakdown (+24h Horizon):
* **Karachi** (Coastal): MAE = **3.91**, RMSE = **5.02**, $R^2$ = **0.663** (Skill Score: **+88.5%** 🏆)
* **Islamabad** (Foothills): MAE = **14.37**, RMSE = **19.11**, $R^2$ = **0.583** (Skill Score: **+45.4%**)
* **Lahore** (Smog Basin): MAE = **19.27**, RMSE = **28.05**, $R^2$ = **0.402** (Skill Score: **+32.6%**)

---

## 💻 Local Installation & Setup

### 1. Prerequisites
* Python 3.11+
* Git
* A free [Supabase](https://supabase.com) account

### 2. Clone the Repository
```bash
git clone https://github.com/IbadTahir/10Pearls_AQI_Predictor.git
cd 10Pearls_AQI_Predictor
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the project root:
```ini
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_KEY="your-supabase-service-or-anon-key"
SUPABASE_TABLE_FEATURES="aqi_features"
SUPABASE_TABLE_MODELS="model_registry"
```

### 6. Initialize Database Schema (Supabase SQL Editor)
Execute the DDL in [`supabase_schema.sql`](supabase_schema.sql) in your Supabase SQL Editor to create tables with composite indices:
* `aqi_features` (table with composite primary key `(city, datetime)`)
* `model_registry` (model metadata, SHAP summary, and binary payload)

### 7. Historical Backfill (1 Year of Continuous Data)
```bash
python historical_backfill.py
```

### 8. Train the Champion Model & Register to Supabase
```bash
python trainingpipeline.py
```

### 9. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚡ Automated CI/CD Pipelines (GitHub Actions)

The project includes two autonomous serverless workflows located in `.github/workflows/`:

1. **Hourly Feature Ingestion (`feature_pipeline.yml`)**:
   * **Trigger**: Scheduled cron at minute 15 of every hour (`15 * * * *`).
   * **Action**: Fetches live atmospheric telemetry for Karachi, Lahore, and Islamabad, engineers 84 features, and performs idempotent upserts into Supabase.
2. **Daily Automated Retraining (`training_pipeline.yml`)**:
   * **Trigger**: Scheduled cron every 24 hours at midnight UTC (`0 0 * * *`).
   * **Action**: Pulls historical features from Supabase, retrains `MultiHorizonXGBoost`, validates against persistence baselines, extracts updated Tree SHAP values, compresses the model payload, and promotes the champion model in the database.

Both workflows can also be triggered manually on demand via `workflow_dispatch`.

---

## 📁 Repository Structure

```
.
├── .github/workflows/
│   ├── feature_pipeline.yml          # Hourly live ingestion workflow (cron: 15 * * * *)
│   └── training_pipeline.yml         # Daily model retraining workflow (cron: 0 0 * * *)
├── app.py                            # Streamlit web dashboard (5 tabs, glassmorphic UI)
├── config.py                         # City coordinates, API endpoints, and Supabase secrets
├── feature_engineering.py            # Physics calculations & 84-feature generator
├── historical_backfill.py            # 1-year historical dataset harvesting script
├── model_architecture.py             # MultiHorizonXGBoost decoupled container
├── requirements.txt                  # Python production dependencies
├── supabase_client.py                # PostgREST wrapper & Base64 model deserializer
├── supabase_schema.sql               # PostgreSQL table schemas and DDL
├── trainingpipeline.py               # Time-series split, training, and SHAP pipeline
└── README.md                         # Project documentation
```

---

## 📄 License & Attribution

This project is distributed under the MIT License. See [LICENSE](LICENSE) for more details.

* **Data Sourcing**: Open-Meteo Air Quality & Weather Reanalysis APIs (ECMWF CAMS & ERA5).
* **Air Quality Standards**: United States Environmental Protection Agency (US EPA) Air Quality Index Breakpoints.
