# 🌍 Air Quality Index (AQI) 3-Day Forecasting System for Pakistan
## Comprehensive End-to-End Technical Project Report

---

### 📋 Executive Summary
Air pollution in major urban centers of Pakistan—particularly **Karachi** (coastal megacity), **Lahore** (Indus plain smog basin), and **Islamabad** (Himalayan foothills)—presents significant public health and economic challenges. Rapid fluctuations in atmospheric boundary layers, wind stagnation, temperature inversions, and transboundary particulate matter necessitate high-resolution, predictive meteorological intelligence.

This project delivers a production-grade, end-to-end Machine Learning and MLOps system that forecasts Air Quality Index (US EPA standard) across three target cities for **+24h (Day 1)**, **+48h (Day 2)**, and **+72h (Day 3)** horizons. The architecture automates hourly live data ingestion, continuous feature engineering, daily model retraining, explainable AI (SHAP), and real-time visualization via a cloud-synchronized Streamlit dashboard.

---

## 1. Data Sourcing & Ingestion Architecture

### 1.1 Data Sources & Frequency
* **Historical Reanalysis Data**: Sourced from the **Open-Meteo Air Quality & Historical Weather APIs**, integrating the **Copernicus Atmosphere Monitoring Service (ECMWF CAMS)** and global atmospheric reanalysis models.
* **Temporal Coverage**: **1 full year (12 months / 8,760+ hourly steps per city)**, providing over **28,000+ continuous hourly records**.
* **Temporal Resolution**: Continuous **1-hour sampling frequency** ($24 \text{ observations/day}$).
* **Target Cities & Geographic Microclimates**:
  * **Karachi** ($24.8607^\circ\text{N}, 67.0011^\circ\text{E}$): Coastal marine boundary layer, high humidity, sea breeze dispersion.
  * **Lahore** ($31.5204^\circ\text{N}, 74.3587^\circ\text{E}$): Continental inland plain, severe winter smog, agricultural biomass burning, low boundary layer height.
  * **Islamabad** ($33.6844^\circ\text{N}, 73.0479^\circ\text{E}$): Sub-Himalayan foothill valley, terrain-trapped particulates, orographic precipitation washing.

### 1.2 Ingested Atmospheric & Pollutant Parameters
1. **Primary Criteria Pollutants**: $\text{PM}_{2.5}$, $\text{PM}_{10}$, Carbon Monoxide ($\text{CO}$), Nitrogen Dioxide ($\text{NO}_2$), Sulphur Dioxide ($\text{SO}_2$), Ground-Level Ozone ($\text{O}_3$).
2. **Standard Target**: United States Environmental Protection Agency ($\text{US AQI}$, $0–500$ scale).
3. **Core Meteorological Variables**: 2-meter Temperature ($\text{}^\circ\text{C}$), Relative Humidity ($\%$), Surface Pressure ($\text{hPa}$), 10-meter Wind Speed ($\text{km/h}$), Wind Direction ($\text{degrees}$).
4. **Boundary & Optical Dynamics**: Boundary Layer Height ($\text{m}$), Aerosol Optical Depth ($\text{AOD}_{550}$), Dust Concentration ($\mu\text{g/m}^3$), Direct Shortwave Radiation ($\text{W/m}^2$), Precipitation ($\text{mm}$), Cloud Cover ($\%$), Dew Point Temperature ($\text{}^\circ\text{C}$), UV Index.

---

## 2. Atmospheric Feature Engineering (84 Domain Features)

Raw meteorological variables alone are insufficient for air quality modeling due to non-linear physical interactions. The feature engineering pipeline derives **84 physical, temporal, and interaction features**:

```
Raw Telemetry (Weather + Pollutants)
    │
    ├──► Domain Physics Engine
    │     ├── Ventilation Index (BLH × Wind Speed)
    │     ├── Wind Vector Decomposition (U = -s·sin(θ), V = -s·cos(θ))
    │     ├── Dew Point Depression & Hygroscopic Aerosol Growth
    │     └── Radiative Inversion Proxies (Temp / Radiation)
    │
    ├──► Temporal & Cyclic Encodings
    │     ├── Diurnal Cyclics: sin/cos(Hour), sin/cos(Day of Week)
    │     ├── Seasonal Cyclics: sin/cos(Month)
    │     └── Binary Flags: is_weekend, is_rush_hour
    │
    ├──► Multi-Scale Historical Lags & Rolling Aggregations
    │     ├── AQI Lags: lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_48h, lag_72h
    │     ├── Rolling Means: rolling_mean_3h, rolling_6h, rolling_12h, rolling_24h, rolling_48h
    │     ├── Rolling Volatility: rolling_std_24h, rolling_min_24h, rolling_max_24h
    │     └── Delta Momentum: aqi_change_rate_24h, pm2_5_delta_24h
    │
    └──► City-Distributed Microclimate Interactions
          └── One-Hot Identifiers (city_Karachi, city_Lahore, city_Islamabad)
```

---

## 3. Model Architecture & Evaluation

### 3.1 Training Methodology & Split Strategy
* **Split Strategy**: **Time-Series Chronological Split** (85% Training, 15% Out-of-Time Testing). Random/K-Fold shuffling was strictly avoided to prevent temporal data leakage and lookahead bias.
* **Architecture**: **Multi-Horizon Gradient Boosted Trees (`MultiHorizonXGBoost`)** containing independent regressors tuned for each forecasting horizon:
  * $H_{24}$: 1-Day Ahead (+24 hours)
  * $H_{48}$: 2-Days Ahead (+48 hours)
  * $H_{72}$: 3-Days Ahead (+72 hours)
* **Hyperparameters**: `n_estimators=450`, `learning_rate=0.03`, `max_depth=6`, `subsample=0.85`, `colsample_bytree=0.80`, `reg_alpha=0.1`, `reg_lambda=1.0`.

### 3.2 Quantitative Evaluation Metrics vs Baseline

| Forecast Horizon | Naïve Persistence MAE | XGBoost MAE | MAE Improvement | Naïve Persistence RMSE | XGBoost RMSE | Skill Score (RMSE Reduction) | Naïve Persistence $R^2$ | **XGBoost $R^2$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **+24h (Day 1)** | 24.18 | **13.08** | **+45.9%** 🚀 | 36.81 | **20.91** | **+43.2%** 🏆 | 0.229 | **0.751** 🌟 |
| **+48h (Day 2)** | 28.52 | **18.17** | **+36.3%** 🚀 | 38.94 | **25.37** | **+34.8%** 🏆 | 0.138 | **0.598** 🌟 |
| **+72h (Day 3)** | 30.66 | **18.80** | **+38.7%** 🚀 | 40.33 | **25.92** | **+35.7%** 🏆 | 0.077 | **0.569** 🌟 |
| **3-Day Overall Average** | **27.79** | **16.68** | **+40.3%** | **38.69** | **24.06** | **+37.9%** | **0.148** | **0.639** |

### 3.3 Per-City Performance Breakdown (+24h Horizon)

| City | Test AQI Range | XGBoost MAE | XGBoost RMSE | XGBoost $R^2$ | Skill Score vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Karachi** | $61 - 109$ (Moderate) | **4.22** | **5.47** | **0.614** | **+88.5%** 🏆 |
| **Islamabad** | $61 - 208$ (Sensitive) | **14.85** | **19.86** | **0.554** | **+45.4%** |
| **Lahore** | $74 - 364$ (Smog/Hazardous) | **20.54** | **30.27** | **0.305** | **+32.6%** |

---

## 4. Model Selection Rationale: Why XGBoost?

During exploratory modeling, multiple candidate algorithms were tested (Linear/Ridge Regression, Random Forest, Multi-Layer Perceptrons, and Gradient Boosting). **XGBoost Regressor was selected as the champion model for the following reasons:**

1. **Non-Linear Meteorological Thresholds**: Atmospheric boundary layers and particulate concentrations exhibit sharp thresholds (e.g. wind speeds below $5\text{ km/h}$ create sudden stagnation spikes). Linear and Ridge models failed because they assume linear relationships.
2. **Handling Highly Collinear Atmospheric Variables**: Temperature, solar radiation, humidity, and dew point are correlated. Tree-based partitioning naturally selects the best split without suffering from matrix instability.
3. **Second-Order Loss Optimization (Newton-Raphson)**: XGBoost uses both first-order gradients and second-order Hessian approximations, achieving faster convergence and tighter predictions than standard Random Forests.
4. **Built-in Regularization**: Explicit $L_1$ (`reg_alpha`) and $L_2$ (`reg_lambda`) penalties prevent the tree ensemble from overfitting high-volatility winter smog anomalies.
5. **Exact Tree SHAP Interpretability**: Compatible with `shap.TreeExplainer`, providing exact, mathematically consistent polynomial-time feature attributions for human-in-the-loop decision making.

---

## 5. Explainable AI: SHAP (SHapley Additive exPlanations)

To ensure transparency and clinical trust, the model integrates **Game-Theoretic Tree SHAP** explanations:

* **Top Global AQI Predictive Drivers**:
  1. `us_aqi` (Current baseline concentration) — $\text{SHAP} \approx 19.77$
  2. `pm2_5` (Fine inhalable particulate matter) — $\text{SHAP} \approx 5.02$
  3. `pm2_5_rolling_mean_24h` (Sustained 24h particulate load) — $\text{SHAP} \approx 3.85$
  4. `pm10` (Coarse respirable dust) — $\text{SHAP} \approx 1.04$
  5. `day` & `month` (Seasonal cyclic variations / crop burning season) — $\text{SHAP} \approx 0.70$
  6. `dust` & `aerosol_optical_depth` — $\text{SHAP} \approx 0.65$
* **Local Waterfall Explanations**: The Streamlit interface dynamically computes local SHAP waterfalls for the active city, breaking down exactly how today's wind speed, humidity, and lag concentrations shift tomorrow's forecast up or down.

---

## 6. Cloud Infrastructure & Supabase Warehouse

All operational data and model binaries are managed via a cloud-native **Supabase PostgreSQL** instance:

```
                  ┌──────────────────────────────────────────────┐
                  │           SUPABASE POSTGRESQL                │
                  │                                              │
                  │   ┌──────────────────────────────────────┐   │
                  │   │   aqi_features (Table)               │   │
                  │   │   - Primary Key: (city, datetime)    │   │
                  │   │   - 28,000+ hourly atmospheric rows  │   │
                  │   │   - Optimized B-Tree Indices         │   │
                  │   └──────────────────────────────────────┘   │
                  │                                              │
                  │   ┌──────────────────────────────────────┐   │
                  │   │   model_registry (Table)             │   │
                  │   │   - Versioning (v1, v2, ... v11)     │   │
                  │   │   - avg_rmse, metrics JSONB          │   │
                  │   │   - shap_summary (Top 25 drivers)    │   │
                  │   │   - model_payload (Compressed Base64)│   │
                  │   └──────────────────────────────────────┘   │
                  └──────────────────────────────────────────────┘
```

* **Universal Database Model Persistence**: Model objects are serialized, compressed via `zlib/joblib`, and stored directly in `model_registry.metrics['model_payload']`. This guarantees that any client, GitHub Actions runner, or web server can download and deserialize the exact model artifact in milliseconds without file-system dependencies.

---

## 7. CI/CD & Automated GitHub Actions Workflows

The system runs autonomously in production using GitHub Actions:

```
[Cron: 15 * * * *] (Hourly) ──► feature_pipeline.yml ──► Fetches Open-Meteo ──► Upserts to Supabase aqi_features
[Cron: 0 0 * * *]  (Daily)  ──► training_pipeline.yml ──► Retrains XGBoost ───► Registers new version & SHAP
```

1. **Hourly Feature Ingestion (`.github/workflows/feature_pipeline.yml`)**:
   * Runs automatically at minute 15 of every hour (`15 * * * *`).
   * Fetches latest observations from Open-Meteo and upserts new rows into `aqi_features`.
2. **Daily Model Retraining (`.github/workflows/training_pipeline.yml`)**:
   * Runs automatically every 24 hours at midnight UTC (`0 0 * * *`).
   * Retrains the Multi-Horizon XGBoost ensemble on the expanding database history, evaluates validation benchmarks, generates SHAP feature importances, and promotes the champion model to `model_registry`.

---

## 8. Dashboard Architecture & User Interface

The web interface is built with **Streamlit** using a dark glassmorphic design language:

* **Header Controls**: Synchronized target city selector (`Karachi`, `Lahore`, `Islamabad`) and cache refresh.
* **EPA Health Action Banners**: Color-coded alert cards (Good, Moderate, Sensitive, Unhealthy, Very Unhealthy, Hazardous) with actionable medical advice for sensitive populations.
* **4-Card KPI Ribbon**: Displays Observed AQI alongside +24h, +48h, and +72h delta changes.
* **5 Interactive Tabs**:
  * **Tab 1: 📈 3-Day Forecast & Continuous Trends** (Interactive Plotly chart linking 7-day observed history to the 3-day future trajectory with EPA threshold lines).
  * **Tab 2: 🧠 Model Explainability (SHAP)** (City-specific waterfall attribution + top 25 global feature drivers).
  * **Tab 3: 🔬 Exploratory Data Analysis (EDA)** (7-day individual pollutant breakdown, meteorological correlation heatmap, and 24h diurnal pattern).
  * **Tab 4: 📊 Model Performance & Benchmarks** (Evaluation table vs Naïve Persistence, RMSE/MAE/$R^2$/Skill Scores, and per-city breakdown).
  * **Tab 5: 📋 Live Feature Store Inspector** (Real-time searchable tabular view queried directly from Supabase).

---

## 9. Conclusion & Project Deliverables

This project successfully operationalizes an enterprise-grade, explainable, and fully automated Air Quality Index forecasting platform for Pakistan. By integrating physical atmospheric dynamics, gradient boosted tree ensembles, automated cloud retraining, and transparent game-theoretic explainability, the system delivers high-precision predictions that significantly outperform traditional persistence baselines across all forecasting horizons.

* **Repository**: [https://github.com/IbadTahir/10Pearls_AQI_Predictor](https://github.com/IbadTahir/10Pearls_AQI_Predictor)
* **Status**: Complete, Verified, and Operational in Production.
