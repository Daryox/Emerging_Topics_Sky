# Forecasting Airport Congestion at Madrid-Barajas

**University Group Project** - Predicting daily airport congestion pressure at Adolfo Suarez Madrid-Barajas Airport (LEMD) using flight, weather, and calendar signals.

## Overview

The full pipeline runs in a single notebook: `notebooks/airport_congestion_forecasting.ipynb`

It covers data collection, EDA, feature engineering, time series modelling, machine learning, clustering, neural networks, and geospatial analysis — producing a trained model with R² = 0.96 on 498 unseen test days.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register Jupyter kernel
python -m ipykernel install --user --name airport-congestion --display-name "Airport Congestion (Python 3.10)"

# 4. Open the notebook
jupyter notebook notebooks/airport_congestion_forecasting.ipynb
```

## Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| Eurocontrol IFR Traffic | Excel file | Daily flight movements 2017–2026 (primary) |
| Open-Meteo | REST API | 9 years of hourly weather, aggregated to daily |
| Nager.Date | REST API | Spanish public holidays |

## Pipeline

```
Part 1  Data Collection
        Eurocontrol IFR movements (3,346 daily records, Jan 2017 – Feb 2026)
        Open-Meteo weather (temperature, wind, precipitation, cloud cover)
        Nager.Date Spanish holidays (99 national dates)

Part 2  EDA
        Temporal patterns (yearly, weekly, monthly)
        COVID structural break analysis
        Weather and correlation analysis

Part 3  Feature Engineering
        53 features: flight volumes, lag/rolling features (1d–365d),
        calendar cyclics (sin/cos), weather variables, holiday flags
        ACPS target: 0–100 congestion pressure score

Part 4  Train / Validation / Test Split (chronological 70/15/15)
        Train: Jan 2017 – Jun 2023
        Valid: Jun 2023 – Oct 2024
        Test:  Oct 2024 – Feb 2026

Part 5  Models
        SARIMAX(1,0,1)(1,0,1,7) with weather exogenous variables
        HistGradientBoosting Regressor (continuous ACPS)
        HistGradientBoosting Classifier (Low / Medium / High congestion)
        K-Means clustering (k=3, traffic pattern discovery)
        ANN / MLP Regressor (64-64 hidden layers)

Part 6  Results
        HGB Regressor: MAE=0.45, RMSE=0.69, R²=0.96 on test set
        HGB Classifier: 90.2% accuracy, no Low↔High misclassifications
        HGB outperforms SARIMAX, ANN, and all baselines
```

## Target Variable

**ACPS (Airport Congestion Pressure Score)** combines absolute traffic volume and relative pressure vs. typical day-of-week traffic, rescaled to 0–100:

```
ACPS = rescale(0.6 * z(total_movements) + 0.4 * z(pressure_ratio))
```

Congestion classes:
```
Low     ACPS < 69.3   (bottom 60%)
Medium  69.3 <= ACPS < 74.5   (60th–85th percentile)
High    ACPS >= 74.5  (top 15%)
```

## Project Structure

```
notebooks/           Single unified pipeline notebook
src/data/            Data fetcher modules
src/features/        Feature engineering modules
src/modeling/        Modelling modules (baselines, SARIMAX, tree models, evaluation)
src/visualization/   Visualization modules
config/              YAML configs (airports, paths, modelling params)
data/raw/            Cached API and source data
data/processed/      Train/valid/test parquet splits
outputs/figures/     20+ report-ready PNG figures
outputs/tables/      model_comparison, feature_importance, test_predictions
outputs/models/      hgb_regressor, hgb_classifier (joblib)
research/            Literature and context notes
report/              Report outline
presentation/        Slide outline
```