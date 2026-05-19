# Wildfire Evacuation Route & Risk Optimization

## Problem Statement

Wildfires in Los Angeles County pose a severe and growing threat to life and infrastructure. Emergency responders and city planners currently lack a systematic, data-driven way to identify *which* communities are most vulnerable before a fire ignites — and *which* roads should be prioritized for evacuation once one does.

This project addresses that gap through supervised machine learning. Given socioeconomic vulnerability data (CDC Social Vulnerability Index) and spatial infrastructure features (fire station proximity, road density, hydrant coverage), we train a classifier to assign each of the **2,493 LA County census tracts** one of four risk tiers:

## Proposed Solution

A Machine Learning project designed to identify high-risk communities and optimize evacuation routes during wildfire events in Los Angeles County.
The problems divide in 3 sub-problems 
1. Risk Classifier for Vulnerable Neighborhoods (this project)
2. Traffic Congestion
3. Route Optimization

The system produces an actionable, data-driven vulnerability assessment across all 2,493 LA County census tracts.

---

## ML Pipeline

### Features

**SVI Composite & Themes (from CDC)**
- `rpl_theme_1` — socioeconomic status
- `rpl_theme_2` — household characteristics
- `rpl_theme_3` — racial & ethnic minority status
- `rpl_theme_4` — housing type & transportation

**Engineered Spatial Features**
- `dist_fire_station_m` — distance from tract centroid to nearest fire station
- `dist_hospital_m` — distance from tract centroid to nearest hospital
- `hydrant_count` — DWP hydrants within tract polygon
- `hydrant_density` — hydrant count / tract area (km²)
- `road_length_m` — total drivable road length clipped to tract
- `road_density` — road length / tract area (km²)
- `tract_area_km2` — tract area

### Risk Classifier

- **Model:** Random Forest (`n_estimators=200`, `max_depth=10`, `class_weight=balanced`)
- **Target:** 3-class risk label — Low [0–0.25), Moderate [0.25–0.5), High [0.5–0.75), Critical [0.75–1.0]
- **Validation:** 5-fold cross-validation, 80/20 train-test split
- **Explainability:** SHAP feature importance

<!-- ### Evacuation Routing

- **Graph:** OSMnx drivable network filtered to motorway → tertiary roads (26,716 nodes, 61,375 edges), projected to EPSG:3310
- **Algorithm:** Dijkstra on `travel_time_s` edge weights
- **Risk penalty:** 5× weight multiplier applied to edges in high-risk tracts

--- -->

## Setup & Usage

### Run the full pipeline (recommended)

# Full pipeline: install deps → data processing → ML training
python run_pipeline.py

# Skip pip install if dependencies are already installed
python run_pipeline.py --skip-install

# Run only the data processing notebooks
python run_pipeline.py --only data

# Run only the ML pipeline notebooks
python run_pipeline.py --only ml
```

`run_pipeline.py` executes the following stages in order:

**Stage 1 — Data Processing** (`src/data_processing/`, run once):
```
CensusTract_SVI.ipynb
Escape_Route.ipynb
FireHydrants.ipynb
FireStations.ipynb
Historic_Wildfire.ipynb
Hospital.ipynb
```

**Stage 2 — ML Pipeline** (`notebooks/`):
```
1. exploration.ipynb         — understand the data
2. feature_engineering.ipynb — build features.geojson
3. model_trainning.ipynb     — train model, save risk_classifier.pkl
```

**Stage 3 — Outputs** (run manually after the pipeline):
```bash
python implementation/predict.py    # compare model to historic wildfire data
python implementation/simulate.py   # compare model to random spawn wildfire data
```

### View outputs
Open any of the generated HTML maps directly in a browser:
```
outputs/maps/01_risk_choropleth.html
outputs/maps/02_risk_with_infrastructure.html
```
