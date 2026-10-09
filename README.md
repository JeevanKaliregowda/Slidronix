# Slidronix: AI-Based Landslide Early Warning System

Slidronix is a geospatial machine-learning project focused on landslide risk assessment in the Western Ghats of India. It combines historical landslide inventories with terrain, satellite, soil and rainfall features to train and evaluate models and generate spatial risk predictions.

## Current Implementation Status

Implemented components include:

- Historical landslide inventory extraction and cleaning.
- Environmental feature preparation using terrain, satellite, soil and rainfall data.
- Integrated and spatially separated datasets for model development.
- Random Forest, MLP, CNN, rainfall Transformer and multimodal Fusion models.
- Model comparison and feature-importance outputs.
- Spatial risk predictions and a React-based GIS dashboard.

The existing dashboard loads predictions from a static JSON file. The complete real-time early-warning and emergency-response workflow is not yet implemented.

## Technology Stack

### Machine Learning and Geospatial Processing
- Python
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Rasterio and GeoPandas
- Copernicus DEM, MODIS/AppEEARS, SoilGrids and CHIRPS rainfall data

### Frontend
- React
- Vite
- Leaflet and React-Leaflet
- Recharts
- Lucide React

## Machine-Learning Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8154 | 0.8286 | 0.7949 | 0.8114 | 0.9093 |
| MLP | 0.8236 | 0.7891 | 0.8828 | 0.8333 | 0.8968 |
| CNN | 0.8124 | 0.7747 | 0.8831 | 0.8254 | 0.8815 |
| Rainfall Transformer | 0.6854 | 0.7144 | 0.6169 | 0.6621 | 0.7322 |
| Multimodal Fusion | 0.8218 | 0.8033 | 0.8543 | 0.8280 | 0.9122 |

These are recorded test-set results. CNN and Fusion use 2,761 test samples; the other models use 2,800. The difference arises from locations without valid 32 × 32 DEM patches, so direct comparisons should account for test-set coverage.

The Fusion model combines:
- A 26-feature tabular input.
- A two-channel, 32 × 32 terrain patch containing elevation and slope.
- A 30-day rainfall sequence.

No single model is best on every metric: MLP has the highest recorded F1 score, while Fusion has the highest recorded ROC-AUC.

## Risk Mapping

The current Phase H output contains 2,761 predictions.

| Risk category | Samples | Percentage |
|---|---:|---:|
| Low | 1,083 | 39.22% |
| Moderate | 490 | 17.75% |
| High | 1,188 | 43.03% |

Current visualization thresholds:
- Low: probability below 0.33.
- Moderate: probability from 0.33 to below 0.66.
- High: probability of 0.66 or greater.

These thresholds are project-defined visualization thresholds, not validated operational warning thresholds. The rainfall sequence currently represents January 1–30, 2022, rather than live rainfall. Consequently, the current map is not a validated real-time warning system.

## Repository Structure

```text
Slidronix/
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── public/
│   │   └── data/
│   │       └── risk_predictions.json
│   └── src/
├── implementation documentation/
├── notebooks/
├── outputs/
│   ├── phase_f/
│   ├── phase_g/
│   └── phase_h/
├── scripts/
│   └── maintenance/
│       └── download_missing_dem.py
├── scratch/
├── src/
├── .gitignore
└── README.md
```

The Python scripts in `src/` cover data extraction, feature engineering, dataset construction, model training, evaluation and risk-map generation. The `outputs/` directory contains saved models, evaluation reports, charts and predictions.

## Running the Existing Frontend

Requirements: Node.js and npm.

From the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Vite will print the local development URL in the terminal.

To build the production frontend:

```powershell
npm run build
```

The generated files are placed in `frontend/dist/`.

## Python Environment

A Python virtual environment is present in the local development setup. To activate it from the repository root on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

The project scripts use relative paths such as `data/processed/` and `outputs/`. Run them from the repository root unless a script explicitly documents otherwise.

Python dependencies have not yet been consolidated into a verified, reproducible requirements file. Model inference should reproduce the preprocessing and normalization used during training.

## Planned Development

The following components remain part of the intended full system and should not be assumed to be implemented:

- FastAPI backend and prediction endpoints.
- PostgreSQL with PostGIS spatial storage.
- User authentication and role-based dashboards.
- Citizen incident reporting and SOS workflows.
- Rescue-team incident management and emergency notifications.
- Progressive Web App push notifications.
- Risk-aware route comparison.
- Live rainfall integration and validated warning logic.
- Automated testing and production deployment.

## Safety and Research Limitations

Slidronix is a research and engineering project. Model probabilities and map categories must not be treated as authoritative emergency instructions. Real-time warnings require reliable live data, appropriate validation, monitoring, calibrated thresholds and coordination with relevant authorities.

## Documentation

Implementation reports and proposed architecture documents are stored in `implementation documentation/`.

