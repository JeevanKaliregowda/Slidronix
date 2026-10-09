# Slidronix: AI-Based Landslide Early Warning System

Slidronix is a geospatial machine-learning project focused on landslide risk assessment in the Western Ghats of India. It combines historical landslide inventories with terrain, satellite, soil and rainfall features to train and evaluate models and generate spatial risk predictions.

## Current Implementation Status

Implemented components include historical landslide data extraction and cleaning, terrain and environmental feature preparation, spatial dataset construction, model training and comparison, risk prediction generation, and a React-based GIS dashboard.

The dashboard currently displays prepared predictions from a static JSON file. Live environmental API integration, real-time inference, and automated emergency alerts are future enhancements.

## Project Pipeline

1. **Phase A — Historical Data:** Processed a 582-page inventory and extracted 34,207 historical landslide records.
2. **Phase B — Terrain:** Used Copernicus DEM GLO-30 to derive elevation and slope.
3. **Phase C — Satellite and Soil:** Integrated MODIS NDVI, land cover and SoilGrids properties.
4. **Phase D — Rainfall:** Generated rainfall features from CHIRPS data using a fixed January 1–30, 2022 reference period.
5. **Phase E — Dataset:** Constructed a balanced spatial dataset with 13,982 samples and spatially separated train, validation and test splits.
6. **Phase F — Models:** Implemented MLP, CNN, rainfall Transformer and multimodal Fusion models.
7. **Phase G — Evaluation:** Compared model performance and generated evaluation outputs.
8. **Phase H — Risk Mapping:** Generated 2,761 aligned risk predictions for the GIS dashboard.

## Technology Stack

- **Languages:** Python, JavaScript
- **Machine learning:** Scikit-learn, PyTorch
- **Data processing:** Pandas, NumPy, pdfplumber
- **Geospatial processing:** Rasterio, GeoPandas
- **Data sources:** Copernicus DEM GLO-30, MODIS/AppEEARS, SoilGrids and CHIRPS
- **Frontend:** React, Vite, Leaflet, Recharts and Lucide React

## Model Evaluation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8154 | 0.8286 | 0.7949 | 0.8114 | 0.9093 |
| MLP | 0.8236 | 0.7891 | 0.8828 | 0.8333 | 0.8968 |
| CNN | 0.8124 | 0.7747 | 0.8831 | 0.8254 | 0.8815 |
| Rainfall Transformer | 0.6854 | 0.7144 | 0.6169 | 0.6621 | 0.7322 |
| Multimodal Fusion | 0.8218 | 0.8033 | 0.8543 | 0.8280 | 0.9122 |

These are recorded test-set results. Random Forest, MLP and Transformer used 2,800 test samples, while CNN and Fusion used 2,761 because some locations lacked valid 32 × 32 terrain patches. Comparisons should account for this difference.

The Fusion model combines 26 tabular features, a two-channel 32 × 32 elevation/slope patch, and a 30-day rainfall sequence. MLP has the highest recorded F1 score, while Fusion has the highest recorded ROC-AUC.

## Risk Prediction Output

The current Phase H output contains 2,761 predictions.

| Risk category | Samples | Percentage |
|---|---:|---:|
| Low | 1,083 | 39.22% |
| Moderate | 490 | 17.75% |
| High | 1,188 | 43.03% |

The dashboard uses project-defined visualization thresholds:
- Low: probability below 0.33.
- Moderate: probability from 0.33 to below 0.66.
- High: probability of 0.66 or greater.

These thresholds are not validated operational warning thresholds. The rainfall input represents a fixed historical reference period rather than live rainfall. The current dashboard is therefore not a validated real-time warning system.

## GIS Dashboard

The React and Leaflet dashboard visualizes prepared risk predictions and associated environmental attributes. Current features include interactive map visualization, risk filtering, point selection and display of available prediction details.

Live inference for arbitrary coordinates, real-time rainfall updates, citizen SOS workflows, role-based dashboards, rescue-team management and automated notifications remain planned enhancements.

Existing dashboard: https://slidronix.netlify.app/

## Repository Structure

- `data/` — raw and processed datasets
- `frontend/` — React/Vite dashboard
- `implementation documentation/` — phase reports and proposed architecture documents
- `notebooks/` — notebooks
- `outputs/` — model artifacts, evaluation results and predictions
- `scripts/maintenance/` — maintenance utilities
- `src/` — data processing, feature engineering, model training and prediction scripts

## Running the Frontend

Requirements: Node.js and npm.

From the repository root:

1. `cd frontend`
2. `npm install`
3. `npm run dev`

To create a production build, run `npm run build` from the `frontend` directory. The output is generated in `frontend/dist/`.

## Python Environment

From the repository root on Windows, activate the existing virtual environment with:

`.\venv\Scripts\Activate.ps1`

Run project scripts from the repository root unless the script specifies otherwise, because scripts may use relative paths such as `data/processed/` and `outputs/`.

Python dependencies have not yet been consolidated into a verified, reproducible requirements file.

## Planned Development

- FastAPI backend and prediction endpoints
- PostgreSQL with PostGIS
- Authentication and role-based dashboards
- Citizen incident reporting and SOS workflows
- Rescue-team incident management and notifications
- Progressive Web App push notifications
- Risk-aware route comparison
- Live rainfall integration and validated warning logic
- Automated tests and production deployment

## Safety and Research Limitations

Slidronix is a research and engineering project. Model probabilities and map categories must not be treated as authoritative emergency instructions. Operational warnings require reliable live data, appropriate validation, calibrated thresholds, monitoring and coordination with relevant authorities.

## Authors

- Jeevan Kaliregowda
- Vikas
- Veerabhadra Prasad R
- Kunguma Sanjutha V

K. S. Institute of Technology, Bengaluru
