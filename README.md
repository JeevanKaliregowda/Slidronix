# Slidronix: AI-Based Landslide Early Warning System

Slidronix is a geospatial machine-learning project focused on landslide risk assessment in the Western Ghats of India. It combines historical landslide inventories with terrain, satellite, soil and rainfall features to train and evaluate models and generate spatial risk predictions.

## Current Implementation Status

The repository contains the historical data and model pipeline, a React/Leaflet dashboard, and a FastAPI backend for JWT authentication, role-protected workspaces, incident and SOS records, alerts, prepared prediction data, and coordinate inference. Public registration creates citizen accounts. An administrator must bootstrap the first administrator account before other workspace roles can be assigned.

The dashboard can show the prepared Phase H risk surface. Coordinate assessment invokes the Fusion inference endpoint when its local data/model assets and external data sources are available. The risk surface and inference are research outputs, not validated operational warnings.

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

## GIS Dashboard and Workspaces

The React and Leaflet dashboard visualizes prepared risk predictions and associated environmental attributes. Signed-in users receive a workspace based on their backend account role: citizen, rescue, researcher, or administrator. Incident and SOS records are account-scoped for citizens; rescue and administrator accounts can review shared response queues. Administrators can assign roles to other accounts.

The coordinate assessment endpoint requires the local Fusion assets and is not guaranteed to be ready. Static predictions are the prepared Phase H dataset. Route geometry is requested from a configurable road-routing provider; route-wide exposure scoring, push notifications, and live sensor integrations are not implemented.

Place search and road routing are available as configurable map-provider integrations described below. The Fusion API supports point inference only; it does not produce a route-wide hazard score, so a computed road route must not be described as safe.

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

During local development Vite proxies the API routes to `http://127.0.0.1:8000`. Set `VITE_API_PROXY_TARGET` to use another backend host. For a deployed frontend, set `VITE_API_BASE_URL` at build time and configure the backend's `CORS_ORIGINS` setting.

### Place search and road routing

The default geocoder is the Photon public demo (`https://photon.komoot.io`), which supports place and address lookup without an API key. Search waits until the user types at least three characters, debounces requests, caches recent results, and spaces requests to stay below one request per second. Photon is a community demo without an availability guarantee; use a permitted hosted provider or self-hosted Photon instance for dependable or higher-volume service. OpenStreetMap is credited in the UI.

Road geometry and maneuver data use the OSRM-compatible demo endpoint at `https://router.project-osrm.org`. A request is sent only when the user submits both selected places. Provider-returned geometry and turn steps are displayed only when a valid route response arrives. The public demo is for modest evaluation use and has no availability or throughput guarantee; configure a self-hosted OSRM server or another compatible service for production. Route timing is an estimate. Slidronix does not calculate risk over every route segment or guarantee a safe path. The optional endpoint assessment runs two independent Fusion coordinate predictions and does not evaluate the corridor.

Override providers in the same PowerShell session used to start Vite:

```powershell
$env:VITE_GEOCODER_BASE_URL = "https://photon.komoot.io"
$env:VITE_ROUTING_BASE_URL = "https://router.project-osrm.org"
npm run dev -- --host 127.0.0.1 --port 5173
```

Set `VITE_GEOCODER_BASE_URL` to another Photon-compatible endpoint or set `VITE_ROUTING_BASE_URL` to an OSRM-compatible route API. Set either variable to `disabled` to show the unavailable state. Provider URLs are compiled into the frontend and must not contain private API keys.

## Running the Backend

Use Python 3.10 or newer. From the repository root, create `backend/.env` with a strong, private `JWT_SECRET_KEY` (at least 32 random bytes encoded as text). Set `DATABASE_URL` only when using a database other than the default `slidronix_demo.db`; the backend creates missing tables and performs an additive ownership-column migration while preserving existing records. Do not commit the environment file.

Install `backend/requirements.txt`, then start the API from the `backend` directory with `python -m uvicorn app.main:app --reload`. The API docs are available at `/docs` while it is running. For a new installation with no administrator, run `python -m app.bootstrap_admin` from `backend`; it prompts for the account and password and refuses to create a second administrator.

The API requires bearer-token authentication for prediction, inference, incident, SOS, and alert routes. Register a citizen account in the frontend, then sign in. Administrators assign rescue/researcher roles in User management. Development checks: run `npm run build` and `npm run lint` from `frontend`, and `python -m pytest` from `backend`.

## Python Environment

From the repository root on Windows, activate the existing virtual environment with:

`.\venv\Scripts\Activate.ps1`

Run project scripts from the repository root unless the script specifies otherwise, because scripts may use relative paths such as `data/processed/` and `outputs/`.

Backend runtime and test dependencies are declared in `backend/requirements.txt`. The manifest matches the current backend imports; a clean environment installation has not been verified in this release check.

## Remaining Development

- Production database operations and deployment hardening
- Push notification delivery
- Route-wide hazard exposure scoring and comparison of risk-aware alternatives
- Validated warning thresholds, live-data monitoring, and operational evaluation
- Further testing of live inference against the required local assets and provider availability

## Safety and Research Limitations

Slidronix is a research and engineering project. Model probabilities and map categories must not be treated as authoritative emergency instructions. Operational warnings require reliable live data, appropriate validation, calibrated thresholds, monitoring and coordination with relevant authorities.

## Authors

- Jeevan Kaliregowda
- Vikas
- Veerabhadra Prasad R
- Kunguma Sanjutha V

K. S. Institute of Technology, Bengaluru
