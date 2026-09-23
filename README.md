# Slidronix — AI-Based Landslide Early Warning System

Slidronix is an AI-based landslide risk assessment and early-warning system developed for the Western Ghats of India.

The system combines **historical landslide records, terrain data, satellite observations, soil properties, and rainfall sequences** to build a multimodal environmental dataset. Machine-learning and deep-learning models are used to estimate landslide risk and visualize predictions through an interactive GIS dashboard.

> **Current status:** Phases **A–H are implemented**. The current Phase H system produces probabilistic risk predictions and GIS visualization from the prepared environmental dataset. Live environmental API integration and real-time rainfall are future enhancements.

---

## 1. Project Objective

Slidronix aims to:
- Process and standardize historical landslide information.
- Extract terrain and environmental features.
- Integrate satellite, soil, and rainfall information.
- Construct a balanced spatial dataset for landslide-risk modelling.
- Learn **tabular, spatial, and temporal** representations.
- Compare machine-learning and deep-learning models.
- Generate probabilistic landslide-risk predictions.
- Visualize risk through an interactive GIS dashboard.

---

## 2. System Pipeline

~~~text
Historical Landslide Inventory
            ↓
Phase A — Foundation + Baseline RF
            ↓
Phase B — DEM + Terrain Features
            ↓
Phase C — NDVI + Land Cover + Soil
            ↓
Phase D — Rainfall Features
            ↓
Phase E — Balanced Spatial Dataset
            ↓
Phase F — MLP + CNN + Transformer + Fusion
            ↓
Phase G — Model Comparison
            ↓
Phase H — Risk Mapping + GIS Dashboard
~~~

---

## 3. Phase-wise Implementation

### Phase A — Historical Data & Baseline Model

- Processed a **582-page historical landslide inventory** using Python and pdfplumber.
- Extracted **34,207 historical landslide records** with 11 attributes.
- Performed coordinate validation, missing-value handling, categorical cleaning and duplicate removal.
- Implemented a **Random Forest baseline classifier**.
- Input: latitude, longitude, state and material involved.
- Target: movement type.
- Train/test: 80/20 stratified split.
- Training: **27,183** | Testing: **6,796**
- Baseline accuracy: **69.60%**

### Phase B — Terrain Features

Used **Copernicus DEM GLO-30** to derive:
- Elevation
- Slope

The terrain dataset contains **14,270 spatial samples**.

### Phase C — Satellite & Soil Features

**Satellite**
- MODIS MOD13Q1.061 → NDVI
- MODIS MCD12Q1.061 → Land Cover

**NDVI features**
- Mean, standard deviation, minimum, maximum, valid observations and range.

**Soil — SoilGrids**
- Clay
- Sand
- Silt
- Soil Organic Carbon
- Bulk Density
- Soil pH

### Phase D — Rainfall

Used **CHIRPS v3 daily satellite rainfall**.

The current implementation uses a fixed **1–30 January 2022 reference period**.

Generated:
- 7-day, 15-day and 30-day accumulated rainfall
- Mean daily rainfall
- Maximum daily rainfall
- Standard deviation
- Maximum rainfall over 7/15/30-day windows
- 30 daily rainfall observations for the temporal Transformer

> **Limitation:** the current rainfall implementation is a fixed reference period. Event-aligned and real-time rainfall integration are future enhancements.

### Phase E — Balanced Spatial Dataset

Constructed a binary dataset using historical landslide locations and spatially generated background locations.

- Study region: Maharashtra, Goa, Karnataka, Kerala and Tamil Nadu.
- Minimum **1 km separation** between background and positive locations.
- Balanced classes.
- Spatial block-based train/validation/test splitting.
- **0 spatial block overlap** between splits.

| Dataset | Samples |
|---|---:|
| Landslide | 6,991 |
| Background | 6,991 |
| **Total** | **13,982** |
| Training | 8,385 |
| Validation | 2,797 |
| Testing | 2,800 |

### Phase F — Deep Learning & Multimodal Fusion

**MLP — Tabular Encoder**
- Uses **26 environmental features**.
- Architecture: 26 → 128 → 64 → 32 → 1

**CNN — Spatial Encoder**
- Uses **32 × 32 terrain patches** containing elevation and slope.

**Transformer — Temporal Encoder**
- Uses the **30-day rainfall sequence**.
- 2 Transformer encoder layers, 64-dimensional representation and 4 attention heads.

**Fusion Model**
Combines the three learned representations:

~~~text
Tabular MLP ─────────┐
Spatial CNN ─────────┼──→ Feature Fusion → Fusion Head
Temporal Transformer ┘
                              ↓
                    Landslide Probability
                              0 ——— 1
~~~

### Phase G — Model Comparison

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| Random Forest | 81.54% | 81.14% | 0.9093 |
| MLP | 82.36% | 83.33% | 0.8968 |
| CNN | 81.24% | 82.54% | 0.8815 |
| Transformer | 68.54% | 66.21% | 0.7322 |
| **Fusion** | **82.18%** | **82.80%** | **0.9122** |

> CNN and Fusion used **2,761 aligned test samples** because valid 32 × 32 terrain patches are required. Other models used 2,800 test samples.

### Phase H — Risk Prediction & GIS Mapping

The Fusion model generates a probability from **0 to 1**.

- Aligned predictions: **2,761**
- Probability range: **0.0001–0.9978**
- Mean probability: **0.4974**

| Risk Level | Samples | Percentage |
|---|---:|---:|
| Low | 1,083 | 39.22% |
| Moderate | 490 | 17.75% |
| High | 1,188 | 43.03% |

The predictions are visualized through an interactive React + Leaflet GIS dashboard.

---

## 4. GIS Dashboard

### Features
- Interactive risk map
- Satellite and street basemaps
- Risk-level filtering
- Point selection
- Risk probability
- Latitude/longitude
- Elevation and slope
- Rainfall
- NDVI
- Soil pH
- Land cover
- Prediction ID
- Reverse-geocoded location information

The current dashboard visualizes generated Phase H predictions from the prepared dataset. It is **not currently a real-time inference system**.

---

## 5. Data Sources

| Data | Source | Purpose |
|---|---|---|
| Historical landslides | Historical landslide inventory | Landslide records |
| DEM | Copernicus DEM GLO-30 | Elevation and slope |
| NDVI | MODIS MOD13Q1.061 | Vegetation |
| Land Cover | MODIS MCD12Q1.061 | Land-cover classification |
| Soil | SoilGrids | Soil properties |
| Rainfall | CHIRPS v3 | Rainfall features |

---

## 6. Technology Stack

**Languages:** Python, JavaScript

**ML/DL:** Scikit-learn, PyTorch, Random Forest, MLP, CNN, Transformer

**Data Processing:** Pandas, NumPy, pdfplumber, Rasterio

**GIS:** Spatial processing, Leaflet

**Frontend:** React, Vite, Leaflet, Lucide React

**Version Control:** Git, GitHub

---

## 7. Repository Structure

~~~text
Slidronix/
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── public/
│   └── src/
├── outputs/
│   ├── charts/
│   ├── maps/
│   ├── phase_f/
│   ├── phase_g/
│   └── phase_h/
├── src/
│   ├── extract_dem_features.py
│   ├── extract_soil_features.py
│   ├── extract_chirps_rainfall.py
│   ├── create_rainfall_features.py
│   ├── create_integrated_dataset.py
│   ├── finalize_phase_e_dataset.py
│   ├── train_baseline_model.py
│   ├── train_mlp.py
│   ├── train_cnn.py
│   ├── train_transformer.py
│   ├── train_fusion.py
│   ├── train_rf_phase_g.py
│   ├── create_phase_g_comparison.py
│   ├── generate_phase_h_risk_predictions.py
│   └── create_phase_h_risk_map.py
└── README.md
~~~

---

## 8. Key Outputs

- **34,207** historical landslide records processed.
- **13,982** balanced modelling samples created.
- Terrain, satellite, soil and rainfall features integrated.
- MLP, CNN, Transformer and Fusion models implemented.
- Five-model comparative evaluation completed.
- **2,761** aligned Phase H predictions generated.
- Interactive GIS risk dashboard implemented.

---

## 9. Limitations

1. Rainfall currently uses a fixed **1–30 January 2022 reference period**.
2. The dashboard currently visualizes prepared predictions rather than performing live inference for arbitrary coordinates.
3. Real-time rainfall and environmental API integration is not yet connected.
4. CNN/Fusion use fewer aligned test samples because valid terrain patches are required.
5. The study-region boundary is a project-derived spatial boundary for the selected Western Ghats states, not an official definition of the Western Ghats.

---

## 10. Future Enhancements

- Event-aligned rainfall histories.
- Real-time rainfall and environmental API integration.
- Coordinate-based feature generation.
- FastAPI inference backend.
- Dynamic GIS risk updates.
- Automated risk-alert engine.
- Additional environmental and historical data.
- Longer/event-centered temporal rainfall sequences.

---

## 11. Project Status

| Phase | Status |
|---|---|
| Phase A — Foundation | ✅ Completed |
| Phase B — Terrain | ✅ Completed |
| Phase C — Satellite + Soil | ✅ Completed |
| Phase D — Rainfall | ✅ Completed |
| Phase E — Dataset Construction | ✅ Completed |
| Phase F — Deep Learning | ✅ Completed |
| Phase G — Model Comparison | ✅ Completed |
| Phase H — Risk Mapping | ✅ Completed |

---

## 12. Running the Dashboard

~~~bash
cd frontend
npm install
npm run dev
~~~

For a production build:

~~~bash
npm run build
~~~

The production build is generated in:

~~~text
frontend/dist/
~~~

check out:
https://slidronix.netlify.app/

---

## 13. Project Summary

~~~text
Historical Data
      ↓
Terrain + Satellite + Soil + Rainfall
      ↓
Balanced Spatial Dataset
      ↓
MLP + CNN + Transformer
      ↓
Multimodal Fusion
      ↓
Landslide Probability
      ↓
Risk Classification
      ↓
GIS Visualization
~~~

Slidronix currently implements the complete **Phase A–H modelling and visualization pipeline**. Live environmental integration and automated real-time alerts remain planned enhancements.

---

## Authors

**Jeevan Kaliregowda** 
**Vikas**
**Veerabhadra Prasad R**
**Kunguma Sanjutha V**
K. S. Institute of Technology, Bengaluru

**Project:** Slidronix — AI-Based Landslide Early Warning System
