# Slidronix Frontend

Slidronix is an AI-based landslide risk assessment and early warning system for the Western Ghats. This frontend presents landslide risk predictions through an interactive dashboard.

## Technology Stack

- React
- Vite
- Leaflet for interactive maps
- Recharts for charts
- Lucide React for icons

## Prerequisites

- Node.js and npm

## Run Locally

From the repository root, run:

1. `cd frontend`
2. `npm install`
3. `npm run dev`

Open the local URL printed by Vite in your terminal.

## Production Build

From the frontend directory, run `npm run build`.

The production build is generated in `frontend/dist/`.

## Risk Prediction Data

The dashboard reads the static prediction dataset from `frontend/public/data/risk_predictions.json`.

The dashboard displays existing predictions; it does not itself train the machine-learning models.

## Important Limitations

- The displayed predictions are based on an existing dataset, not a live rainfall monitoring pipeline.
- Risk categories should not be treated as independently validated operational warnings.
- The dashboard supports project demonstration and risk visualization; it does not replace official disaster-management advisories.
