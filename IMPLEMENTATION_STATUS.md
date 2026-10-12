# Slidronix implementation status

Updated: 2026-10-12

## UI/UX and feature update

- Refined the existing dashboard theme with warm off-white surfaces, forest-green navigation, restrained blue actions, larger type, stronger spacing and borders, and a low-contrast contour texture behind the workspace. The existing sidebar, role workspaces, and responsive page layout remain in place.
- Added a map-led Check My Area flow. Users can search for a place, select a point on the Leaflet map, explicitly request browser location, or open the manual-coordinate fallback. The selected label and exact coordinates are shown. A new point clears any previous result; the UI shows a successful assessment only after `/predict/coordinate` returns one, including actual class, probability, source, environmental inputs and warning.
- Added place-name route planning. The user selects origin and destination search results. A route is drawn only from actual OSRM-compatible GeoJSON; returned maneuvers are presented as directions alongside distance and estimated time. Failed, empty or disabled provider responses show an unavailable state and never produce a straight-line substitute.
- Added optional, separately labelled Fusion estimates at the two endpoints. The model API supports point inference only; these estimates do not assess the road corridor or establish that a route is safe.
- Provider endpoints can be configured using `VITE_GEOCODER_BASE_URL`, `VITE_GEOCODER_ATTRIBUTION`, and `VITE_ROUTING_BASE_URL`. URLs must not contain private keys because these frontend settings are public at build time.
- Kept backend authorization and API contracts unchanged. No dependency packages, environment files, databases, trained models, or existing backups were replaced or deleted.

## Providers and known limits

- Place search defaults to Photon at `https://photon.komoot.io`, backed by OpenStreetMap data. Requests are user-triggered, debounced, cached and globally spaced to at most one request per 1.1 seconds in the page runtime. The public Photon service describes its endpoint as a fair-use demo with no uptime guarantee; use an appropriate provider or self-hosted instance for larger or dependable deployments. OpenStreetMap attribution is visible in the UI.
- Road routing defaults to the OSRM demo at `https://router.project-osrm.org`. Only an explicit route submission sends a request. The public demo has no availability/throughput guarantee; use a suitable hosted or self-hosted OSRM-compatible service for production.
- A computed road route is not a hazard assessment. The existing Fusion endpoint predicts one coordinate at a time, so no segment/corridor score, safe alternative comparison, or safety guarantee is provided.
- Browser geolocation is requested only after the user selects “Use my location.”
- Push delivery and PostgreSQL/PostGIS deployment remain unconfigured. Live coordinate inference requires the current local model/data/DEM assets and external weather/terrain availability; its environmental reference values are approximate where described in the API response.

## Verification

- `frontend`: `npm.cmd run test:location` — 3 passed. Tests cover Photon feature-to-coordinate parsing, OSRM route geometry/turn conversion, and no-route, disabled-provider, HTTP failure and invalid-coordinate handling.
- `frontend`: `npm.cmd run lint` — passed.
- `frontend`: `npm.cmd run build` — passed; Vite transformed 1,925 modules.
- `backend`: `..\venv\Scripts\python.exe -m pytest -q` — 6 passed, 1 upstream Starlette/httpx deprecation warning.
- Live geocoder request for Chikkamagaluru returned 3 Photon features.
- Live OSRM request returned `code=Ok`, 24,379.3 m, 687 geometry points and 4 route steps. The frontend route adapter was separately exercised against OSRM and returned usable Leaflet coordinates and directions.
- On a temporary local API with a disposable `%TEMP%` SQLite database and process-only JWT key, Fusion readiness returned HTTP 200 (`ready=true`, 26 features). A genuine inference at 13.3161, 75.772 returned `fresh_fusion_model_inference`, `Low`, probability 0.0768767893. An out-of-coverage coordinate returned HTTP 422 and an unauthenticated request returned HTTP 401. No prediction fallback was used. The configured database and `backend/.env` values were not used or displayed.
- Route-provider disabled, HTTP 503 and `NoRoute` responses were verified with the provider adapter tests; none yielded geometry.
- Git whitespace check: `git diff --check` passes. A source scan found no malformed UTF-8 sequences in active frontend source.
- No browser was exposed by the computer-use session, and no Playwright/Puppeteer package or browser binary is installed. Therefore place selection, map clicking, geolocation permission, rendered error states, visual overflow, and browser-console behavior could not be driven or observed in a browser. Provider and API adapters were tested directly; this UI interaction check remains outstanding.

## Commands

From repository root, PowerShell terminal 1:

```powershell
.\venv\Scripts\Activate.ps1
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

PowerShell terminal 2:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`. Existing Vite proxy points to `http://127.0.0.1:8000`. The existing `backend/.env` must retain a private `JWT_SECRET_KEY`; do not replace it. Backend startup creates missing tables and applies the existing additive ownership migration, so choose the intended database deliberately.

To override providers, set variables in terminal 2 before starting Vite, for example:

```powershell
$env:VITE_GEOCODER_BASE_URL = "https://photon.komoot.io"
$env:VITE_GEOCODER_ATTRIBUTION = "Place data attribution"
$env:VITE_ROUTING_BASE_URL = "https://router.project-osrm.org"
npm run dev -- --host 127.0.0.1 --port 5173
```

Set `VITE_ROUTING_BASE_URL` to `disabled` to show routing unavailable. Provider URLs must implement Photon-compatible search/reverse and OSRM-compatible routing responses, respectively.

Final local checks:

```powershell
cd frontend
npm.cmd run test:location
npm.cmd run lint
npm.cmd run build
cd ..\backend
..\venv\Scripts\python.exe -m pytest -q
```

## Preservation checkpoint

Created the final source-only checkpoint at `backups/source_checkpoint_20261012_uiux_verified.zip`. It contains active frontend/backend source, package metadata, tests and documentation; it excludes `.env` files, databases, model weights, datasets, generated build output and existing backups. Earlier source checkpoints remain untouched.
