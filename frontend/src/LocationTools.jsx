import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, CircleMarker, Polyline, Popup, TileLayer, useMap, useMapEvents } from "react-leaflet";
import { Activity, AlertTriangle, Check, Clock3, LocateFixed, MapPin, Navigation, Route, Search, Target } from "lucide-react";
import { photonFeatureToPlace, photonResultsToPlaces, requestRoadRoute } from "./locationServices";
import "./LocationTools.css";

const GEOCODER_URL = import.meta.env.VITE_GEOCODER_BASE_URL ?? "https://photon.komoot.io";
const DEFAULT_GEOCODER = "https://photon.komoot.io";
const GEOCODER_ATTRIBUTION = import.meta.env.VITE_GEOCODER_ATTRIBUTION ?? "Place and address data from OpenStreetMap";
const ROUTER_URL = import.meta.env.VITE_ROUTING_BASE_URL ?? "https://router.project-osrm.org";
const USING_OSRM_DEMO = ROUTER_URL.replace(/\/$/, "") === "https://router.project-osrm.org";
const geocodeCache = new Map();
let geocodeQueue = Promise.resolve();
let lastGeocodeStarted = 0;

function queuedGeocode(url, signal) {
  const task = geocodeQueue.catch(() => {}).then(async () => {
    const wait = Math.max(0, 1100 - (Date.now() - lastGeocodeStarted));
    if (wait) await new Promise((resolve) => setTimeout(resolve, wait));
    if (signal?.aborted) throw new DOMException("Search cancelled", "AbortError");
    lastGeocodeStarted = Date.now();
    const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Place search is unavailable (HTTP ${response.status}).`);
    return response.json();
  });
  geocodeQueue = task.then(() => undefined, () => undefined);
  return task;
}

function searchPlaces(query, signal) {
  if (!GEOCODER_URL || GEOCODER_URL.toLowerCase() === "disabled") return Promise.reject(new Error("Place search is not configured. Set VITE_GEOCODER_BASE_URL to a compatible Photon service."));
  const key = query.trim().toLowerCase();
  if (geocodeCache.has(key)) return Promise.resolve(geocodeCache.get(key));
  const url = new URL(`${GEOCODER_URL.replace(/\/$/, "")}/api/`);
  url.searchParams.set("q", query.trim());
  url.searchParams.set("limit", "5");
  url.searchParams.set("lang", "en");
  return queuedGeocode(url, signal).then((data) => {
    const places = photonResultsToPlaces(data);
    if (geocodeCache.size > 80) geocodeCache.clear();
    geocodeCache.set(key, places);
    return places;
  });
}

function reversePlace(point, signal) {
  if (!GEOCODER_URL || GEOCODER_URL.toLowerCase() === "disabled") return Promise.reject(new Error("Place search is not configured."));
  const url = new URL(`${GEOCODER_URL.replace(/\/$/, "")}/reverse`);
  url.searchParams.set("lat", String(point.latitude));
  url.searchParams.set("lon", String(point.longitude));
  url.searchParams.set("lang", "en");
  return queuedGeocode(url, signal).then((data) => photonFeatureToPlace(data.features?.[0]));
}

function PlaceSearch({ label, placeholder, selected, onSelect }) {
  const [query, setQuery] = useState(selected?.label || "");
  const [places, setPlaces] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [resultsFor, setResultsFor] = useState("");
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);

  useEffect(() => {
    const term = query.trim();
    if (selected || term.length < 3) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setLoading(true); setMessage("");
      searchPlaces(term, controller.signal)
        .then((results) => { setPlaces(results); setResultsFor(term); setOpen(true); setMessage(results.length ? "" : "No matching places. Try adding a town or state."); })
        .catch((error) => { if (error.name !== "AbortError") { setPlaces([]); setResultsFor(term); setMessage(error.message?.startsWith("Place search") ? error.message : `Place search could not reach its provider. Check the provider URL and connection, then retry. (${error.message || "network error"})`); } })
        .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    }, 900);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query, selected]);

  useEffect(() => {
    function dismiss(event) { if (!wrapperRef.current?.contains(event.target)) setOpen(false); }
    document.addEventListener("pointerdown", dismiss);
    return () => document.removeEventListener("pointerdown", dismiss);
  }, []);

  return <div className="place-search" ref={wrapperRef}>
    <label className="field"><span>{label}</span><span className="place-search-input"><Search size={16}/><input value={selected?.label || query} placeholder={placeholder} autoComplete="off" onFocus={() => setOpen(true)} onChange={(event) => { setQuery(event.target.value); onSelect(null); setOpen(true); }}/>{loading && query.trim().length >= 3 && !selected && <span className="search-spinner" aria-label="Searching"/>}</span></label>
    {open && query.trim().length >= 3 && !selected && (places.length > 0 || message) && <div className="place-suggestions" role="listbox" aria-label={`${label} suggestions`}>
      {resultsFor === query.trim() && places.map((place, index) => <button type="button" role="option" aria-selected="false" className="place-option" key={`${place.latitude}-${place.longitude}-${index}`} onClick={() => { onSelect(place); setQuery(place.label); setPlaces([]); setMessage(""); setOpen(false); }}><MapPin size={15}/><span><strong>{place.label}</strong><small>{place.latitude.toFixed(5)}, {place.longitude.toFixed(5)}</small></span></button>)}
      {resultsFor === query.trim() && message && <div className="place-search-message" role="status">{message}</div>}
      <div className="provider-credit">{!GEOCODER_URL || GEOCODER_URL.toLowerCase() === "disabled" ? "Place search provider is disabled" : GEOCODER_URL.replace(/\/$/, "") === DEFAULT_GEOCODER ? <>Place search by <a href="https://github.com/komoot/photon" target="_blank" rel="noreferrer">Photon</a> · OpenStreetMap data</> : `Configured place search · ${GEOCODER_ATTRIBUTION}`}</div>
    </div>}
  </div>;
}

function RecenterMap({ locations, routeCoordinates }) {
  const map = useMap();
  useEffect(() => {
    const endpointPoints = locations.filter(Boolean).map((p) => [p.latitude, p.longitude]);
    const routePoints = routeCoordinates || [];
    const pts = [...endpointPoints, ...routePoints];
    if (pts.length > 1) map.fitBounds(pts, { padding: [36, 36], maxZoom: 13 });
    else if (pts.length === 1) map.setView(pts[0], Math.max(map.getZoom(), 12));
  }, [locations, routeCoordinates, map]);
  return null;
}

function PointClick({ onPick }) {
  useMapEvents({ click(event) { onPick({ latitude: event.latlng.lat, longitude: event.latlng.lng }); } });
  return null;
}

function MapBase({ center = [13.05, 75.7], zoom = 7, locations = [], onPick, routeCoordinates }) {
  return <div className="location-map-frame"><MapContainer center={center} zoom={zoom} scrollWheelZoom className="risk-map" attributionControl>
    <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"/>
    <RecenterMap locations={locations} routeCoordinates={routeCoordinates}/>
    {onPick && <PointClick onPick={onPick}/>}
    {routeCoordinates?.length > 1 && <Polyline positions={routeCoordinates} pathOptions={{ color: "#397b78", weight: 6, opacity: 0.88, lineCap: "round", lineJoin: "round" }}/>}
    {locations.filter(Boolean).map((point, index) => <CircleMarker key={`${point.latitude}-${point.longitude}-${index}`} center={[point.latitude, point.longitude]} radius={index === 0 ? 9 : 8} pathOptions={{ color: "#fff", weight: 3, fillColor: index === 0 ? "#356e58" : "#356fa4", fillOpacity: 1 }}><Popup><strong>{point.label}</strong><br/>{point.latitude.toFixed(5)}, {point.longitude.toFixed(5)}</Popup></CircleMarker>)}
  </MapContainer></div>;
}

function LocationSummary({ point }) {
  if (!point) return <div className="location-summary empty-location"><MapPin size={17}/><span>Search for a place or select a point on the map.</span></div>;
  return <div className="location-summary"><MapPin size={17}/><span><strong>{point.label}</strong><small>{point.latitude.toFixed(5)}, {point.longitude.toFixed(5)}</small></span><Check size={17} className="location-check"/></div>;
}

function InferenceResult({ result, label }) {
  if (!result) return null;
  const percent = Math.max(0, Math.min(100, Number(result.landslide_probability) * 100));
  return <article className="inference-result">
    <div className="inference-result-heading"><span className="eyebrow">{label}</span><span className="inference-source"><Check size={14}/> {result.is_new_model_inference ? "Fresh Fusion inference" : result.source || "Model response"}</span></div>
    <div className={`inference-risk risk-${String(result.risk_level || "unknown").toLowerCase()}`}><div><strong>{result.risk_level || "Unclassified"}</strong><small>Modelled landslide risk</small></div><strong className="inference-probability">{percent.toFixed(2)}<small>%</small></strong></div>
    <div className="inference-meter"><i style={{ width: `${percent}%` }}/></div>
    <div className="inference-meta"><span>Coordinates</span><strong>{Number(result.latitude).toFixed(5)}, {Number(result.longitude).toFixed(5)}</strong></div>
    {result.data_sources && <div className="source-grid">{Object.entries(result.data_sources).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><strong>{String(value)}</strong></div>)}</div>}
    {Object.keys(result.features || {}).length > 0 && <details className="feature-details"><summary>Environmental inputs returned by the model</summary><div className="feature-grid feature-grid-full">{Object.entries(result.features).map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><strong>{Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 }) : "Unavailable"}</strong></div>)}</div></details>}
    <p className="research-warning"><AlertTriangle size={15}/>{result.warning || "Research estimate only. It is not an official warning."}</p>
  </article>;
}

export function CheckAreaTool({ api, onAssessment }) {
  const [point, setPoint] = useState(null);
  const [manualOpen, setManualOpen] = useState(false);
  const [manualLat, setManualLat] = useState("");
  const [manualLon, setManualLon] = useState("");
  const [searchReset, setSearchReset] = useState(0);
  const [reverseBusy, setReverseBusy] = useState(false);
  const [reverseMessage, setReverseMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const reverseId = useRef(0);
  const reverseController = useRef(null);
  const assessmentId = useRef(0);
  const selectedLocations = useMemo(() => point ? [point] : [], [point]);

  useEffect(() => () => reverseController.current?.abort(), []);

  async function pickCoordinates(latitude, longitude, label = "Selected map point") {
    const selected = { latitude: Number(latitude), longitude: Number(longitude), label };
    if (!Number.isFinite(selected.latitude) || selected.latitude < -90 || selected.latitude > 90 || !Number.isFinite(selected.longitude) || selected.longitude < -180 || selected.longitude > 180) {
      setError("Choose valid coordinates between latitude −90 and 90 and longitude −180 and 180."); return;
    }
    reverseController.current?.abort(); assessmentId.current += 1; setBusy(false); setPoint(selected); setResult(null); setError(""); setReverseMessage(""); setSearchReset((value) => value + 1);
    setManualLat(String(selected.latitude.toFixed(6))); setManualLon(String(selected.longitude.toFixed(6)));
    const id = ++reverseId.current;
    if (label !== "Selected map point" && label !== "My current location" && label !== "Manual coordinate selection") return;
    const controller = new AbortController(); reverseController.current = controller;
    setReverseBusy(true);
    try {
      const resolved = await reversePlace(selected, controller.signal);
      if (id === reverseId.current && resolved) setPoint({ ...selected, label: resolved.label });
      else if (id === reverseId.current) setReverseMessage("Address lookup returned no place name; the selected coordinates remain available.");
    } catch (lookupError) {
      if (id === reverseId.current) setReverseMessage(lookupError.name === "AbortError" ? "" : "Place name unavailable. Assessment can still use these coordinates.");
    } finally { if (id === reverseId.current) { reverseController.current = null; setReverseBusy(false); } }
  }

  function selectPlace(place) {
    reverseController.current?.abort(); reverseController.current = null;
    if (!place) { reverseId.current += 1; assessmentId.current += 1; setReverseBusy(false); setBusy(false); setPoint(null); setResult(null); setError(""); setReverseMessage(""); return; }
    reverseId.current += 1; assessmentId.current += 1; setReverseBusy(false); setBusy(false);
    setPoint(place); setResult(null); setError(""); setReverseMessage("");
    setManualLat(String(place.latitude.toFixed(6))); setManualLon(String(place.longitude.toFixed(6)));
  }

  function useMyLocation() {
    setError("");
    if (!navigator.geolocation) { setError("This browser does not support location access."); return; }
    navigator.geolocation.getCurrentPosition((position) => pickCoordinates(position.coords.latitude, position.coords.longitude, "My current location"), (locationError) => {
      const reason = locationError.code === 1 ? "Location permission was declined." : locationError.code === 3 ? "Location request timed out." : "Your current location could not be determined.";
      setError(`${reason} You can search for a place or choose a map point instead.`);
    }, { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 });
  }

  async function assess(event) {
    event?.preventDefault();
    if (!point) { setError("Select a place or map point before starting an assessment."); return; }
    const requestId = ++assessmentId.current;
    setBusy(true); setResult(null); setError("");
    try {
      const response = await api("/predict/coordinate", { method: "POST", body: JSON.stringify({ latitude: point.latitude, longitude: point.longitude }) });
      if (requestId === assessmentId.current) { setResult(response); onAssessment?.(response); }
    } catch (assessmentError) { if (requestId === assessmentId.current) setError(`Assessment unavailable: ${assessmentError.message || "The model request failed."}`); }
    finally { if (requestId === assessmentId.current) setBusy(false); }
  }

  return <>
    <section className="location-tool-grid">
      <div className="surface-card location-map-card">
        <div className="location-panel-head"><div><span className="eyebrow">POINT SELECTION</span><h3>Choose a location to assess</h3><p>Search for a place or click anywhere on the map to move the marker.</p></div><button type="button" className="btn btn-secondary" onClick={useMyLocation}><LocateFixed size={16}/> Use my location</button></div>
        <div className="location-search-row"><PlaceSearch key={searchReset} label="Place or address" placeholder="Search a town, landmark or address" selected={point} onSelect={selectPlace}/>{point && <button type="button" className="btn btn-quiet" onClick={() => { selectPlace(null); setSearchReset((value) => value + 1); }}>Clear selection</button>}</div>
        <MapBase center={point ? [point.latitude, point.longitude] : [13.05, 75.7]} zoom={point ? 12 : 7} locations={selectedLocations} onPick={(picked) => pickCoordinates(picked.latitude, picked.longitude)}/>
        <div className="map-provider-line">Map © OpenStreetMap contributors · Search by Photon · select a point directly on the map</div>
      </div>
      <aside className="surface-card assessment-panel">
        <div className="eyebrow">ASSESSMENT</div><h3>Selected location</h3><LocationSummary point={point}/>
        {reverseBusy && <div className="inline-status"><span className="mini-spinner"/> Looking up place name…</div>}
        {reverseMessage && <div className="inline-note" role="status">{reverseMessage}</div>}
        <details className="manual-coordinates" open={manualOpen} onToggle={(event) => setManualOpen(event.currentTarget.open)}><summary>Enter coordinates manually</summary><form className="manual-coordinate-fields" onSubmit={(event) => { event.preventDefault(); pickCoordinates(manualLat, manualLon, "Manual coordinate selection"); }}><label className="field"><span>Latitude</span><input type="number" step="any" min="-90" max="90" required value={manualLat} onChange={(event) => setManualLat(event.target.value)} placeholder="13.3161"/></label><label className="field"><span>Longitude</span><input type="number" step="any" min="-180" max="180" required value={manualLon} onChange={(event) => setManualLon(event.target.value)} placeholder="75.7720"/></label><button type="submit" className="btn btn-secondary">Use these coordinates</button></form></details>
        {error && <div className="location-error" role="alert"><AlertTriangle size={16}/><span>{error}</span></div>}
        <button type="button" className="btn btn-primary assess-button" disabled={!point || busy} onClick={assess}>{busy ? <><span className="button-spinner"/> Assessing with Fusion…</> : <><Target size={17}/> Assess selected location</>}</button>
        {!result && !busy && <p className="assessment-note">No estimate is shown until the authenticated coordinate-inference API returns a result.</p>}
        <InferenceResult result={result} label="SELECTED LOCATION"/>
      </aside>
    </section>
  </>;
}

export function SafeRouteTool({ api }) {
  const [origin, setOrigin] = useState(null);
  const [destination, setDestination] = useState(null);
  const [route, setRoute] = useState(null);
  const [routeError, setRouteError] = useState("");
  const [routeBusy, setRouteBusy] = useState(false);
  const [endpointBusy, setEndpointBusy] = useState(false);
  const [endpointResults, setEndpointResults] = useState(null);
  const routeRequestId = useRef(0);
  const endpointRequestId = useRef(0);

  function changeEndpoint(setter, place) {
    routeRequestId.current += 1; endpointRequestId.current += 1;
    setRouteBusy(false); setEndpointBusy(false); setRoute(null); setRouteError(""); setEndpointResults(null); setter(place);
  }

  async function findRoute(event) {
    event.preventDefault(); setRoute(null); setRouteError(""); setEndpointResults(null);
    if (!origin || !destination) { setRouteError("Choose both an origin and destination from the place search results."); return; }
    if (!ROUTER_URL || ROUTER_URL.toLowerCase() === "disabled") { setRouteError("No routing service is configured. Set VITE_ROUTING_BASE_URL to a compatible OSRM service to request a road route."); return; }
    const requestId = ++routeRequestId.current;
    setRouteBusy(true);
    try {
      const actual = await requestRoadRoute(ROUTER_URL, origin, destination);
      if (requestId === routeRequestId.current) setRoute(actual);
    } catch (error) {
      if (requestId !== routeRequestId.current) return;
      const timedOut = error.name === "AbortError";
      setRouteError(timedOut ? "The routing request timed out. Try again later or configure a reachable OSRM-compatible service." : `Route unavailable: ${error.message || "the routing service could not be reached"}. Check the provider URL and try again.`);
    } finally { if (requestId === routeRequestId.current) setRouteBusy(false); }
  }

  async function assessEndpoints() {
    const requestId = ++endpointRequestId.current;
    setEndpointBusy(true); setEndpointResults(null);
    const outcomes = await Promise.all([origin, destination].map(async (point) => {
      try { return { place: point, result: await api("/predict/coordinate", { method: "POST", body: JSON.stringify({ latitude: point.latitude, longitude: point.longitude }) }) }; }
      catch (error) { return { place: point, error: error.message || "Endpoint assessment failed." }; }
    }));
    if (requestId === endpointRequestId.current) { setEndpointResults(outcomes); setEndpointBusy(false); }
  }

  const locations = useMemo(() => [origin, destination].filter(Boolean), [origin, destination]);
  const configured = Boolean(ROUTER_URL && ROUTER_URL.toLowerCase() !== "disabled");

  return <>
    <section className="surface-card route-intro-card"><div className="route-intro-icon"><Route size={21}/></div><div><span className="eyebrow">ROAD NETWORK ROUTE</span><h3>Plan a route between places</h3><p>Request road geometry and turn-by-turn steps from the configured routing provider. A computed route is not a safety guarantee.</p></div></section>
    <section className="route-layout">
      <div className="surface-card route-controls">
        <form onSubmit={findRoute}>
          <PlaceSearch label="Origin" placeholder="Search a starting place" selected={origin} onSelect={(place) => changeEndpoint(setOrigin, place)}/>
          <PlaceSearch label="Destination" placeholder="Search a destination" selected={destination} onSelect={(place) => changeEndpoint(setDestination, place)}/>
          <button type="submit" className="btn btn-primary assess-button" disabled={routeBusy || !origin || !destination}>{routeBusy ? <><span className="button-spinner"/> Requesting road route…</> : <><Navigation size={16}/> Show road route</>}</button>
        </form>
        <div className={`provider-state ${configured ? "provider-configured" : "provider-unavailable"}`}><span className="provider-dot"/><div><strong>{configured ? "Routing endpoint configured" : "Routing provider not configured"}</strong><span>{configured ? USING_OSRM_DEMO ? "OSRM public demo. Availability and throughput are not guaranteed." : "OSRM-compatible service. Reachability is checked when a route is requested." : "Set VITE_ROUTING_BASE_URL to a compatible OSRM service."}</span></div></div>
        {routeError && <div className="location-error route-error" role="alert"><AlertTriangle size={16}/><span>{routeError}</span></div>}
        {route && <div className="route-summary"><div className="route-metrics"><div><span>Road distance</span><strong>{(route.distance / 1000).toFixed(1)} km</strong></div><div><span>Estimated drive time</span><strong><Clock3 size={15}/>{Math.max(1, Math.round(route.duration / 60))} min</strong></div></div><p className="route-source-note"><Check size={14}/> Geometry and maneuvers returned by the routing provider. Travel time is a routing estimate.</p><h4>Directions</h4><ol className="direction-list">{route.steps.map((step, index) => <li key={`${index}-${step}`}><span>{index + 1}</span><p>{step}</p></li>)}</ol><button type="button" className="btn btn-secondary endpoint-risk-button" onClick={assessEndpoints} disabled={endpointBusy}>{endpointBusy ? <><span className="button-spinner"/> Running point inference…</> : <><Activity size={15}/> Assess endpoint conditions</>}</button></div>}
      </div>
      <div className="surface-card route-map-card"><div className="location-panel-head"><div><span className="eyebrow">{route ? "COMPUTED ROAD ROUTE" : "ROUTE PREVIEW"}</span><h3>{route ? "Provider-returned geometry" : "Selected endpoints"}</h3></div>{route && <span className="route-status-badge"><Check size={14}/> Route computed</span>}</div><MapBase center={origin ? [origin.latitude, origin.longitude] : [13.05, 75.7]} zoom={origin ? 11 : 7} locations={locations} routeCoordinates={route?.coordinates}/><div className="map-provider-line">Map © OpenStreetMap contributors · Road geometry from configured OSRM-compatible provider</div>
        <div className="route-risk-notice"><AlertTriangle size={17}/><div><strong>No route-wide hazard assessment</strong><p>The model API predicts a selected coordinate, not the road corridor. Endpoint estimates cannot establish the risk of every segment or identify a guaranteed safe route.</p></div></div>
        {endpointResults && <div className="endpoint-results"><h4>Independent endpoint estimates</h4><p>These are two point assessments from the Fusion model, not a corridor or route risk score.</p>{endpointResults.map((item) => item.result ? <InferenceResult key={`${item.place.latitude}-${item.place.longitude}`} result={item.result} label={item.place.label}/> : <div className="location-error" role="alert" key={`${item.place.latitude}-${item.place.longitude}`}><AlertTriangle size={15}/><span>{item.place.label}: {item.error}</span></div>)}</div>}
      </div>
    </section>
  </>;
}

