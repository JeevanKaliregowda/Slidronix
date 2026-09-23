import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Database,
  Info,
  Map,
  MapPin,
  Mountain,
  Navigation,
  Satellite,
  ShieldCheck,
  SlidersHorizontal,
  TrendingUp,
  Waves,
} from "lucide-react";

import {
  CircleMarker,
  MapContainer,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
  LayersControl,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";

const { BaseLayer } = LayersControl;

const SATELLITE_URL =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";

const STREET_URL =
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

const RISK_COLORS = {
  Low: "#2F6B4F",
  Moderate: "#A16B16",
  High: "#B43B3B",
};

function FlyToLocation({ location }) {
  const map = useMap();

  useEffect(() => {
    if (location) {
      map.flyTo([location.Latitude, location.Longitude], 10, {
        duration: 0.8,
      });
    }
  }, [location, map]);

  return null;
}

function MapClickBlocker() {
  useMapEvents({
    click() {
      // Intentionally do nothing.
      // Only prediction points are selectable.
    },
  });

  return null;
}

function App() {
  const [points, setPoints] = useState([]);
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState("All");
  const [locationInfo, setLocationInfo] = useState(null);
  const [loadingLocation, setLoadingLocation] = useState(false);

  useEffect(() => {
    fetch("/data/risk_predictions.json")
      .then((response) => response.json())
      .then((data) => setPoints(data))
      .catch((error) => console.error("Prediction data error:", error));
  }, []);

  const filteredPoints = useMemo(() => {
    if (filter === "All") return points;
    return points.filter((point) => point.Risk_Level === filter);
  }, [points, filter]);

  const statistics = useMemo(() => {
    return {
      total: points.length,
      low: points.filter((p) => p.Risk_Level === "Low").length,
      moderate: points.filter((p) => p.Risk_Level === "Moderate").length,
      high: points.filter((p) => p.Risk_Level === "High").length,
    };
  }, [points]);

  async function selectPoint(point) {
    setSelected(point);
    setLocationInfo(null);
    setLoadingLocation(true);

    try {
      const url =
        `https://nominatim.openstreetmap.org/reverse` +
        `?format=jsonv2` +
        `&lat=${point.Latitude}` +
        `&lon=${point.Longitude}` +
        `&zoom=18` +
        `&addressdetails=1` +
        `&accept-language=en`;

      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setLocationInfo(data);
      }
    } catch (error) {
      console.error("Reverse geocoding failed:", error);
    } finally {
      setLoadingLocation(false);
    }
  }

  function scrollToSection(id) {
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  return (
    <div className="app-shell">

      {/* NAVIGATION */}
      <header className="top-nav">
        <div className="nav-inner">

          <button
            className="brand"
            onClick={() => scrollToSection("top")}
          >
            <span className="brand-mark">
              <Mountain size={21} strokeWidth={2.2} />
            </span>

            <span>
              <strong>SLIDRONIX</strong>
              <small>LANDSLIDE INTELLIGENCE</small>
            </span>
          </button>

          <nav className="nav-links">
            <button onClick={() => scrollToSection("risk-map")}>
              <Map size={17} />
              Risk Map
            </button>

            <button onClick={() => scrollToSection("model")}>
              <Activity size={17} />
              Model
            </button>

            <button onClick={() => scrollToSection("data")}>
              <Database size={17} />
              Data
            </button>

            <button onClick={() => scrollToSection("about")}>
              <Info size={17} />
              About
            </button>
          </nav>

        </div>
      </header>

      <main id="top">

        {/* HERO */}
        <section className="hero-section section-container">

          <div className="hero-copy">

            <div className="eyebrow">
              <ShieldCheck size={17} />
              WESTERN GHATS · RISK INTELLIGENCE
            </div>

            <h1>
              Landslide risk,
              <br />
              <span>mapped with data.</span>
            </h1>

            <p className="hero-description">
              Slidronix combines terrain, satellite, soil and rainfall
              information with a multimodal machine-learning model to
              estimate landslide probability across the study region.
            </p>

            <div className="hero-actions">
              <button
                className="primary-button"
                onClick={() => scrollToSection("risk-map")}
              >
                <Map size={18} />
                Explore Risk Map
              </button>

              <button
                className="secondary-button"
                onClick={() => scrollToSection("model")}
              >
                View Model
              </button>
            </div>

          </div>

          <div className="hero-visual">

            <div className="hero-stat hero-stat-one">
              <span>LOCATIONS</span>
              <strong>{statistics.total.toLocaleString()}</strong>
            </div>

            <div className="hero-stat hero-stat-two">
              <span>MODEL INPUTS</span>
              <strong>26</strong>
            </div>

            <div className="hero-map-preview">
              <Satellite size={38} />
              <span>WESTERN GHATS</span>
              <small>GIS RISK ANALYSIS</small>
            </div>

          </div>

        </section>

        {/* QUICK STATS */}
        <section className="stats-section section-container">

          <div className="stat-card">
            <span>Total Locations</span>
            <strong>{statistics.total.toLocaleString()}</strong>
            <small>Phase H prediction points</small>
          </div>

          <div className="stat-card">
            <span>High Risk</span>
            <strong>{statistics.high.toLocaleString()}</strong>
            <small>Model classification</small>
          </div>

          <div className="stat-card">
            <span>Moderate Risk</span>
            <strong>{statistics.moderate.toLocaleString()}</strong>
            <small>Model classification</small>
          </div>

          <div className="stat-card">
            <span>Low Risk</span>
            <strong>{statistics.low.toLocaleString()}</strong>
            <small>Model classification</small>
          </div>

        </section>

        {/* RISK MAP */}
        <section id="risk-map" className="content-section section-container">

          <div className="section-heading">

            <div>
              <div className="section-label">
                <Map size={17} />
                RISK MAP
              </div>

              <h2>Spatial landslide risk</h2>

              <p>
                Select a prediction point to inspect its location and
                environmental model inputs.
              </p>
            </div>

            <div className="filter-control">
              <SlidersHorizontal size={17} />

              <select
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              >
                <option value="All">All risk levels</option>
                <option value="Low">Low</option>
                <option value="Moderate">Moderate</option>
                <option value="High">High</option>
              </select>
            </div>

          </div>

          <div className="map-layout">

            <div className="map-panel">

              <MapContainer
                center={[12.3, 76.2]}
                zoom={7}
                scrollWheelZoom={true}
                className="risk-map"
              >

                <LayersControl position="topright">

                  <BaseLayer checked name="Satellite">
                    <TileLayer
                      url={SATELLITE_URL}
                      attribution="Tiles &copy; Esri"
                    />
                  </BaseLayer>

                  <BaseLayer name="Street Map">
                    <TileLayer
                      url={STREET_URL}
                      attribution="&copy; OpenStreetMap contributors"
                    />
                  </BaseLayer>

                </LayersControl>

                <MapClickBlocker />

                {selected && <FlyToLocation location={selected} />}

                {filteredPoints.map((point, index) => (
                  <CircleMarker
                    key={`${point.ID}-${index}`}
                    center={[point.Latitude, point.Longitude]}
                    radius={
                      selected?.ID === point.ID ? 9 : 5
                    }
                    pathOptions={{
                      color: "#FFFFFF",
                      weight: 1.5,
                      fillColor:
                        RISK_COLORS[point.Risk_Level] || "#737574",
                      fillOpacity:
                        selected?.ID === point.ID ? 1 : 0.82,
                    }}
                    eventHandlers={{
                      click: () => selectPoint(point),
                    }}
                  >
                    <Popup>
                      <strong>{point.Risk_Level} Risk</strong>
                      <br />
                      Probability:{" "}
                      {Number(
                        point.Landslide_Probability
                      ).toFixed(4)}
                    </Popup>
                  </CircleMarker>
                ))}

              </MapContainer>

              <div className="map-legend">

                <span>
                  <i style={{ background: RISK_COLORS.Low }} />
                  Low
                </span>

                <span>
                  <i style={{ background: RISK_COLORS.Moderate }} />
                  Moderate
                </span>

                <span>
                  <i style={{ background: RISK_COLORS.High }} />
                  High
                </span>

              </div>

            </div>

            {/* LOCATION PANEL */}
            <aside className="location-panel">

              {!selected ? (
                <div className="empty-location">

                  <MapPin size={32} />

                  <h3>Select a location</h3>

                  <p>
                    Click one of the prediction points on the map
                    to inspect the model output and environmental
                    features.
                  </p>

                </div>
              ) : (

                <>
                  <div className="location-header">

                    <div>
                      <span className="panel-label">
                        SELECTED LOCATION
                      </span>

                      <h3>
                        {loadingLocation
                          ? "Loading location..."
                          : locationInfo?.address?.village ||
                            locationInfo?.address?.town ||
                            locationInfo?.address?.city ||
                            locationInfo?.address?.county ||
                            "Mapped location"}
                      </h3>
                    </div>

                    <div
                      className="risk-badge"
                      style={{
                        color:
                          RISK_COLORS[selected.Risk_Level],
                        borderColor:
                          RISK_COLORS[selected.Risk_Level],
                      }}
                    >
                      {selected.Risk_Level}
                    </div>

                  </div>

                  {locationInfo && (
                    <div className="place-details">

                      <MapPin size={17} />

                      <span>
                        {locationInfo.display_name}
                      </span>

                    </div>
                  )}

                  <div className="probability-box">

                    <span>LANDSLIDE PROBABILITY</span>

                    <strong>
                      {(
                        Number(
                          selected.Landslide_Probability
                        ) * 100
                      ).toFixed(2)}
                      %
                    </strong>

                  </div>

                  <div className="coordinate-box">

                    <div>
                      <span>LATITUDE</span>
                      <strong>
                        {Number(selected.Latitude).toFixed(6)}
                      </strong>
                    </div>

                    <div>
                      <span>LONGITUDE</span>
                      <strong>
                        {Number(selected.Longitude).toFixed(6)}
                      </strong>
                    </div>

                  </div>

                  <div className="feature-grid">

                    <Feature
                      label="Elevation"
                      value={`${Number(
                        selected.Elevation_m
                      ).toFixed(1)} m`}
                    />

                    <Feature
                      label="Slope"
                      value={`${Number(
                        selected.Slope_deg
                      ).toFixed(1)}°`}
                    />

                    <Feature
                      label="30-day rainfall"
                      value={`${Number(
                        selected.rainfall_total_30d_mm
                      ).toFixed(1)} mm`}
                    />

                    <Feature
                      label="NDVI"
                      value={
                        selected.NDVI_mean == null
                          ? "N/A"
                          : Number(
                              selected.NDVI_mean
                            ).toFixed(3)
                      }
                    />

                    <Feature
                      label="Soil pH"
                      value={
                        selected.Soil_phh2o_0_5cm == null
                          ? "N/A"
                          : Number(
                              selected.Soil_phh2o_0_5cm
                            ).toFixed(2)
                      }
                    />

                    <Feature
                      label="Land cover"
                      value={
                        selected.Land_Cover_Type ?? "N/A"
                      }
                    />

                  </div>

                  <div className="point-id">
                    <Navigation size={15} />
                    {selected.ID}
                  </div>
                </>

              )}

            </aside>

          </div>

        </section>

        {/* MODEL */}
<section id="model" className="content-section section-container">

  <div className="section-heading">
    <div>
      <div className="section-label">
        <Activity size={17} />
        MODEL
      </div>

      <h2>Multimodal machine-learning pipeline</h2>

      <p>
        Slidronix evaluates the same spatially separated dataset
        through multiple model architectures before combining
        complementary representations through multimodal fusion.
      </p>
    </div>
  </div>

  <div className="model-architecture">

    <ModelBlock
      icon={<BarChart3 />}
      title="MLP"
      text="26-feature tabular encoder"
    />

    <ModelBlock
      icon={<Mountain />}
      title="CNN"
      text="32 × 32 terrain patches"
    />

    <ModelBlock
      icon={<Waves />}
      title="Transformer"
      text="30-day rainfall sequence"
    />

    <ModelBlock
      icon={<TrendingUp />}
      title="Fusion Model"
      text="Tabular + CNN + Transformer"
    />

  </div>

  <div className="model-details">

    <div className="model-detail">
      <strong>MLP</strong>
      <span>
        Processes terrain, rainfall summaries, NDVI,
        land cover and soil features.
      </span>
    </div>

    <div className="model-detail">
      <strong>CNN</strong>
      <span>
        Learns spatial patterns from 32 × 32 elevation
        and slope patches.
      </span>
    </div>

    <div className="model-detail">
      <strong>Transformer</strong>
      <span>
        Encodes the 30-observation rainfall sequence
        using temporal attention.
      </span>
    </div>

    <div className="model-detail">
      <strong>Fusion</strong>
      <span>
        Combines the tabular, spatial and rainfall
        representations to estimate landslide probability.
      </span>
    </div>

  </div>

</section>

       {/* DATA */}
<section id="data" className="content-section section-container">

  <div className="section-heading">
    <div>
      <div className="section-label">
        <Database size={17} />
        DATA
      </div>

      <h2>Environmental data sources</h2>

      <p>
        Slidronix combines geospatial, satellite, soil and
        rainfall datasets to construct the model feature space.
      </p>
    </div>
  </div>

  <div className="data-source-grid">

    <DataSource
      name="Copernicus DEM GLO-30"
      category="Terrain"
      description="Digital elevation data used to derive elevation and slope."
      icon={<Mountain />}
    />

    <DataSource
      name="MODIS MOD13Q1.061"
      category="Satellite · NDVI"
      description="16-day vegetation index observations used to derive NDVI features."
      icon={<Satellite />}
    />

    <DataSource
      name="MODIS MCD12Q1.061"
      category="Satellite · Land Cover"
      description="Annual land-cover classification used as an environmental feature."
      icon={<Map />}
    />

    <DataSource
      name="SoilGrids"
      category="Soil"
      description="0–5 cm soil properties including clay, sand, silt, SOC, bulk density and pH."
      icon={<Database />}
    />

    <DataSource
      name="CHIRPS v3"
      category="Rainfall"
      description="Daily satellite rainfall used to construct the 30-day rainfall sequence and summary features."
      icon={<Waves />}
    />

    <DataSource
      name="Historical Landslide Inventory"
      category="Reference Dataset"
      description="Historical landslide records used to construct the positive and background spatial dataset."
      icon={<MapPin />}
    />

  </div>

  <div className="data-summary">

    <div>
      <strong>26</strong>
      <span>Tabular features</span>
    </div>

    <div>
      <strong>30</strong>
      <span>Daily rainfall observations</span>
    </div>

    <div>
      <strong>32×32</strong>
      <span>Terrain patch size</span>
    </div>

    <div>
      <strong>13,982</strong>
      <span>Phase E balanced samples</span>
    </div>

  </div>

</section>
        {/* ABOUT */}
        <section id="about" className="content-section section-container">

          <div className="about-section">

            <div>
              <div className="section-label">
                <Info size={17} />
                ABOUT SLIDRONIX
              </div>

              <h2>From environmental data to risk information.</h2>
            </div>

            <div className="about-copy">

              <p>
                Slidronix is an AI-based landslide risk assessment
                system focused on the Western Ghats study region.
              </p>

              <p>
                The project integrates geospatial terrain features,
                satellite-derived vegetation and land-cover
                information, soil properties and rainfall sequences
                into machine-learning and deep-learning models.
              </p>

              <div className="future-note">

                <strong>Next development stage</strong>

                <span>
                  The dashboard can be extended so a user selects
                  any coordinate, the system retrieves the latest
                  available environmental data, builds the required
                  model features and performs a fresh inference.
                </span>

              </div>

            </div>

          </div>

        </section>

      </main>

      <footer className="footer">
        <div className="section-container footer-inner">
          <strong>SLIDRONIX</strong>
          <span>AI Landslide Risk Assessment</span>
        </div>
      </footer>

    </div>
  );
}

function Feature({ label, value }) {
  return (
    <div className="feature-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ModelBlock({ icon, title, text }) {
  return (
    <div className="model-block">
      <div className="model-icon">{icon}</div>
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}

function DataSource({ name, category, description, icon }) {
  return (
    <div className="data-source-card">

      <div className="data-source-icon">
        {icon}
      </div>

      <div className="data-source-content">

        <span className="data-source-category">
          {category}
        </span>

        <h3>{name}</h3>

        <p>{description}</p>

      </div>

    </div>
  );
}
function DataRow({ title, value }) {
  return (
    <div className="data-row">
      <strong>{title}</strong>
      <span>{value}</span>
    </div>
  );
}

export default App;