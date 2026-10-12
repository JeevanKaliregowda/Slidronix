export function photonFeatureToPlace(feature) {
  const properties = feature?.properties || {};
  const parts = [
    properties.name,
    properties.housenumber && properties.street ? `${properties.housenumber} ${properties.street}` : properties.street,
    properties.suburb,
    properties.city || properties.locality || properties.district,
    properties.state,
    properties.country,
  ].filter(Boolean).filter((part, index, all) => all.indexOf(part) === index);
  const [longitude, latitude] = feature?.geometry?.coordinates || [];
  if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90 || !Number.isFinite(longitude) || longitude < -180 || longitude > 180) return null;
  return { label: parts.join(", ") || `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`, latitude, longitude };
}

export function photonResultsToPlaces(data) {
  return (data?.features || []).map(photonFeatureToPlace).filter(Boolean);
}

export function maneuverInstruction(step) {
  const { type, modifier } = step.maneuver || {};
  if (type === "arrive") return "Arrive at your destination";
  const road = step.name ? ` onto ${step.name}` : "";
  const actions = {
    depart: "Start", turn: `Turn ${modifier || "ahead"}`, "new name": "Continue", continue: "Continue",
    merge: "Merge", fork: `Keep ${modifier || "straight"}`, "on ramp": "Take the ramp",
    "off ramp": "Take the exit", roundabout: "Enter the roundabout", rotary: "Enter the rotary",
    "end of road": `Turn ${modifier || "at the end of the road"}`,
  };
  const action = actions[type] || "Continue";
  return `${action}${road}${step.distance > 0 ? ` for ${(step.distance / 1000).toFixed(step.distance >= 1000 ? 1 : 2)} km` : ""}`;
}

export async function requestRoadRoute(baseUrl, origin, destination, fetcher = fetch) {
  if (!baseUrl || baseUrl.toLowerCase() === "disabled") throw new Error("No routing service is configured.");
  const coordinates = `${destinationSafe(origin.longitude, -180, 180)},${destinationSafe(origin.latitude, -90, 90)};${destinationSafe(destination.longitude, -180, 180)},${destinationSafe(destination.latitude, -90, 90)}`;
  const url = `${baseUrl.replace(/\/$/, "")}/route/v1/driving/${coordinates}?overview=full&geometries=geojson&steps=true&alternatives=false`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 25000);
  let response;
  try { response = await fetcher(url, { signal: controller.signal, headers: { Accept: "application/json" } }); }
  finally { clearTimeout(timeout); }
  if (!response.ok) throw new Error(`Routing service returned HTTP ${response.status}.`);
  const data = await response.json();
  const route = data?.routes?.[0];
  const coordinatesOut = route?.geometry?.coordinates;
  if (data?.code !== "Ok" || !coordinatesOut?.length || !Array.isArray(route.legs)) throw new Error(data?.message || "No road route was returned for those places.");
  const points = coordinatesOut.filter((point) => Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]));
  if (points.length < 2) throw new Error("The routing provider returned incomplete route geometry.");
  return {
    distance: route.distance,
    duration: route.duration,
    coordinates: points.map(([longitude, latitude]) => [latitude, longitude]),
    steps: route.legs.flatMap((leg) => leg.steps || []).map(maneuverInstruction),
  };
}

function destinationSafe(value, minimum, maximum) {
  const coordinate = Number(value);
  if (!Number.isFinite(coordinate) || coordinate < minimum || coordinate > maximum) throw new Error("Choose valid origin and destination coordinates.");
  return coordinate;
}
