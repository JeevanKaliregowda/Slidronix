import test from "node:test";
import assert from "node:assert/strict";
import { photonResultsToPlaces, requestRoadRoute } from "../src/locationServices.js";

test("Photon results resolve real feature coordinates and omit malformed geometry", () => {
  const places = photonResultsToPlaces({ features: [
    { properties: { name: "Chikkamagaluru", state: "Karnataka", country: "India" }, geometry: { coordinates: [75.78, 13.32] } },
    { properties: { name: "Bad location" }, geometry: { coordinates: [null, null] } },
    { properties: { name: "Invalid bounds" }, geometry: { coordinates: [190, 95] } },
  ] });
  assert.equal(places.length, 1);
  assert.deepEqual(places[0], { label: "Chikkamagaluru, Karnataka, India", latitude: 13.32, longitude: 75.78 });
});

test("OSRM route response becomes real Leaflet geometry and maneuver steps", async () => {
  const response = { ok: true, json: async () => ({
    code: "Ok",
    routes: [{ distance: 2400, duration: 600, geometry: { coordinates: [[75.7, 13.3], [75.71, 13.31]] }, legs: [{ steps: [
      { name: "Market Road", distance: 1000, maneuver: { type: "depart" } },
      { name: "Hill Street", distance: 1400, maneuver: { type: "turn", modifier: "left" } },
      { name: "", distance: 0, maneuver: { type: "arrive" } },
    ] }] }],
  }) };
  const result = await requestRoadRoute("https://router.example", { latitude: 13.3, longitude: 75.7 }, { latitude: 13.31, longitude: 75.71 }, async (url) => {
    assert.match(url, /75\.7,13\.3;75\.71,13\.31/);
    assert.match(url, /geometries=geojson/);
    assert.match(url, /steps=true/);
    return response;
  });
  assert.deepEqual(result.coordinates, [[13.3, 75.7], [13.31, 75.71]]);
  assert.deepEqual(result.steps, ["Start onto Market Road for 1.0 km", "Turn left onto Hill Street for 1.4 km", "Arrive at your destination"]);
  assert.equal(result.distance, 2400);
  assert.equal(result.duration, 600);
});

test("routing failure and no-route responses never create substitute geometry", async () => {
  await assert.rejects(requestRoadRoute("https://router.example", { latitude: 13, longitude: 75 }, { latitude: 14, longitude: 76 }, async () => ({
    ok: true, json: async () => ({ code: "NoRoute", message: "No route found", routes: [] }),
  })), /No route found/);
  await assert.rejects(requestRoadRoute("https://router.example", { latitude: 13, longitude: 75 }, { latitude: 14, longitude: 76 }, async () => ({ ok: false, status: 503 })), /HTTP 503/);
  await assert.rejects(requestRoadRoute("disabled", { latitude: 13, longitude: 75 }, { latitude: 14, longitude: 76 }), /configured/);
  await assert.rejects(requestRoadRoute("https://router.example", { latitude: "bad", longitude: 75 }, { latitude: 14, longitude: 76 }), /valid origin and destination/);
  await assert.rejects(requestRoadRoute("https://router.example", { latitude: 91, longitude: 75 }, { latitude: 14, longitude: 76 }), /valid origin and destination/);
});
