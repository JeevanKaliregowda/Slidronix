import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from pathlib import Path
from shapely.geometry import MultiPoint
from sklearn.neighbors import BallTree


# --------------------------------------------------
# PATHS AND CONSTANTS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landslide_inventory_cleaned.csv"
)

GEOJSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "india_district.geojson"
)

OUTPUT_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "landslide_binary_dataset.csv"
)

OUTPUT_CHART_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
    / "landslide_positive_vs_background.png"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "binary_dataset_summary.txt"
)

TARGET_STATES = [
    "Maharashtra",
    "Goa",
    "Karnataka",
    "Kerala",
    "Tamil Nadu"
]

MIN_SEPARATION_KM = 1.0
EARTH_RADIUS_KM = 6371.0088
RANDOM_SEED = 42


# --------------------------------------------------
# 1. LOAD POSITIVE INVENTORY & INVESTIGATE DUPLICATES
# --------------------------------------------------

def load_positive_dataset_and_inspect_duplicates(csv_path):
    print("=" * 65)
    print("PHASE 0.5: WESTERN GHATS POLYGON BACKGROUND SAMPLING")
    print("=" * 65)
    
    df_raw = pd.read_csv(csv_path)
    df_raw["State"] = df_raw["State"].astype(str).str.strip()
    
    # Filter to 5 Western Ghats states
    positive_df = df_raw[df_raw["State"].isin(TARGET_STATES)].copy()
    positive_df["Landslide"] = 1
    positive_df["Sample_Type"] = "Landslide"
    
    pos_count = len(positive_df)
    
    # Duplicate investigation
    dup_groups = positive_df[positive_df.duplicated(subset=["Latitude", "Longitude"], keep=False)]
    grouped = dup_groups.groupby(["Latitude", "Longitude"])
    
    num_dup_groups = len(grouped)
    num_dup_records = len(dup_groups)
    
    diff_movement_groups = 0
    diff_time_groups = 0
    diff_both_groups = 0
    
    for _, grp in grouped:
        has_diff_mov = grp["Movement Type"].dropna().nunique() > 1
        has_diff_time = (grp["Initiation_Year"].dropna().nunique() > 1) or (grp["History_date"].dropna().nunique() > 1)
        
        if has_diff_mov:
            diff_movement_groups += 1
        if has_diff_time:
            diff_time_groups += 1
        if has_diff_mov and has_diff_time:
            diff_both_groups += 1
            
    dup_stats = {
        "pos_count": pos_count,
        "num_dup_groups": num_dup_groups,
        "num_dup_records": num_dup_records,
        "diff_movement_groups": diff_movement_groups,
        "diff_time_groups": diff_time_groups,
        "diff_both_groups": diff_both_groups
    }
    
    print(f"\nPositive Records in 5 Target States: {pos_count}")
    print("Positive Duplicate Coordinates Analysis:")
    print(f"  - Duplicate Coordinate Groups: {num_dup_groups}")
    print(f"  - Total Records Involved: {num_dup_records}")
    print(f"  - Groups with Different Movement Types: {diff_movement_groups} ({diff_movement_groups/max(1, num_dup_groups)*100:.1f}%)")
    print(f"  - Groups with Different Years/Events: {diff_time_groups} ({diff_time_groups/max(1, num_dup_groups)*100:.1f}%)")
    print(f"  - Groups with Both Different Types & Events: {diff_both_groups} ({diff_both_groups/max(1, num_dup_groups)*100:.1f}%)")
    
    return positive_df, dup_stats


# --------------------------------------------------
# 2. CONSTRUCT WESTERN GHATS STUDY AREA POLYGON
# --------------------------------------------------

def construct_western_ghats_polygon(positive_df, geojson_path):
    print("\n" + "-" * 65)
    print("CONSTRUCTING WESTERN GHATS STUDY BOUNDARY POLYGON")
    print("-" * 65)
    
    # 1. Load official India district GeoJSON
    districts_gdf = gpd.read_file(geojson_path)
    wg_districts = districts_gdf[districts_gdf["NAME_1"].isin(TARGET_STATES)].copy()
    union_districts = wg_districts.union_all()
    
    # 2. Concave hull around landslide points with 0.18° (~20 km) smooth spatial buffer
    points = MultiPoint(list(zip(positive_df["Longitude"], positive_df["Latitude"])))
    concave_bnd = shapely.concave_hull(points, ratio=0.20)
    buffered_concave = concave_bnd.buffer(0.18)
    
    # 3. Intersection of district boundaries and buffered concave hull
    wg_polygon = union_districts.intersection(buffered_concave)
    wg_gdf = gpd.GeoDataFrame({"geometry": [wg_polygon]}, crs="EPSG:4326")
    
    # Calculate area using Equal Area projection EPSG:7755 (India Albers Equal Area)
    wg_projected = wg_gdf.to_crs("EPSG:7755")
    area_sq_km = float(wg_projected.geometry.area.sum() / 1e6)
    
    print(f"Authoritative Geospatial Source: India District GeoJSON (geohacker/DataMeet) + Landslide Spatial Envelope")
    print(f"Calculated Western Ghats Study Area Polygon: {area_sq_km:,.2f} sq km")
    
    # Check positive containment
    pos_gdf = gpd.GeoDataFrame(
        positive_df,
        geometry=gpd.points_from_xy(positive_df["Longitude"], positive_df["Latitude"]),
        crs="EPSG:4326"
    )
    inside_count = int(pos_gdf.within(wg_polygon).sum())
    print(f"Positive Locations Inside Study Polygon: {inside_count} / {len(positive_df)} ({inside_count/len(positive_df)*100:.2f}%)")
    
    return wg_polygon, wg_gdf, area_sq_km


# --------------------------------------------------
# 3. POLYGON-BASED REPRODUCIBLE BACKGROUND SAMPLING
# --------------------------------------------------

def generate_polygon_background_samples(positive_df, wg_polygon, n_samples=7885, min_dist_km=1.0, seed=42):
    print("\n" + "-" * 65)
    print(f"SAMPLING {n_samples} BACKGROUND POINTS INSIDE WESTERN GHATS POLYGON")
    print("-" * 65)
    
    rng = np.random.default_rng(seed)
    
    # Spatial BallTree of positive coordinates in radians
    pos_coords_rad = np.radians(positive_df[["Latitude", "Longitude"]].values)
    tree = BallTree(pos_coords_rad, metric="haversine")
    min_dist_rad = min_dist_km / EARTH_RADIUS_KM
    
    # Bounding box of the Western Ghats polygon
    minx, miny, maxx, maxy = wg_polygon.bounds
    
    accepted_lats = []
    accepted_lons = []
    seen_coords = set(zip(positive_df["Latitude"].round(6), positive_df["Longitude"].round(6)))
    
    batch_size = 30000
    total_tested = 0
    valid_candidates_in_polygon = 0
    
    while len(accepted_lats) < n_samples and total_tested < 500000:
        cand_lons = rng.uniform(minx, maxx, batch_size)
        cand_lats = rng.uniform(miny, maxy, batch_size)
        total_tested += batch_size
        
        # Point-in-polygon check
        points_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(cand_lons, cand_lats),
            crs="EPSG:4326"
        )
        inside_mask = points_gdf.within(wg_polygon).values
        valid_candidates_in_polygon += int(inside_mask.sum())
        
        inside_lats = cand_lats[inside_mask]
        inside_lons = cand_lons[inside_mask]
        
        if len(inside_lats) == 0:
            continue
            
        # Distance check (>= 1.0 km from positive coordinates)
        cand_coords_rad = np.radians(np.column_stack([inside_lats, inside_lons]))
        dist_rad, _ = tree.query(cand_coords_rad, k=1)
        dist_rad = dist_rad.flatten()
        
        valid_dist_mask = dist_rad >= min_dist_rad
        
        for lat, lon in zip(inside_lats[valid_dist_mask], inside_lons[valid_dist_mask]):
            lat_r, lon_r = round(float(lat), 6), round(float(lon), 6)
            if (lat_r, lon_r) not in seen_coords:
                seen_coords.add((lat_r, lon_r))
                accepted_lats.append(lat_r)
                accepted_lons.append(lon_r)
                if len(accepted_lats) == n_samples:
                    break
                    
    print(f"Total Candidates Generated in Bounding Box: {total_tested:,}")
    print(f"Candidates Falling Inside Polygon: {valid_candidates_in_polygon:,}")
    print(f"Accepted Valid Background Samples: {len(accepted_lats):,}")
    
    bg_df = pd.DataFrame({
        "S.No": range(34118, 34118 + len(accepted_lats)),
        "Landslide": 0,
        "Sample_Type": "Background",
        "Latitude": accepted_lats,
        "Longitude": accepted_lons,
        "State": "Western Ghats Region",
        "District": "Background / Unspecified",
        "Subdivision Or Taluk": np.nan,
        "Material Involved": np.nan,
        "Movement Type": np.nan,
        "Initiation_Year": np.nan,
        "History_date": np.nan,
        "Slide_Name": np.nan
    })
    
    return bg_df


# --------------------------------------------------
# 4. VALIDATION & DISTANCE METRICS
# --------------------------------------------------

def validate_binary_dataset(positive_df, background_df, area_sq_km, min_dist_km=1.0):
    print("\n" + "-" * 65)
    print("VALIDATING POLYGON BINARY DATASET METRICS")
    print("-" * 65)
    
    columns = [
        "S.No", "Landslide", "Sample_Type", "Latitude", "Longitude",
        "State", "District", "Subdivision Or Taluk", "Material Involved",
        "Movement Type", "Initiation_Year", "History_date", "Slide_Name"
    ]
    
    combined_df = pd.concat([positive_df[columns], background_df[columns]], ignore_index=True)
    
    pos_count = int((combined_df["Landslide"] == 1).sum())
    bg_count = int((combined_df["Landslide"] == 0).sum())
    total_count = len(combined_df)
    class_ratio = f"{pos_count} : {bg_count} ({pos_count/total_count*100:.1f}% : {bg_count/total_count*100:.1f}%)"
    
    pos_dup_count = int(positive_df.duplicated(subset=["Latitude", "Longitude"]).sum())
    bg_dup_count = int(background_df.duplicated(subset=["Latitude", "Longitude"]).sum())
    
    pos_coords_set = set(zip(positive_df["Latitude"].round(6), positive_df["Longitude"].round(6)))
    bg_coords_set = set(zip(background_df["Latitude"].round(6), background_df["Longitude"].round(6)))
    overlap_dup_count = len(pos_coords_set.intersection(bg_coords_set))
    
    # Haversine distance to nearest positive point
    pos_coords_rad = np.radians(positive_df[["Latitude", "Longitude"]].values)
    bg_coords_rad = np.radians(background_df[["Latitude", "Longitude"]].values)
    
    tree = BallTree(pos_coords_rad, metric="haversine")
    distances_rad, _ = tree.query(bg_coords_rad, k=1)
    min_distances_km = distances_rad.flatten() * EARTH_RADIUS_KM
    
    actual_min_dist_km = float(min_distances_km.min())
    mean_dist_km = float(min_distances_km.mean())
    max_dist_km = float(min_distances_km.max())
    
    sample_density = bg_count / area_sq_km
    
    lat_range = (float(combined_df["Latitude"].min()), float(combined_df["Latitude"].max()))
    lon_range = (float(combined_df["Longitude"].min()), float(combined_df["Longitude"].max()))
    
    print(f"Total Positive Samples ($Y=1$): {pos_count}")
    print(f"Total Background Samples ($Y=0$): {bg_count}")
    print(f"Total Binary Samples: {total_count}")
    print(f"Class Balance Ratio: {class_ratio}")
    print(f"Background Internal Duplicates: {bg_dup_count}")
    print(f"Background-Positive Overlaps: {overlap_dup_count}")
    print(f"Minimum Pos-Bg Distance: {actual_min_dist_km:.4f} km (Required >= {min_dist_km} km)")
    print(f"Mean Nearest-Positive Distance: {mean_dist_km:.4f} km")
    print(f"Max Nearest-Positive Distance: {max_dist_km:.4f} km")
    print(f"Western Ghats Area Represented: {area_sq_km:,.2f} sq km ({sample_density:.4f} background samples/sq km)")
    
    assert actual_min_dist_km >= min_dist_km, f"Distance violation! Min dist {actual_min_dist_km:.4f} km < {min_dist_km} km"
    assert bg_dup_count == 0, f"Background duplicate coordinates detected: {bg_dup_count}"
    assert overlap_dup_count == 0, f"Background-Positive overlap detected: {overlap_dup_count}"
    
    stats = {
        "pos_count": pos_count,
        "bg_count": bg_count,
        "total_count": total_count,
        "class_ratio": class_ratio,
        "pos_dup_count": pos_dup_count,
        "bg_dup_count": bg_dup_count,
        "overlap_dup_count": overlap_dup_count,
        "actual_min_dist_km": actual_min_dist_km,
        "mean_dist_km": mean_dist_km,
        "max_dist_km": max_dist_km,
        "area_sq_km": area_sq_km,
        "sample_density": sample_density,
        "lat_range": lat_range,
        "lon_range": lon_range
    }
    
    return combined_df, stats


# --------------------------------------------------
# 5. SAVE OUTPUTS & GENERATE POLYGON VISUALIZATION
# --------------------------------------------------

def save_outputs_and_polygon_visualization(combined_df, stats, dup_stats, wg_gdf):
    print("\n" + "-" * 65)
    print("SAVING DATASET, SUMMARY REPORT, AND POLYGON MAP")
    print("-" * 65)
    
    # 1. Save binary dataset CSV
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[OK] Saved binary dataset to: {OUTPUT_CSV_PATH}")
    
    # 2. Save validation summary report
    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("SLIDRONIX PHASE 0.5: WESTERN GHATS POLYGON BINARY DATASET SUMMARY\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Total Positive Samples (Landslide = 1): {stats['pos_count']}\n")
        f.write(f"Total Background Samples (Landslide = 0): {stats['bg_count']}\n")
        f.write(f"Total Dataset Size: {stats['total_count']}\n")
        f.write(f"Class Balance Ratio: {stats['class_ratio']}\n\n")
        
        f.write("DUPLICATE POSITIVE COORDINATE INVESTIGATION:\n")
        f.write(f"  - Duplicate Coordinate Groups: {dup_stats['num_dup_groups']}\n")
        f.write(f"  - Total Positive Records Involved: {dup_stats['num_dup_records']}\n")
        f.write(f"  - Groups with Different Movement Types: {dup_stats['diff_movement_groups']} ({dup_stats['diff_movement_groups']/max(1, dup_stats['num_dup_groups'])*100:.1f}%)\n")
        f.write(f"  - Groups with Different Initiation Years/Events: {dup_stats['diff_time_groups']} ({dup_stats['diff_time_groups']/max(1, dup_stats['num_dup_groups'])*100:.1f}%)\n")
        f.write(f"  - Groups with Both Different Types & Events: {dup_stats['diff_both_groups']} ({dup_stats['diff_both_groups']/max(1, dup_stats['num_dup_groups'])*100:.1f}%)\n\n")
        
        f.write("BACKGROUND DEDUPLICATION & SPATIAL SEPARATION METRICS:\n")
        f.write(f"  - Background Internal Duplicate Coordinates: {stats['bg_dup_count']}\n")
        f.write(f"  - Background-Positive Overlapping Coordinates: {stats['overlap_dup_count']}\n")
        f.write(f"  - Minimum Distance (Background to Positive): {stats['actual_min_dist_km']:.4f} km\n")
        f.write(f"  - Mean Nearest-Positive Distance: {stats['mean_dist_km']:.4f} km\n")
        f.write(f"  - Max Distance (Background to Positive): {stats['max_dist_km']:.4f} km\n\n")
        
        f.write("STUDY AREA POLYGON COVERAGE:\n")
        f.write(f"  - Western Ghats Polygon Area: {stats['area_sq_km']:,.2f} sq km\n")
        f.write(f"  - Sampling Density: {stats['sample_density']:.4f} background samples/sq km\n")
        f.write(f"  - Latitude Range: [{stats['lat_range'][0]:.6f}, {stats['lat_range'][1]:.6f}]\n")
        f.write(f"  - Longitude Range: [{stats['lon_range'][0]:.6f}, {stats['lon_range'][1]:.6f}]\n\n")
        
        f.write("METHODOLOGICAL ASSUMPTIONS & DESIGN DECISIONS:\n")
        f.write("1. Western Ghats Study Boundary is derived by intersecting official Indian district geometries with a 20 km smooth spatial envelope of historical landslides.\n")
        f.write("2. Background samples are generated strictly inside the Western Ghats polygon domain, avoiding rectangular plain/ocean sampling.\n")
        f.write("3. Minimum geodesic distance of 1.0 km ensures background points stay outside immediate slope trigger zones.\n")
        f.write("4. Duplicate positive coordinates represent distinct historical landslide events or movement types recorded at shared regional coordinates.\n")
    print(f"[OK] Saved summary report to: {OUTPUT_SUMMARY_PATH}")
    
    # 3. Generate high-resolution map with Western Ghats polygon boundary
    OUTPUT_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Plot Western Ghats Polygon Boundary
    wg_gdf.plot(
        ax=ax,
        facecolor="#e0f2fe",
        edgecolor="#0284c7",
        linewidth=1.5,
        alpha=0.4,
        label="Western Ghats Study Boundary Polygon"
    )
    
    pos_mask = combined_df["Landslide"] == 1
    bg_mask = combined_df["Landslide"] == 0
    
    # Plot Background Samples
    ax.scatter(
        combined_df.loc[bg_mask, "Longitude"],
        combined_df.loc[bg_mask, "Latitude"],
        c="#2563eb",
        s=7,
        alpha=0.45,
        label="Background Samples (Landslide = 0)"
    )
    
    # Plot Positive Landslide Locations
    ax.scatter(
        combined_df.loc[pos_mask, "Longitude"],
        combined_df.loc[pos_mask, "Latitude"],
        c="#dc2626",
        s=9,
        alpha=0.65,
        label="Positive Landslides (Landslide = 1)"
    )
    
    ax.set_title("Western Ghats Landslide Susceptibility Study Domain\nPositive Observations, Background Samples, and Polygon Boundary", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", edgecolor="none")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved polygon spatial visualization to: {OUTPUT_CHART_PATH}")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():
    positive_df, dup_stats = load_positive_dataset_and_inspect_duplicates(INPUT_CSV_PATH)
    wg_polygon, wg_gdf, area_sq_km = construct_western_ghats_polygon(positive_df, GEOJSON_PATH)
    
    background_df = generate_polygon_background_samples(
        positive_df=positive_df,
        wg_polygon=wg_polygon,
        n_samples=len(positive_df),
        min_dist_km=MIN_SEPARATION_KM,
        seed=RANDOM_SEED
    )
    
    combined_df, stats = validate_binary_dataset(
        positive_df=positive_df,
        background_df=background_df,
        area_sq_km=area_sq_km,
        min_dist_km=MIN_SEPARATION_KM
    )
    
    save_outputs_and_polygon_visualization(combined_df, stats, dup_stats, wg_gdf)
    
    print("\n" + "=" * 65)
    print("PHASE 0.5 POLYGON REWORK SUCCESSFULLY COMPLETED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
