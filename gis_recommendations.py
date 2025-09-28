import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# ----------------------------
# CONFIG (tune as you like)
# ----------------------------
RADIUS_M = 1000  # 1 km catchment
WEIGHTS = {
    "hospital_boost": 200,
    "alfa_near_boost": 150,
    "no_our_branches_boost": 100,
    "our_branch_penalty_each": -80,
}
# UTM metric CRS for Alexandria (around 29.95E -> UTM zone 35N)
CRS_WGS84 = 4326
CRS_METRIC = 32635

# ----------------------------
# FILES (must exist in same folder)
# ----------------------------
FILES = {
    "population": "hex_alex_centroidsnew.csv",  # columns: Latitude, Longitude, VALUE_sum
    "mokhtabar":  "AlMokhtabar.csv",
    "borg":       "AlBorg.csv",
    "alfa":       "Alfa.csv",
    "hospitals":  "Hospitals.csv",
}

OUTPUTS = [
    "Recommended_Branches.csv",
    "Recommended_Branches.geojson",
    "All_Population_Scored.geojson",
]

# ----------------------------
# HELPERS
# ----------------------------
def load_points_csv(path, lat_col="Latitude", lon_col="Longitude"):
    """Load a CSV into a GeoDataFrame of points (WGS84)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    # standardize column names
    df.columns = [c.strip() for c in df.columns]
    if lat_col not in df.columns or lon_col not in df.columns:
        lat_guess = next((c for c in df.columns if c.lower() in ["lat", "latitude"]), None)
        lon_guess = next((c for c in df.columns if c.lower() in ["lng", "lon", "longitude"]), None)
        if not lat_guess or not lon_guess:
            raise ValueError(f"{path}: latitude/longitude columns not found. Columns: {df.columns.tolist()}")
        lat_col, lon_col = lat_guess, lon_guess

    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(x=df[lon_col], y=df[lat_col]),
        crs=f"EPSG:{CRS_WGS84}",
    )
    return gdf

def to_metric(gdf):
    return gdf.to_crs(epsg=CRS_METRIC)

def nearest_distance_m(source_points_metric, target_points_metric):
    """For each source point, get distance (meters) to nearest target point. np.inf if target empty."""
    if target_points_metric.empty:
        return np.full(len(source_points_metric), np.inf)
    out = gpd.sjoin_nearest(
        source_points_metric, target_points_metric[["geometry"]],
        how="left", distance_col="__dist_m"
    )
    return out["__dist_m"].fillna(np.inf).to_numpy()

def count_within_radius(source_points_metric, target_points_metric, radius_m):
    """For each source point, count how many target points fall within 'radius_m'."""
    if target_points_metric.empty:
        return np.zeros(len(source_points_metric), dtype=int)
    buffers = source_points_metric.copy()
    buffers["geometry"] = buffers.buffer(radius_m)
    joined = gpd.sjoin(target_points_metric, buffers[["geometry"]], predicate="within", how="left")
    counts = joined.groupby(joined.index_right).size()
    return buffers.assign(_tmp_idx=range(len(buffers))).set_index("_tmp_idx").index.to_series().map(counts).fillna(0).astype(int).values

# ----------------------------
# MAIN
# ----------------------------
def main():
    # --- Auto-clean old outputs ---
    for f in OUTPUTS:
        if os.path.exists(f):
            os.remove(f)
            print(f"🧹 Removed old {f}")

    pop = load_points_csv(FILES["population"])
    if "VALUE_sum" not in pop.columns:
        pop_col = next((c for c in pop.columns if c.lower() in ["value", "value_sum", "population", "pop", "weight"]), None)
        if not pop_col:
            raise ValueError("Population file must include VALUE_sum column (or a recognizable alternative).")
        pop = pop.rename(columns={pop_col: "VALUE_sum"})

    mok = load_points_csv(FILES["mokhtabar"])
    brg = load_points_csv(FILES["borg"])
    alf = load_points_csv(FILES["alfa"])
    hos = load_points_csv(FILES["hospitals"])

    # Project to metric CRS
    pop_m  = to_metric(pop)
    mok_m  = to_metric(mok)
    brg_m  = to_metric(brg)
    alf_m  = to_metric(alf)
    hos_m  = to_metric(hos)

    # Distances (meters)
    d_mok_m = nearest_distance_m(pop_m, mok_m)
    d_brg_m = nearest_distance_m(pop_m, brg_m)
    d_alf_m = nearest_distance_m(pop_m, alf_m)
    d_hos_m = nearest_distance_m(pop_m, hos_m)

    # Counts within radius
    c_mok = count_within_radius(pop_m, mok_m, RADIUS_M)
    c_brg = count_within_radius(pop_m, brg_m, RADIUS_M)
    c_alf = count_within_radius(pop_m, alf_m, RADIUS_M)
    c_hos = count_within_radius(pop_m, hos_m, RADIUS_M)

    # Score
    score = pop["VALUE_sum"].astype(float).to_numpy().copy()
    score += (c_hos > 0).astype(int) * WEIGHTS["hospital_boost"]
    score += (c_alf > 0).astype(int) * WEIGHTS["alfa_near_boost"]
    our_near = (c_mok + c_brg) > 0
    score += (~our_near).astype(int) * WEIGHTS["no_our_branches_boost"]
    score += c_mok * WEIGHTS["our_branch_penalty_each"]
    score += c_brg * WEIGHTS["our_branch_penalty_each"]

    # Build results
    out = pop.copy()
    out["Population"] = pop["VALUE_sum"].astype(float)
    out["Dist_Mokhtabar_m"] = np.round(d_mok_m, 1)
    out["Dist_Borg_m"]      = np.round(d_brg_m, 1)
    out["Dist_Alfa_m"]      = np.round(d_alf_m, 1)
    out["Dist_Hospital_m"]  = np.round(d_hos_m, 1)
    out["Count_Mokhtabar_1km"] = c_mok
    out["Count_Borg_1km"]      = c_brg
    out["Count_Alfa_1km"]      = c_alf
    out["Count_Hospital_1km"]  = c_hos
    out["Gap_Score"] = np.round(score, 2)

    # Rank & Top 10
    out_sorted = out.sort_values("Gap_Score", ascending=False).reset_index(drop=True)
    top10 = out_sorted.head(10).copy()

    # Always reset Name column
    if "Name" in top10.columns:
        top10 = top10.drop(columns=["Name"])
    top10.insert(0, "Name", top10.index.map(lambda i: f"Recommended Site #{i+1}"))

    # Save for HTML map
    cols_for_map = ["Name", "Latitude", "Longitude", "Population", "Gap_Score",
                    "Dist_Mokhtabar_m", "Dist_Borg_m", "Dist_Alfa_m", "Dist_Hospital_m",
                    "Count_Mokhtabar_1km", "Count_Borg_1km", "Count_Alfa_1km", "Count_Hospital_1km"]
    top10[cols_for_map].to_csv("Recommended_Branches.csv", index=False, encoding="utf-8-sig")

    # Save GeoJSONs
    top10_gdf = gpd.GeoDataFrame(top10, geometry=gpd.points_from_xy(top10["Longitude"], top10["Latitude"]), crs=f"EPSG:{CRS_WGS84}")
    top10_gdf.to_file("Recommended_Branches.geojson", driver="GeoJSON")

    full_gdf = gpd.GeoDataFrame(out_sorted, geometry=gpd.points_from_xy(out_sorted["Longitude"], out_sorted["Latitude"]), crs=f"EPSG:{CRS_WGS84}")
    full_gdf.to_file("All_Population_Scored.geojson", driver="GeoJSON")

    print("✅ Done.")
    print("  - Recommended_Branches.csv (for HTML map)")
    print("  - Recommended_Branches.geojson (Top 10 for QGIS)")
    print("  - All_Population_Scored.geojson (all scored)")
    print(top10[cols_for_map])

if __name__ == "__main__":
    main()
