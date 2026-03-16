import os
import ast
import glob
import math
import random
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from pyproj import CRS
import warnings
from shapely.errors import GEOSException, TopologicalError

try:
    # Shapely 2.x
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None

try:
    from shapely import union_all as _union_all  # Shapely 2.x
except Exception:
    from shapely.ops import unary_union as _union_all  # Shapely 1.x fallback


from shapely.geometry import GeometryCollection
from shapely.prepared import prep
from concurrent.futures import ProcessPoolExecutor
from shapely import wkb  # for fast geometry (de)serialization
import sys, time, datetime, os
import multiprocessing as mp

# Decide the safest start method for your platform
if sys.platform.startswith("linux"):
    MP_CTX = mp.get_context("fork")  # supports CoW sharing
else:
    MP_CTX = mp.get_context("spawn")  # safe fallback (macOS/Windows)

_BOUND_AEZ = None  # which AEZ is currently cached in the parent (for fork mode)


# make stdout line-buffered so prints show up promptly (esp. with multiprocessing)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    pid = os.getpid()
    print(f"[{ts}][PID {pid}] {msg}", flush=True)


MAX_WORKERS = min(12, os.cpu_count() or 1)  # tune if RAM is tight
# --- Global cache used by workers ---
_BOUND_GDF = None
_BOUND_CRS = None

# =======================
# CONFIG
# =======================

# Where your AEZ boundary files are stored (can be a parent dir with subfolders)
BOUNDARY_SEARCH_ROOT = ""  # e.g., "./assets" or "./AEZ_boundaries"
# Where AEZ_<no>/status.csv and AEZ_<no>/samples live
AEZ_DATA_ROOT = Path(".")
# Pattern each AEZ will search for; will be combined with globbing extensions below
BOUNDARY_NAME_PATTERN = "AEZ_{aez}_boundaries*"

# Supported vector extensions (add/remove as needed)
VECTOR_EXTS = [".gpkg", ".geojson", ".shp", ".feather", ".parquet"]

# Path to your local LULC raster (GeoTIFF) covering India
LULC_RASTER_PATH = "data/pan_india_lulc_v3_2023_2024.tif"

# LULC classes considered "of interest" for scrubland exclusion (same as GEE list)
CLASSES_OF_INTEREST = {8, 9, 10, 11}

# Stratified sample targets per class label (0=rest, 1=farm, 2=plantation, 3=scrubland)
CLASS_TARGETS = {0: 0, 1: 150, 2: 150, 3: 150}

# Random seed for reproducibility
RANDOM_SEED = 42

# Are your "points" in status.csv stored as ((lon, lat), (lon, lat)) ?
# If they are ((lat, lon), (lat, lon)), set this to False.
POINTS_ARE_LON_LAT = False

# Buffer distance for farm clustering (meters)
FARM_CLUSTER_BUFFER_M = 10.0

# Negative tile buffer before filtering (meters) to mimic ee.Geometry.Polygon(tile).buffer(-5)
TILE_INNER_BUFFER_M = 5.0

# =======================
# HELPERS
# =======================


def aez_dir(aez_no: int) -> Path:
    return AEZ_DATA_ROOT / f"AEZ_{aez_no}"


def sample_dir(aez_no: int) -> Path:
    return aez_dir(aez_no) / "samples"


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    """Return a metric UTM CRS suitable for buffering/area at a given lon/lat."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    south = lat < 0
    epsg = 32700 + zone if south else 32600 + zone
    return CRS.from_epsg(epsg)


def _find_boundary_files(aez_no: int) -> list[str]:
    """
    Find all vector files matching AEZ_<no>_boundaries*.{ext} recursively under BOUNDARY_SEARCH_ROOT.
    For shapefiles, we only collect the .shp (GDAL will read the .dbf/.prj/.shx automatically).
    """
    base = f"AEZ_{aez_no}_boundaries"
    patterns = []

    # Handle numbered parts and the base (with or without suffix)
    for ext in VECTOR_EXTS:
        # numbered suffixes like *_0.shp, *_1.shp ...
        patterns.append(os.path.join(BOUNDARY_SEARCH_ROOT, f"**/{base}_*{ext}"))
        # non-numbered single file like AEZ_1_boundaries.shp
        patterns.append(os.path.join(BOUNDARY_SEARCH_ROOT, f"**/{base}{ext}"))

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    # Keep only the primary file for shapefiles (avoid .dbf/.prj/etc; glob already limits to .shp)
    files = sorted(set(files))
    return files


def _safe_fix_invalid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fix invalid geometries via buffer(0) where needed, with care to preserve empties.
    """
    if gdf.empty:
        return gdf
    geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            geoms.append(geom)
            continue
        if geom.is_valid:
            geoms.append(geom)
            continue
        try:
            geoms.append(geom.buffer(0))
        except TopologicalError:
            # As a last resort, skip the feature
            geoms.append(None)
    out = gdf.copy()
    out.geometry = geoms
    out = out[~out.geometry.isna() & ~out.geometry.is_empty]
    return out


def load_aez_boundaries(aez_no: int) -> gpd.GeoDataFrame:
    """
    Load and merge all boundary parts for the given AEZ number.
    - Supports files like AEZ_1_boundaries_0.shp, AEZ_1_boundaries_1.shp, etc.
    - Normalizes to a common CRS (EPSG:4326 by default).
    - Unions schemas (missing columns become NaN).
    - Fixes invalid geometries.
    """
    t0 = time.time()
    log(
        f"[AEZ {aez_no}] Scanning boundary files in '{BOUNDARY_SEARCH_ROOT or os.getcwd()}' ..."
    )
    files = _find_boundary_files(aez_no)
    if not files:
        raise FileNotFoundError(
            f"No boundary files found for AEZ {aez_no} under '{BOUNDARY_SEARCH_ROOT}'. "
            f"Expected patterns like 'AEZ_{aez_no}_boundaries_0.shp'."
        )
    log(f"[AEZ {aez_no}] Found {len(files)} boundary part(s). Loading...")

    gdfs: list[gpd.GeoDataFrame] = []
    for fp in files:
        try:
            if fp.endswith(".feather"):
                gdf = gpd.read_feather(fp)
            elif fp.endswith(".parquet"):
                gdf = gpd.read_parquet(fp)
            else:
                gdf = gpd.read_file(fp)
        except Exception as e:
            warnings.warn(f"Skipping '{fp}' due to read error: {e}")
            continue

        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)

        gdf = _safe_fix_invalid(gdf)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not gdf.empty:
            gdfs.append(gdf)

    if not gdfs:
        raise ValueError(
            f"All boundary files for AEZ {aez_no} were unreadable or empty after cleaning."
        )

    target_crs = next((g.crs for g in gdfs if g.crs is not None), "EPSG:4326")
    gdfs = [g.to_crs(target_crs) for g in gdfs]

    all_cols = set().union(*[set(g.columns) for g in gdfs])
    all_cols = [c for c in all_cols if c != "geometry"] + ["geometry"]

    gdfs_norm = [g.reindex(columns=all_cols, fill_value=pd.NA) for g in gdfs]
    merged = pd.concat(gdfs_norm, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=target_crs)
    merged = merged[~merged.geometry.is_empty & merged.geometry.notna()].reset_index(
        drop=True
    )

    dt = time.time() - t0
    log(
        f"[AEZ {aez_no}] Boundaries loaded: {len(merged)} feature(s), CRS={merged.crs}, parts={len(files)} in {dt:.2f}s."
    )
    return merged


def parse_points_to_tile(points_str: str, assume_lonlat: bool = True) -> Polygon:
    """
    'points' looks like '((lon1, lat1), (lon2, lat2))' (or (lat, lon)).
    Returns a rectangular tile polygon (lon,lat order).
    """
    tup = ast.literal_eval(points_str)  # ((a, b), (c, d))
    (a, b), (c, d) = tup
    if assume_lonlat:
        lon1, lat1 = a, b
        lon2, lat2 = c, d
    else:
        lat1, lon1 = a, b
        lat2, lon2 = c, d

    min_lon, max_lon = (lon1, lon2) if lon1 <= lon2 else (lon2, lon1)
    min_lat, max_lat = (lat1, lat2) if lat1 <= lat2 else (lat2, lat1)
    return box(min_lon, min_lat, max_lon, max_lat)


def clip_gdf_to_tile(gdf: gpd.GeoDataFrame, tile: Polygon) -> gpd.GeoDataFrame:
    """Spatially filter features that intersect the tile."""
    return gdf[gdf.geometry.intersects(tile)].copy()


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """If a column is missing, add with NaN."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def to_utm(
    gdf: gpd.GeoDataFrame, lon: float, lat: float
) -> Tuple[gpd.GeoDataFrame, CRS]:
    """Project to local UTM for correct meters-based ops."""
    crs = utm_crs_for_lonlat(lon, lat)
    return gdf.to_crs(crs), crs


def negative_buffer_polygon(
    poly: Polygon, meters: float, utm_crs: CRS, src_crs: CRS
) -> Polygon:
    """Apply a negative buffer in meters by projecting to UTM then back."""
    gdf = gpd.GeoDataFrame(geometry=[poly], crs=src_crs)
    gdf_utm = gdf.to_crs(utm_crs)
    geom_buf = gdf_utm.geometry.iloc[0].buffer(-meters)
    return gpd.GeoSeries([geom_buf], crs=utm_crs).to_crs(src_crs).iloc[0]


def cluster_farms(farm_gdf: gpd.GeoDataFrame, utm_crs: CRS) -> gpd.GeoDataFrame:
    """
    Buffer farms slightly, union to clusters, keep clusters with >=3 farm polys,
    then keep only farms that intersect such clusters.
    All in UTM (meters).
    """
    if farm_gdf.empty:
        return farm_gdf

    # Work in UTM
    farm_utm = farm_gdf.to_crs(utm_crs)
    # Small outward buffer to connect touching farms
    buffed = farm_utm.geometry.buffer(FARM_CLUSTER_BUFFER_M)
    merged = _union_all(buffed.values)

    cluster_polys: List[Polygon] = []
    if isinstance(merged, (Polygon, MultiPolygon)):
        if isinstance(merged, Polygon):
            cluster_polys = [merged]
        else:
            cluster_polys = list(merged.geoms)
    else:
        # Fallback: create GeoSeries directly
        cluster_polys = [Polygon(geom) for geom in merged]

    clusters = gpd.GeoDataFrame(geometry=cluster_polys, crs=utm_crs)

    # Count farms per cluster
    # Use bounding-box prefilter then intersects for speed
    counts = []
    for geom in clusters.geometry:
        subset = farm_utm.sindex.query(geom, predicate="intersects")
        count = farm_utm.iloc[subset].geometry.intersects(geom).sum()
        counts.append(count)
    clusters["count"] = counts
    good_clusters = clusters[clusters["count"] >= 3]
    if good_clusters.empty:
        return farm_gdf.iloc[0:0]

    # Keep farms intersecting any good cluster
    hits = gpd.sjoin(
        farm_utm, good_clusters[["geometry"]], predicate="intersects", how="inner"
    )
    keep = farm_utm.loc[hits.index.unique()]
    return keep.to_crs(farm_gdf.crs)


def percent_interest_in_polygon(
    poly: Polygon, raster_path: str, classes_of_interest: set
) -> float:
    """
    Returns percent of pixels inside 'poly' whose value is in classes_of_interest.
    """
    with rasterio.open(raster_path) as src:
        poly_geojson = [
            gpd.GeoSeries([poly], crs="EPSG:4326")
            .to_crs(src.crs)
            .iloc[0]
            .__geo_interface__
        ]
        out_img, out_transform = mask(src, poly_geojson, crop=True, filled=True)
        data = out_img[0]  # assume single-band LULC
        # Pixels outside polygon are set to src.nodata or 0 by mask; we need the mask array to exclude those
        # Re-run mask to get a boolean of valid (inside) pixels
        mask_arr = geometry_mask(
            poly_geojson, transform=out_transform, invert=True, out_shape=data.shape
        )
        inside = data[mask_arr]
        if inside.size == 0:
            return 0.0
        # Interest is membership in classes_of_interest
        interest = np.isin(inside, list(classes_of_interest))
        return float(interest.sum()) / float(inside.size) * 100.0


def difference_all(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if a.empty:
        return a
    if b.empty:
        return a
    b_union = _union_all(b.geometry.values)
    out = a.copy()
    try:
        out.geometry = out.geometry.difference(b_union)
    except GEOSException:
        # Fallback piecewise
        out.geometry = [
            g.difference(b_union) if g is not None else None for g in out.geometry
        ]
    # Fix invalids
    out.geometry = [_fix_polygonal_geom(g) for g in out.geometry]
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    out = out.explode(index_parts=False, ignore_index=True)
    return out


def dissolve_all(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    dis = gdf.dissolve()
    dis = dis.explode(index_parts=False, ignore_index=True)
    # Fix invalids post-dissolve
    dis.geometry = [_fix_polygonal_geom(g) for g in dis.geometry]
    dis = dis[dis.geometry.notna() & ~dis.geometry.is_empty]
    return gpd.GeoDataFrame(dis, geometry="geometry", crs=gdf.crs)


def _fix_polygonal_geom(g):
    if g is None or g.is_empty:
        return None
    # Extract polygonal part if GeometryCollection
    if isinstance(g, GeometryCollection):
        polys = [h for h in g.geoms if isinstance(h, (Polygon, MultiPolygon))]
        if not polys:
            return None
        g = unary_union(polys)

    # Make valid (prefer make_valid; fallback to buffer(0))
    if not g.is_valid:
        try:
            g = _make_valid(g) if _make_valid is not None else g.buffer(0)
        except Exception:
            # If still broken, drop it
            return None

    if g.is_empty:
        return None
    return g


def clean_for_sampling(gdf: gpd.GeoDataFrame, to_crs: CRS) -> gpd.GeoDataFrame:
    """
    Reproject to 'to_crs', make polygons valid, drop empties, explode multipolygons,
    and remove tiny slivers that cause numerical issues.
    """
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=to_crs)

    gdf = gdf.to_crs(to_crs)
    geoms = [_fix_polygonal_geom(g) for g in gdf.geometry]
    out = gdf.copy()
    out.geometry = geoms
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    if out.empty:
        return out

    out = out.explode(index_parts=False, ignore_index=True)

    # Remove vanishingly small slivers (units are meters in UTM)
    out = out[out.geometry.area > 1e-2]  # ~1 cm²
    return out


def weighted_random_points_in_gdf(
    gdf_utm: gpd.GeoDataFrame, n: int, rng: random.Random
) -> List[Point]:
    """
    Sample ~n points inside polygons (UTM CRS). Assumes geometries are already cleaned/valid.
    Uses area weights and prepared geometries. If a polygon triggers a GEOS error during
    containment check, it is down-weighted to zero and excluded thereafter.
    """
    if n <= 0 or gdf_utm.empty:
        return []

    # Keep only non-empty polygons
    gdf_utm = gdf_utm[~gdf_utm.geometry.is_empty].copy()
    if gdf_utm.empty:
        return []

    areas = gdf_utm.geometry.area.values.astype(float)
    total = areas.sum()
    if total <= 0:
        return []

    probs = areas / total
    polys = list(gdf_utm.geometry.values)
    prepped = [prep(p) for p in polys]
    active = np.arange(len(polys))

    result: List[Point] = []
    max_attempts = n * 100

    while len(result) < n and max_attempts > 0 and active.size > 0:
        # choose among active indices
        idx = rng.choices(active.tolist(), weights=probs[active].tolist(), k=1)[0]
        poly = polys[idx]
        minx, miny, maxx, maxy = poly.bounds

        accepted = False
        for _ in range(400):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            p = Point(x, y)
            try:
                if prepped[idx].contains(p):
                    result.append(p)
                    accepted = True
                    break
            except GEOSException:
                # Permanently remove this polygon from sampling weights
                probs[idx] = 0.0
                active = np.where(probs > 0)[0]
                break  # break inner loop; outer will re-check active set

        max_attempts -= 1

        # Renormalize probs occasionally (only if we've zeroed something)
        if active.size > 0 and probs.sum() != 0 and not np.isclose(probs.sum(), 1.0):
            probs = probs / probs.sum()

    return result[:n]


def prepare_class_geometries(
    all_gdf: gpd.GeoDataFrame, tile_poly: Polygon, lulc_path: str, src_crs: CRS
) -> dict:
    """
    Returns dict of label -> GeoDataFrame with exclusive geometries inside tile,
    following precedence: farm (1), scrubland (3), plantation (2) with last paint winning.
    """

    # Filter to tile
    subset = clip_gdf_to_tile(all_gdf, tile_poly).copy()
    if subset.empty:
        return {
            0: gpd.GeoDataFrame(geometry=[], crs=src_crs),
            1: gpd.GeoDataFrame(geometry=[], crs=src_crs),
            2: gpd.GeoDataFrame(geometry=[], crs=src_crs),
            3: gpd.GeoDataFrame(geometry=[], crs=src_crs),
        }

    # Make sure required columns exist
    ensure_columns(subset, ["rect", "size", "ent", "class"])

    # Plantation: compute area (m^2) first
    tile_centroid = tile_poly.centroid
    utm = utm_crs_for_lonlat(tile_centroid.x, tile_centroid.y)
    subset_utm = subset.to_crs(utm)
    subset["area_m2"] = subset_utm.geometry.area.values  # scalar array

    # Easy filters
    farm = subset[
        (subset["rect"] >= 0.67)
        & (subset["size"] > 500)
        & (subset["size"] < 2000)
        & (subset["ent"] < 1.0)
    ].copy()

    scrubland = subset[(subset["size"] >= 60000) & (subset["size"] <= 5_000_000)].copy()

    plantation = subset[
        (subset["class"].astype(str).str.lower() == "plantation")
    ].copy()

    # Plantation area filters (m^2)
    plantation = plantation[
        (plantation["area_m2"] > 1000) & (plantation["area_m2"] < 20000)
    ].copy()

    # Farm clustering (≥3)
    farm_clustered = cluster_farms(farm, utm)

    # Scrubland interest-percent filter (<50%)
    if not scrubland.empty:
        keep_idx = []
        for i, geom in scrubland.geometry.items():
            try:
                pct = percent_interest_in_polygon(geom, lulc_path, CLASSES_OF_INTEREST)
            except Exception:
                pct = 100.0  # conservative: drop if we can't compute
            if pct < 50.0:
                keep_idx.append(i)
        scrubland = scrubland.loc[keep_idx].copy()

    # Apply tile inner negative buffer (to mimic GEE .buffer(-5) on tile)
    if TILE_INNER_BUFFER_M > 0:
        # negative buffer the tile
        tile_shrunk = negative_buffer_polygon(
            tile_poly, TILE_INNER_BUFFER_M, utm, src_crs
        )
        if not tile_shrunk.is_empty:
            farm_clustered = clip_gdf_to_tile(farm_clustered, tile_shrunk)
            scrubland = clip_gdf_to_tile(scrubland, tile_shrunk)
            plantation = clip_gdf_to_tile(plantation, tile_shrunk)

    # Dissolve to clean geometries
    farm_d = dissolve_all(farm_clustered[["geometry"]])
    scrub_d = dissolve_all(scrubland[["geometry"]])
    plant_d = dissolve_all(plantation[["geometry"]])

    # Precedence: start with farm (1), paint scrubland (3) over it, then plantation (2) over both.
    # Implement as exclusive differences:
    scrub_excl = difference_all(scrub_d, plant_d)  # scrub minus plantation
    farm_excl = difference_all(
        farm_d,
        gpd.GeoDataFrame(
            pd.concat([scrub_excl, plant_d], ignore_index=True),
            geometry="geometry",
            crs=farm_d.crs,
        ),
    )
    plant_excl = plant_d  # highest precedence

    return {
        1: farm_excl,
        2: plant_excl,
        3: scrub_excl,
    }


def stratified_sample_points(
    class_geoms: dict,
    tile_poly: Polygon,
    aez_no: int,
    tile_index: int,
    seed: int = RANDOM_SEED,
) -> gpd.GeoDataFrame:
    """
    Generate stratified random points per class (labels 1,2,3).
    Sampling is performed in local UTM for area-correctness, then reprojected to EPSG:4326.
    """
    rng = random.Random(seed + 1000 * aez_no + tile_index)

    # UTM based on tile centroid
    centroid = tile_poly.centroid
    utm = utm_crs_for_lonlat(centroid.x, centroid.y)

    samples_frames = []
    for label, target in CLASS_TARGETS.items():
        if label == 0 or target <= 0:
            continue
        gdf = class_geoms.get(label, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        if gdf.empty:
            continue
        # Work in UTM
        gdf_utm = clean_for_sampling(gdf, utm)
        pts_utm = weighted_random_points_in_gdf(gdf_utm, target, rng)
        if not pts_utm:
            continue
        pts_gdf = gpd.GeoDataFrame(
            {"label": [label] * len(pts_utm)}, geometry=pts_utm, crs=utm
        ).to_crs(epsg=4326)
        pts_gdf["tile_index"] = tile_index
        pts_gdf["aez_no"] = aez_no
        samples_frames.append(pts_gdf)

    if samples_frames:
        out = pd.concat(samples_frames, ignore_index=True)
        out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    else:
        out = gpd.GeoDataFrame(
            columns=["label", "tile_index", "aez_no", "geometry"], crs="EPSG:4326"
        )

    return out


def write_samples_csv(gdf_points: gpd.GeoDataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if gdf_points.empty:
        # Keep headers consistent and lat before lon
        pd.DataFrame(columns=["lat", "lon", "label", "tile_index", "aez_no"]).to_csv(
            out_csv, index=False
        )
        log(f"Wrote EMPTY tile CSV: {out_csv}")
        return

    df = pd.DataFrame(
        {
            "lat": gdf_points.geometry.y,  # ⬅️ lat first
            "lon": gdf_points.geometry.x,  # ⬅️ then lon
            "label": gdf_points["label"].astype(int),
            "tile_index": gdf_points["tile_index"].astype(int),
            "aez_no": gdf_points["aez_no"].astype(int),
        }
    )
    df.to_csv(out_csv, index=False)
    log(f"Wrote {len(df)} row(s) → {out_csv}")


def collate_samples_to_gee_csv(aez_list: List[int], out_csv: Path) -> None:
    """
    Scan AEZ_<no>/samples/tile_*.csv (lat,lon,label,tile_index,aez_no),
    combine to a single CSV with column names that GEE auto-detects:
    latitude,longitude,label,tile_index,aez_no.
    """
    t0 = time.time()
    frames = []
    file_count = 0
    for aez in aez_list:
        samples_path = sample_dir(aez)
        if not samples_path.exists():
            log(f"[AEZ {aez}] No 'samples' directory found.")
            continue
        for p in sorted(samples_path.glob("tile_*.csv")):
            file_count += 1
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue

            # accept either new (lat/lon) or any older order; coerce types
            cols = {c.lower(): c for c in df.columns}
            if "lat" not in cols or "lon" not in cols:
                # try legacy names
                if "latitude" in cols and "longitude" in cols:
                    df.rename(
                        columns={cols["latitude"]: "lat", cols["longitude"]: "lon"},
                        inplace=True,
                    )
                else:
                    # skip malformed files
                    continue

            # Keep canonical order + types
            keep = ["lat", "lon", "label", "tile_index", "aez_no"]
            df = df[[c for c in keep if c in df.columns]].copy()
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            if "label" in df.columns:
                df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(
                    "Int64"
                )
            if "tile_index" in df.columns:
                df["tile_index"] = pd.to_numeric(
                    df["tile_index"], errors="coerce"
                ).astype("Int64")
            if "aez_no" in df.columns:
                df["aez_no"] = pd.to_numeric(df["aez_no"], errors="coerce").astype(
                    "Int64"
                )

            df = df.dropna(subset=["lat", "lon"])
            if not df.empty:
                frames.append(df)

    if not frames:
        log("No sample CSVs found to collate.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # Rename for GEE auto-detection
    all_df.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)
    dt = time.time() - t0
    log(
        f"Collated {len(frames)} file(s) (scanned={file_count}) → {len(all_df)} row(s) → {out_csv.resolve()} in {dt:.2f}s."
    )


# =======================
# MAIN
# =======================


def process_tile(aez_no: int, tile_poly: Polygon, tile_index: int) -> pd.DataFrame:
    """
    Process one tile: filter classes, build exclusive masks, sample, write CSV.
    Returns a small summary row.
    """
    log(f"[AEZ {aez_no}][Tile {tile_index}] START")
    # Load boundaries for this AEZ
    all_gdf = load_aez_boundaries(
        aez_no
    )  # CRS assumed geographic (EPSG:4326) unless provided

    # Prepare exclusive geometries per class inside tile
    class_geoms = prepare_class_geometries(
        all_gdf, tile_poly, LULC_RASTER_PATH, all_gdf.crs
    )

    # Stratified samples
    points_gdf = stratified_sample_points(
        class_geoms, tile_poly, aez_no, tile_index, seed=RANDOM_SEED
    )

    # Write
    out_csv = sample_dir(aez_no) / f"tile_{tile_index}.csv"
    write_samples_csv(points_gdf, out_csv)

    n1 = int((points_gdf["label"] == 1).sum()) if not points_gdf.empty else 0
    n2 = int((points_gdf["label"] == 2).sum()) if not points_gdf.empty else 0
    n3 = int((points_gdf["label"] == 3).sum()) if not points_gdf.empty else 0
    log(
        f"[AEZ {aez_no}][Tile {tile_index}] DONE → farm:{n1} plant:{n2} scrub:{n3} → {out_csv}"
    )

    return pd.DataFrame(
        [
            {
                "aez_no": aez_no,
                "tile_index": tile_index,
                "n_class_1_farm": (
                    int((points_gdf["label"] == 1).sum()) if not points_gdf.empty else 0
                ),
                "n_class_2_plantation": (
                    int((points_gdf["label"] == 2).sum()) if not points_gdf.empty else 0
                ),
                "n_class_3_scrubland": (
                    int((points_gdf["label"] == 3).sum()) if not points_gdf.empty else 0
                ),
                "csv_path": str(out_csv.resolve()),
            }
        ]
    )


def _worker_init(aez_no: int):
    global _BOUND_GDF, _BOUND_CRS
    log(f"[AEZ {aez_no}] Worker init: loading boundaries (spawn mode)...")
    _BOUND_GDF = load_aez_boundaries(aez_no)
    _BOUND_CRS = _BOUND_GDF.crs
    log(f"[AEZ {aez_no}] Worker ready with {len(_BOUND_GDF)} features.")


def _process_tile_worker(args):
    """
    Worker-side tile execution. Args: (aez_no, tile_wkb, tile_index)
    Rebuilds tile polygon from WKB, uses global _BOUND_GDF, and writes CSV.
    Returns a small summary dict.
    """
    aez_no, tile_wkb, tile_index = args
    log(f"[AEZ {aez_no}][Tile {tile_index}] (worker) START")
    tile_poly = wkb.loads(tile_wkb)  # reconstruct geometry

    # Prepare exclusive geometries per class inside tile
    class_geoms = prepare_class_geometries(
        _BOUND_GDF, tile_poly, LULC_RASTER_PATH, _BOUND_CRS
    )

    # Stratified samples
    points_gdf = stratified_sample_points(
        class_geoms, tile_poly, aez_no, tile_index, seed=RANDOM_SEED
    )

    # Write CSV
    out_csv = sample_dir(aez_no) / f"tile_{tile_index}.csv"
    n1 = int((points_gdf["label"] == 1).sum()) if not points_gdf.empty else 0
    n2 = int((points_gdf["label"] == 2).sum()) if not points_gdf.empty else 0
    n3 = int((points_gdf["label"] == 3).sum()) if not points_gdf.empty else 0
    log(
        f"[AEZ {aez_no}][Tile {tile_index}] (worker) DONE → farm:{n1} plant:{n2} scrub:{n3} → {out_csv}"
    )
    write_samples_csv(points_gdf, out_csv)

    # Summary back to parent
    return {
        "aez_no": aez_no,
        "tile_index": tile_index,
        "n_class_1_farm": (
            int((points_gdf["label"] == 1).sum()) if not points_gdf.empty else 0
        ),
        "n_class_2_plantation": (
            int((points_gdf["label"] == 2).sum()) if not points_gdf.empty else 0
        ),
        "n_class_3_scrubland": (
            int((points_gdf["label"] == 3).sum()) if not points_gdf.empty else 0
        ),
        "csv_path": str(out_csv.resolve()),
    }


def sample_points_from_tiles(aez_no: int):
    """
    Read AEZ_<no>/status.csv; build tiles; run processing for each tile.
    """
    status_csv = aez_dir(aez_no) / "status.csv"
    if not status_csv.exists():
        raise FileNotFoundError(f"Missing status file: {status_csv}")

    df = pd.read_csv(status_csv)
    if "points" not in df.columns:
        raise ValueError(f"'points' column not found in {status_csv}")

    # Build tile polygons
    df["tile_geom"] = df["points"].apply(
        lambda s: parse_points_to_tile(s, assume_lonlat=POINTS_ARE_LON_LAT)
    )

    # Process each tile
    summaries = []
    for idx, tile in enumerate(df["tile_geom"]):
        # Optional: skip tiles where points sampling already exists
        # out_csv = sample_dir(aez_no) / f"tile_{idx}.csv"
        # if out_csv.exists(): continue

        # Process
        summary_row = process_tile(aez_no, tile, idx)
        summaries.append(summary_row)

    if summaries:
        out = pd.concat(summaries, ignore_index=True)
        print(out)
    else:
        print(f"No tiles processed for AEZ {aez_no} (no rows in status.csv?).")


def _dynamic_workers_for_aez(aez_no: int) -> int:
    """
    Scale workers inversely with number of boundary parts:
      - 5 parts -> 12 workers (capped by CPU)
      - more parts -> fewer workers
    Also clamps to CPU count and is more conservative on 'spawn'.
    """
    base_cap = min(12, os.cpu_count() or 1)  # your current ceiling
    n_parts = max(1, len(_find_boundary_files(aez_no)))

    # Inverse scaling anchored at (parts=5 -> base_cap workers)
    workers = int(math.floor(base_cap * 4 / n_parts))

    # On spawn platforms, boundaries are loaded per worker → be extra conservative
    if MP_CTX.get_start_method() != "fork":
        workers = int(math.floor(workers * 0.6))

    # Sane bounds
    workers = max(2, min(workers, base_cap))  # at least 2 (unless CPU < 2)
    return workers


def sample_points_from_tiles_parallel(aez_no: int):
    """
    Read AEZ_<no>/status.csv; build tiles; run processing for each tile in parallel.
    On Linux (fork) we load boundaries ONCE in the parent and share to children via CoW.
    On spawn platforms, we load once per worker via initializer.
    """
    status_csv = aez_dir(aez_no) / "status.csv"
    if not status_csv.exists():
        raise FileNotFoundError(f"Missing status file: {status_csv}")

    df = pd.read_csv(status_csv)
    if "points" not in df.columns:
        raise ValueError(f"'points' column not found in {status_csv}")

    # Build tile polygons (in parent)
    df["tile_geom"] = df["points"].apply(
        lambda s: parse_points_to_tile(s, assume_lonlat=POINTS_ARE_LON_LAT)
    )
    n_parts = max(1, len(_find_boundary_files(aez_no)))
    num_workers = _dynamic_workers_for_aez(aez_no)
    log(
        f"[AEZ {aez_no}] Preparing {len(df)} tile(s). "
        f"Boundary parts={n_parts}. Using {num_workers} worker(s). "
        f"Start method: {MP_CTX.get_start_method()}"
    )

    # Preload boundaries once in parent if we can use 'fork'
    initializer = None
    initargs = ()
    global _BOUND_GDF, _BOUND_CRS, _BOUND_AEZ

    if MP_CTX.get_start_method() == "fork":
        # Load in parent, share via CoW
        if _BOUND_AEZ != aez_no or _BOUND_GDF is None:
            log(
                f"[AEZ {aez_no}] (parent) Loading boundaries for shared use across workers..."
            )
            _BOUND_GDF = load_aez_boundaries(aez_no)
            _BOUND_CRS = _BOUND_GDF.crs
            _BOUND_AEZ = aez_no
            log(
                f"[AEZ {aez_no}] (parent) Shared cache ready: {len(_BOUND_GDF)} features."
            )
        else:
            log(
                f"[AEZ {aez_no}] (parent) Reusing existing shared cache with {len(_BOUND_GDF)} features."
            )
        # No initializer needed; children inherit memory on fork
    else:
        # On spawn, children don't inherit; load per worker once
        initializer = _worker_init
        initargs = (aez_no,)
        log(f"[AEZ {aez_no}] Using spawn; each worker will load boundaries once.")

    # Prepare compact payloads
    tasks = [(aez_no, geom.wkb, idx) for idx, geom in enumerate(df["tile_geom"])]

    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=MP_CTX,
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        results = list(ex.map(_process_tile_worker, tasks, chunksize=1))
    dt = time.time() - t0

    # Summaries
    if results:
        out = pd.DataFrame(results).sort_values("tile_index").reset_index(drop=True)
        log(f"[AEZ {aez_no}] Completed {len(out)} tile(s) in {dt:.2f}s.")
        print(out)
    else:
        log(f"[AEZ {aez_no}] No tiles processed.")


def _clear_parent_cache():
    import gc

    global _BOUND_GDF, _BOUND_CRS, _BOUND_AEZ
    _BOUND_GDF = None
    _BOUND_CRS = None
    _BOUND_AEZ = None
    gc.collect()


def parse_int_list(values: List[str], field_name: str) -> List[int]:
    parsed = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid integer '{token}' in {field_name}") from exc
    if not parsed:
        raise ValueError(f"No values provided for {field_name}")
    return parsed


def run_sampling_for_aez_list(aez_list: List[int]) -> None:
    for aez_no in aez_list:
        log(f"=== AEZ {aez_no} START ===")
        sample_points_from_tiles_parallel(aez_no)
        _clear_parent_cache()
        log(f"=== AEZ {aez_no} FINISH ===")


def resolve_single_collate_out(aez_no: int, collate_out: Optional[str]) -> Path:
    """
    For single-AEZ mode, keep collated output inside that AEZ folder by default.
    If a relative path is provided, resolve it under AEZ_<no>/.
    """
    if not collate_out:
        return aez_dir(aez_no) / f"gee_samples_aez_{aez_no}.csv"
    out_path = Path(collate_out)
    if out_path.is_absolute():
        return out_path
    return aez_dir(aez_no) / out_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local samples from AEZ boundaries and status.csv tiles."
    )
    parser.add_argument(
        "--aez-root",
        default=".",
        help="Base directory containing AEZ_<no> folders with status.csv and samples/.",
    )
    parser.add_argument(
        "--boundary-root",
        default="",
        help="Root directory to recursively search AEZ boundary files. Defaults to current directory.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    all_parser = subparsers.add_parser("all", help="Run sampling for multiple AEZs.")
    all_parser.add_argument(
        "--aezs",
        nargs="+",
        help="AEZ list, e.g. --aezs 1 2 3 or --aezs 1,2,3. Default is 1..19.",
    )
    all_parser.add_argument(
        "--skip-collate",
        action="store_true",
        help="Skip final collation step.",
    )
    all_parser.add_argument(
        "--collate-out",
        default="export/gee_samples_all.csv",
        help="Output CSV path for collated samples.",
    )

    single_parser = subparsers.add_parser("single", help="Run sampling for one AEZ.")
    single_parser.add_argument(
        "--aez", type=int, required=True, help="AEZ number to process."
    )
    single_parser.add_argument(
        "--skip-collate",
        action="store_true",
        help="Skip final collation step for this AEZ.",
    )
    single_parser.add_argument(
        "--collate-out",
        default=None,
        help="Optional output file name/path for collated single-AEZ samples. "
        "Relative paths are saved inside AEZ_<no>/.",
    )
    return parser


if __name__ == "__main__":
    print("Inside sampling_locally_workers script")
    args = build_argument_parser().parse_args()

    AEZ_DATA_ROOT = Path(args.aez_root)
    BOUNDARY_SEARCH_ROOT = args.boundary_root

    if args.mode == "all":
        print("Running for all")
        aez_list = (
            parse_int_list(args.aezs, "--aezs") if args.aezs else list(range(1, 20))
        )
        run_sampling_for_aez_list(aez_list)
        if not args.skip_collate:
            collate_samples_to_gee_csv(aez_list, Path(args.collate_out))
    else:
        aez_list = [args.aez]
        print("Running for single ", str(aez_list))
        run_sampling_for_aez_list(aez_list)
        if not args.skip_collate:
            single_out = resolve_single_collate_out(args.aez, args.collate_out)
            collate_samples_to_gee_csv(aez_list, single_out)
