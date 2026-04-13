import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


in_shp = r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer.shp"
naip_tif = r"Z:\Modeling\ESPA_Soil_Water_Balance\Rasters\naip\naip_clipped.tif"

out_circular_shp = r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\circular_fields_cleaned_auto.shp"
out_sector_shp = r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\sector_fields_auto.shp"
out_noncircular_shp = r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\non_circular_fields_auto.shp"
out_review_shp = r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\uncertain_review_auto.shp"
out_rejects_shp = r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\rejected_candidates_auto.shp"
out_uncertain_shp = r"Z:\Modeling\ESPA_Soil_Water_Balance\Shapefiles\2008_Irrigated_Lands_for_the_Eastern_Snake_Plain_Aquifer\classified_shp\uncertain_candidates_auto.shp"

fix_invalid = True
min_area = 1000.0

circularity_thresh = 0.75 # pretty close version was 0.5, try between 0.5-0.75
centroid_tol = 60.0

pacman_circ_min = 0.25
pacman_circ_max = 0.85
pacman_solidity_min = 0.68
pacman_aspect_ratio_max = 1.65
pacman_fill_ratio_min = 0.28

pacman_centroid_dedupe_tol = 80.0
chip_pad_pixels = 20

fit_buffer_mult = 1.25
fit_buffer_min = 60.0

circle_iou_min = 0.72
circle_support_min = 0.45
circle_center_offset_frac_max = 0.25
circle_area_ratio_min = 0.70
circle_area_ratio_max = 1.25

sector_iou_min = 0.58
sector_support_min = 0.40
sector_center_offset_frac_max = 0.30
sector_area_ratio_min = 0.45
sector_area_ratio_max = 1.40
sector_min_bin_run = 10

simplify_tol = 1.0
ndvi_thresh = 0.18
exg_thresh = 20.0
smooth_dist = 4.0
sector_hist_bins = 180
sector_min_bin_run = 18   
circle_resolution = 128
sector_resolution = 180



def explode_multiparts(gdf):
    return gdf.explode(index_parts=False).reset_index(drop=True)

def circularity(geom):
    if geom is None or geom.is_empty:
        return np.nan
    a = geom.area
    p = geom.length
    if a <= 0 or p <= 0:
        return np.nan
    return 4.0 * np.pi * a / (p ** 2)

def equiv_radius(area):
    if area <= 0:
        return np.nan
    return np.sqrt(area / np.pi)

def solidity(geom):
    if geom is None or geom.is_empty:
        return np.nan
    hull = geom.convex_hull
    ha = hull.area
    if ha <= 0:
        return np.nan
    return geom.area / ha

def bbox_metrics(geom):
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        return np.nan, np.nan, np.nan
    aspect = max(w, h) / min(w, h)
    bbox_area = w * h
    fill = geom.area / bbox_area if bbox_area > 0 else np.nan
    return w, h, aspect, fill

def pd_concat_safe(gdfs, crs=None):
    gdfs = [g for g in gdfs if g is not None and len(g) > 0]
    if len(gdfs) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    out = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs if crs is not None else gdfs[0].crs)

def dedupe_by_centroid_keep_smallest_with_drops(gdf, centroid_tol=60.0):
    if len(gdf) == 0:
        return gdf.copy(), set(), set()

    gdf = gdf.copy()
    gdf["cx"] = gdf.geometry.centroid.x
    gdf["cy"] = gdf.geometry.centroid.y
    gdf["area_tmp"] = gdf.geometry.area

    coords = gdf[["cx", "cy"]].to_numpy()
    n = len(gdf)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if d <= centroid_tol:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    keep_idx = []
    drop_idx = []

    for idxs in groups.values():
        sub = gdf.iloc[idxs].copy()
        best_idx = sub["area_tmp"].idxmin()
        keep_idx.append(best_idx)
        for idx in sub.index:
            if idx != best_idx:
                drop_idx.append(idx)

    out = gdf.loc[keep_idx].copy().reset_index(drop=True)
    kept_ids = set(out["src_id"].astype(int).tolist())
    dropped_ids = set(gdf.loc[drop_idx, "src_id"].astype(int).tolist())
    out = out.drop(columns=["cx", "cy", "area_tmp"], errors="ignore")
    return out, kept_ids, dropped_ids

def sample_boundary_points(geom, n=300):
    line = geom.exterior
    if line.length == 0:
        return np.empty((0, 2), dtype=float)
    dists = np.linspace(0, line.length, n, endpoint=False)
    pts = [line.interpolate(d) for d in dists]
    return np.array([[p.x, p.y] for p in pts], dtype=float)

def fit_circle_kasa(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r = np.sqrt(max(0.0, c0 + cx**2 + cy**2))
    return cx, cy, r

def make_circle(cx, cy, r, resolution=circle_resolution):
    return Point(cx, cy).buffer(r, resolution=resolution)

def iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return inter / union

def point_angles(points, cx, cy):
    ang = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    return np.mod(ang, 2*np.pi)

def longest_true_run(mask):
    if len(mask) == 0:
        return None
    doubled = np.r_[mask, mask]
    best_len = 0
    best_start = None
    cur_len = 0
    cur_start = None
    for i, v in enumerate(doubled):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
            cur_start = None
    if best_len > len(mask):
        best_len = len(mask)
    if best_start is None:
        return None
    return best_start % len(mask), best_len

def make_sector(cx, cy, r, ang1, ang2, n=sector_resolution):
    if ang2 < ang1:
        ang2 += 2*np.pi
    angles = np.linspace(ang1, ang2, n)
    arc = [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles]
    coords = [(cx, cy)] + arc + [(cx, cy)]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def fit_sector_from_circle(poly, cx, cy, r):
    pts = sample_boundary_points(poly, n=500)
    if len(pts) < 20:
        return None

    ang = point_angles(pts, cx, cy)
    hist, edges = np.histogram(ang, bins=sector_hist_bins, range=(0, 2*np.pi))

    thresh = max(2, int(np.percentile(hist, 60)))
    occupied = hist >= thresh

    run = longest_true_run(occupied)
    if run is None:
        return None

    start_bin, run_len = run
    if run_len < sector_min_bin_run:
        return None

    end_bin = (start_bin + run_len) % sector_hist_bins
    ang1 = edges[start_bin]
    ang2 = edges[end_bin] if end_bin > start_bin else edges[end_bin] + 2*np.pi

    sector = make_sector(cx, cy, r, ang1, ang2)
    return {
        "geometry": sector,
        "ang1": ang1,
        "ang2": ang2,
        "span_deg": np.degrees(ang2 - ang1),
    }

def geometry_to_mask(geom, transform, out_shape):
    if len(out_shape) != 2:
        raise ValueError(f"out_shape must be 2D, got {out_shape}")

    h, w = out_shape
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=bool)

    arr = features.rasterize(
        [(geom, 1)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return arr.astype(bool)

def read_local_chip(src, geom, pad):
    minx, miny, maxx, maxy = geom.bounds

    left = minx - pad
    bottom = miny - pad
    right = maxx + pad
    top = maxy + pad

    src_left, src_bottom, src_right, src_top = src.bounds

    left = max(left, src_left)
    right = min(right, src_right)
    bottom = max(bottom, src_bottom)
    top = min(top, src_top)

    if right <= left or top <= bottom:
        return None, None

    window = from_bounds(left, bottom, right, top, src.transform)
    window = window.round_offsets().round_lengths()

    col_off = int(window.col_off)
    row_off = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    if width <= 0 or height <= 0:
        return None, None

    window = rasterio.windows.Window(col_off, row_off, width, height)

    data = src.read(window=window)

    if data.ndim != 3 or data.shape[1] <= 0 or data.shape[2] <= 0:
        return None, None

    transform = src.window_transform(window)
    return data, transform

def build_veg_mask(chip):
    chip = chip.astype("float32")

    if chip.shape[0] >= 4:
        red = chip[0]
        green = chip[1]
        nir = chip[3]
        denom = nir + red
        ndvi = np.where(denom != 0, (nir - red) / denom, -1.0)
        veg = ndvi > ndvi_thresh
        return veg

    red = chip[0]
    green = chip[1]
    blue = chip[2]
    exg = 2 * green - red - blue
    veg = exg > exg_thresh
    return veg

def candidate_support_score(candidate_geom, chip, transform):
    veg = build_veg_mask(chip)

    if veg.ndim != 2 or veg.shape[0] <= 0 or veg.shape[1] <= 0:
        return 0.0

    cand_mask = geometry_to_mask(candidate_geom, transform, veg.shape)

    n = cand_mask.sum()
    if n == 0:
        return 0.0

    return float((veg & cand_mask).sum() / n)

def score_circle_fit(poly, chip, transform):
    poly_fit = smooth_geom_for_metrics(poly, simplify_tol=1.0, smooth_dist=4.0)

    pts = sample_boundary_points(poly_fit, n=120)
    if len(pts) < 20:
        return None

    cx, cy, r = fit_circle_kasa(pts)
    if not np.isfinite(r) or r <= 0:
        return None

    fitted = make_circle(cx, cy, r)
    fit_iou = iou(poly, fitted)
    center_offset = Point(cx, cy).distance(poly.centroid)
    area_ratio = poly.area / fitted.area if fitted.area > 0 else np.nan
    support = candidate_support_score(fitted, chip, transform)

    return {
        "geometry": fitted,
        "cx": cx,
        "cy": cy,
        "r": r,
        "fit_iou": fit_iou,
        "center_offset": center_offset,
        "area_ratio": area_ratio,
        "support": support,
    }


def accept_circle_fit(res):
    if res is None:
        return False
    r = res["r"]
    return (
        res["fit_iou"] >= circle_iou_min and
        res["support"] >= circle_support_min and
        res["center_offset"] <= circle_center_offset_frac_max * r and
        circle_area_ratio_min <= res["area_ratio"] <= circle_area_ratio_max
    )

def read_chip(src, poly, chip_pad_pixels=20):
    minx, miny, maxx, maxy = poly.bounds
    resx = abs(src.transform.a)
    resy = abs(src.transform.e)

    pad_x = chip_pad_pixels * resx
    pad_y = chip_pad_pixels * resy

    window = rasterio.windows.from_bounds(
        minx - pad_x,
        miny - pad_y,
        maxx + pad_x,
        maxy + pad_y,
        transform=src.transform
    )

    window = window.round_offsets().round_lengths()
    chip = src.read(window=window)
    transform = src.window_transform(window)

    return chip, transform

def score_sector_fit(poly, chip, transform, circle_res):
    if circle_res is None:
        return None

    sector_res = fit_sector_from_circle(
        poly,
        circle_res["cx"],
        circle_res["cy"],
        circle_res["r"],
    )
    if sector_res is None:
        return None

    fitted = sector_res["geometry"]
    fit_iou = iou(poly, fitted)
    center_offset = Point(circle_res["cx"], circle_res["cy"]).distance(poly.centroid)
    area_ratio = poly.area / fitted.area if fitted.area > 0 else np.nan
    support = candidate_support_score(fitted, chip, transform)

    sector_res.update({
        "fit_iou": fit_iou,
        "center_offset": center_offset,
        "area_ratio": area_ratio,
        "support": support,
        "cx": circle_res["cx"],
        "cy": circle_res["cy"],
        "r": circle_res["r"],
    })
    return sector_res

def accept_sector_fit(res):
    if res is None:
        return False
    r = res["r"]
    return (
        res["fit_iou"] >= sector_iou_min and
        res["support"] >= sector_support_min and
        res["center_offset"] <= sector_center_offset_frac_max * r and
        sector_area_ratio_min <= res["area_ratio"] <= sector_area_ratio_max
    )
def explode_to_polygons(gdf):
    if len(gdf) == 0:
        return gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
    out = gdf.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.notnull()].copy()
    out = out[~out.geometry.is_empty].copy()
    out = out[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return out

def compactness(geom):
    a = geom.area
    p = geom.length
    if a <= 0 or p <= 0:
        return 0.0
    return 4.0 * np.pi * a / (p ** 2)

def smooth_geom_for_metrics(geom, simplify_tol=1.0, smooth_dist=4.0):
    if geom is None or geom.is_empty:
        return geom
    g = geom.simplify(simplify_tol, preserve_topology=True)
    g = g.buffer(smooth_dist).buffer(-smooth_dist)
    if not g.is_valid:
        g = g.buffer(0)
    return g


def subtract_pivots_from_noncircular(
    non_circular,
    pivot_gdf,
    min_remnant_area=3000.0,
    min_compactness=0.08,
    min_solidity=0.55,
    min_fill=0.18,
    max_aspect_ratio=6.0
):
    if len(non_circular) == 0 or len(pivot_gdf) == 0:
        return non_circular.copy()

    pivot_union = unary_union(pivot_gdf.geometry.tolist())

    cleaned_rows = []
    for _, row in non_circular.iterrows():
        diff = row.geometry.difference(pivot_union)

        if diff.is_empty:
            continue

        if diff.geom_type == "Polygon":
            parts = [diff]
        elif diff.geom_type == "MultiPolygon":
            parts = list(diff.geoms)
        else:
            continue

        for part in parts:
            if part.is_empty:
                continue

            if part.area < min_remnant_area:
                continue

            comp = compactness(part)
            sol = solidity(part)
            _, _, aspect, fill = bbox_metrics(part)

            if comp < min_compactness:
                continue
            if sol < min_solidity:
                continue
            if aspect > max_aspect_ratio:
                continue
            if fill < min_fill:
                continue

            rec = row.copy()
            rec.geometry = part
            cleaned_rows.append(rec)

    if len(cleaned_rows) == 0:
        return gpd.GeoDataFrame(columns=non_circular.columns, geometry=[], crs=non_circular.crs)

    out = gpd.GeoDataFrame(cleaned_rows, geometry="geometry", crs=non_circular.crs)
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.notnull()].copy()
    out = out[~out.geometry.is_empty].copy()
    out = out[out.geometry.area >= min_remnant_area].copy()
    return out.reset_index(drop=True)

def remove_small_pivot_adjacent_slivers(non_circular, pivot_gdf, area_max=12000.0, touch_buffer=8.0):
    if len(non_circular) == 0 or len(pivot_gdf) == 0:
        return non_circular.copy()

    pivot_touch = unary_union([g.buffer(touch_buffer) for g in pivot_gdf.geometry])

    keep_rows = []
    for _, row in non_circular.iterrows():
        geom = row.geometry

        if geom.area <= area_max and geom.intersects(pivot_touch):
            continue

        keep_rows.append(row)

    if len(keep_rows) == 0:
        return gpd.GeoDataFrame(columns=non_circular.columns, geometry=[], crs=non_circular.crs)

    return gpd.GeoDataFrame(keep_rows, geometry="geometry", crs=non_circular.crs).reset_index(drop=True)

gdf = gpd.read_file(in_shp)

if gdf.empty:
    raise ValueError("Input shapefile is empty.")
if gdf.crs is None:
    raise ValueError("Input shapefile has no CRS. Use projected CRS.")

with rasterio.open(naip_tif) as src:
    naip_crs = src.crs

if gdf.crs != naip_crs:
    gdf = gdf.to_crs(naip_crs)

gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()
gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

if fix_invalid:
    gdf["geometry"] = gdf.geometry.buffer(0)

gdf = explode_multiparts(gdf)
gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

gdf["area_m2"] = gdf.geometry.area
gdf = gdf[gdf["area_m2"] >= min_area].copy().reset_index(drop=True)

gdf["perim_m"] = gdf.geometry.length
gdf["circ_raw"] = gdf.geometry.apply(circularity)
gdf["eq_radius"] = gdf["area_m2"].apply(equiv_radius)
gdf["solidity_raw"] = gdf.geometry.apply(solidity)
gdf["src_id"] = np.arange(len(gdf), dtype=int)

bbox_vals_raw = gdf.geometry.apply(bbox_metrics)
gdf["bbox_w_raw"] = [v[0] for v in bbox_vals_raw]
gdf["bbox_h_raw"] = [v[1] for v in bbox_vals_raw]
gdf["aspect_ratio_raw"] = [v[2] for v in bbox_vals_raw]
gdf["fill_ratio_raw"] = [v[3] for v in bbox_vals_raw]

candidate_mask = (
    (gdf["circ_raw"] >= 0.30) &
    (gdf["solidity_raw"] >= 0.75) &
    (gdf["aspect_ratio_raw"] <= 1.50)
)

gdf["geom_smooth"] = gdf.geometry
gdf.loc[candidate_mask, "geom_smooth"] = gdf.loc[candidate_mask, "geometry"].apply(
    lambda g: smooth_geom_for_metrics(g, simplify_tol=1.0, smooth_dist=4.0)
)

gdf["area_m2"] = gdf["geom_smooth"].area
gdf["perim_m"] = gdf["geom_smooth"].length
gdf["circ"] = gdf["geom_smooth"].apply(circularity)
gdf["solidity"] = gdf["geom_smooth"].apply(solidity)

bbox_vals = gdf["geom_smooth"].apply(bbox_metrics)
gdf["bbox_w"] = [v[0] for v in bbox_vals]
gdf["bbox_h"] = [v[1] for v in bbox_vals]
gdf["aspect_ratio"] = [v[2] for v in bbox_vals]
gdf["fill_ratio"] = [v[3] for v in bbox_vals]

print(f"Total cleaned input polygons: {len(gdf)}")
print(f"Smoothed candidate polygons: {int(candidate_mask.sum())}")



circular = gdf[
    (gdf["circ"] >= circularity_thresh) &
    (gdf["aspect_ratio"] <= 1.20) &
    (gdf["fill_ratio"] >= 0.60) &
    (gdf["solidity"] >= 0.90)
].copy().reset_index(drop=True)

circular_clean, circular_keep_ids, circular_drop_ids = (
    dedupe_by_centroid_keep_smallest_with_drops(
        circular,
        centroid_tol=centroid_tol
    )
)
circular_clean["out_class"] = "circular_clean"
circular_clean["fit_method"] = "original_geom"

print(f"Initial circular candidates: {len(circular)}")
print(f"Circular after dedupe: {len(circular_clean)}")
print(f"Circular duplicate drops: {len(circular_drop_ids)}")



pacman = gdf[
    (~gdf["src_id"].isin(circular_keep_ids.union(circular_drop_ids))) &
    (
        (
            (gdf["solidity"] >= 0.68) &
            (gdf["aspect_ratio"] <= 1.65) &
            (gdf["fill_ratio"] >= 0.28)
        )
        |
        (
            (gdf["circ"] >= 0.25) &
            (gdf["aspect_ratio"] <= 1.75)
        )
    )
].copy().reset_index(drop=True)

print(f"Filtered pacman candidate count (pre-dedupe): {len(pacman)}")

pacman, pacman_body_keep_ids, pacman_body_drop_ids = (
    dedupe_by_centroid_keep_smallest_with_drops(
        pacman,
        centroid_tol=pacman_centroid_dedupe_tol
    )
)

print(f"Filtered pacman candidate count (post-dedupe): {len(pacman)}")
print(f"Pacman body duplicate drops: {len(pacman_body_drop_ids)}")



accepted_rows = []
sector_rows = []
rejected_rows = []
uncertain_rows = []

with rasterio.open(naip_tif) as src:
    for _, row in pacman.iterrows():
        poly = row.geometry
        src_id = row["src_id"]

        try:
            chip, transform = read_chip(src, poly, chip_pad_pixels=chip_pad_pixels)
        except Exception:
            rejected_rows.append({
                "src_id": src_id,
                "reason": "chip_read_failed",
                "geometry": poly
            })
            continue

        cf = score_circle_fit(poly, chip, transform)

        accepted_circle = False
        if cf is not None:
            r = cf["r"]
            center_offset_frac = (
                cf["center_offset"] / r
                if (r is not None and np.isfinite(r) and r > 0)
                else np.inf
            )

            if (
                cf["fit_iou"] >= circle_iou_min and
                cf["support"] >= circle_support_min and
                center_offset_frac <= circle_center_offset_frac_max and
                circle_area_ratio_min <= cf["area_ratio"] <= circle_area_ratio_max
            ):
                accepted_rows.append({
                    "src_id": src_id,
                    "geometry": cf["geometry"],
                    "fit_iou": cf["fit_iou"],
                    "support": cf["support"],
                    "center_offset": cf["center_offset"],
                    "center_offset_frac": center_offset_frac,
                    "area_ratio": cf["area_ratio"],
                    "r": cf["r"],
                    "kind": "fullfit"
                })
                accepted_circle = True
            else:
                uncertain_rows.append({
                    "src_id": src_id,
                    "geometry": poly,
                    "fit_iou": cf["fit_iou"],
                    "support": cf["support"],
                    "center_offset": cf["center_offset"],
                    "center_offset_frac": center_offset_frac,
                    "area_ratio": cf["area_ratio"],
                    "r": cf["r"],
                    "reason": "circle_near_miss"
                })

        if accepted_circle:
            continue

        sf = score_sector_fit(poly, chip, transform, cf)

        accepted_sector = False
        if sf is not None:
            r = sf["r"]
            center_offset_frac = (
                sf["center_offset"] / r
                if (r is not None and np.isfinite(r) and r > 0)
                else np.inf
            )

            if (
                sf["fit_iou"] >= sector_iou_min and
                sf["support"] >= sector_support_min and
                center_offset_frac <= sector_center_offset_frac_max and
                sector_area_ratio_min <= sf["area_ratio"] <= sector_area_ratio_max
            ):
                sector_rows.append({
                    "src_id": src_id,
                    "geometry": sf["geometry"],
                    "fit_iou": sf["fit_iou"],
                    "support": sf["support"],
                    "center_offset": sf["center_offset"],
                    "center_offset_frac": center_offset_frac,
                    "area_ratio": sf["area_ratio"],
                    "r": sf["r"],
                    "theta0": sf.get("theta0", np.nan),
                    "theta1": sf.get("theta1", np.nan),
                    "kind": "sector"
                })
                accepted_sector = True
            else:
                uncertain_rows.append({
                    "src_id": src_id,
                    "geometry": poly,
                    "fit_iou": sf["fit_iou"],
                    "support": sf["support"],
                    "center_offset": sf["center_offset"],
                    "center_offset_frac": center_offset_frac,
                    "area_ratio": sf["area_ratio"],
                    "r": sf["r"],
                    "theta0": sf.get("theta0", np.nan),
                    "theta1": sf.get("theta1", np.nan),
                    "reason": "sector_near_miss"
                })

        if not accepted_circle and not accepted_sector:
            rejected_rows.append({
                "src_id": src_id,
                "reason": "fit_failed",
                "geometry": poly
            })

pivot_fullfit = gpd.GeoDataFrame(accepted_rows, geometry="geometry", crs=gdf.crs)
pivot_sector = gpd.GeoDataFrame(sector_rows, geometry="geometry", crs=gdf.crs)
fit_rejects = gpd.GeoDataFrame(rejected_rows, geometry="geometry", crs=gdf.crs)
uncertain_review = gpd.GeoDataFrame(uncertain_rows, geometry="geometry", crs=gdf.crs)

print(f"Accepted full-circle fits: {len(pivot_fullfit)}")
print(f"Accepted sector fits: {len(pivot_sector)}")
print(f"Rejected fit candidates: {len(fit_rejects)}")
print(f"Uncertain review candidates: {len(uncertain_review)}")


auto_ids = set()
if len(pivot_fullfit):
    auto_ids |= set(pivot_fullfit["src_id"].astype(int).tolist())
if len(pivot_sector):
    auto_ids |= set(pivot_sector["src_id"].astype(int).tolist())
if len(uncertain_review):
    auto_ids |= set(uncertain_review["src_id"].astype(int).tolist())

exclude_ids = (
    circular_keep_ids
    .union(circular_drop_ids)
    .union(pacman_body_keep_ids)
    .union(pacman_body_drop_ids)
    .union(auto_ids)
)

non_circular = gdf[~gdf["src_id"].isin(exclude_ids)].copy().reset_index(drop=True)
non_circular["out_class"] = "non_circular"
non_circular["fit_method"] = "original_geom"

print(f"Non-circular output count: {len(non_circular)}")


circular_out = pd_concat_safe([circular_clean, pivot_fullfit], crs=gdf.crs)
accepted_pivots = pd_concat_safe([circular_out, pivot_sector], crs=gdf.crs)

non_circular = subtract_pivots_from_noncircular(
    non_circular,
    accepted_pivots,
    min_remnant_area=3000.0,
    min_compactness=0.08,
    min_solidity=0.55,
    min_fill=0.18,
    max_aspect_ratio=6.0
)

non_circular = remove_small_pivot_adjacent_slivers(
    non_circular,
    accepted_pivots,
    area_max=12000.0,
    touch_buffer=8.0
)

print(f"Non-circular after halo/corner cleanup: {len(non_circular)}")


def strip_extra_geometry_cols(gdf):
    gdf = gdf.copy()
    extra_geom_cols = []

    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        if str(gdf[col].dtype) == "geometry":
            extra_geom_cols.append(col)

    if extra_geom_cols:
        gdf = gdf.drop(columns=extra_geom_cols)

    return gdf



circular_out_export = strip_extra_geometry_cols(circular_out)
non_circular_export = strip_extra_geometry_cols(non_circular)
pivot_sector_export = strip_extra_geometry_cols(pivot_sector)
fit_rejects_export = strip_extra_geometry_cols(fit_rejects)
uncertain_review_export = strip_extra_geometry_cols(uncertain_review)

circular_out_export.to_file(out_circular_shp)
non_circular_export.to_file(out_noncircular_shp)

if len(pivot_sector_export) > 0:
    pivot_sector_export.to_file(out_sector_shp)

if len(fit_rejects_export) > 0:
    fit_rejects_export.to_file(out_rejects_shp)

if len(uncertain_review_export) > 0:
    uncertain_review_export.to_file(out_uncertain_shp)




