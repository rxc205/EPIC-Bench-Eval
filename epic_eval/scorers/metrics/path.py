# -*- coding: utf-8 -*-
"""
FeasiblePath path score (point-sequence planning).

This implementation follows the legacy bench_eval rules and adapts to std_format:
- feasible area mask: GT field masklet_path
- target area mask: GT field masklet_target_area
- perspective is determined by task_type:
  - FeasiblePath_Ego -> FirstPerspective (single target region)
  - FeasiblePath_Exo -> ThirdPerspective (two target regions)
- coordinate_system comes from each sample.

Weighted components (legacy-consistent):
- start_end / feasibility / away_start / toward_end / continuity
- weights: first point (1,0,0,0,0); last point (0.3,0.4,0.1,0.1,0.1);
  middle points (0,0.4,0.2,0.2,0.2)
- final score = mean per-point scores over filtered points.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.coords import normalize_points, resolve_actual_coord_mode
from ...utils.masks import decode_rle_list_to_mask
from ...utils.parsers import is_no_valid_response, is_not_required, parse_path_points

# ---------------------------------------------------------------------------
# Constants (legacy-consistent)
# ---------------------------------------------------------------------------
WEIGHTS_FIRST_POINT = (1.0, 0.0, 0.0, 0.0, 0.0)
WEIGHTS_LAST_POINT = (0.3, 0.4, 0.1, 0.1, 0.1)
WEIGHTS_MIDDLE_POINT = (0.0, 0.4, 0.2, 0.2, 0.2)

START_END_DISTANCE_K = 3.0
NEAR_POINT_RATIO = 1.0 / 30.0
CONTINUITY_MAX_RATIO = 2.0 / 3.0


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Nearby point filtering
# ---------------------------------------------------------------------------
def filter_nearby_points(
    points: List[Tuple[float, float]],
    height: int,
    width: int,
) -> Tuple[List[Tuple[float, float]], List[bool]]:
    if len(points) <= 1:
        return points, [True] * len(points)
    diag = (height ** 2 + width ** 2) ** 0.5
    min_dist = diag * NEAR_POINT_RATIO
    out: List[Tuple[float, float]] = [points[0]]
    out_indices: List[int] = [0]
    for i in range(1, len(points)):
        if _dist(out[-1], points[i]) >= min_dist:
            out.append(points[i])
            out_indices.append(i)
    participated = [i in set(out_indices) for i in range(len(points))]
    return out, participated


# ---------------------------------------------------------------------------
# Mask geometry utilities
# ---------------------------------------------------------------------------
def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    if mask is None or mask.size == 0:
        return 0.0, 0.0
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return h / 2.0, w / 2.0
    return float(np.mean(ys)), float(np.mean(xs))


def _point_to_mask_min_distance(point: Tuple[float, float], mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return float("inf")
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return float("inf")
    px, py = point[0], point[1]
    row, col = int(round(py)), int(round(px))
    h, w = mask.shape
    if 0 <= row < h and 0 <= col < w and mask[row, col] > 0:
        return 0.0
    return float(np.min(np.sqrt((xs - px) ** 2 + (ys - py) ** 2)))


def _point_to_region_distance(
    point: Tuple[float, float],
    region_mask: Optional[np.ndarray],
    fallback_point: Optional[Tuple[float, float]],
) -> float:
    if region_mask is not None and region_mask.size > 0:
        return _point_to_mask_min_distance(point, region_mask)
    if fallback_point is not None:
        return _dist(point, fallback_point)
    return float("inf")


def _mask_to_mask_min_distance(
    m1: Optional[np.ndarray],
    m2: Optional[np.ndarray],
    fb1: Optional[Tuple[float, float]],
    fb2: Optional[Tuple[float, float]],
) -> float:
    if m1 is None or m1.size == 0:
        if fb1 is not None and m2 is not None and m2.size > 0:
            return _point_to_mask_min_distance(fb1, m2)
        if fb1 is not None and fb2 is not None:
            return _dist(fb1, fb2)
        return float("inf")
    if m2 is None or m2.size == 0:
        if fb2 is not None:
            return _point_to_mask_min_distance(fb2, m1)
        return float("inf")
    if np.any(np.logical_and(m1 > 0, m2 > 0)):
        return 0.0
    ys1, xs1 = np.where(m1 > 0)
    ys2, xs2 = np.where(m2 > 0)
    if len(ys1) == 0 or len(ys2) == 0:
        return float("inf")
    max_samples = 1000
    if len(xs1) > max_samples:
        idx = np.random.choice(len(xs1), max_samples, replace=False)
        xs1, ys1 = xs1[idx], ys1[idx]
    if len(xs2) > max_samples:
        idx = np.random.choice(len(xs2), max_samples, replace=False)
        xs2, ys2 = xs2[idx], ys2[idx]
    min_dist = float("inf")
    for i in range(len(xs1)):
        d = float(np.min(np.sqrt((xs2 - xs1[i]) ** 2 + (ys2 - ys1[i]) ** 2)))
        if d < min_dist:
            min_dist = d
            if min_dist == 0:
                break
    return min_dist


def _segment_in_mask_ratio(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    mask: np.ndarray,
    num_samples: int = 50,
) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    h, w = mask.shape
    inside = 0
    for i in range(num_samples):
        t = (i + 0.5) / num_samples
        x = p1[0] * (1 - t) + p2[0] * t
        y = p1[1] * (1 - t) + p2[1] * t
        col, row = int(round(x)), int(round(y))
        if 0 <= row < h and 0 <= col < w and mask[row, col] > 0:
            inside += 1
    return inside / num_samples


# ---------------------------------------------------------------------------
# Split target_area mask into two logical regions (ThirdPerspective)
# ---------------------------------------------------------------------------
def _split_into_two_regions(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        from scipy import ndimage  # type: ignore
    except ImportError:
        return mask, None

    labeled, n = ndimage.label(mask > 0)
    if n < 2:
        return mask, None

    components = []
    for label_id in range(1, n + 1):
        ys, xs = np.where(labeled == label_id)
        if len(ys) == 0:
            continue
        components.append({
            "label_id": label_id,
            "centroid": (float(np.mean(xs)), float(np.mean(ys))),
            "pixel_count": int(len(ys)),
        })
    if len(components) < 2:
        return mask, None

    # Pick the two farthest connected components as clustering seeds
    seed1, seed2 = 0, 1
    max_d = -1.0
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            c1, c2 = components[i]["centroid"], components[j]["centroid"]
            d = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
            if d > max_d:
                max_d = d
                seed1, seed2 = i, j
    cen1 = components[seed1]["centroid"]
    cen2 = components[seed2]["centroid"]

    cluster1, cluster2 = [], []
    for comp in components:
        c = comp["centroid"]
        d1 = (c[0] - cen1[0]) ** 2 + (c[1] - cen1[1]) ** 2
        d2 = (c[0] - cen2[0]) ** 2 + (c[1] - cen2[1]) ** 2
        (cluster1 if d1 <= d2 else cluster2).append(comp["label_id"])

    region1 = np.zeros_like(mask, dtype=np.uint8)
    region2 = np.zeros_like(mask, dtype=np.uint8)
    for lid in cluster1:
        region1[labeled == lid] = 1
    for lid in cluster2:
        region2[labeled == lid] = 1
    if int(region1.sum()) == 0 or int(region2.sum()) == 0:
        return mask, None
    return region1, region2


# ---------------------------------------------------------------------------
# Determine start/end regions
# ---------------------------------------------------------------------------
def _resolve_start_end_regions(
    target_mask: Optional[np.ndarray],
    perspective: str,
    points: List[Tuple[float, float]],
    height: int,
    width: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[float, float]]:
    """
    perspective ∈ {'FirstPerspective', 'ThirdPerspective'}
    Return (start_mask, end_mask, fallback_start_point).
    - FirstPerspective: no explicit start mask (use image bottom-center as a fallback), end = target_mask
    - ThirdPerspective: split into two regions; the region closer to the first predicted point is start
    """
    fallback_start = (width / 2.0, float(height - 1))
    if target_mask is None:
        return None, None, fallback_start
    if perspective == "FirstPerspective":
        return None, target_mask, fallback_start
    region1, region2 = _split_into_two_regions(target_mask)
    if region2 is None:
        return None, region1, fallback_start
    if not points:
        return region1, region2, fallback_start
    first = points[0]
    d1 = _point_to_mask_min_distance(first, region1)
    d2 = _point_to_mask_min_distance(first, region2)
    if d1 <= d2:
        return region1, region2, fallback_start
    return region2, region1, fallback_start


# ---------------------------------------------------------------------------
# Main entry: path score
# ---------------------------------------------------------------------------
def path_score(
    formatted_response_path: Any,
    path_mask_rle: Any,
    target_area_mask_rle: Any,
    task_type: str,
    coord_mode: str,
) -> Tuple[float, Dict[str, Any], Optional[str]]:
    perspective = "ThirdPerspective" if task_type == "FeasiblePath_Exo" else "FirstPerspective"
    details: Dict[str, Any] = {
        "perspective": perspective,
        "coord_mode": coord_mode,
        "actual_coord_mode": coord_mode if coord_mode != "auto" else None,
        "num_points_before_filter": 0,
        "num_points_after_filter": 0,
        "start_end_score": None,
        "feasibility_score": None,
        "away_start_score": None,
        "toward_end_score": None,
        "continuity_score": None,
        "point_details": [],
    }

    if is_not_required(formatted_response_path):
        return 0.0, {**details, "message": "Not Required"}, None
    if is_no_valid_response(formatted_response_path):
        return 0.0, {**details, "no_valid_response": True}, "No Valid Response"

    feasible_mask = decode_rle_list_to_mask(path_mask_rle)
    if feasible_mask is None:
        return 0.0, details, "Failed to decode feasible area mask (masklet_path) or mask is missing"

    height, width = feasible_mask.shape[0], feasible_mask.shape[1]

    raw_pts = parse_path_points(formatted_response_path)
    if not raw_pts:
        details["actual_coord_mode"] = resolve_actual_coord_mode([], height, width, coord_mode)
        return 0.0, details, "No path points parsed"
    flat_vals = [v for p in raw_pts for v in p]
    details["actual_coord_mode"] = resolve_actual_coord_mode(flat_vals, height, width, coord_mode)
    points_abs = normalize_points(raw_pts, height, width, coord_mode=coord_mode)
    points, participated = filter_nearby_points(points_abs, height, width)
    details["num_points_before_filter"] = len(points_abs)
    details["num_points_after_filter"] = len(points)

    if len(points) < 2:
        details["point_details"] = _build_empty_point_details(raw_pts, participated)
        return 0.0, {**details, "message": "Fewer than 2 points; score is undefined"}, None

    target_mask = decode_rle_list_to_mask(target_area_mask_rle)
    start_mask, end_mask, start_point = _resolve_start_end_regions(
        target_mask, perspective, points, height, width
    )

    diag = (height ** 2 + width ** 2) ** 0.5
    start_end_dist = _mask_to_mask_min_distance(start_mask, end_mask, start_point, None)
    if start_end_dist == float("inf") or start_end_dist <= 0:
        # Degenerate case: use centroid distance
        if start_mask is not None:
            cy, cx = _mask_centroid(start_mask)
            sp = (cx, cy)
        else:
            sp = start_point
        if end_mask is not None:
            cy, cx = _mask_centroid(end_mask)
            ep = (cx, cy)
        else:
            ep = (width / 2.0, height / 2.0)
        start_end_dist = _dist(sp, ep)

    min_segment = diag * NEAR_POINT_RATIO
    max_segment = start_end_dist * CONTINUITY_MAX_RATIO if start_end_dist > 0 else diag

    n = len(points)

    # Per-point distances to start/end regions
    dist_to_start = [_point_to_region_distance(pt, start_mask, start_point) for pt in points]
    dist_to_end = [_point_to_region_distance(pt, end_mask, None) for pt in points]

    # 1. start_end
    d1_norm = dist_to_start[0] / diag if diag > 0 else 0.0
    d2_norm = dist_to_end[-1] / diag if diag > 0 else 0.0
    s_se_first = max(0.0, 1.0 - START_END_DISTANCE_K * d1_norm)
    s_se_last = max(0.0, 1.0 - START_END_DISTANCE_K * d2_norm)
    details["start_end_score"] = (s_se_first + s_se_last) / 2.0

    # 2. feasibility per point i (>=1)
    feasibility_per = [0.0] * n
    for i in range(1, n):
        feasibility_per[i] = _segment_in_mask_ratio(points[i - 1], points[i], feasible_mask)
    details["feasibility_score"] = sum(feasibility_per) / max(1, n - 1) if n > 1 else 0.0

    # 3. away_start
    away_per = [0.0] * n
    for i in range(1, n):
        if dist_to_start[i] > dist_to_start[i - 1]:
            away_per[i] = 1.0
    details["away_start_score"] = sum(away_per[1:]) / max(1, n - 1) if n >= 2 else None

    # 4. toward_end
    toward_per = [0.0] * n
    for i in range(1, n):
        if dist_to_end[i] < dist_to_end[i - 1]:
            toward_per[i] = 1.0
    details["toward_end_score"] = sum(toward_per[1:]) / max(1, n - 1) if n >= 2 else None

    # 5) continuity (middle points)
    continuity_per = [0.0] * n
    for i in range(1, n - 1):
        d = _dist(points[i - 1], points[i])
        if min_segment <= d <= max_segment:
            continuity_per[i] = 1.0
    details["continuity_score"] = sum(continuity_per[1:n - 1]) / max(1, n - 2) if n >= 3 else None

    # Aggregate per-point scores with positional weights
    filtered_point_scores: List[Dict[str, Any]] = []
    for i in range(n):
        if i == 0:
            w_se, w_fea, w_away, w_toward, w_cont = WEIGHTS_FIRST_POINT
        elif i == n - 1:
            w_se, w_fea, w_away, w_toward, w_cont = WEIGHTS_LAST_POINT
        else:
            w_se, w_fea, w_away, w_toward, w_cont = WEIGHTS_MIDDLE_POINT

        is_middle = (0 < i < n - 1)

        s_se = s_se_first if i == 0 else (s_se_last if i == n - 1 else 0.0)
        s_fea = feasibility_per[i] if i > 0 else 0.0
        s_away = away_per[i] if i >= 1 else 0.0
        s_toward = toward_per[i] if i >= 1 else 0.0
        s_cont = continuity_per[i] if is_middle else 0.0

        total_here = 0.0
        w_sum = 0.0
        w_se_act = w_fea_act = w_away_act = w_toward_act = w_cont_act = 0.0

        if i == 0 or i == n - 1:
            total_here += w_se * s_se
            w_sum += w_se
            w_se_act = w_se
        if i > 0:
            total_here += w_fea * s_fea
            w_sum += w_fea
            w_fea_act = w_fea
        if i >= 1:
            total_here += w_away * s_away + w_toward * s_toward
            w_sum += w_away + w_toward
            w_away_act = w_away
            w_toward_act = w_toward
        if is_middle:
            total_here += w_cont * s_cont
            w_sum += w_cont
            w_cont_act = w_cont
        elif i == n - 1 and n >= 3:
            last_d = _dist(points[n - 2], points[n - 1])
            s_cont_last = 1.0 if min_segment <= last_d <= max_segment else 0.0
            total_here += w_cont * s_cont_last
            w_sum += w_cont
            w_cont_act = w_cont
            s_cont = s_cont_last

        pt_score = total_here / w_sum if w_sum > 0 else 0.0
        filtered_point_scores.append({
            "filtered_idx": i,
            "score": pt_score,
            "start_end": s_se,
            "feasibility": s_fea,
            "away_start": s_away,
            "toward_end": s_toward,
            "continuity": s_cont,
            "dist_to_start": dist_to_start[i],
            "dist_to_end": dist_to_end[i],
            "weights": {
                "weight_sum": w_sum,
                "start_end_weight": w_se_act,
                "feasibility_weight": w_fea_act,
                "away_start_weight": w_away_act,
                "toward_end_weight": w_toward_act,
                "continuity_weight": w_cont_act,
            },
        })

    total = sum(ps["score"] for ps in filtered_point_scores) / n if filtered_point_scores else 0.0

    # Merge raw point details (keep all original points + participation flag)
    point_details: List[Dict[str, Any]] = []
    filtered_idx = 0
    for i in range(len(raw_pts)):
        entry: Dict[str, Any] = {
            "raw_coord": list(raw_pts[i]),
            "participated_in_scoring": bool(participated[i]),
            "scores": None,
        }
        if participated[i] and filtered_idx < len(filtered_point_scores):
            entry["scores"] = filtered_point_scores[filtered_idx]
            filtered_idx += 1
        point_details.append(entry)
    details["point_details"] = point_details

    return float(total), details, None


def _build_empty_point_details(raw_pts, participated):
    if not raw_pts:
        return []
    return [
        {
            "raw_coord": list(raw_pts[i]),
            "participated_in_scoring": bool(participated[i]) if i < len(participated) else False,
            "scores": None,
        }
        for i in range(len(raw_pts))
    ]
