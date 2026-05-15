# -*- coding: utf-8 -*-
"""
Normalize bbox/point coordinates to absolute pixels according to coordinate_system.

coordinate_system values (from std_format):
- "auto": infer from numeric range
- "absolute": absolute pixels
- "normalized_0_1": normalized to [0, 1]
- "normalized_0_1000": normalized to [0, 1000]

This module keeps backward-compatible aliases ("normalized", "normalized_1000").
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]
Bbox = Tuple[Number, Number, Number, Number]
Point = Tuple[Number, Number]


# Legacy -> canonical name aliases
_COORD_ALIAS = {
    "normalized": "normalized_0_1",
    "normalized_1000": "normalized_0_1000",
}


def canonical_coord_mode(mode: str) -> str:
    if not mode:
        return "auto"
    return _COORD_ALIAS.get(mode, mode)


def resolve_actual_coord_mode(
    values: Sequence[Number],
    height: int,
    width: int,
    coord_mode: str,
) -> str:
    """Resolve (possibly 'auto') coord_mode into an actual mode.

    Returns one of: absolute / normalized_0_1 / normalized_0_1000.
    If values is empty and coord_mode is auto, fall back to absolute.
    """
    canon = canonical_coord_mode(coord_mode)
    if canon in ("absolute", "normalized_0_1", "normalized_0_1000"):
        return canon
    # auto
    if not values:
        return "absolute"
    max_val = max(values)
    min_val = min(values)
    max_dim = max(height, width)
    if max_val <= 1.0 and min_val >= 0:
        return "normalized_0_1"
    if max_val > max_dim:
        return "normalized_0_1000"
    return "absolute"


def _detect_scale(values: Sequence[Number], height: int, width: int) -> Tuple[float, float]:
    """Infer scaling factors in auto mode from numeric range."""
    if not values:
        return 1.0, 1.0
    max_val = max(values)
    min_val = min(values)
    max_dim = max(height, width)
    if max_val <= 1.0 and min_val >= 0:
        return float(width), float(height)
    if max_val > max_dim:
        return width / 1000.0, height / 1000.0
    return 1.0, 1.0


def _scale_for_mode(mode: str, height: int, width: int, all_vals: Sequence[Number]) -> Tuple[float, float]:
    mode = canonical_coord_mode(mode)
    if mode == "absolute":
        return 1.0, 1.0
    if mode == "normalized_0_1":
        return float(width), float(height)
    if mode == "normalized_0_1000":
        return width / 1000.0, height / 1000.0
    return _detect_scale(all_vals, height, width)


def normalize_bboxes(
    bboxes: Sequence[Bbox],
    height: int,
    width: int,
    coord_mode: str = "auto",
) -> List[Tuple[int, int, int, int]]:
    """Normalize bboxes into [(x1,y1,x2,y2), ...] integer pixels, with clipping."""
    if not bboxes:
        return []
    flat = [v for b in bboxes for v in b]
    sx, sy = _scale_for_mode(coord_mode, height, width, flat)
    out: List[Tuple[int, int, int, int]] = []
    for b in bboxes:
        if not b or len(b) < 4:
            continue
        x1, y1, x2, y2 = float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
        if x2 > x1 and y2 > y1:
            out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def normalize_points(
    points: Sequence[Point],
    height: int,
    width: int,
    coord_mode: str = "auto",
) -> List[Tuple[float, float]]:
    """Normalize points into [(x,y), ...] pixel coordinates, with clipping."""
    if not points:
        return []
    flat = [v for p in points for v in p]
    sx, sy = _scale_for_mode(coord_mode, height, width, flat)
    out: List[Tuple[float, float]] = []
    for p in points:
        if not p or len(p) < 2:
            continue
        x = max(0.0, min(float(p[0]) * sx, float(width - 1)))
        y = max(0.0, min(float(p[1]) * sy, float(height - 1)))
        out.append((x, y))
    return out
