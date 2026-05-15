# -*- coding: utf-8 -*-
"""Shared scorer helpers to build score_details / response_details."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _first(v: Any) -> Any:
    if isinstance(v, list):
        return v[0] if v else None
    return v


def make_score_details(
    *,
    localization: float = 0.0,
    count: float = 0.0,
    path: float = 0.0,
    binary: float = 0.0,
    w_loc: float = 0.0,
    w_count: float = 0.0,
    w_path: float = 0.0,
    w_binary: float = 0.0,
) -> Dict[str, Any]:
    """Build unified score_details (4 scores + 4 weights)."""
    return {
        "localization_score": float(localization),
        "count_score": float(count),
        "path_score": float(path),
        "binary_score": float(binary),
        "weight_localization": float(w_loc),
        "weight_count": float(w_count),
        "weight_path": float(w_path),
        "weight_binary": float(w_binary),
    }


def make_response_details(
    sample: Dict[str, Any],
    actual_coord_mode: Optional[str],
) -> Dict[str, Any]:
    """Collect std_format response fields and metadata for result logging."""
    fmt = _first(sample.get("format_type")) or ""
    coord = actual_coord_mode or _first(sample.get("coordinate_system")) or ""
    return {
        "format_type": fmt,
        "coordinate_system": coord,
        "formatted_response_localization": sample.get("formatted_response_localization"),
        "formatted_response_count": sample.get("formatted_response_count"),
        "formatted_response_path": sample.get("formatted_response_path"),
        "formatted_response_binary": sample.get("formatted_response_binary"),
    }
