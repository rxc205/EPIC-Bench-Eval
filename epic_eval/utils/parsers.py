# -*- coding: utf-8 -*-
"""
Parse the four `formatted_response_*` fields in std_format.

Conventions (aligned with `tools/formatting/format_response.py`):
- "Not Required": the task does not require this field; the metric is skipped.
- "No Valid Response" (or None): invalid/unparseable response; the metric is scored as 0.
- Empty list `[]`: valid response that explicitly indicates "no objects/path/boxes".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from ..config import NOT_REQUIRED, NO_VALID_RESPONSE


def is_not_required(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == NOT_REQUIRED.lower()


def is_no_valid_response(value: Any) -> bool:
    """
    Return True if the field should be treated as "no valid response".

    Note the distinction:
    - "No Valid Response" string / None: invalid response (invalid_response=True).
    - Empty list []: valid response that explicitly means "no targets / no path / no boxes".
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in (
            NO_VALID_RESPONSE.lower(),
            "no_valid_response",
            "invalid response",
            "no response",
            "",
        )
    return False


# ---------------------------------------------------------------------------
# Localization (bbox / mask)
# ---------------------------------------------------------------------------
def parse_localization_bboxes(value: Any) -> List[List[float]]:
    """Parse bbox list from formatted_response_localization. Expects [[x1,y1,x2,y2], ...]."""
    if not isinstance(value, list):
        return []
    out: List[List[float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            try:
                quad = [float(item[0]), float(item[1]), float(item[2]), float(item[3])]
            except (TypeError, ValueError):
                continue
            out.append(quad)
    return out


def parse_localization_mask(value: Any) -> Any:
    """
    In format_type='mask', formatted_response_localization is assumed to be either:
    - compressed RLE list: [{"size":[H,W], "counts": "..."}, ...], or
    - a single compressed RLE dict: {"size":[H,W], "counts":"..."}

    We return the raw value here; decoding is handled by utils.masks.decode_response_mask.
    """
    return value


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------
def parse_count(value: Any) -> Optional[int]:
    """Parse predicted count from formatted_response_count. For list, uses first element."""
    raw = value
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            return int(s)
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
def parse_path_points(value: Any) -> List[Tuple[float, float]]:
    """Parse point sequence from formatted_response_path. Expects [[x,y], ...]."""
    if not isinstance(value, list):
        return []
    pts: List[Tuple[float, float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return pts


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------
def parse_binary(value: Any) -> Optional[bool]:
    """Parse boolean from formatted_response_binary. For list, uses first element."""
    raw = value
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None
