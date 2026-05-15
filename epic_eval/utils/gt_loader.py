# -*- coding: utf-8 -*-
"""
Ground-truth (GT) loader utilities.

Workflow:
1) Scorers first read required GT fields from sample["ground_truth"] directly.
2) If missing, fall back to the raw GT JSON referenced by:
     sample["ground_truth"]["gt_json_path"][0]
   (written by tools/formatting/format_response.py), and read from its
   mask_annotation / text_annotation.

In most std_format samples, ground_truth only contains gt_json_path, so the fallback
must be robust.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ..config import get_gt_fields


# ---------------------------------------------------------------------------
# Cached gt_json_path reading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _load_gt_json_cached(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _get_gt_json_path(ground_truth: Dict[str, Any]) -> Optional[str]:
    raw = ground_truth.get("gt_json_path") if isinstance(ground_truth, dict) else None
    if isinstance(raw, list) and raw:
        return str(raw[0]) if raw[0] is not None else None
    if isinstance(raw, str):
        return raw
    return None


def load_raw_gt_json(ground_truth: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load raw GT JSON (cached)."""
    p = _get_gt_json_path(ground_truth or {})
    if not p:
        return None
    return _load_gt_json_cached(p)


# ---------------------------------------------------------------------------
# Field resolution: mask / count / bool, etc.
# ---------------------------------------------------------------------------
def _get_first(value: Any) -> Any:
    """std_format fields are usually lists; take the first element."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _resolve_from_raw(raw: Optional[Dict[str, Any]], field: str) -> Any:
    """Resolve a field from raw GT JSON's text_annotation / mask_annotation."""
    if not raw:
        return None
    text_ann = raw.get("text_annotation") or {}
    mask_ann = raw.get("mask_annotation") or {}
    if field in text_ann:
        return text_ann[field]
    if field in mask_ann:
        return mask_ann[field]
    # PlacementRegion compatibility: std_format uses placement_feasibility; raw JSON uses placement_feasible
    if field == "placement_feasibility" and "placement_feasible" in text_ann:
        return text_ann["placement_feasible"]
    return None


def _get_field(
    ground_truth: Dict[str, Any],
    field: str,
    raw_gt: Optional[Dict[str, Any]] = None,
) -> Any:
    """Resolve a GT field with raw-GT-first policy.

    Prefer the raw GT JSON (referenced by ground_truth.gt_json_path). If the field is
    missing or the raw GT cannot be loaded, fall back to the std_format field.

    Rationale: std_format ground_truth fields are copies produced by the formatter and
    may occasionally diverge from raw annotations.
    """
    v = _resolve_from_raw(raw_gt, field)
    if v is not None:
        return v
    if isinstance(ground_truth, dict) and field in ground_truth:
        v = ground_truth[field]
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# High-level API: collect all GT info needed for scoring a task
# ---------------------------------------------------------------------------
class GTContext:
    """Parse one sample's GT into a scorer-friendly object.

    - localization_mask / target_area_mask / path_mask: list[RLE dict] or None
    - count_gt: int or None
    - precise_count: bool
    - count_in_description: bool
    - feasibility: bool or None
    """

    def __init__(self, sample: Dict[str, Any]):
        self.sample = sample
        self.task_category: str = (sample.get("task_category") or [""])[0] if isinstance(sample.get("task_category"), list) else (sample.get("task_category") or "")
        self.task_type: str = (sample.get("task_type") or [""])[0] if isinstance(sample.get("task_type"), list) else (sample.get("task_type") or "")
        self.ground_truth: Dict[str, Any] = sample.get("ground_truth") or {}
        self.raw_gt: Optional[Dict[str, Any]] = load_raw_gt_json(self.ground_truth)
        self._fields = get_gt_fields(self.task_category, self.task_type)

    # ----- mask fields -----
    def get_mask_field(self, key: str) -> Any:
        """key ∈ {'localization_mask', 'target_area_mask', 'path_mask'}."""
        field_name = self._fields.get(key)
        if not field_name:
            return None
        return _get_field(self.ground_truth, field_name, self.raw_gt)

    @property
    def localization_mask_rle(self) -> Any:
        return self.get_mask_field("localization_mask")

    @property
    def target_area_mask_rle(self) -> Any:
        return self.get_mask_field("target_area_mask")

    @property
    def path_mask_rle(self) -> Any:
        return self.get_mask_field("path_mask")

    # ----- count -----
    @property
    def count_gt(self) -> Optional[int]:
        field_name = self._fields.get("count_field")
        if not field_name:
            return None
        v = _get_field(self.ground_truth, field_name, self.raw_gt)
        v = _get_first(v)
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return None
        return None

    @property
    def precise_count(self) -> bool:
        for k in ("precise_count_field", "precise_count_field_alt"):
            field_name = self._fields.get(k)
            if not field_name:
                continue
            v = _get_field(self.ground_truth, field_name, self.raw_gt)
            v = _get_first(v)
            if v is None:
                continue
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1")
        return False

    @property
    def count_in_description(self) -> bool:
        field_name = self._fields.get("count_in_description_field")
        if not field_name:
            return False
        v = _get_field(self.ground_truth, field_name, self.raw_gt)
        v = _get_first(v)
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1")
        return False

    # ----- binary -----
    @property
    def feasibility(self) -> Optional[bool]:
        for k in ("feasibility_field", "feasibility_field_alt"):
            field_name = self._fields.get(k)
            if not field_name:
                continue
            v = _get_field(self.ground_truth, field_name, self.raw_gt)
            v = _get_first(v)
            if v is None:
                continue
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "yes", "1"):
                    return True
                if s in ("false", "no", "0"):
                    return False
        return None
