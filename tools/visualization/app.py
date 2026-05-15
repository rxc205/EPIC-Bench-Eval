# -*- coding: utf-8 -*-
"""
EPIC-Bench evaluation visualization (Streamlit app).

This app is compatible with detailed results produced by `epic_eval` (e.g. `*_full.json`
or `*_full.jsonl`). Each sample contains:
- task_category / task_type / score / score_details / response_details
- ground_truth.gt_json_path (raw GT JSON path)

The app loads raw GT fields from the JSON referenced by gt_json_path
(mask_annotation / text_annotation).

Run:
  streamlit run tools/visualization/app.py
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Optional: reuse epic_eval utilities (coords / RLE decoding).
# If import fails, this file falls back to local equivalents.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from epic_eval.utils.coords import normalize_bboxes as _eval_normalize_bboxes
    from epic_eval.utils.coords import normalize_points as _eval_normalize_points
    from epic_eval.utils.coords import resolve_actual_coord_mode as _eval_resolve_coord_mode
    from epic_eval.utils.masks import decode_rle_list_to_mask as _eval_decode_rle
    from epic_eval.utils.masks import decode_response_mask as _eval_decode_resp_mask
except Exception:
    _eval_normalize_bboxes = None
    _eval_normalize_points = None
    _eval_resolve_coord_mode = None
    _eval_decode_rle = None
    _eval_decode_resp_mask = None

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    from pycocotools import mask as mask_utils  # type: ignore
    _HAS_PYCOCO = True
except Exception:
    _HAS_PYCOCO = False


# =========================================================================== #
# Visualization config: decide how to display text / masks / predictions per task.
# - text_fields: fields read from raw GT text_annotation
# - text_label_field: primary title field
# - reference_mask_field/color: reference object mask (blue)
# - primary_mask_field/color: primary GT mask (red)
# - secondary_mask_field/color: secondary GT mask (green; FeasiblePath)
# - response_kind: "bbox" / "point" / "bbox_binary"
# - is_dual_image: True for dual-image tasks (VisualMatching)
# - image1_mask_field: GT mask overlay for image1
# - image2_mask_field: GT mask overlay for image2 (where prediction bbox is drawn)
# =========================================================================== #
COLOR_REFERENCE = (0, 120, 255)   # reference object: blue
COLOR_PRIMARY = (255, 0, 0)       # primary GT: red
COLOR_SECONDARY = (0, 255, 0)     # secondary GT: green
COLOR_PRED_BBOX = (0, 255, 0)     # predicted bbox: green
COLOR_PRED_POINT = (0, 255, 255)  # predicted point: cyan
COLOR_PRED_MASK = (255, 215, 0)   # predicted mask: gold

TL_TEXT_FIELDS = [
    "target_object_description_en",
    "target_object_description_cn",
    "count_target_instance",
    "precise_count",
    "count_in_description",
]

TASK_VIZ_CONFIG: Dict[Tuple[str, str], Dict[str, Any]] = {
    # TargetLocalization: all 14 leaf tasks share one config
    ("TargetLocalization", "*"): {
        "text_label_field": "target_object_description_en",
        "text_fields": TL_TEXT_FIELDS,
        "primary_mask_field": "masklet_target",
        "primary_mask_label": "Target (masklet_target)",
        "primary_mask_color": COLOR_PRIMARY,
        "response_kind": "bbox",
    },
    ("Navigation", "GroundDetection"): {
        "text_label_field": "scene_description_en",
        "text_fields": ["scene_description_en", "scene_description_cn", "precise_description"],
        "primary_mask_field": "masklet_ground",
        "primary_mask_label": "Ground (masklet_ground)",
        "primary_mask_color": COLOR_PRIMARY,
        "response_kind": "bbox",
    },
    ("Navigation", "FeasiblePath_Ego"): {
        "text_label_field": "target_area_description_en",
        "text_fields": [
            "target_area_description_en",
            "target_area_description_cn",
            "visually_blocked",
            "precise_description",
        ],
        "primary_mask_field": "masklet_path",
        "primary_mask_label": "Feasible area (masklet_path)",
        "primary_mask_color": COLOR_PRIMARY,
        "secondary_mask_field": "masklet_target_area",
        "secondary_mask_label": "Target area (masklet_target_area)",
        "secondary_mask_color": COLOR_SECONDARY,
        "response_kind": "point",
    },
    ("Navigation", "FeasiblePath_Exo"): {
        "text_label_field": "target_area_description_en",
        "text_fields": [
            "target_area_description_en",
            "target_area_description_cn",
            "visually_blocked",
            "precise_description",
        ],
        "primary_mask_field": "masklet_path",
        "primary_mask_label": "Feasible area (masklet_path)",
        "primary_mask_color": COLOR_PRIMARY,
        "secondary_mask_field": "masklet_target_area",
        "secondary_mask_label": "Target area (masklet_target_area)",
        "secondary_mask_color": COLOR_SECONDARY,
        "response_kind": "point",
    },
    ("Navigation", "VisualMatching"): {
        "text_label_field": "reference_area_description_en",
        "text_fields": [
            "reference_area_description_en",
            "reference_area_description_cn",
            "count_matching_area",
            "precise_description",
        ],
        "is_dual_image": True,
        "image1_mask_field": "masklet_reference_area",
        "image1_mask_label": "Reference area (masklet_reference_area)",
        "image1_mask_color": COLOR_REFERENCE,
        "image2_mask_field": "masklet_matching_area",
        "image2_mask_label": "Target area (masklet_matching_area)",
        "image2_mask_color": COLOR_PRIMARY,
        # Prediction rendering on image2 in dual-image mode
        "response_kind": "bbox",
    },
    ("Manipulation", "AffordanceRegion"): {
        "text_label_field": "target_object_description_en",
        "text_fields": [
            "target_object_description_en",
            "target_object_description_cn",
            "reference_object_description_en",
            "reference_object_description_cn",
            "action_en",
            "action_zh",
            "num_hands",
            "precise_description",
        ],
        "reference_mask_field": "masklet_reference_area",
        "reference_mask_label": "Reference area (masklet_reference_area)",
        "reference_mask_color": COLOR_REFERENCE,
        "primary_mask_field": "masklet_affordance_region",
        "primary_mask_label": "Affordance region (masklet_affordance_region)",
        "primary_mask_color": COLOR_PRIMARY,
        "response_kind": "bbox",
    },
    ("Manipulation", "ContactRelationship_TypeOne"): {
        "text_label_field": "reference_object_description_en",
        "text_fields": [
            "reference_object_description_en",
            "reference_object_description_cn",
            "count_contact_target",
            "precise_description",
        ],
        "reference_mask_field": "masklet_reference_object",
        "reference_mask_label": "Reference object (masklet_reference_object)",
        "reference_mask_color": COLOR_REFERENCE,
        "primary_mask_field": "masklet_contact_target",
        "primary_mask_label": "Contact target (masklet_contact_target)",
        "primary_mask_color": COLOR_PRIMARY,
        "response_kind": "bbox",
    },
    ("Manipulation", "ContactRelationship_TypeTwo"): {
        "text_label_field": "reference_object_description_en",
        "text_fields": [
            "reference_object_description_en",
            "reference_object_description_cn",
            "count_contact_target",
            "precise_description",
        ],
        "reference_mask_field": "masklet_reference_object",
        "reference_mask_label": "Reference object (masklet_reference_object)",
        "reference_mask_color": COLOR_REFERENCE,
        "primary_mask_field": "masklet_contact_target",
        "primary_mask_label": "Contact target (masklet_contact_target)",
        "primary_mask_color": COLOR_PRIMARY,
        "response_kind": "bbox",
    },
    ("Manipulation", "ContactRelationship_TypeThree"): {
        "text_label_field": "reference_object_description_en",
        "text_fields": [
            "reference_object_description_en",
            "reference_object_description_cn",
            "count_contact_target",
            "precise_description",
        ],
        "reference_mask_field": "masklet_reference_object",
        "reference_mask_label": "Reference object (masklet_reference_object)",
        "reference_mask_color": COLOR_REFERENCE,
        "primary_mask_field": "masklet_contact_target",
        "primary_mask_label": "Contact target (masklet_contact_target)",
        "primary_mask_color": COLOR_PRIMARY,
        "response_kind": "bbox",
    },
    ("Manipulation", "PlacementRegion"): {
        "text_label_field": "target_object_description_en",
        "text_fields": [
            "target_object_description_en",
            "target_object_description_cn",
            "reference_object_description_en",
            "reference_object_description_cn",
            "placement_region_description_en",
            "placement_region_description_cn",
            "placement_feasibility",
            "precise_description",
        ],
        "reference_mask_field": "masklet_reference_object",
        "reference_mask_label": "Reference object (masklet_reference_object)",
        "reference_mask_color": COLOR_REFERENCE,
        "primary_mask_field": "masklet_placement_region",
        "primary_mask_label": "Placement region (masklet_placement_region)",
        "primary_mask_color": COLOR_PRIMARY,
        "response_kind": "bbox_binary",
    },
}


def get_viz_config(task_category: str, task_type: str) -> Dict[str, Any]:
    """Get visualization config by (category, type). TargetLocalization tasks share one config."""
    if task_category == "TargetLocalization":
        return TASK_VIZ_CONFIG.get(("TargetLocalization", "*"), {})
    cfg = TASK_VIZ_CONFIG.get((task_category, task_type))
    if cfg is not None:
        return cfg
    # Legacy compatibility: FeasiblePath (no _Ego/_Exo) / Affordance / Placement / ContactRelationship
    if (task_category, task_type) == ("Navigation", "FeasiblePath"):
        return TASK_VIZ_CONFIG[("Navigation", "FeasiblePath_Ego")]
    if (task_category, task_type) == ("Manipulation", "Affordance"):
        return TASK_VIZ_CONFIG[("Manipulation", "AffordanceRegion")]
    if (task_category, task_type) == ("Manipulation", "Placement"):
        return TASK_VIZ_CONFIG[("Manipulation", "PlacementRegion")]
    if (task_category, task_type) == ("Manipulation", "ContactRelationship"):
        return TASK_VIZ_CONFIG[("Manipulation", "ContactRelationship_TypeOne")]
    return {}


# =========================================================================== #
# Key text fields (shown in the primary info panel).
# Note: some legacy annotations may use slightly different names; light fallbacks apply.
# =========================================================================== #
KEY_TEXT_FIELDS: Dict[Tuple[str, str], List[str]] = {
    ("TargetLocalization", "*"): ["target_object_description_en", "count_target_instance", "precise_count"],
    ("Navigation", "VisualMatching"): ["count_matching_area", "precise_count"],
    ("Manipulation", "AffordanceRegion"): ["task_description_en", "reference_object_description_en"],
    ("Manipulation", "ContactRelationship"): ["count_contact_target", "precise_count"],
    ("Manipulation", "PlacementRegion"): [
        "reference_object_description_en",
        "placement_region_description_en",
        "placement_feasibility",
    ],
}


def get_key_text_fields(task_category: str, task_type: str) -> List[str]:
    if task_category == "TargetLocalization":
        return KEY_TEXT_FIELDS.get(("TargetLocalization", "*"), [])
    if (task_category, task_type) in KEY_TEXT_FIELDS:
        return KEY_TEXT_FIELDS[(task_category, task_type)]
    if task_category == "Manipulation" and task_type.startswith("ContactRelationship"):
        return KEY_TEXT_FIELDS.get(("Manipulation", "ContactRelationship"), [])
    return []


def _read_text_field_with_fallback(text_ann: Dict[str, Any], field: str) -> Any:
    """Get a field from text_annotation with small task-specific fallbacks."""
    if field in text_ann:
        return text_ann.get(field)
    # AffordanceRegion: some annotations use target_object_description_en for task description
    if field == "task_description_en":
        return text_ann.get("target_object_description_en") or text_ann.get("text_label")
    # PlacementRegion: sometimes uses placement_feasible naming
    if field == "placement_feasibility":
        return text_ann.get("placement_feasible")
    return None


# =========================================================================== #
# GT loader utilities (cached)
# =========================================================================== #
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


def get_gt_json_path(sample: Dict[str, Any]) -> Optional[str]:
    gt = sample.get("ground_truth") or {}
    raw = gt.get("gt_json_path")
    if isinstance(raw, list) and raw:
        return str(raw[0]) if raw[0] else None
    if isinstance(raw, str):
        return raw
    return None


def load_raw_gt(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = get_gt_json_path(sample)
    if not p:
        return None
    return _load_gt_json_cached(p)


def get_first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def get_text_annotation_field(raw_gt: Optional[Dict[str, Any]], field: str) -> Any:
    if not raw_gt:
        return None
    text_ann = raw_gt.get("text_annotation") or {}
    if field in text_ann:
        return text_ann[field]
    # Compatibility: placement_feasibility may appear as placement_feasible in legacy GT
    if field == "placement_feasibility" and "placement_feasible" in text_ann:
        return text_ann["placement_feasible"]
    return None


def get_mask_field(raw_gt: Optional[Dict[str, Any]], field: str) -> Any:
    if not raw_gt:
        return None
    mask_ann = raw_gt.get("mask_annotation") or {}
    return mask_ann.get(field)


def resolve_media_path(gt_json_path: Optional[str], path_value: Any) -> Optional[str]:
    """
    Resolve an image path stored in raw GT JSON into a readable local absolute path.

    - If absolute path exists: return it.
    - If relative path (e.g. EPIC_Bench/Manipulation/.../xxx_image.jpg), try:
      1) dirname(gt_json) / relative_path
      2) dirname(gt_json) / basename(relative_path)  (common when image sits next to mask_rle.json)
      3) walk up parent directories from dirname(gt_json) and repeat (1)(2)
    """
    if path_value is None:
        return None
    if not isinstance(path_value, str):
        path_value = str(path_value)
    s = path_value.strip()
    if not s:
        return None

    p_try = Path(s)
    if p_try.is_absolute():
        try:
            resolved = p_try.resolve()
            if resolved.is_file():
                return str(resolved)
        except (OSError, RuntimeError):
            pass

    rel = Path(s)
    candidates: List[Path] = []
    seen: set[str] = set()

    def add(c: Path) -> None:
        try:
            r = c.resolve()
            key = str(r)
            if key in seen:
                return
            seen.add(key)
            candidates.append(r)
        except (OSError, RuntimeError):
            return

    if gt_json_path:
        try:
            gt_dir = Path(gt_json_path).resolve().parent
        except (OSError, RuntimeError):
            gt_dir = None
        if gt_dir is not None:
            add(gt_dir / rel)
            add(gt_dir / rel.name)
            depth = 0
            anc = gt_dir
            while depth < 80:
                add(anc / rel)
                add(anc / rel.name)
                parent = anc.parent
                if parent == anc:
                    break
                anc = parent
                depth += 1

    add(Path.cwd() / rel)
    add(Path.cwd() / rel.name)

    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return None


def get_image_paths(raw_gt: Optional[Dict[str, Any]], gt_json_path: Optional[str] = None) -> List[str]:
    """Prefer image_path_saved, else image_path_original; resolve via resolve_media_path()."""
    if not raw_gt:
        return []
    saved = raw_gt.get("image_path_saved") or []
    orig = raw_gt.get("image_path_original") or []
    raw_list: List[str] = []
    if isinstance(saved, list) and saved:
        raw_list = [str(p) for p in saved if p]
    elif isinstance(orig, list) and orig:
        raw_list = [str(p) for p in orig if p]
    elif isinstance(saved, str) and saved.strip():
        raw_list = [saved]
    elif isinstance(orig, str) and orig.strip():
        raw_list = [orig]

    out: List[str] = []
    for item in raw_list:
        resolved = resolve_media_path(gt_json_path, item)
        if resolved:
            out.append(resolved)
    return out


# =========================================================================== #
# Image/mask utilities
# =========================================================================== #
def load_image(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or Image is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.array(Image.open(p).convert("RGB"))
    except Exception:
        return None


def _decode_single_rle_local(rle: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(rle, dict):
        return None
    size = rle.get("size")
    counts = rle.get("counts")
    if not size or counts is None:
        return None
    try:
        h, w = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not _HAS_PYCOCO:
        return None
    try:
        if isinstance(counts, str):
            counts_b = counts.encode("utf-8")
        elif isinstance(counts, bytes):
            counts_b = counts
        else:
            # Uncompressed RLE: convert to compressed first
            rle_obj = mask_utils.frPyObjects([{"size": [h, w], "counts": counts}], h, w)
            m = mask_utils.decode(rle_obj)
            if m is None:
                return None
            if m.ndim == 3:
                m = m[..., 0]
            return (m > 0).astype(np.uint8)
        m = mask_utils.decode({"size": [h, w], "counts": counts_b})
        if m is None or m.size == 0:
            return None
        return (m > 0).astype(np.uint8)
    except Exception:
        return None


def decode_rle_list(rle_list: Any) -> Optional[np.ndarray]:
    """RLE list / single dict -> binary mask (union if multiple)."""
    if _eval_decode_rle is not None:
        try:
            return _eval_decode_rle(rle_list)
        except Exception:
            pass
    if rle_list is None:
        return None
    items: List[Dict[str, Any]] = []
    if isinstance(rle_list, list):
        items = [x for x in rle_list if isinstance(x, dict)]
    elif isinstance(rle_list, dict):
        items = [rle_list]
    if not items:
        return None
    masks: List[np.ndarray] = []
    h_out, w_out = None, None
    for it in items:
        m = _decode_single_rle_local(it)
        if m is None:
            continue
        if h_out is None:
            h_out, w_out = m.shape[0], m.shape[1]
        if m.shape == (h_out, w_out):
            masks.append(m)
    if not masks or h_out is None:
        return None
    out = np.zeros((h_out, w_out), dtype=np.uint8)
    for m in masks:
        np.maximum(out, m, out=out)
    return out


def decode_response_mask_any(value: Any) -> Optional[np.ndarray]:
    if _eval_decode_resp_mask is not None:
        try:
            return _eval_decode_resp_mask(value)
        except Exception:
            pass
    if value is None:
        return None
    if isinstance(value, list):
        return decode_rle_list([x for x in value if isinstance(x, dict)])
    if isinstance(value, dict):
        return decode_rle_list([value])
    return None


def resize_mask_to_shape(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    m = (mask > 0).astype(np.uint8)
    pil_m = Image.fromarray(m).resize((target_w, target_h), Image.NEAREST)
    return (np.array(pil_m) > 0).astype(np.uint8)


def overlay_mask(img: np.ndarray, mask: Optional[np.ndarray], color: Tuple[int, int, int],
                 alpha: float = 0.4) -> np.ndarray:
    if img is None or mask is None:
        return img
    if img.shape[:2] != mask.shape[:2]:
        mask = resize_mask_to_shape(mask, img.shape[0], img.shape[1])
    if mask is None:
        return img
    out = img.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    mask_f = (mask > 0).astype(np.float32)[:, :, np.newaxis]
    out = out * (1 - alpha * mask_f) + c * alpha * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_bboxes(img: np.ndarray, bboxes: Sequence[Tuple[int, int, int, int]],
                color: Tuple[int, int, int] = COLOR_PRED_BBOX, width: int = 3) -> np.ndarray:
    if img is None or not bboxes:
        return img
    out = img.copy()
    h, w = out.shape[0], out.shape[1]
    for box in bboxes:
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 > x1 and y2 > y1:
            out[y1: y1 + width, x1: x2] = color
            out[max(0, y2 - width): y2, x1: x2] = color
            out[y1: y2, x1: x1 + width] = color
            out[y1: y2, max(0, x2 - width): x2] = color
    return out


def draw_points(img: np.ndarray, points: Sequence[Tuple[float, float]],
                color: Tuple[int, int, int] = COLOR_PRED_POINT, radius: int = 6,
                connect: bool = True, line_width: int = 2) -> np.ndarray:
    if img is None or not points:
        return img
    out = img.copy()
    h, w = out.shape[0], out.shape[1]

    if connect and len(points) >= 2:
        for i in range(len(points) - 1):
            x0, y0 = int(points[i][0]), int(points[i][1])
            x1, y1 = int(points[i + 1][0]), int(points[i + 1][1])
            x0 = min(max(0, x0), w - 1); y0 = min(max(0, y0), h - 1)
            x1 = min(max(0, x1), w - 1); y1 = min(max(0, y1), h - 1)
            n = max(abs(x1 - x0), abs(y1 - y0)) + 1
            xs = np.linspace(x0, x1, n).astype(int)
            ys = np.linspace(y0, y1, n).astype(int)
            for dy in range(-line_width // 2, line_width // 2 + 1):
                for dx in range(-line_width // 2, line_width // 2 + 1):
                    yy = np.clip(ys + dy, 0, h - 1)
                    xx = np.clip(xs + dx, 0, w - 1)
                    out[yy, xx] = color

    for (px, py) in points:
        x = min(max(0, int(px)), w - 1)
        y = min(max(0, int(py)), h - 1)
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        out[y1:y2, x1:x2] = color
    return out


# =========================================================================== #
# Coordinate normalization to absolute pixels
# =========================================================================== #
COORD_CHOICES = ("from_sample", "auto", "absolute", "normalized_0_1", "normalized_0_1000")
COORD_LABELS = {
    "from_sample": "From sample (response_details.coordinate_system)",
    "auto": "Auto detect",
    "absolute": "Absolute pixels",
    "normalized_0_1": "Normalized [0, 1]",
    "normalized_0_1000": "Normalized [0, 1000]",
}


def _scale_for_mode(mode: str, height: int, width: int, all_vals: Sequence[float]) -> Tuple[float, float]:
    if mode == "absolute":
        return 1.0, 1.0
    if mode == "normalized_0_1":
        return float(width), float(height)
    if mode == "normalized_0_1000":
        return width / 1000.0, height / 1000.0
    # auto
    if not all_vals:
        return 1.0, 1.0
    max_val = max(all_vals)
    min_val = min(all_vals)
    max_dim = max(height, width)
    if max_val <= 1.0 and min_val >= 0:
        return float(width), float(height)
    if max_val > max_dim:
        return width / 1000.0, height / 1000.0
    return 1.0, 1.0


def normalize_bboxes(bboxes: Sequence[Sequence[float]], height: int, width: int,
                     coord_mode: str) -> List[Tuple[int, int, int, int]]:
    if _eval_normalize_bboxes is not None and coord_mode != "from_sample":
        try:
            return list(_eval_normalize_bboxes(bboxes, height, width, coord_mode))
        except Exception:
            pass
    if not bboxes:
        return []
    flat: List[float] = []
    for b in bboxes:
        for v in b:
            try:
                flat.append(float(v))
            except (TypeError, ValueError):
                pass
    sx, sy = _scale_for_mode(coord_mode, height, width, flat)
    out: List[Tuple[int, int, int, int]] = []
    for b in bboxes:
        if not b or len(b) < 4:
            continue
        try:
            x1 = float(b[0]) * sx; y1 = float(b[1]) * sy
            x2 = float(b[2]) * sx; y2 = float(b[3]) * sy
        except (TypeError, ValueError):
            continue
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
        if x2 > x1 and y2 > y1:
            out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def normalize_points(points: Sequence[Sequence[float]], height: int, width: int,
                     coord_mode: str) -> List[Tuple[float, float]]:
    if _eval_normalize_points is not None and coord_mode != "from_sample":
        try:
            return list(_eval_normalize_points(points, height, width, coord_mode))
        except Exception:
            pass
    if not points:
        return []
    flat: List[float] = []
    for p in points:
        for v in p:
            try:
                flat.append(float(v))
            except (TypeError, ValueError):
                pass
    sx, sy = _scale_for_mode(coord_mode, height, width, flat)
    out: List[Tuple[float, float]] = []
    for p in points:
        if not p or len(p) < 2:
            continue
        try:
            x = float(p[0]) * sx
            y = float(p[1]) * sy
        except (TypeError, ValueError):
            continue
        x = max(0.0, min(x, float(width - 1)))
        y = max(0.0, min(y, float(height - 1)))
        out.append((x, y))
    return out


def resolve_coord_mode(sample: Dict[str, Any], user_choice: str) -> str:
    """If user_choice=='from_sample', use response_details.coordinate_system; otherwise use the chosen mode."""
    if user_choice != "from_sample":
        return user_choice
    rd = sample.get("response_details") or {}
    cs = rd.get("coordinate_system") or "auto"
    return str(cs)


# =========================================================================== #
# Parse model response fields
# =========================================================================== #
NOT_REQUIRED_VALUES = {"not required", "not_required"}
NO_VALID_VALUES = {"no valid response", "no_valid_response", "invalid response", "no response", ""}


def is_not_required(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in NOT_REQUIRED_VALUES


def is_no_valid(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in NO_VALID_VALUES
    return False


def parse_bboxes(value: Any) -> List[List[float]]:
    if not isinstance(value, list):
        return []
    out: List[List[float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            try:
                out.append([float(item[0]), float(item[1]), float(item[2]), float(item[3])])
            except (TypeError, ValueError):
                continue
    return out


def parse_path_points(value: Any) -> List[Tuple[float, float]]:
    if not isinstance(value, list):
        return []
    out: List[Tuple[float, float]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def parse_count(value: Any) -> Optional[int]:
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


def parse_binary(value: Any) -> Optional[bool]:
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


# =========================================================================== #
# Data loading (upload / path)
# =========================================================================== #
def load_results_data(uploaded, path_text: Optional[str]) -> Optional[Dict[str, Any]]:
    if uploaded is not None:
        try:
            return json.load(uploaded)
        except Exception as e:
            st.sidebar.error(f"Failed to parse uploaded JSON: {e}")
            return None
    if path_text and Path(path_text).exists():
        p = Path(path_text)
        try:
            if p.suffix.lower() == ".jsonl":
                return _load_eval_jsonl(p)
            with open(path_text, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.sidebar.error(f"Failed to read result file: {e}")
            return None
    return None


def get_samples(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not data:
        return []
    details = data.get("details") or {}
    if isinstance(details, dict) and isinstance(details.get("by_samples"), list):
        return details["by_samples"]
    # Compatibility: older format (top-level "samples")
    if isinstance(data.get("samples"), list):
        return data["samples"]
    return []


def get_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    return (data or {}).get("summary") or {}


def _load_eval_jsonl(p: Path) -> Optional[Dict[str, Any]]:
    """Load epic_eval JSONL output (first line summary, following lines kind=sample)."""
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    samples: List[Dict[str, Any]] = []
    summary: Any = None
    std_ref: Any = None
    coord: Any = None
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = obj.get("kind")
        if k == "summary":
            summary = obj.get("summary")
            std_ref = obj.get("std_response_jsonl")
            coord = obj.get("coord_override")
        elif k == "sample":
            row = {x: y for x, y in obj.items() if x != "kind"}
            samples.append(row)
        elif isinstance(obj, dict) and ("index" in obj or "task_category" in obj):
            samples.append(obj)
    if summary is None and not samples:
        return None
    out: Dict[str, Any] = {
        "summary": summary or {},
        "details": {"by_samples": samples},
    }
    if std_ref is not None:
        out["std_response_jsonl"] = std_ref
    if coord is not None:
        out["coord_override"] = coord
    return out


def discover_default_results_path() -> str:
    """Discover a default results path from EPIC_VIS_SCORES_ROOT / EPIC_VIS_RESULTS_JSON."""
    ex = os.environ.get("EPIC_VIS_RESULTS_JSON", "").strip()
    if ex:
        p = Path(ex)
        if p.is_file() and p.exists():
            return str(p.resolve())
    root = os.environ.get("EPIC_VIS_SCORES_ROOT", "").strip()
    if not root:
        return ""
    r = Path(root)
    if r.is_file() and r.exists():
        return str(r.resolve())
    if not r.is_dir():
        return ""
    for pattern in ("**/*_full.json", "**/*_full.jsonl", "**/*.json", "**/*.jsonl"):
        found = sorted(r.glob(pattern))
        if found:
            return str(found[0].resolve())
    return ""


# =========================================================================== #
# Streamlit page
# =========================================================================== #
st.set_page_config(page_title="EPIC-Bench Visualization", layout="wide")

st.title("EPIC-Bench Visualization")
st.caption(
    "Visualize detailed results produced by epic_eval. Each sample shows images, GT fields/masks, "
    "model prediction overlays, and scoring breakdowns. Raw GT is loaded from `ground_truth.gt_json_path`."
)

# --------------- Data source ---------------
_default_for_path = discover_default_results_path()
if "epic_viz_radio" not in st.session_state:
    st.session_state["epic_viz_radio"] = (
        "Enter JSON path" if _default_for_path else "Upload JSON"
    )
if "epic_viz_json_path" not in st.session_state:
    st.session_state["epic_viz_json_path"] = _default_for_path or ""

data_src = st.sidebar.radio(
    "Data source",
    ["Upload JSON", "Enter JSON path"],
    horizontal=True,
    key="epic_viz_radio",
)
data: Optional[Dict[str, Any]] = None

if data_src == "Upload JSON":
    uploaded = st.sidebar.file_uploader("Choose a detailed result JSON", type=["json"])
    data = load_results_data(uploaded, None)
else:
    json_path = st.sidebar.text_input(
        "Detailed result JSON / JSONL path",
        placeholder="e.g. outputs/scores/<model_family>/<name>_full.json or _full.jsonl",
        key="epic_viz_json_path",
    )
    data = load_results_data(None, json_path)

samples = get_samples(data) if data else []
summary = get_summary(data) if data else {}

if not samples:
    st.info(
        "Upload a file or enter a path on the left. Supported formats: epic_eval detailed **JSON** or **JSONL** "
        "(`*_full.json` / `*_full.jsonl`, containing `details.by_samples`).\n\n"
        "If launched via `scripts/visualization.sh`, the app scans `outputs/scores` and pre-fills the first match.\n\n"
        "Make sure you saved detailed results (`details.by_samples`). If evaluation ran without details, there is "
        "nothing to visualize here."
    )
    st.stop()

# --------------- Coordinate mode ---------------
st.sidebar.divider()
st.sidebar.subheader("Coordinate mode")
selected_coord_mode = st.sidebar.selectbox(
    "Choose coordinate mode",
    options=COORD_CHOICES,
    format_func=lambda x: f"{COORD_LABELS.get(x, x)}",
    index=0,
    help="If from_sample, use each sample's response_details.coordinate_system; otherwise override.",
)

# --------------- Filters ---------------
st.sidebar.subheader("Filters")
all_categories = sorted({(s.get("task_category") or "") for s in samples if s.get("task_category")})
all_types = sorted({(s.get("task_type") or "") for s in samples if s.get("task_type")})
filter_cat = st.sidebar.selectbox("task_category", ["All"] + all_categories)

# task_type options depend on selected category
if filter_cat != "All":
    type_options = sorted({(s.get("task_type") or "") for s in samples
                           if (s.get("task_category") or "") == filter_cat and s.get("task_type")})
else:
    type_options = all_types
filter_type = st.sidebar.selectbox("task_type", ["All"] + type_options)

# format_type
all_format_types = sorted({
    (s.get("response_details") or {}).get("format_type") or ""
    for s in samples
    if (s.get("response_details") or {}).get("format_type")
})
filter_fmt = st.sidebar.selectbox("format_type", ["All"] + list(all_format_types))

# validity filter
invalid_filter = st.sidebar.selectbox("Validity", ["All", "Valid only", "Invalid only"])

# score range
score_min, score_max = st.sidebar.slider("Score range", 0.0, 1.0, (0.0, 1.0), 0.01)

# apply filters
filtered: List[Dict[str, Any]] = samples
if filter_cat != "All":
    filtered = [s for s in filtered if (s.get("task_category") or "") == filter_cat]
if filter_type != "All":
    filtered = [s for s in filtered if (s.get("task_type") or "") == filter_type]
if filter_fmt != "All":
    filtered = [s for s in filtered if ((s.get("response_details") or {}).get("format_type") or "") == filter_fmt]
if invalid_filter == "Valid only":
    filtered = [s for s in filtered if not s.get("invalid_response")]
elif invalid_filter == "Invalid only":
    filtered = [s for s in filtered if s.get("invalid_response")]
filtered = [s for s in filtered if score_min <= float(s.get("score") or 0.0) <= score_max]

# --------------- Search ---------------
def _searchable_text(sample: Dict[str, Any]) -> str:
    parts: List[str] = []
    p = get_gt_json_path(sample) or ""
    parts.append(p)
    parts.append(p.replace("_", " "))
    raw = load_raw_gt(sample) or {}
    text_ann = raw.get("text_annotation") or {}
    for key in ("text_label", "target_object_description_en", "target_area_description_en",
                "reference_object_description_en", "reference_area_description_en",
                "scene_description_en", "placement_region_description_en"):
        v = text_ann.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v if x is not None)
        elif v is not None:
            parts.append(str(v))
    return " ".join(parts).lower()


st.sidebar.subheader("Search (text/path)")
search_text = st.sidebar.text_input(
    "Keyword",
    placeholder="e.g. 'balloon' or a path fragment",
    key="search_text",
    help="Matches gt_json_path and text fields in raw GT JSON. Both spaces and underscores are searchable.",
)
if search_text and search_text.strip():
    needle = search_text.strip().lower()
    filtered = [s for s in filtered if needle in _searchable_text(s)]
st.sidebar.caption(f"After filtering: **{len(filtered)}** samples")

if not filtered:
    st.warning("No samples match the current filters/search.")
    st.stop()

# --------------- Sorting ---------------
sort_choice = st.sidebar.selectbox(
    "Sort",
    ["Index (asc)", "Score (asc)", "Score (desc)"],
    index=0,
)
if sort_choice == "Score (asc)":
    filtered = sorted(filtered, key=lambda s: float(s.get("score") or 0.0))
elif sort_choice == "Score (desc)":
    filtered = sorted(filtered, key=lambda s: float(s.get("score") or 0.0), reverse=True)
else:
    filtered = sorted(filtered, key=lambda s: int(s.get("index") or 0))

# --------------- Current sample ---------------
num = len(filtered)
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
st.session_state.current_index = max(0, min(st.session_state.current_index, num - 1))
current_index = st.session_state.current_index

sample_options = [
    f"#{s.get('index', i)} | {s.get('task_category', '')}/{s.get('task_type', '')} | "
    f"score={float(s.get('score') or 0.0):.3f}"
    + (" [INVALID]" if s.get("invalid_response") else "")
    for i, s in enumerate(filtered)
]
sel_dropdown = st.sidebar.selectbox(
    "Select sample",
    range(num),
    index=current_index,
    format_func=lambda i: sample_options[i],
)
if sel_dropdown != current_index:
    st.session_state.current_index = int(sel_dropdown)
    st.rerun()
current_index = st.session_state.current_index


def _go_to(idx: int) -> None:
    st.session_state.current_index = max(0, min(idx, num - 1))
    st.rerun()


st.sidebar.subheader("Navigate")
b1, b2, b3, b4 = st.sidebar.columns(4)
with b1:
    if st.button("First", use_container_width=True):
        _go_to(0)
with b2:
    if st.button("Prev", use_container_width=True):
        _go_to(current_index - 1)
with b3:
    if st.button("Next", use_container_width=True):
        _go_to(current_index + 1)
with b4:
    if st.button("Last", use_container_width=True):
        _go_to(num - 1)
st.sidebar.caption(f"**{current_index + 1} / {num}**")

# --------------- Sample view ---------------
sample = filtered[current_index]
task_category = str(sample.get("task_category") or "")
task_type = str(sample.get("task_type") or "")
viz_cfg = get_viz_config(task_category, task_type)
response_details = sample.get("response_details") or {}
score_details = sample.get("score_details") or {}
total_score = float(sample.get("score") or 0.0)
invalid_response = bool(sample.get("invalid_response"))
err = sample.get("error")
fmt_type = str(response_details.get("format_type") or "")
sample_coord_sys = str(response_details.get("coordinate_system") or "auto")
effective_coord = resolve_coord_mode(sample, selected_coord_mode)

raw_gt = load_raw_gt(sample)

st.subheader(
    f"Sample #{sample.get('index', '')} · {task_category} / {task_type} · format_type={fmt_type or '-'}"
)
if invalid_response or err:
    st.error(f"⚠️ This sample is marked as an invalid response. error={err!r}")

st.info(
    f"**Effective coordinate mode**: {COORD_LABELS.get(effective_coord, effective_coord)} (source="
    + ("sample field response_details.coordinate_system" if selected_coord_mode == "from_sample" else "sidebar override")
    + f"; raw sample field=`{sample_coord_sys}`)"
)

# --------------- Score cards ---------------
st.markdown("### Score")
s_loc = score_details.get("localization_score")
s_count = score_details.get("count_score")
s_path = score_details.get("path_score")
s_bin = score_details.get("binary_score")
w_loc = score_details.get("weight_localization")
w_count = score_details.get("weight_count")
w_path = score_details.get("weight_path")
w_bin = score_details.get("weight_binary")

cols = st.columns(5)


def _fmt(v: Any, p: int = 4) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{p}f}"
    except (TypeError, ValueError):
        return str(v)


cols[0].metric("Total", _fmt(total_score))
cols[1].metric("Localization", _fmt(s_loc))
cols[2].metric("Count", _fmt(s_count))
cols[3].metric("Path", _fmt(s_path))
cols[4].metric("Binary", _fmt(s_bin))

# Show weights under each score
with cols[0]:
    st.caption("Weight: —")
with cols[1]:
    st.caption(f"Weight: **{_fmt(w_loc, 2)}**")
with cols[2]:
    st.caption(f"Weight: **{_fmt(w_count, 2)}**")
with cols[3]:
    st.caption(f"Weight: **{_fmt(w_path, 2)}**")
with cols[4]:
    st.caption(f"Weight: **{_fmt(w_bin, 2)}**")

text_ann = (raw_gt or {}).get("text_annotation") or {}
mask_ann = (raw_gt or {}).get("mask_annotation") or {}

# --------------- Main info: key text + response_details ---------------
st.markdown("### Key information")

main_left, main_right = st.columns([2, 2])

with main_left:
    st.markdown("**GT key text fields**")
    if not raw_gt:
        st.warning("Failed to load raw GT JSON pointed to by gt_json_path.")
        st.code(get_gt_json_path(sample) or "(empty)")
    else:
        key_fields = get_key_text_fields(task_category, task_type)
        if not key_fields:
            st.caption("No extra key fields for this task.")
        else:
            for f in key_fields:
                v = _read_text_field_with_fallback(text_ann, f)
                if isinstance(v, list):
                    v_show = v[0] if v else None
                else:
                    v_show = v
                st.markdown(f"- **{f}**: {v_show if v_show is not None else '—'}")

with main_right:
    st.markdown("**Model formatted responses (response_details)**")
    st.caption(f"format_type=`{fmt_type or '-'}` · coordinate_system=`{sample_coord_sys}`")

    loc_value = response_details.get("formatted_response_localization")
    count_value = response_details.get("formatted_response_count")
    path_value = response_details.get("formatted_response_path")
    bin_value = response_details.get("formatted_response_binary")

    def _show_line(label: str, value: Any, extra: Optional[str] = None) -> None:
        cols = st.columns([1, 3])
        with cols[0]:
            st.markdown(f"**{label}**")
        with cols[1]:
            st.code(json.dumps(value, ensure_ascii=False) if value is not None else "Not Required")
            if extra:
                st.caption(extra)

    pc = parse_count(count_value)
    pp = parse_path_points(path_value) if isinstance(path_value, list) else []
    pb = parse_binary(bin_value)

    _show_line("localization", loc_value)
    _show_line("count", count_value, extra=(f"Parsed predicted count: **{pc}**" if pc is not None else None))
    _show_line(
        "path",
        path_value,
        extra=(f"Points: {len(pp)} (coord: {COORD_LABELS.get(effective_coord, effective_coord)})" if pp else None),
    )
    _show_line("binary", bin_value, extra=(f"Parsed: **{pb}**" if pb is not None else None))

# --------------- Images and overlays ---------------
st.markdown("### Images and overlays")
_gt_json_path = get_gt_json_path(sample)
image_paths = get_image_paths(raw_gt, _gt_json_path)

# Overlay controls
ctrl_cols = st.columns([1, 1, 1, 1, 2])
with ctrl_cols[0]:
    show_primary = st.checkbox(
        "GT primary mask",
        value=True,
        key="show_primary",
        help=viz_cfg.get("primary_mask_label") or viz_cfg.get("image2_mask_label") or "-",
    )
with ctrl_cols[1]:
    aux_help = " + ".join([
        x for x in [
            viz_cfg.get("reference_mask_label") or viz_cfg.get("image1_mask_label"),
            viz_cfg.get("secondary_mask_label"),
        ]
        if x
    ]) or "-"
    show_aux = st.checkbox(
        "reference masks",
        value=True,
        key="show_aux",
        help=aux_help,
    )
with ctrl_cols[2]:
    show_pred = st.checkbox("Model Response", value=True, key="show_pred")
with ctrl_cols[3]:
    mask_alpha = st.slider("Mask opacity", 0.0, 1.0, 0.45, 0.05, key="mask_alpha")
with ctrl_cols[4]:
    pass


def render_overlay_on(img: np.ndarray, *,
                      gt_primary: Optional[np.ndarray] = None,
                      gt_primary_color: Tuple[int, int, int] = COLOR_PRIMARY,
                      gt_secondary: Optional[np.ndarray] = None,
                      gt_secondary_color: Tuple[int, int, int] = COLOR_SECONDARY,
                      gt_reference: Optional[np.ndarray] = None,
                      gt_reference_color: Tuple[int, int, int] = COLOR_REFERENCE,
                      pred_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
                      pred_points: Optional[Sequence[Tuple[float, float]]] = None,
                      pred_mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = img
    if show_aux and gt_reference is not None:
        out = overlay_mask(out, gt_reference, gt_reference_color, mask_alpha)
    if show_aux and gt_secondary is not None:
        out = overlay_mask(out, gt_secondary, gt_secondary_color, mask_alpha)
    if show_primary and gt_primary is not None:
        out = overlay_mask(out, gt_primary, gt_primary_color, mask_alpha)
    if show_pred and pred_mask is not None:
        out = overlay_mask(out, pred_mask, COLOR_PRED_MASK, max(mask_alpha, 0.5))
    if show_pred and pred_bboxes:
        out = draw_bboxes(out, pred_bboxes, COLOR_PRED_BBOX, width=3)
    if show_pred and pred_points:
        out = draw_points(out, pred_points, COLOR_PRED_POINT, radius=6, connect=True, line_width=2)
    return out


def get_pred_overlays(img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[float, float]], Optional[np.ndarray]]:
    """Parse predictions from response_details into overlays: bbox / point / mask."""
    h, w = img.shape[0], img.shape[1]
    pred_bboxes: List[Tuple[int, int, int, int]] = []
    pred_points: List[Tuple[float, float]] = []
    pred_mask: Optional[np.ndarray] = None

    loc_value = response_details.get("formatted_response_localization")
    path_value = response_details.get("formatted_response_path")

    # Localization field
    if not is_not_required(loc_value) and not is_no_valid(loc_value):
        if fmt_type == "bbox":
            raw = parse_bboxes(loc_value)
            pred_bboxes = normalize_bboxes(raw, h, w, effective_coord)
        elif fmt_type == "mask":
            pred_mask = decode_response_mask_any(loc_value)
        elif fmt_type == "point":
            raw = parse_path_points(loc_value)
            pred_points = normalize_points(raw, h, w, effective_coord)

    # Path field (FeasiblePath)
    if not is_not_required(path_value) and not is_no_valid(path_value):
        raw = parse_path_points(path_value)
        if raw:
            pred_points.extend(normalize_points(raw, h, w, effective_coord))

    return pred_bboxes, pred_points, pred_mask


# --------------- Dual-image mode (VisualMatching) ---------------
if viz_cfg.get("is_dual_image"):
    img1_path = image_paths[0] if len(image_paths) >= 1 else None
    img2_path = image_paths[1] if len(image_paths) >= 2 else None
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    img1_mask = decode_rle_list(mask_ann.get(viz_cfg.get("image1_mask_field") or "")) if mask_ann else None
    img2_mask = decode_rle_list(mask_ann.get(viz_cfg.get("image2_mask_field") or "")) if mask_ann else None

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"**Image 1 (reference)**: {img1_path or '(no path)'}")
        if img1 is not None:
            out1 = render_overlay_on(
                img1,
                gt_reference=img1_mask,
                gt_reference_color=viz_cfg.get("image1_mask_color", COLOR_REFERENCE),
            )
            st.image(out1, use_container_width=True)
        else:
            st.warning(f"Failed to load image 1: {img1_path or '(no path)'}")
    with col_b:
        st.caption(f"**Image 2 (target)**: {img2_path or '(no path)'}")
        if img2 is not None:
            pred_bboxes, pred_points, pred_mask = get_pred_overlays(img2)
            out2 = render_overlay_on(
                img2,
                gt_primary=img2_mask,
                gt_primary_color=viz_cfg.get("image2_mask_color", COLOR_PRIMARY),
                pred_bboxes=pred_bboxes,
                pred_points=pred_points,
                pred_mask=pred_mask,
            )
            st.image(out2, use_container_width=True)
        else:
            st.warning(f"Failed to load image 2: {img2_path or '(no path)'}")
    legend_parts = ["blue=reference area", "red=GT target area", "green box=pred bbox"]
    st.caption("**Legend:** " + ", ".join(legend_parts))

# --------------- Single-image mode ---------------
else:
    main_img_path = image_paths[0] if image_paths else None
    img = load_image(main_img_path)
    if img is None:
        st.warning(f"Failed to load image: {main_img_path or '(no path)'}")
    else:
        primary_mask = decode_rle_list(mask_ann.get(viz_cfg.get("primary_mask_field") or "")) if mask_ann else None
        secondary_mask = decode_rle_list(mask_ann.get(viz_cfg.get("secondary_mask_field") or "")) if mask_ann else None
        reference_mask = decode_rle_list(mask_ann.get(viz_cfg.get("reference_mask_field") or "")) if mask_ann else None

        pred_bboxes, pred_points, pred_mask = get_pred_overlays(img)
        out = render_overlay_on(
            img,
            gt_primary=primary_mask,
            gt_primary_color=viz_cfg.get("primary_mask_color", COLOR_PRIMARY),
            gt_secondary=secondary_mask,
            gt_secondary_color=viz_cfg.get("secondary_mask_color", COLOR_SECONDARY),
            gt_reference=reference_mask,
            gt_reference_color=viz_cfg.get("reference_mask_color", COLOR_REFERENCE),
            pred_bboxes=pred_bboxes,
            pred_points=pred_points,
            pred_mask=pred_mask,
        )
        st.image(out, use_container_width=True)
        st.caption(f"**Image path**: {main_img_path}")
        legend = []
        if reference_mask is not None:
            legend.append(f"blue={viz_cfg.get('reference_mask_label', 'reference')}")
        if secondary_mask is not None:
            legend.append(f"green={viz_cfg.get('secondary_mask_label', 'secondary GT')}")
        if primary_mask is not None:
            legend.append(f"red={viz_cfg.get('primary_mask_label', 'primary GT')}")
        if viz_cfg.get("response_kind") == "point":
            legend.append("cyan=pred points (connected)")
        else:
            legend.append("green box=pred bbox")
        if pred_mask is not None:
            legend.append("gold=pred mask")
        if legend:
            st.caption("**Legend:** " + ", ".join(legend))

with st.expander("Raw JSON (debug)", expanded=False):
    raw_cols = st.columns(3)
    with raw_cols[0]:
        st.markdown("**score_details**")
        st.json(score_details)
    with raw_cols[1]:
        st.markdown("**response_details**")
        st.json(response_details)
    with raw_cols[2]:
        st.markdown("**ground_truth**")
        st.json(sample.get("ground_truth"))

    if raw_gt is not None:
        st.divider()
        st.markdown("**gt_json_path**")
        st.code(get_gt_json_path(sample) or "")
        st.markdown("**text_annotation**")
        st.json({k: v for k, v in (raw_gt.get("text_annotation") or {}).items()})
        st.markdown("**mask_annotation keys**")
        st.json(list((raw_gt.get("mask_annotation") or {}).keys()))

# --------------- Summary ---------------
st.sidebar.divider()
if summary:
    st.sidebar.caption("Summary")
    overall = summary.get("overall") or {}
    short_summary = {
        "total_processed": overall.get("total_processed") or summary.get("total_samples"),
        "num_invalid_response": overall.get("num_invalid_response"),
        "average_score": overall.get("average_score") or overall.get("mean_score"),
    }
    st.sidebar.json({k: v for k, v in short_summary.items() if v is not None})
    by_cat = summary.get("by_category")
    if by_cat:
        with st.sidebar.expander("by_category"):
            st.json(by_cat)
    by_type = summary.get("by_type")
    if by_type:
        with st.sidebar.expander("by_type"):
            st.json(by_type)
