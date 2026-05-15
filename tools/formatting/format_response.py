#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert raw ms-swift model responses (JSON/JSONL) into the EPIC-Bench standard format
for scoring.

Standard format (per record):
{
  "task_category": ["TargetLocalization" | "Navigation" | "Manipulation"],
  "task_type":     [...],
  "format_type":       ["mask" | "point" | "bbox" | "img"],
  "coordinate_system": ["auto" | "absolute" | "normalized_0_1" | "normalized_0_1000"],
  "formatted_response_localization": [...],
  "formatted_response_count":        [...],
  "formatted_response_path":         [...],
  "formatted_response_binary":       [...],
  "images":       ["..."],
  "ground_truth": {"gt_json_path": ["..."]}
}

Notes:
- task_category / task_type are inferred from image paths in the record.
- Output can be a JSON array or JSONL (use --jsonl).

Examples:
  python tools/formatting/format_response.py input.jsonl -o std_format.jsonl --jsonl
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Task category / task type constants
# ---------------------------------------------------------------------------
TASK_CATEGORIES = ("TargetLocalization", "Navigation", "Manipulation")

TL_TASK_TYPES = {
    "Category", "Color", "Geometry", "Material", "ObjectProjections",
    "RelativePerception", "ObjectState", "PartWhole", "PartWholeHuman",
    "Spoken", "Orientation", "OrientationHuman", "SpatialRelation", "SpatialRelationHuman",
}
NVG_TASK_TYPES = {"GroundDetection", "FeasiblePath", "VisualMatching"}
MANIP_TASK_TYPES = {"AffordanceRegion", "ContactRelationship", "PlacementRegion"}

# Sub-types for FeasiblePath / ContactRelationship (final task_type)
FEASIBLEPATH_SUBTYPES = {"FeasiblePath_Ego", "FeasiblePath_Exo"}
CONTACTRELATIONSHIP_SUBTYPES = {
    "ContactRelationship_TypeOne",
    "ContactRelationship_TypeTwo",
    "ContactRelationship_TypeThree",
}

# Legacy name -> canonical name (used only when parsing paths)
_MANIP_LEGACY_NORMALIZE = {
    "Affordance": "AffordanceRegion",
    "Placement": "PlacementRegion",
}

NO_VALID_RESPONSE = "No Valid Response"

# Supported format_type / coordinate_system values (written to output)
SUPPORTED_FORMAT_TYPES = ("mask", "point", "bbox", "img")
SUPPORTED_COORDINATE_SYSTEMS = ("auto", "absolute", "normalized_0_1", "normalized_0_1000")

# CLI-only sentinel: means "auto-select defaults per task"
_FORMAT_TYPE_AUTO = "auto"
_COORDINATE_SYSTEM_AUTO_SENTINEL = "__cli_auto__"  # distinct from field value "auto"


def default_format_type_for_task(
    task_category: Optional[str], task_type: Optional[str]
) -> str:
    """Return default format_type for a task.
    - Navigation / FeasiblePath (FeasiblePath_Ego / FeasiblePath_Exo) -> point
    - Otherwise -> bbox
    """
    if task_category == "Navigation" and task_type in (
        "FeasiblePath",
        "FeasiblePath_Ego",
        "FeasiblePath_Exo",
    ):
        return "point"
    return "bbox"


# ---------------------------------------------------------------------------
# task_category / task_type parsing
# ---------------------------------------------------------------------------
def parse_task_from_path(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse (task_category, task_type) from a single image path.
    We locate the first occurrence of a task_category, then infer task_type from
    its following path components.
    """
    parts = Path(image_path).parts
    cat_idx = None
    task_category = None
    for i, part in enumerate(parts):
        if part in TASK_CATEGORIES:
            cat_idx = i
            task_category = part
            break
    if cat_idx is None or task_category is None:
        return None, None

    if task_category == "TargetLocalization":
        # .../TargetLocalization/<group>/<task_type>/...
        if cat_idx + 2 < len(parts):
            cand = parts[cat_idx + 2]
            if cand in TL_TASK_TYPES:
                return task_category, cand
        return task_category, None

    if task_category == "Navigation":
        if cat_idx + 1 >= len(parts):
            return task_category, None
        sub = parts[cat_idx + 1]
        if sub in ("GroundDetection", "VisualMatching"):
            return task_category, sub
        if sub == "FeasiblePath":
            if cat_idx + 2 < len(parts):
                fp_sub = parts[cat_idx + 2]
                if fp_sub in FEASIBLEPATH_SUBTYPES:
                    return task_category, fp_sub
            return task_category, "FeasiblePath"
        return task_category, sub if sub in NVG_TASK_TYPES else None

    # Manipulation
    if cat_idx + 1 >= len(parts):
        return task_category, None
    sub = parts[cat_idx + 1]
    sub = _MANIP_LEGACY_NORMALIZE.get(sub, sub)
    if sub in ("AffordanceRegion", "PlacementRegion"):
        return task_category, sub
    if sub == "ContactRelationship":
        if cat_idx + 2 < len(parts):
            cr_sub = parts[cat_idx + 2]
            if cr_sub in CONTACTRELATIONSHIP_SUBTYPES:
                return task_category, cr_sub
        return task_category, "ContactRelationship"
    return task_category, sub if sub in MANIP_TASK_TYPES else None


def get_task_from_record(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Parse (task_category, task_type) from a record."""
    images = record.get("images") or []
    for img in images:
        path = img.get("path") if isinstance(img, dict) else (img if isinstance(img, str) else None)
        if not path:
            continue
        task_category, task_type = parse_task_from_path(path)
        if task_category:
            return task_category, task_type
    return None, None


def base_task_type(task_type: Optional[str]) -> Optional[str]:
    """Collapse sub-typed task_type to its base type for extractor lookup.
    Example: FeasiblePath_Ego -> FeasiblePath; ContactRelationship_TypeOne -> ContactRelationship.
    """
    if task_type is None:
        return None
    if task_type in FEASIBLEPATH_SUBTYPES:
        return "FeasiblePath"
    if task_type in CONTACTRELATIONSHIP_SUBTYPES:
        return "ContactRelationship"
    return task_type


# ---------------------------------------------------------------------------
# Generic JSON extraction/normalization utilities
# ---------------------------------------------------------------------------
def _strip_think_blocks(text: str) -> Tuple[str, bool]:
    """Remove <think>...</think> blocks.

    Returns (remaining_text, think_only_flag) where think_only_flag is True when
    the response contains only <think> without a valid payload.
    """
    if "<think>" not in text:
        return text, False
    if "</think>" not in text:
        return text, True
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return stripped.strip(), False


def _to_number(val: Any) -> Any:
    """Convert to a numeric type while preserving precision as much as possible."""
    if val is None:
        return None
    try:
        f = float(val)
        if not math.isfinite(f):
            return None
        if f == int(f) and abs(f) < 1e15:
            return int(f)
        return f
    except (TypeError, ValueError, OverflowError):
        return None


def _one_bbox_to_quad(item: Any) -> Optional[List]:
    """Convert one bbox to [x1, y1, x2, y2]. Supports common nested shapes."""
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and len(item) == 4:
        vals = [_to_number(item[i]) for i in range(4)]
        if None not in vals:
            return vals
    if isinstance(item, (list, tuple)) and len(item) == 2:
        a, b = item[0], item[1]
        if isinstance(a, (list, tuple)) and len(a) >= 2 and isinstance(b, (list, tuple)) and len(b) >= 2:
            vals = [_to_number(a[0]), _to_number(a[1]), _to_number(b[0]), _to_number(b[1])]
            if None not in vals:
                return vals
    if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (list, tuple)) and len(item[0]) >= 4:
        t = item[0]
        vals = [_to_number(t[i]) for i in range(4)]
        if None not in vals:
            return vals
    return None


def _normalize_bbox_value(val: Any) -> List[List]:
    """Normalize bboxes into [[x1,y1,x2,y2], ...]."""
    result: List[List] = []
    if not isinstance(val, list):
        return result
    for item in val:
        quad = _one_bbox_to_quad(item)
        if quad is not None:
            result.append(quad)
    return result


def _one_point_to_pair(item: Any) -> Optional[List]:
    """Convert one point to [x, y]."""
    if item is None:
        return None
    if isinstance(item, (list, tuple)) and len(item) == 2:
        vals = [_to_number(item[0]), _to_number(item[1])]
        if None not in vals:
            return vals
    if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (list, tuple)) and len(item[0]) >= 2:
        t = item[0]
        vals = [_to_number(t[0]), _to_number(t[1])]
        if None not in vals:
            return vals
    return None


def _normalize_points_value(val: Any) -> List[List]:
    """Normalize points into [[x,y], ...]."""
    result: List[List] = []
    if not isinstance(val, list):
        return result
    for item in val:
        pair = _one_point_to_pair(item)
        if pair is not None:
            result.append(pair)
    return result


def _normalize_python_to_json(text: str) -> str:
    """Convert Python-ish literals into JSON-compatible text (True/False/None, quotes)."""
    text = text.replace("\\'", "'")

    result = []
    i = 0
    in_string = False
    string_quote = None
    escape_next = False

    while i < len(text):
        char = text[i]

        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
            continue

        if char in ('"', "'"):
            if not in_string:
                in_string = True
                string_quote = char
                result.append('"')
                i += 1
                continue
            elif char == string_quote:
                in_string = False
                string_quote = None
                result.append('"')
                i += 1
                continue
            else:
                result.append(char)
                i += 1
                continue

        if in_string:
            if char == '"' and string_quote == "'":
                result.append('\\"')
            else:
                result.append(char)
            i += 1
            continue

        if text[i:i + 4] == 'True' and (i + 4 >= len(text) or not text[i + 4].isalnum()):
            result.append('true')
            i += 4
            continue
        if text[i:i + 5] == 'False' and (i + 5 >= len(text) or not text[i + 5].isalnum()):
            result.append('false')
            i += 5
            continue
        if text[i:i + 4] == 'None' and (i + 4 >= len(text) or not text[i + 4].isalnum()):
            result.append('null')
            i += 4
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def _replace_tuples_in_field(text: str, field_name: str) -> str:
    """Replace (a,b,...) tuples with [a,b,...] in specific JSON fields (bboxes/points)."""
    pattern = r'("' + re.escape(field_name) + r'"\s*:\s*\[)(.*?)(\]\s*[,}])'
    return re.sub(
        pattern,
        lambda m: m.group(1) + re.sub(r"\(([^)]*)\)", r"[\1]", m.group(2)) + m.group(3),
        text,
        flags=re.DOTALL,
    )


# ---------------------------------------------------------------------------
# Find JSON snippets in text (prefer ```json``` / ```python``` blocks)
# ---------------------------------------------------------------------------
def _find_json_with_keys(
    text: str,
    must_have_keys: Tuple[str, ...] = (),
    any_of_keys: Tuple[str, ...] = (),
) -> Optional[str]:
    """
    Pick the best-matching JSON candidate:
    - Prefer ```json```/```python``` code blocks containing all must_have_keys.
      If none, look for blocks containing any_of_keys.
    - Otherwise, fall back to a balanced-braces scan over the full text.
    """
    code_block_iter = list(re.finditer(r"```(?:json|python)?\s*\n?(.*?)\n?```", text, re.DOTALL))

    if must_have_keys:
        for block in code_block_iter:
            cand = block.group(1).strip()
            if all(k in cand for k in must_have_keys):
                return cand

    if any_of_keys:
        for block in code_block_iter:
            cand = block.group(1).strip()
            if any(k in cand for k in any_of_keys):
                return cand

    search_keys = list(must_have_keys) + list(any_of_keys)
    if not search_keys:
        return None
    start = -1
    for k in search_keys:
        s = text.find(k)
        if s != -1:
            start = s
            break
    if start == -1:
        return None
    brace_start = text.rfind("{", 0, start)
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                cand = text[brace_start:i + 1]
                if must_have_keys and not all(k in cand for k in must_have_keys):
                    if any_of_keys and any(k in cand for k in any_of_keys):
                        return cand
                    return None
                if any_of_keys and not any(k in cand for k in any_of_keys) and not must_have_keys:
                    return None
                return cand
    return None


# ---------------------------------------------------------------------------
# Per-task extractors: parse and normalize into a canonical JSON string
# ---------------------------------------------------------------------------
def _parse_and_normalize_tl_json(candidate: str) -> Optional[str]:
    """TargetLocalization: number_of_objects + bounding_boxes (bounding_boxes defaults to [])."""
    normalized = _normalize_python_to_json(candidate)
    normalized = _replace_tuples_in_field(normalized, "bounding_boxes")
    try:
        obj = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    num = obj.get("number_of_objects")
    if num is None:
        return None
    try:
        n = int(num)
    except (TypeError, ValueError):
        return None
    bboxes = obj.get("bounding_boxes")
    bbox_list = _normalize_bbox_value(bboxes) if bboxes is not None else []
    return json.dumps({"number_of_objects": n, "bounding_boxes": bbox_list}, ensure_ascii=False)


def extract_formatted_response_target_localization(raw_response: str) -> str:
    if not raw_response or not isinstance(raw_response, str):
        return NO_VALID_RESPONSE
    text = raw_response.strip()
    if not text:
        return NO_VALID_RESPONSE
    content, think_only = _strip_think_blocks(text)
    if think_only:
        return NO_VALID_RESPONSE
    candidate = _find_json_with_keys(
        content,
        must_have_keys=("number_of_objects", "bounding_boxes"),
        any_of_keys=("number_of_objects",),
    )
    if not candidate:
        return NO_VALID_RESPONSE
    result = _parse_and_normalize_tl_json(candidate)
    return result if result is not None else NO_VALID_RESPONSE


def _parse_and_normalize_gd_json(candidate: str) -> Optional[str]:
    """GroundDetection / AffordanceRegion: bounding_boxes only."""
    normalized = _normalize_python_to_json(candidate)
    normalized = _replace_tuples_in_field(normalized, "bounding_boxes")
    try:
        obj = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    bboxes = obj.get("bounding_boxes")
    if bboxes is None:
        return None
    bbox_list = _normalize_bbox_value(bboxes)
    return json.dumps({"bounding_boxes": bbox_list}, ensure_ascii=False)


def extract_formatted_response_ground_detection(raw_response: str) -> str:
    if not raw_response or not isinstance(raw_response, str):
        return NO_VALID_RESPONSE
    text = raw_response.strip()
    if not text:
        return NO_VALID_RESPONSE
    content, think_only = _strip_think_blocks(text)
    if think_only:
        return NO_VALID_RESPONSE
    candidate = _find_json_with_keys(content, any_of_keys=("bounding_boxes",))
    if not candidate:
        return NO_VALID_RESPONSE
    result = _parse_and_normalize_gd_json(candidate)
    return result if result is not None else NO_VALID_RESPONSE


def _parse_and_normalize_pr_json(candidate: str) -> Optional[str]:
    """PlacementRegion：bounding_boxes + placement_feasibility。"""
    normalized = _normalize_python_to_json(candidate)
    normalized = _replace_tuples_in_field(normalized, "bounding_boxes")
    try:
        obj = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    bboxes = obj.get("bounding_boxes")
    feas = obj.get("placement_feasibility")
    if bboxes is None or feas is None:
        return None
    bbox_list = _normalize_bbox_value(bboxes)
    if isinstance(feas, bool):
        placement_feasibility = feas
    else:
        s = str(feas).strip().lower()
        placement_feasibility = s in ("true", "1", "yes")
    return json.dumps(
        {"bounding_boxes": bbox_list, "placement_feasibility": placement_feasibility},
        ensure_ascii=False,
    )


def extract_formatted_response_placement_region(raw_response: str) -> str:
    if not raw_response or not isinstance(raw_response, str):
        return NO_VALID_RESPONSE
    text = raw_response.strip()
    if not text:
        return NO_VALID_RESPONSE
    content, think_only = _strip_think_blocks(text)
    if think_only:
        return NO_VALID_RESPONSE
    candidate = _find_json_with_keys(
        content,
        must_have_keys=("bounding_boxes", "placement_feasibility"),
        any_of_keys=("placement_feasibility",),
    )
    if not candidate:
        return NO_VALID_RESPONSE
    result = _parse_and_normalize_pr_json(candidate)
    return result if result is not None else NO_VALID_RESPONSE


def _parse_and_normalize_fp_json(candidate: str) -> Optional[str]:
    """FeasiblePath: points list, at least 3 points."""
    normalized = _normalize_python_to_json(candidate)
    normalized = _replace_tuples_in_field(normalized, "points")
    try:
        obj = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    points = obj.get("points")
    if points is None:
        return None
    point_list = _normalize_points_value(points)
    if len(point_list) < 3:
        return None
    return json.dumps({"points": point_list}, ensure_ascii=False)


def extract_formatted_response_feasible_path(raw_response: str) -> str:
    if not raw_response or not isinstance(raw_response, str):
        return NO_VALID_RESPONSE
    text = raw_response.strip()
    if not text:
        return NO_VALID_RESPONSE
    content, think_only = _strip_think_blocks(text)
    if think_only:
        return NO_VALID_RESPONSE
    candidate = _find_json_with_keys(content, any_of_keys=("points",))
    if not candidate:
        return NO_VALID_RESPONSE
    result = _parse_and_normalize_fp_json(candidate)
    return result if result is not None else NO_VALID_RESPONSE


def _parse_and_normalize_vm_json(candidate: str) -> Optional[str]:
    """VisualMatching / ContactRelationship：number_of_instances + bounding_boxes。"""
    normalized = _normalize_python_to_json(candidate)
    normalized = _replace_tuples_in_field(normalized, "bounding_boxes")
    try:
        obj = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    num = obj.get("number_of_instances")
    bboxes = obj.get("bounding_boxes")
    if num is None and bboxes is None:
        return None
    if num is not None:
        try:
            n = int(num)
        except (TypeError, ValueError):
            return None
    else:
        n = None
    bbox_list = _normalize_bbox_value(bboxes) if bboxes is not None else []
    out: Dict[str, Any] = {}
    if n is not None:
        out["number_of_instances"] = n
    out["bounding_boxes"] = bbox_list
    return json.dumps(out, ensure_ascii=False)


def extract_formatted_response_visual_matching(raw_response: str) -> str:
    if not raw_response or not isinstance(raw_response, str):
        return NO_VALID_RESPONSE
    text = raw_response.strip()
    if not text:
        return NO_VALID_RESPONSE
    content, think_only = _strip_think_blocks(text)
    if think_only:
        return NO_VALID_RESPONSE
    candidate = _find_json_with_keys(
        content,
        must_have_keys=("number_of_instances", "bounding_boxes"),
        any_of_keys=("number_of_instances", "bounding_boxes"),
    )
    if not candidate:
        return NO_VALID_RESPONSE
    result = _parse_and_normalize_vm_json(candidate)
    return result if result is not None else NO_VALID_RESPONSE


def extract_formatted_response_default(raw_response: str) -> str:
    """Default extractor: unwrap a ```json ... ``` fenced block if present."""
    if not raw_response or not isinstance(raw_response, str):
        return raw_response or ""
    text = raw_response.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Extractor routing based on (task_category, base_task_type)
# ---------------------------------------------------------------------------
RESPONSE_EXTRACTORS: Dict[Tuple[Optional[str], Optional[str]], Callable[[str], str]] = {
    ("TargetLocalization", None): extract_formatted_response_target_localization,
    ("Navigation", "GroundDetection"): extract_formatted_response_ground_detection,
    ("Navigation", "FeasiblePath"): extract_formatted_response_feasible_path,
    ("Navigation", "VisualMatching"): extract_formatted_response_visual_matching,
    ("Manipulation", "AffordanceRegion"): extract_formatted_response_ground_detection,
    ("Manipulation", "ContactRelationship"): extract_formatted_response_visual_matching,
    ("Manipulation", "PlacementRegion"): extract_formatted_response_placement_region,
}


def get_response_extractor(
    task_category: Optional[str], task_type: Optional[str]
) -> Callable[[str], str]:
    base = base_task_type(task_type)
    extractor = RESPONSE_EXTRACTORS.get((task_category, base))
    if extractor is not None:
        return extractor
    extractor = RESPONSE_EXTRACTORS.get((task_category, None))
    if extractor is not None:
        return extractor
    return extract_formatted_response_default


def extract_formatted_response_by_task(
    raw_response: str, task_category: Optional[str], task_type: Optional[str]
) -> str:
    return get_response_extractor(task_category, task_type)(raw_response or "")


# ---------------------------------------------------------------------------
# Task requirement matrix and mapping into 4 formatted_response_* fields
# ---------------------------------------------------------------------------
NOT_REQUIRED = "Not Required"

# Internal keys and output field names
_FIELD_KEYS = ("localization", "count", "path", "binary")
_FIELD_NAMES = {
    "localization": "formatted_response_localization",
    "count": "formatted_response_count",
    "path": "formatted_response_path",
    "binary": "formatted_response_binary",
}


def get_required_response_fields(
    task_category: Optional[str], task_type: Optional[str]
) -> Dict[str, bool]:
    """Return which formatted_response_* fields are required by this (task_category, task_type)."""
    required = {k: False for k in _FIELD_KEYS}

    if not task_category:
        return required

    if task_category == "TargetLocalization":
        required["localization"] = True
        required["count"] = True
    elif task_category == "Navigation":
        if task_type == "GroundDetection":
            required["localization"] = True
        elif task_type in ("FeasiblePath", "FeasiblePath_Ego", "FeasiblePath_Exo"):
            required["path"] = True
        elif task_type == "VisualMatching":
            required["localization"] = True
            required["count"] = True
    elif task_category == "Manipulation":
        if task_type in ("Affordance", "AffordanceRegion"):
            required["localization"] = True
        elif task_type in (
            "ContactRelationship",
            "ContactRelationship_TypeOne",
            "ContactRelationship_TypeTwo",
            "ContactRelationship_TypeThree",
        ):
            required["localization"] = True
            required["count"] = True
        elif task_type in ("Placement", "PlacementRegion"):
            required["localization"] = True
            required["binary"] = True

    return required


def build_response_fields(
    formatted_response: str, required: Dict[str, bool]
) -> Dict[str, List[Any]]:
    """
    Map the extracted formatted_response string into 4 output fields based on the task requirement matrix.

    - Field not required -> ["Not Required"]
    - Field required but response invalid -> ["No Valid Response"]
    - Field required and parsed -> parsed value (may be an empty list for a valid "no target" answer)

    Rules:
    - If extractor already returned NO_VALID_RESPONSE or formatted_response is not a valid JSON dict:
      all required fields become ["No Valid Response"].
    - Otherwise extract from dict:
      - localization <- bounding_boxes
      - count <- number_of_instances / number_of_objects (prefer number_of_instances)
      - path <- points
      - binary <- placement_feasibility
      Missing/null/unparseable values become ["No Valid Response"].
    """
    out: Dict[str, List[Any]] = {fname: [] for fname in _FIELD_NAMES.values()}

    is_invalid = (not formatted_response) or (formatted_response == NO_VALID_RESPONSE)
    obj: Optional[Dict[str, Any]] = None
    if not is_invalid:
        try:
            parsed = json.loads(formatted_response)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            obj = parsed
        else:
            is_invalid = True

    for key in _FIELD_KEYS:
        fname = _FIELD_NAMES[key]
        if not required[key]:
            out[fname] = [NOT_REQUIRED]
            continue
        if is_invalid or obj is None:
            out[fname] = [NO_VALID_RESPONSE]
            continue

        if key == "localization":
            if "bounding_boxes" in obj and obj["bounding_boxes"] is not None:
                out[fname] = _normalize_bbox_value(obj["bounding_boxes"])
            else:
                out[fname] = [NO_VALID_RESPONSE]
        elif key == "count":
            cnt_val = None
            for k in ("number_of_instances", "number_of_objects"):
                if k in obj and obj[k] is not None:
                    cnt_val = obj[k]
                    break
            if cnt_val is None:
                out[fname] = [NO_VALID_RESPONSE]
            else:
                try:
                    out[fname] = [int(cnt_val)]
                except (TypeError, ValueError):
                    out[fname] = [NO_VALID_RESPONSE]
        elif key == "path":
            if "points" in obj and obj["points"] is not None:
                out[fname] = _normalize_points_value(obj["points"])
            else:
                out[fname] = [NO_VALID_RESPONSE]
        elif key == "binary":
            if "placement_feasibility" in obj and obj["placement_feasibility"] is not None:
                feas = obj["placement_feasibility"]
                if isinstance(feas, bool):
                    out[fname] = [feas]
                else:
                    s = str(feas).strip().lower()
                    out[fname] = [s in ("true", "1", "yes")]
            else:
                out[fname] = [NO_VALID_RESPONSE]

    return out


# ---------------------------------------------------------------------------
# images / ground_truth construction
# ---------------------------------------------------------------------------
def build_images_list(images: List[Any]) -> List[str]:
    """Extract image path strings from the raw images list."""
    paths = []
    for img in images or []:
        if isinstance(img, dict) and img.get("path"):
            paths.append(img["path"])
        elif isinstance(img, str):
            paths.append(img)
    return paths


def get_gt_json_path(images: List[str], task_type: Optional[str]) -> Optional[str]:
    """
    Infer the GT JSON path from image paths and task_type.

    - Generic: pick the file whose stem ends with _image; replace _image -> _mask_rle and use .json.
    - VisualMatching: take two *_image entries in order and build:
        {base_first}__{base_second}_mask_rle.json
    """
    if not images:
        return None

    image_suffix = "_image"

    if task_type == "VisualMatching":
        image_candidates: List[Tuple[str, Path, str]] = []
        for p in images:
            path = Path(p)
            stem = path.stem
            if stem.endswith(image_suffix):
                image_candidates.append((p, path.parent, stem))
        if len(image_candidates) < 2:
            return None
        first_image = image_candidates[0]
        third_image = image_candidates[1] if len(image_candidates) == 2 else image_candidates[-1]
        base_first = first_image[2][: -len(image_suffix)]
        base_third = third_image[2][: -len(image_suffix)]
        dir_path = first_image[1]
        return str(dir_path / f"{base_first}__{base_third}_mask_rle.json")

    candidates: List[Tuple[str, Path, str]] = []
    for p in images:
        path = Path(p)
        name_no_ext = path.stem
        if name_no_ext.endswith(image_suffix):
            candidates.append((p, path.parent, name_no_ext))
    if len(candidates) != 1:
        return None
    _, dir_path, name_no_ext = candidates[0]
    gt_name = name_no_ext[: -len(image_suffix)] + "_mask_rle.json"
    return str(dir_path / gt_name)


# ---------------------------------------------------------------------------
# Single-record conversion
# ---------------------------------------------------------------------------
def record_to_standard_item(
    record: Dict[str, Any],
    format_type: Optional[str] = None,
    coordinate_system: str = "auto",
) -> Dict[str, Any]:
    """Convert one raw JSONL record into one std_format output item.

    If format_type is None, choose per-task defaults (FeasiblePath_* -> "point", others -> "bbox").
    If provided, it overrides all tasks.
    """
    task_category, task_type = get_task_from_record(record)

    raw_response = record.get("response") or ""
    formatted_response = extract_formatted_response_by_task(
        raw_response, task_category, task_type
    )
    required = get_required_response_fields(task_category, task_type)
    decomposed = build_response_fields(formatted_response, required)

    images = build_images_list(record.get("images"))
    gt_path = get_gt_json_path(images, task_type)

    resolved_format_type = (
        format_type if format_type is not None
        else default_format_type_for_task(task_category, task_type)
    )

    item = {
        "task_category": [task_category] if task_category else [],
        "task_type": [task_type] if task_type else [],
        "format_type": [resolved_format_type],
        "coordinate_system": [coordinate_system],
        "formatted_response_localization": decomposed["formatted_response_localization"],
        "formatted_response_count": decomposed["formatted_response_count"],
        "formatted_response_path": decomposed["formatted_response_path"],
        "formatted_response_binary": decomposed["formatted_response_binary"],
        "images": images,
        "ground_truth": {
            "gt_json_path": [gt_path] if gt_path else [],
        },
    }
    return item


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warn: skipping invalid JSONL line: {e}", file=sys.stderr)
    return records


def convert_one_file(
    in_path: Path,
    out_path: Path,
    *,
    as_jsonl: bool,
    indent: int,
    format_type: Optional[str],
    coordinate_system: str,
) -> int:
    """Convert one JSONL file and write to out_path. Returns number of output records.

    If format_type is None, choose per-task defaults (FeasiblePath_* -> point, others -> bbox).
    """
    records = load_jsonl(str(in_path))
    result = [
        record_to_standard_item(r, format_type, coordinate_system)
        for r in records
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if as_jsonl:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(item, ensure_ascii=False) for item in result))
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=indent)
    return len(result)


def _resolve_output_name(in_file: Path, out_dir: Path, in_root: Path, name_prefix: str) -> Path:
    """Resolve batch output path by mirroring relative structure under in_root and adding name_prefix."""
    rel = in_file.relative_to(in_root)
    new_name = f"{name_prefix}{rel.name}" if name_prefix else rel.name
    return out_dir / rel.parent / new_name


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw model response JSONL into EPIC-Bench std_format (JSON array or JSONL).\n"
            "Supports single-file conversion and batch directory conversion."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single file -> JSONL\n"
            "  python format_response.py input.jsonl -o output.jsonl --jsonl\n\n"
            "  # Batch directory (recursive): mirror directory structure and add std_format_ prefix\n"
            "  python format_response.py \\\n"
            "      model_response/swift_format/version_1 \\\n"
            "      -o model_response/std_format/version_1 --jsonl\n"
        ),
    )
    parser.add_argument(
        "input_jsonl",
        help="Input JSONL file path, or a directory containing JSONL files (batch mode).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (single-file mode) or output directory (batch mode required).",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (JSON array output only).")
    parser.add_argument("--jsonl", action="store_true", help="Write output in JSONL (one JSON per line).")
    parser.add_argument(
        "--format-type",
        default=_FORMAT_TYPE_AUTO,
        choices=(_FORMAT_TYPE_AUTO,) + SUPPORTED_FORMAT_TYPES,
        help=(
            "format_type field. Default 'auto' chooses per-task defaults "
            "(FeasiblePath_Ego/FeasiblePath_Exo -> 'point', others -> 'bbox'). "
            "If explicitly set to mask/point/bbox/img, it overrides all tasks."
        ),
    )
    parser.add_argument(
        "--coordinate-system",
        default="auto",
        choices=SUPPORTED_COORDINATE_SYSTEMS,
        help="coordinate_system field (default: auto)",
    )
    parser.add_argument(
        "--name-prefix",
        default="std_format_",
        help="Batch output filename prefix (default: 'std_format_'; empty string disables).",
    )
    parser.add_argument(
        "--glob",
        default="**/*.jsonl",
        help="Input glob pattern in batch mode (default: '**/*.jsonl')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in batch mode (default: skip).",
    )
    args = parser.parse_args()

    # Convert CLI 'auto' sentinel -> None (record_to_standard_item decides per task)
    cli_format_type: Optional[str] = (
        None if args.format_type == _FORMAT_TYPE_AUTO else args.format_type
    )

    in_path = Path(args.input_jsonl)

    # Batch mode: input is a directory
    if in_path.is_dir():
        if not args.output:
            parser.error("Batch mode (directory input) requires -o to specify an output directory.")
        out_dir = Path(args.output)
        in_files = sorted(in_path.glob(args.glob))
        if not in_files:
            print(f"No files matched {args.glob} under {in_path}", file=sys.stderr)
            return

        total_files = 0
        total_records = 0
        skipped = 0
        failed: List[Tuple[str, str]] = []
        for in_file in in_files:
            out_file = _resolve_output_name(in_file, out_dir, in_path, args.name_prefix)
            if out_file.exists() and not args.overwrite:
                skipped += 1
                print(f"[skip] exists: {out_file}", file=sys.stderr)
                continue
            try:
                n = convert_one_file(
                    in_file, out_file,
                    as_jsonl=args.jsonl,
                    indent=args.indent,
                    format_type=cli_format_type,
                    coordinate_system=args.coordinate_system,
                )
            except Exception as e:
                failed.append((str(in_file), repr(e)))
                print(f"[fail] {in_file}: {e}", file=sys.stderr)
                continue
            total_files += 1
            total_records += n
            print(f"[ok] {in_file} -> {out_file} ({n} records)", file=sys.stderr)

        print(
            f"\nBatch done: ok {total_files} files / total {total_records} records; "
            f"skipped {skipped}; failed {len(failed)}",
            file=sys.stderr,
        )
        if failed:
            sys.exit(1)
        return

    # Single-file mode
    records = load_jsonl(str(in_path))
    result = [
        record_to_standard_item(r, cli_format_type, args.coordinate_system)
        for r in records
    ]

    if args.jsonl:
        lines = [json.dumps(item, ensure_ascii=False) for item in result]
        out_content = "\n".join(lines)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_content)
            print(f"Wrote {len(result)} records to {args.output} (JSONL)", file=sys.stderr)
        else:
            print(out_content)
    else:
        out_json = json.dumps(result, ensure_ascii=False, indent=args.indent)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_json)
            print(f"Wrote {len(result)} records to {args.output}", file=sys.stderr)
        else:
            print(out_json)


if __name__ == "__main__":
    main()
