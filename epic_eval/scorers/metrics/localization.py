# -*- coding: utf-8 -*-
"""
Localization score: compute IoU between model formatted_response_localization and GT masks.

Supported format_type:
- "bbox": convert predicted bboxes into a mask (respecting coordinate_system), then IoU with GT.
- "mask": decode predicted RLE mask(s) and IoU with GT.

Returns (score, details_dict, error_msg).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...utils.coords import normalize_bboxes, resolve_actual_coord_mode
from ...utils.masks import (
    bboxes_to_mask,
    decode_response_mask,
    decode_rle_list_to_mask,
    mask_iou,
)
from ...utils.parsers import (
    is_no_valid_response,
    is_not_required,
    parse_localization_bboxes,
    parse_localization_mask,
)


def _gt_is_empty(gt_mask_rle: Any) -> bool:
    """Return True if GT mask is empty (i.e., "no target").

    - empty list / None / empty dict are treated as empty annotations.
    - if decoding fails but there is data, it should NOT be treated as empty.
    """
    if gt_mask_rle is None:
        return True
    if isinstance(gt_mask_rle, list) and len(gt_mask_rle) == 0:
        return True
    if isinstance(gt_mask_rle, dict) and not gt_mask_rle:
        return True
    return False


def localization_score(
    formatted_response_localization: Any,
    gt_mask_rle: Any,
    format_type: str,
    coord_mode: str,
    gt_count: Optional[int] = None,
) -> Tuple[float, Dict[str, Any], Optional[str]]:
    """Return (score, details, error).

    details includes 'actual_coord_mode' (when coord_mode is auto).

    Edge cases:
    - "No Valid Response" -> invalid, score=0 with error.
    - empty [] / {} (a valid "no target" response):
      - if GT empty -> 1.0
      - else -> 0.0 (missed all targets)
    - if gt_count is provided and equals 0, treat GT as empty even if mask has tiny noise.
    - if GT mask decoding fails: treat as empty only if GT is truly empty; otherwise error.
    """
    details: Dict[str, Any] = {
        "format_type": format_type,
        "coord_mode": coord_mode,
        "actual_coord_mode": coord_mode if coord_mode != "auto" else None,
        "gt_count": gt_count,
        "pred_bbox_count": 0,
        "pred_mask_decoded": False,
        "gt_mask_decoded": False,
        "gt_is_empty": False,
        "pred_is_empty": False,
    }

    if is_not_required(formatted_response_localization):
        return 0.0, {**details, "message": "Not Required"}, None

    if is_no_valid_response(formatted_response_localization):
        return 0.0, {**details, "no_valid_response": True}, "No Valid Response"

    gt_empty = _gt_is_empty(gt_mask_rle) or (gt_count is not None and int(gt_count) == 0)
    details["gt_is_empty"] = gt_empty

    # Decode GT mask
    gt_mask = None if gt_empty else decode_rle_list_to_mask(gt_mask_rle)
    if not gt_empty and gt_mask is None:
        return 0.0, details, "Failed to decode GT mask or GT mask is missing"
    if gt_mask is not None:
        h, w = gt_mask.shape[0], gt_mask.shape[1]
        details["gt_mask_decoded"] = True
    else:
        # GT is empty: use a placeholder size (gt_mask won't be used for IoU)
        h, w = 1, 1

    # ----- mask mode -----
    if format_type == "mask":
        details["actual_coord_mode"] = "n/a"
        raw = parse_localization_mask(formatted_response_localization)
        # pred explicitly empty ([] / {})
        if raw is None or (isinstance(raw, list) and not raw) or (isinstance(raw, dict) and not raw):
            details["pred_is_empty"] = True
            score = 1.0 if gt_empty else 0.0
            return score, {**details, "iou": score}, None
        pred_mask = decode_response_mask(raw)
        if pred_mask is None:
            return 0.0, details, "Failed to decode predicted mask response"
        details["pred_mask_decoded"] = True
        # all-zero predicted mask
        if int(pred_mask.sum()) == 0:
            details["pred_is_empty"] = True
            score = 1.0 if gt_empty else 0.0
            return score, {**details, "iou": score}, None
        if gt_empty:
            return 0.0, {**details, "iou": 0.0}, None
        if pred_mask.shape != gt_mask.shape:
            return 0.0, {**details, "pred_shape": list(pred_mask.shape), "gt_shape": [h, w]}, "Pred mask shape does not match GT mask shape"
        score = mask_iou(pred_mask, gt_mask)
        return score, {**details, "iou": score}, None

    # ----- bbox mode (default) -----
    pred_bboxes_raw = parse_localization_bboxes(formatted_response_localization)
    details["pred_bbox_count"] = len(pred_bboxes_raw)
    details["actual_coord_mode"] = resolve_actual_coord_mode(
        [v for b in pred_bboxes_raw for v in b], h, w, coord_mode,
    )

    # predicted bbox list is explicitly empty ("no target")
    if not pred_bboxes_raw:
        details["pred_is_empty"] = True
        score = 1.0 if gt_empty else 0.0
        return score, {**details, "iou": score}, None

    # GT is empty but pred has boxes -> false positive
    if gt_empty:
        return 0.0, {**details, "iou": 0.0}, None

    pred_bboxes = normalize_bboxes(pred_bboxes_raw, h, w, coord_mode=coord_mode)
    pred_mask = bboxes_to_mask(pred_bboxes, h, w)
    score = mask_iou(pred_mask, gt_mask)
    return score, {**details, "iou": score, "pred_bbox_normalized_count": len(pred_bboxes)}, None
