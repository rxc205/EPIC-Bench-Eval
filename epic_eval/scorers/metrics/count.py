# -*- coding: utf-8 -*-
"""
Count score (consistent with legacy rules).

Exact matching is required if:
- precise_count is True, OR
- count_in_description is True

Exact mode: pred == gt -> 1.0, else 0.0
Approx mode: 1 - min(1, |gt - pred| / max(1, gt)); if gt==0 then pred==0 -> 1.0 else 0.0
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ...utils.parsers import is_no_valid_response, is_not_required, parse_count


def _exact_required(precise: bool, in_desc: bool) -> bool:
    return bool(precise) or bool(in_desc)


def count_score(
    formatted_response_count: Any,
    gt_count: Optional[int],
    precise_count: bool,
    count_in_description: bool,
) -> Tuple[float, Dict[str, Any], Optional[str]]:
    exact = _exact_required(precise_count, count_in_description)
    details: Dict[str, Any] = {
        "precise_count": bool(precise_count),
        "count_in_description": bool(count_in_description),
        "exact_required": exact,
        "gt_count": gt_count,
        "pred_count": None,
    }

    if is_not_required(formatted_response_count):
        return 0.0, {**details, "message": "Not Required"}, None
    if is_no_valid_response(formatted_response_count):
        return 0.0, {**details, "no_valid_response": True}, "No Valid Response"

    pred = parse_count(formatted_response_count)
    details["pred_count"] = pred
    if pred is None or gt_count is None:
        return 0.0, details, "Missing pred_count or gt_count"

    if exact:
        return (1.0 if int(pred) == int(gt_count) else 0.0), {**details, "mode": "exact"}, None

    # Approx mode
    gt = int(gt_count)
    if gt <= 0:
        return (1.0 if int(pred) == 0 else 0.0), {**details, "mode": "approx"}, None
    diff_ratio = abs(gt - int(pred)) / float(max(1, gt))
    score = float(max(0.0, 1.0 - min(1.0, diff_ratio)))
    return score, {**details, "mode": "approx"}, None


def get_target_localization_weights(precise_count: bool, count_in_description: bool):
    """Return (w_loc, w_count) for TargetLocalization, consistent with legacy rules."""
    if count_in_description:
        w_count = 0.1
    elif precise_count:
        w_count = 0.5
    else:
        w_count = 0.3
    w_loc = 1.0 - w_count
    return w_loc, w_count
