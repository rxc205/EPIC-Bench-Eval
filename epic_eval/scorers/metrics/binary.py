# -*- coding: utf-8 -*-
"""Binary classification score for PlacementRegion placement_feasibility."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ...utils.parsers import is_no_valid_response, is_not_required, parse_binary


def binary_score(
    formatted_response_binary: Any,
    gt_binary: Optional[bool],
) -> Tuple[float, Dict[str, Any], Optional[str]]:
    details: Dict[str, Any] = {"gt_binary": gt_binary, "pred_binary": None}
    if is_not_required(formatted_response_binary):
        return 0.0, {**details, "message": "Not Required"}, None
    if is_no_valid_response(formatted_response_binary):
        return 0.0, {**details, "no_valid_response": True}, "No Valid Response"
    pred = parse_binary(formatted_response_binary)
    details["pred_binary"] = pred
    if pred is None or gt_binary is None:
        return 0.0, details, "Missing pred_binary or gt_binary"
    return (1.0 if bool(pred) == bool(gt_binary) else 0.0), details, None
