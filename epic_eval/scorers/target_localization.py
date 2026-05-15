# -*- coding: utf-8 -*-
"""TargetLocalization scorers.

All 14 leaf task types (Category, Color, ..., SpatialRelationHuman) share the same rule:
  score = w_loc * localization_score + w_count * count_score
Weights follow legacy settings determined by precise_count / count_in_description.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import TL_GROUP_NAMES, TL_LEAF_TASKS
from ..utils.gt_loader import GTContext
from ._helpers import _first, make_response_details, make_score_details
from .base import BaseScorer, ScoreResult
from .metrics.count import count_score, get_target_localization_weights
from .metrics.localization import localization_score
from .registry import register_scorer


class TargetLocalizationScorer(BaseScorer):
    def score_one(self, sample: Dict[str, Any], index: int) -> ScoreResult:
        gt = GTContext(sample)
        format_type = _first(sample.get("format_type")) or "bbox"
        coord_mode = _first(sample.get("coordinate_system")) or "auto"

        loc, loc_md, loc_err = localization_score(
            sample.get("formatted_response_localization"),
            gt.localization_mask_rle,
            format_type=format_type,
            coord_mode=coord_mode,
            gt_count=gt.count_gt,
        )
        cnt, cnt_md, cnt_err = count_score(
            sample.get("formatted_response_count"),
            gt_count=gt.count_gt,
            precise_count=gt.precise_count,
            count_in_description=gt.count_in_description,
        )

        w_loc, w_count = get_target_localization_weights(gt.precise_count, gt.count_in_description)
        total = w_loc * loc + w_count * cnt
        err = "; ".join([e for e in (loc_err, cnt_err) if e]) or None
        invalid = bool(err) and (("No Valid Response" in (err or "")) or ("No path points parsed" in (err or "")) or ("Missing" in (err or "")))

        actual_coord = loc_md.get("actual_coord_mode")
        return ScoreResult(
            sample_index=index,
            task_category=self.task_category,
            task_type=self.task_type,
            score=float(total),
            score_details=make_score_details(
                localization=loc, count=cnt, w_loc=w_loc, w_count=w_count
            ),
            response_details=make_response_details(sample, actual_coord),
            metric_details={"localization": loc_md, "count": cnt_md},
            error=err,
            invalid_response=invalid,
        )


for _t in TL_LEAF_TASKS + TL_GROUP_NAMES:
    register_scorer("TargetLocalization", _t, TargetLocalizationScorer)
