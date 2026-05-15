# -*- coding: utf-8 -*-
"""Manipulation scorers."""

from __future__ import annotations

from typing import Any, Dict

from ..utils.gt_loader import GTContext
from ._helpers import _first, make_response_details, make_score_details
from .base import BaseScorer, ScoreResult
from .metrics.binary import binary_score
from .metrics.count import count_score, get_target_localization_weights
from .metrics.localization import localization_score
from .registry import register_scorer


def _is_invalid(err):
    if not err:
        return False
    return ("No Valid Response" in err) or ("No path points parsed" in err) or ("Missing" in err)


class AffordanceRegionScorer(BaseScorer):
    def score_one(self, sample: Dict[str, Any], index: int) -> ScoreResult:
        gt = GTContext(sample)
        format_type = _first(sample.get("format_type")) or "bbox"
        coord_mode = _first(sample.get("coordinate_system")) or "auto"
        loc, loc_md, err = localization_score(
            sample.get("formatted_response_localization"),
            gt.localization_mask_rle,
            format_type=format_type,
            coord_mode=coord_mode,
            gt_count=gt.count_gt,
        )
        return ScoreResult(
            sample_index=index,
            task_category=self.task_category,
            task_type=self.task_type,
            score=float(loc),
            score_details=make_score_details(localization=loc, w_loc=1.0),
            response_details=make_response_details(sample, loc_md.get("actual_coord_mode")),
            metric_details={"localization": loc_md},
            error=err,
            invalid_response=_is_invalid(err),
        )


class ContactRelationshipScorer(BaseScorer):
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
        return ScoreResult(
            sample_index=index,
            task_category=self.task_category,
            task_type=self.task_type,
            score=float(total),
            score_details=make_score_details(
                localization=loc, count=cnt, w_loc=w_loc, w_count=w_count
            ),
            response_details=make_response_details(sample, loc_md.get("actual_coord_mode")),
            metric_details={"localization": loc_md, "count": cnt_md},
            error=err,
            invalid_response=_is_invalid(err),
        )


class PlacementRegionScorer(BaseScorer):
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
        bin_score_v, bin_md, bin_err = binary_score(
            sample.get("formatted_response_binary"),
            gt_binary=gt.feasibility,
        )
        w_loc, w_bin = 0.5, 0.5
        total = w_loc * loc + w_bin * bin_score_v
        err = "; ".join([e for e in (loc_err, bin_err) if e]) or None
        return ScoreResult(
            sample_index=index,
            task_category=self.task_category,
            task_type=self.task_type,
            score=float(total),
            score_details=make_score_details(
                localization=loc, binary=bin_score_v, w_loc=w_loc, w_binary=w_bin
            ),
            response_details=make_response_details(sample, loc_md.get("actual_coord_mode")),
            metric_details={"localization": loc_md, "binary": bin_md},
            error=err,
            invalid_response=_is_invalid(err),
        )


register_scorer("Manipulation", "AffordanceRegion", AffordanceRegionScorer)
register_scorer("Manipulation", "Affordance", AffordanceRegionScorer)

register_scorer("Manipulation", "ContactRelationship", ContactRelationshipScorer)
register_scorer("Manipulation", "ContactRelationship_TypeOne", ContactRelationshipScorer)
register_scorer("Manipulation", "ContactRelationship_TypeTwo", ContactRelationshipScorer)
register_scorer("Manipulation", "ContactRelationship_TypeThree", ContactRelationshipScorer)

register_scorer("Manipulation", "PlacementRegion", PlacementRegionScorer)
register_scorer("Manipulation", "Placement", PlacementRegionScorer)
