# -*- coding: utf-8 -*-
"""Placeholder scorer for unregistered tasks."""

from typing import Any, Dict

from .base import BaseScorer, ScoreResult


class PlaceholderScorer(BaseScorer):
    def score_one(self, sample: Dict[str, Any], index: int) -> ScoreResult:
        return ScoreResult(
            sample_index=index,
            task_category=self.task_category,
            task_type=self.task_type,
            score=0.0,
            score_details={"message": "No scorer registered for this task (placeholder)."},
        )
