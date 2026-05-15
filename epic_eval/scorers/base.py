# -*- coding: utf-8 -*-
"""Base scorer and score result data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScoreResult:
    """Per-sample scoring result."""
    sample_index: int
    task_category: str
    task_type: str
    score: float
    score_details: Dict[str, Any] = field(default_factory=dict)
    response_details: Dict[str, Any] = field(default_factory=dict)
    metric_details: Dict[str, Any] = field(default_factory=dict)  # metric-level details (debug)
    error: Optional[str] = None
    invalid_response: bool = False  # whether counted in num_invalid_response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "task_category": self.task_category,
            "task_type": self.task_type,
            "score": float(self.score),
            "score_details": self.score_details,
            "response_details": self.response_details,
            "metric_details": self.metric_details,
            "error": self.error,
            "invalid_response": self.invalid_response,
        }


class BaseScorer(ABC):
    """Base scorer. Subclasses implement score_one for (task_category, task_type)."""

    def __init__(self, task_category: str, task_type: str):
        self.task_category = task_category
        self.task_type = task_type

    @abstractmethod
    def score_one(self, sample: Dict[str, Any], index: int) -> ScoreResult:
        ...

    def score_batch(self, samples: List[Dict[str, Any]], start_index: int = 0) -> List[ScoreResult]:
        return [self.score_one(s, start_index + i) for i, s in enumerate(samples)]
