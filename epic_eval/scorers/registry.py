# -*- coding: utf-8 -*-
"""Register scorers by (task_category, task_type)."""

from __future__ import annotations

from typing import Dict, List, Tuple, Type

from .base import BaseScorer

_registry: Dict[Tuple[str, str], Type[BaseScorer]] = {}


def register_scorer(task_category: str, task_type: str, scorer_cls: Type[BaseScorer]) -> None:
    _registry[(task_category, task_type)] = scorer_cls


def get_scorer(task_category: str, task_type: str) -> BaseScorer:
    key = (task_category, task_type)
    if key in _registry:
        return _registry[key](task_category, task_type)
    # Fallback: default scorer for task_category (when task_type is missing)
    fallback = _registry.get((task_category, "*"))
    if fallback is not None:
        return fallback(task_category, task_type)
    from .placeholder import PlaceholderScorer
    return PlaceholderScorer(task_category, task_type)


def list_registered() -> List[Tuple[str, str]]:
    return list(_registry.keys())
