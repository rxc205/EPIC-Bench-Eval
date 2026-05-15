# -*- coding: utf-8 -*-
"""Scorer registry entrypoint.

Importing this module triggers registration for all task scorers.
"""

from .base import BaseScorer, ScoreResult  # noqa: F401
from .registry import get_scorer, list_registered, register_scorer  # noqa: F401

# Trigger scorer registration for all tasks
from . import target_localization  # noqa: F401
from . import navigation  # noqa: F401
from . import manipulation  # noqa: F401

__all__ = [
    "BaseScorer",
    "ScoreResult",
    "get_scorer",
    "list_registered",
    "register_scorer",
]
