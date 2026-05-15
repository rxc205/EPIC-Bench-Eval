# -*- coding: utf-8 -*-
"""
EPIC-Bench evaluation package (std_format).

Modules:
- config: task taxonomy and GT field mapping
- utils.coords: coordinate normalization (bbox/point)
- utils.masks: RLE decoding and mask utilities
- utils.parsers: parse formatted_response_* fields
- utils.gt_loader: load GT from sample.ground_truth, fallback to gt_json_path
- scorers: task-specific scorers dispatched by (task_category, task_type)
"""

from .scorers import get_scorer, list_registered  # noqa: F401
