# -*- coding: utf-8 -*-
"""
Resolve a sample's (task_category, task_type, gt_json_path) into:
- leaf: the finest-grained leaf task type (one of 23)
- group: group name (one of 9)
- by_type_key: ordered key used in summary.by_type
- by_group_key: ordered key used in summary.by_group

Key design notes:
- In TargetLocalization, task_type may be a group name (BasicAttributes / ...) or a leaf
  type (Category / ...). For group-name cases, we infer the leaf from gt_json_path.
- In Navigation/Manipulation, task_type is usually already a leaf (FeasiblePath_Ego,
  ContactRelationship_TypeOne, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ..config import (
    BY_TYPE_ORDER,
    LEAF_TASKS,
    LEAF_TO_INFO,
    TL_GROUP_NAMES,
    TL_LEAF_TASKS,
)


def _parse_leaf_from_gt_path(gt_json_path: Optional[str], group: Optional[str] = None) -> Optional[str]:
    """Parse leaf task type from gt_json_path.

    Path pattern: .../<task_category>/<group>/<leaf>/<sample_dir>/...
    """
    if not gt_json_path:
        return None
    parts = Path(gt_json_path).parts
    leaf_set = set(TL_LEAF_TASKS) | {item[2] for item in LEAF_TASKS}
    # Prefer: take the component right after group
    if group:
        for i, p in enumerate(parts):
            if p == group and i + 1 < len(parts):
                cand = parts[i + 1]
                if cand in leaf_set:
                    return cand
    # Fallback: scan full path for the first leaf token
    for p in parts:
        if p in leaf_set:
            return p
    return None


def resolve(
    task_category: str,
    task_type: str,
    gt_json_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (leaf, group, by_type_key, by_group_key), or Nones if unresolved."""
    if not task_category:
        return None, None, None, None

    # TargetLocalization
    if task_category == "TargetLocalization":
        leaf: Optional[str] = None
        if task_type in TL_LEAF_TASKS:
            leaf = task_type
        elif task_type in TL_GROUP_NAMES:
            leaf = _parse_leaf_from_gt_path(gt_json_path, group=task_type)
        else:
            # Fallback: scan gt_json_path
            leaf = _parse_leaf_from_gt_path(gt_json_path)
        if leaf and leaf in LEAF_TO_INFO:
            cat, grp, by_type, by_grp = LEAF_TO_INFO[leaf]
            return leaf, grp, by_type, by_grp
        return None, None, None, None

    # Navigation
    if task_category == "Navigation":
        leaf = task_type
        # If FeasiblePath has no _Ego/_Exo suffix, infer from gt_json_path
        if leaf == "FeasiblePath":
            inferred = _parse_leaf_from_gt_path(gt_json_path)
            if inferred in ("FeasiblePath_Ego", "FeasiblePath_Exo"):
                leaf = inferred
        if leaf in LEAF_TO_INFO:
            cat, grp, by_type, by_grp = LEAF_TO_INFO[leaf]
            return leaf, grp, by_type, by_grp
        return None, None, None, None

    # Manipulation
    if task_category == "Manipulation":
        leaf = task_type
        if leaf == "Affordance":
            leaf = "AffordanceRegion"
        elif leaf == "Placement":
            leaf = "PlacementRegion"
        elif leaf == "ContactRelationship":
            inferred = _parse_leaf_from_gt_path(gt_json_path)
            if inferred and inferred.startswith("ContactRelationship_"):
                leaf = inferred
        if leaf in LEAF_TO_INFO:
            cat, grp, by_type, by_grp = LEAF_TO_INFO[leaf]
            return leaf, grp, by_type, by_grp
        return None, None, None, None

    return None, None, None, None
