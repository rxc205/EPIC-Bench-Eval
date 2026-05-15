# -*- coding: utf-8 -*-
"""
Task taxonomy and GT field mapping.

In the std_format schema, each sample explicitly includes:
- task_category: TargetLocalization / Navigation / Manipulation
- task_type:
  - TargetLocalization: 14 leaf task types (Category, Color, ... SpatialRelationHuman)
  - Navigation: GroundDetection / FeasiblePath_Ego / FeasiblePath_Exo / VisualMatching
  - Manipulation: AffordanceRegion / ContactRelationship_TypeOne|Two|Three / PlacementRegion

GT field names are defined in GT_FIELDS.
Scorers use (task_category, task_type) to look up the correct GT fields.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Task categories
# ---------------------------------------------------------------------------
TASK_CATEGORIES: Tuple[str, ...] = ("TargetLocalization", "Navigation", "Manipulation")

# ---------------------------------------------------------------------------
# TargetLocalization: 14 leaf task types grouped by attributes
# ---------------------------------------------------------------------------
TL_GROUPS: Dict[str, List[str]] = {
    "BasicAttributes": [
        "Category",
        "Color",
        "Geometry",
        "Material",
        "ObjectProjections",
        "RelativePerception",
    ],
    "EmbodiedCompositionalAttributes": [
        "ObjectState",
        "PartWhole",
        "PartWholeHuman",
        "Spoken",
    ],
    "SpatialRelatedAttributes": [
        "Orientation",
        "OrientationHuman",
        "SpatialRelation",
        "SpatialRelationHuman",
    ],
}
TL_LEAF_TASKS: Tuple[str, ...] = tuple(t for sub in TL_GROUPS.values() for t in sub)

# Legacy compatibility: older std_format may use group names as task_type
TL_GROUP_NAMES: Tuple[str, ...] = tuple(TL_GROUPS.keys())

# ---------------------------------------------------------------------------
# Navigation task types
# ---------------------------------------------------------------------------
NAV_SUB_TASKS: Tuple[str, ...] = (
    "GroundDetection",
    "FeasiblePath_Ego",
    "FeasiblePath_Exo",
    "VisualMatching",
)
# NAV_SUB_TASKS: Tuple[str, ...] = (
#     "GroundDetection",
#     "FeasiblePath",
#     "FeasiblePath_Ego",
#     "FeasiblePath_Exo",
#     "VisualMatching",
# )

# ---------------------------------------------------------------------------
# Manipulation task types
# ---------------------------------------------------------------------------
MANIP_SUB_TASKS: Tuple[str, ...] = (
    "AffordanceRegion",
    "ContactRelationship",
    "ContactRelationship_TypeOne",
    "ContactRelationship_TypeTwo",
    "ContactRelationship_TypeThree",
    "PlacementRegion",
)

# MANIP_SUB_TASKS: Tuple[str, ...] = (
#     "Affordance",
#     "AffordanceRegion",
#     "ContactRelationship",
#     "ContactRelationship_TypeOne",
#     "ContactRelationship_TypeTwo",
#     "ContactRelationship_TypeThree",
#     "Placement",
#     "PlacementRegion",
# )

SUB_TASKS_BY_CATEGORY: Dict[str, Tuple[str, ...]] = {
    "TargetLocalization": TL_LEAF_TASKS + TL_GROUP_NAMES,
    "Navigation": NAV_SUB_TASKS,
    "Manipulation": MANIP_SUB_TASKS,
}

# ---------------------------------------------------------------------------
# GT field mapping: per-task field names (read from sample.ground_truth; fall back to gt_json_path if missing)
# - localization_mask: GT mask (list of RLE dicts) for localization
# - count_field: GT count field
# - precise_count_field: whether precise counting is required (bool)
# - count_in_description_field: whether the description contains a count (bool; precomputed in annotations)
# - feasibility_field: PlacementRegion only: placement feasibility (bool)
# - target_area_mask: FeasiblePath only: target area mask (for start/end checks)
# - path_mask: FeasiblePath only: feasible path mask
# ---------------------------------------------------------------------------
GT_FIELDS: Dict[Tuple[str, str], Dict[str, str]] = {
    # TargetLocalization: all 14 leaf tasks share the same GT fields
    ("TargetLocalization", "*"): {
        "localization_mask": "masklet_target",
        "count_field": "count_target_instance",
        "precise_count_field": "precise_count",
        "count_in_description_field": "count_in_description",
    },
    ("Navigation", "GroundDetection"): {
        "localization_mask": "masklet_ground",
    },
    # FeasiblePath: path_mask is feasible area; target_area_mask is the target (or start/end) area
    ("Navigation", "FeasiblePath_Ego"): {
        "path_mask": "masklet_path",
        "target_area_mask": "masklet_target_area",
    },
    ("Navigation", "FeasiblePath_Exo"): {
        "path_mask": "masklet_path",
        "target_area_mask": "masklet_target_area",
    },
    ("Navigation", "FeasiblePath"): {
        "path_mask": "masklet_path",
        "target_area_mask": "masklet_target_area",
    },
    ("Navigation", "VisualMatching"): {
        "localization_mask": "masklet_matching_area",
        "count_field": "count_matching_area",
        # Compatibility: some legacy GT used "Precise_Count"
        "precise_count_field": "precise_count",
        "precise_count_field_alt": "Precise_Count",
        "count_in_description_field": "count_in_description",
    },
    ("Manipulation", "AffordanceRegion"): {
        "localization_mask": "masklet_affordance_region",
    },
    ("Manipulation", "ContactRelationship_TypeOne"): {
        "localization_mask": "masklet_contact_target",
        "count_field": "count_contact_target",
        "precise_count_field": "precise_count",
        "count_in_description_field": "count_in_description",
    },
    ("Manipulation", "ContactRelationship_TypeTwo"): {
        "localization_mask": "masklet_contact_target",
        "count_field": "count_contact_target",
        "precise_count_field": "precise_count",
        "count_in_description_field": "count_in_description",
    },
    ("Manipulation", "ContactRelationship_TypeThree"): {
        "localization_mask": "masklet_contact_target",
        "count_field": "count_contact_target",
        "precise_count_field": "precise_count",
        "count_in_description_field": "count_in_description",
    },
    ("Manipulation", "PlacementRegion"): {
        "localization_mask": "masklet_placement_region",
        # Compatibility: raw GT JSON uses "placement_feasible" (no "-ity"),
        # while std_format uses "placement_feasibility".
        "feasibility_field": "placement_feasibility",
        "feasibility_field_alt": "placement_feasible",
    },
}


def get_gt_fields(task_category: str, task_type: str) -> Dict[str, str]:
    """Return GT field mapping for (task_category, task_type). TargetLocalization shares one mapping."""
    if task_category == "TargetLocalization":
        return GT_FIELDS[("TargetLocalization", "*")]
    # Legacy compatibility: Manipulation/Affordance, Manipulation/Placement
    if (task_category, task_type) == ("Manipulation", "Affordance"):
        return GT_FIELDS[("Manipulation", "AffordanceRegion")]
    if (task_category, task_type) == ("Manipulation", "Placement"):
        return GT_FIELDS[("Manipulation", "PlacementRegion")]
    if (task_category, task_type) == ("Manipulation", "ContactRelationship"):
        return GT_FIELDS[("Manipulation", "ContactRelationship_TypeOne")]
    return GT_FIELDS.get((task_category, task_type), {})


def required_metrics(task_category: str, task_type: str) -> Dict[str, bool]:
    """Which metrics are used for this task (aligned with format_response required fields)."""
    out = {"localization": False, "count": False, "path": False, "binary": False}
    if task_category == "TargetLocalization":
        out["localization"] = True
        out["count"] = True
    elif task_category == "Navigation":
        if task_type == "GroundDetection":
            out["localization"] = True
        elif task_type in ("FeasiblePath", "FeasiblePath_Ego", "FeasiblePath_Exo"):
            out["path"] = True
        elif task_type == "VisualMatching":
            out["localization"] = True
            out["count"] = True
    elif task_category == "Manipulation":
        if task_type in ("Affordance", "AffordanceRegion"):
            out["localization"] = True
        elif task_type in (
            "ContactRelationship",
            "ContactRelationship_TypeOne",
            "ContactRelationship_TypeTwo",
            "ContactRelationship_TypeThree",
        ):
            out["localization"] = True
            out["count"] = True
        elif task_type in ("Placement", "PlacementRegion"):
            out["localization"] = True
            out["binary"] = True
    return out


# ---------------------------------------------------------------------------
# Sentinels: not implemented / invalid response markers (aligned with format_response.py)
# ---------------------------------------------------------------------------
NOT_REQUIRED = "Not Required"
NO_VALID_RESPONSE = "No Valid Response"

# Coordinate modes (aligned with format_response.py)
COORDINATE_SYSTEMS: Tuple[str, ...] = (
    "auto",
    "absolute",
    "normalized_0_1",
    "normalized_0_1000",
)

# format_type
FORMAT_TYPES: Tuple[str, ...] = ("bbox", "mask", "point")


# ---------------------------------------------------------------------------
# Output aggregation: ordered keys for by_category / by_type / by_group
# - by_category: 3 categories
# - by_type: 23 leaf tasks (path-like keys with category/group prefix)
# - by_group: 9 groups (path-like keys)
# Order matches README/templates.
# ---------------------------------------------------------------------------
BY_CATEGORY_ORDER: Tuple[str, ...] = ("TargetLocalization", "Navigation", "Manipulation")

# (category, group, leaf, by_type_key, by_group_key) — order is fixed
LEAF_TASKS: Tuple[Tuple[str, str, str, str, str], ...] = (
    # TargetLocalization / BasicAttributes
    ("TargetLocalization", "BasicAttributes", "Category",
     "TargetLocalization/BasicAttributes/Category", "TargetLocalization/BasicAttributes"),
    ("TargetLocalization", "BasicAttributes", "Color",
     "TargetLocalization/BasicAttributes/Color", "TargetLocalization/BasicAttributes"),
    ("TargetLocalization", "BasicAttributes", "Geometry",
     "TargetLocalization/BasicAttributes/Geometry", "TargetLocalization/BasicAttributes"),
    ("TargetLocalization", "BasicAttributes", "Material",
     "TargetLocalization/BasicAttributes/Material", "TargetLocalization/BasicAttributes"),
    ("TargetLocalization", "BasicAttributes", "ObjectProjections",
     "TargetLocalization/BasicAttributes/ObjectProjections", "TargetLocalization/BasicAttributes"),
    ("TargetLocalization", "BasicAttributes", "RelativePerception",
     "TargetLocalization/BasicAttributes/RelativePerception", "TargetLocalization/BasicAttributes"),
    # TargetLocalization / SpatialRelatedAttributes
    ("TargetLocalization", "SpatialRelatedAttributes", "Orientation",
     "TargetLocalization/SpatialRelatedAttributes/Orientation",
     "TargetLocalization/SpatialRelatedAttributes"),
    ("TargetLocalization", "SpatialRelatedAttributes", "OrientationHuman",
     "TargetLocalization/SpatialRelatedAttributes/OrientationHuman",
     "TargetLocalization/SpatialRelatedAttributes"),
    ("TargetLocalization", "SpatialRelatedAttributes", "SpatialRelation",
     "TargetLocalization/SpatialRelatedAttributes/SpatialRelation",
     "TargetLocalization/SpatialRelatedAttributes"),
    ("TargetLocalization", "SpatialRelatedAttributes", "SpatialRelationHuman",
     "TargetLocalization/SpatialRelatedAttributes/SpatialRelationHuman",
     "TargetLocalization/SpatialRelatedAttributes"),
    # TargetLocalization / EmbodiedCompositionalAttributes
    ("TargetLocalization", "EmbodiedCompositionalAttributes", "ObjectState",
     "TargetLocalization/EmbodiedCompositionalAttributes/ObjectState",
     "TargetLocalization/EmbodiedCompositionalAttributes"),
    ("TargetLocalization", "EmbodiedCompositionalAttributes", "PartWhole",
     "TargetLocalization/EmbodiedCompositionalAttributes/PartWhole",
     "TargetLocalization/EmbodiedCompositionalAttributes"),
    ("TargetLocalization", "EmbodiedCompositionalAttributes", "PartWholeHuman",
     "TargetLocalization/EmbodiedCompositionalAttributes/PartWholeHuman",
     "TargetLocalization/EmbodiedCompositionalAttributes"),
    ("TargetLocalization", "EmbodiedCompositionalAttributes", "Spoken",
     "TargetLocalization/EmbodiedCompositionalAttributes/Spoken",
     "TargetLocalization/EmbodiedCompositionalAttributes"),
    # Navigation
    ("Navigation", "GroundDetection", "GroundDetection",
     "Navigation/GroundDetection", "Navigation/GroundDetection"),
    ("Navigation", "FeasiblePath", "FeasiblePath_Ego",
     "Navigation/FeasiblePath/FeasiblePath_Ego", "Navigation/FeasiblePath"),
    ("Navigation", "FeasiblePath", "FeasiblePath_Exo",
     "Navigation/FeasiblePath/FeasiblePath_Exo", "Navigation/FeasiblePath"),
    ("Navigation", "VisualMatching", "VisualMatching",
     "Navigation/VisualMatching", "Navigation/VisualMatching"),
    # Manipulation
    ("Manipulation", "AffordanceRegion", "AffordanceRegion",
     "Manipulation/AffordanceRegion", "Manipulation/AffordanceRegion"),
    ("Manipulation", "ContactRelationship", "ContactRelationship_TypeOne",
     "Manipulation/ContactRelationship/ContactRelationship_TypeOne",
     "Manipulation/ContactRelationship"),
    ("Manipulation", "ContactRelationship", "ContactRelationship_TypeTwo",
     "Manipulation/ContactRelationship/ContactRelationship_TypeTwo",
     "Manipulation/ContactRelationship"),
    ("Manipulation", "ContactRelationship", "ContactRelationship_TypeThree",
     "Manipulation/ContactRelationship/ContactRelationship_TypeThree",
     "Manipulation/ContactRelationship"),
    ("Manipulation", "PlacementRegion", "PlacementRegion",
     "Manipulation/PlacementRegion", "Manipulation/PlacementRegion"),
)

BY_TYPE_ORDER: Tuple[str, ...] = tuple(item[3] for item in LEAF_TASKS)

# Preserve de-duplicated by_group order
def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)

BY_GROUP_ORDER: Tuple[str, ...] = _dedup_keep_order(item[4] for item in LEAF_TASKS)


# Reverse lookup: leaf -> (category, group, by_type_key, by_group_key)
LEAF_TO_INFO: Dict[str, Tuple[str, str, str, str]] = {
    leaf: (cat, grp, by_type, by_grp)
    for (cat, grp, leaf, by_type, by_grp) in LEAF_TASKS
}


def leaf_to_group(leaf: str) -> str:
    info = LEAF_TO_INFO.get(leaf)
    return info[1] if info else ""
