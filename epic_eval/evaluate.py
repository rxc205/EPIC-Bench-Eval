# -*- coding: utf-8 -*-
"""
Evaluation entrypoint for EPIC-Bench (std_format inputs).

Output schema (JSON):
{
  "std_response_jsonl": "<input path>",
  "summary": { ... },
  "details": { "by_samples": [ ... ] }
}

If `--jsonl` is set (or output path ends with `.jsonl`), writes JSONL:
- first line: kind=summary
- following lines: kind=sample
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    BY_CATEGORY_ORDER,
    BY_GROUP_ORDER,
    BY_TYPE_ORDER,
    COORDINATE_SYSTEMS,
    LEAF_TASKS,
    LEAF_TO_INFO,
    TL_GROUPS,
)
from .scorers import get_scorer
from .scorers.base import ScoreResult
from .utils.gt_loader import load_raw_gt_json
from .utils.leaf_resolver import resolve as resolve_leaf_keys


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_std_format(path: str) -> List[Dict[str, Any]]:
    """Load std_format file. Supports JSONL and JSON array."""
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        try:
            data = json.loads(text)
            return [s for s in data if isinstance(s, dict)]
        except json.JSONDecodeError:
            return []
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                out.append(item)
        except json.JSONDecodeError:
            continue
    return out


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


# ---------------------------------------------------------------------------
# Filtering / overrides
# ---------------------------------------------------------------------------
def filter_samples(
    samples: List[Dict[str, Any]],
    categories: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    group: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cats = set(categories) if categories else None
    types = set(task_types) if task_types else None
    if group is not None:
        types_in_group = set(TL_GROUPS.get(group, []))
        types = (types & types_in_group) if types else types_in_group
        if cats is None:
            cats = {"TargetLocalization"}
    out: List[Dict[str, Any]] = []
    for s in samples:
        c = _first(s.get("task_category")) or ""
        t = _first(s.get("task_type")) or ""
        if cats and c not in cats:
            continue
        if types and t not in types:
            continue
        out.append(s)
    return out


def _apply_coord_override(samples: List[Dict[str, Any]], coord_mode: str) -> None:
    if not coord_mode:
        return
    for s in samples:
        s["coordinate_system"] = [coord_mode]


# ---------------------------------------------------------------------------
# Parse expected_counts
# ---------------------------------------------------------------------------
def load_expected_counts(path: Optional[str]) -> Dict[str, Any]:
    """Expected-counts config. May include 'overall' / 'by_category' / 'by_type' / 'by_group'."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_eval_jsonl(payload: Dict[str, Any], output_path: Path) -> None:
    """Write JSONL: first line kind=summary; remaining lines kind=sample."""
    with open(output_path, "w", encoding="utf-8") as f:
        header: OrderedDict[str, Any] = OrderedDict()
        header["kind"] = "summary"
        header["std_response_jsonl"] = payload["std_response_jsonl"]
        if "coord_override" in payload:
            header["coord_override"] = payload["coord_override"]
        header["summary"] = payload["summary"]
        f.write(json.dumps(header, ensure_ascii=False) + "\n")

        details = payload.get("details") or {}
        for row in details.get("by_samples") or []:
            line: OrderedDict[str, Any] = OrderedDict([("kind", "sample")])
            line.update(row)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def run_evaluation(
    input_path: str,
    categories: Optional[Sequence[str]] = None,
    task_types: Optional[Sequence[str]] = None,
    group: Optional[str] = None,
    output_path: Optional[str] = None,
    coord_mode: Optional[str] = None,
    expected_counts: Optional[Dict[str, Any]] = None,
    include_details: bool = True,
    output_jsonl: bool = False,
) -> Dict[str, Any]:
    samples = load_std_format(input_path)
    filtered = filter_samples(samples, categories=categories, task_types=task_types, group=group)
    if coord_mode:
        _apply_coord_override(filtered, coord_mode)

    expected_counts = expected_counts or {}
    expected_overall = expected_counts.get("overall")
    expected_by_cat = expected_counts.get("by_category", {}) or {}
    expected_by_type = expected_counts.get("by_type", {}) or {}
    expected_by_group = expected_counts.get("by_group", {}) or {}

    # ---- scoring (also resolves leaf/group/by_type/by_group keys) ----
    results: List[Tuple[ScoreResult, Dict[str, Any], Dict[str, Optional[str]]]] = []
    for i, sample in enumerate(filtered):
        c = _first(sample.get("task_category")) or ""
        t = _first(sample.get("task_type")) or ""
        scorer = get_scorer(c, t)
        try:
            res = scorer.score_one(sample, i)
        except Exception as e:  # pragma: no cover
            res = ScoreResult(
                sample_index=i, task_category=c, task_type=t, score=0.0,
                error=f"{type(e).__name__}: {e}", invalid_response=True,
            )
        gt = sample.get("ground_truth") or {}
        gt_path_raw = gt.get("gt_json_path")
        gt_path = gt_path_raw[0] if isinstance(gt_path_raw, list) and gt_path_raw else gt_path_raw
        leaf, grp, by_type, by_grp = resolve_leaf_keys(c, t, gt_path)
        results.append((res, sample, {"leaf": leaf, "group": grp, "by_type": by_type, "by_group": by_grp}))

    # ---- aggregation ----
    cat_scores: Dict[str, List[float]] = defaultdict(list)
    cat_invalid: Dict[str, int] = defaultdict(int)
    type_scores: Dict[str, List[float]] = defaultdict(list)
    type_invalid: Dict[str, int] = defaultdict(int)
    group_scores: Dict[str, List[float]] = defaultdict(list)
    group_invalid: Dict[str, int] = defaultdict(int)

    overall_invalid = 0
    overall_scores: List[float] = []

    for res, _sample, keys in results:
        c = res.task_category
        cat_scores[c].append(res.score)
        cat_invalid[c] += int(res.invalid_response)
        if keys["by_type"]:
            type_scores[keys["by_type"]].append(res.score)
            type_invalid[keys["by_type"]] += int(res.invalid_response)
        if keys["by_group"]:
            group_scores[keys["by_group"]].append(res.score)
            group_invalid[keys["by_group"]] += int(res.invalid_response)
        overall_scores.append(res.score)
        overall_invalid += int(res.invalid_response)

    def _bucket(scores: List[float], invalid: int, expected: Optional[int]) -> Dict[str, Any]:
        n = len(scores)
        return OrderedDict([
            ("total_expected", int(expected) if expected is not None else n),
            ("total_processed", n),
            ("num_invalid_response", int(invalid)),
            ("mean_score", float(sum(scores) / n) if n > 0 else 0.0),
        ])

    summary = OrderedDict()
    summary["overall"] = OrderedDict([
        ("total_expected", int(expected_overall) if expected_overall is not None else len(overall_scores)),
        ("total_processed", len(overall_scores)),
        ("num_invalid_response", int(overall_invalid)),
        ("average_score", float(sum(overall_scores) / len(overall_scores)) if overall_scores else 0.0),
    ])

    # by_category (ordered by BY_CATEGORY_ORDER)
    by_cat = OrderedDict()
    for cat in BY_CATEGORY_ORDER:
        if cat in cat_scores:
            by_cat[cat] = _bucket(cat_scores[cat], cat_invalid[cat], expected_by_cat.get(cat))
    summary["by_category"] = by_cat

    # by_type (ordered by BY_TYPE_ORDER; show present or expected items)
    by_type = OrderedDict()
    for key in BY_TYPE_ORDER:
        if key in type_scores or key in expected_by_type:
            by_type[key] = _bucket(type_scores.get(key, []), type_invalid.get(key, 0), expected_by_type.get(key))
    summary["by_type"] = by_type

    # by_group (ordered by BY_GROUP_ORDER)
    by_group = OrderedDict()
    for key in BY_GROUP_ORDER:
        if key in group_scores or key in expected_by_group:
            by_group[key] = _bucket(group_scores.get(key, []), group_invalid.get(key, 0), expected_by_group.get(key))
    summary["by_group"] = by_group

    # ---- details.by_samples ----
    payload: Dict[str, Any] = OrderedDict()
    payload["std_response_jsonl"] = str(Path(input_path).resolve())
    if coord_mode:
        payload["coord_override"] = coord_mode
    payload["summary"] = summary

    if include_details:
        by_samples: List[Dict[str, Any]] = []
        for res, sample, keys in results:
            entry: Dict[str, Any] = OrderedDict()
            entry["index"] = res.sample_index
            entry["task_category"] = res.task_category
            entry["task_type"] = res.task_type
            # Also include resolved leaf/by_type/by_group for downstream analysis
            # if keys["leaf"]:
            #     entry["leaf_task"] = keys["leaf"]
            # if keys["by_type"]:
            #     entry["by_type_key"] = keys["by_type"]
            # if keys["by_group"]:
            #     entry["by_group_key"] = keys["by_group"]
            entry["score"] = float(res.score)
            entry["score_details"] = res.score_details
            entry["response_details"] = res.response_details
            entry["ground_truth"] = {
                "gt_json_path": (sample.get("ground_truth") or {}).get("gt_json_path"),
            }
            entry["error"] = res.error
            entry["invalid_response"] = bool(res.invalid_response)
            by_samples.append(entry)
        payload["details"] = OrderedDict([("by_samples", by_samples)])

    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        use_jsonl = output_jsonl or outp.suffix.lower() == ".jsonl"
        if use_jsonl:
            _write_eval_jsonl(payload, outp)
        else:
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EPIC-Bench evaluation: compute weighted metrics by task_category / task_type",
    )
    parser.add_argument("input", help="Path to std_format JSONL (or JSON array)")
    parser.add_argument("--categories", nargs="+", choices=list(BY_CATEGORY_ORDER), default=None,
                        help="Evaluate only the specified top-level task categories")
    parser.add_argument("--task-types", nargs="+", default=None,
                        help="Evaluate only the specified task_types")
    parser.add_argument("--group", choices=list(TL_GROUPS.keys()), default=None,
                        help="TargetLocalization only: filter by attribute group")
    parser.add_argument("--coord-mode", choices=list(COORDINATE_SYSTEMS), default=None,
                        help="Override coordinate_system for all samples (default: use per-sample value)")
    parser.add_argument("--expected-counts", default=None,
                        help="Path to expected-counts JSON containing overall/by_category/by_type/by_group")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for summary+details. Use .jsonl or --jsonl to write JSONL.")
    parser.add_argument("--jsonl", action="store_true",
                        help="Write JSONL output (first line kind=summary, then kind=sample)")
    parser.add_argument("--no-details", action="store_true", help="Do not output details.by_samples (summary only)")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List registered (task_category, task_type) and exit")
    args = parser.parse_args()

    if args.list_tasks:
        from .scorers import list_registered
        for c, t in sorted(list_registered()):
            print(f"{c}/{t}")
        return

    expected = load_expected_counts(args.expected_counts) if args.expected_counts else {}
    use_jsonl = bool(args.jsonl)
    if args.output and Path(args.output).suffix.lower() == ".jsonl":
        use_jsonl = True
    payload = run_evaluation(
        args.input,
        categories=args.categories,
        task_types=args.task_types,
        group=args.group,
        output_path=args.output,
        coord_mode=args.coord_mode,
        expected_counts=expected,
        include_details=not args.no_details,
        output_jsonl=use_jsonl,
    )
    # Print summary only to avoid large console output
    out_for_console = OrderedDict()
    out_for_console["std_response_jsonl"] = payload["std_response_jsonl"]
    out_for_console["summary"] = payload["summary"]
    print(json.dumps(out_for_console, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
