#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation driver (std_format).

Expected input directory layout (example; any nesting is supported):
    <INPUT_BASE>/<ModelFamily>/xxx.jsonl
You may also use the legacy naming: std_format_<ModelName>.jsonl

This script recursively finds all *.jsonl under the input root (excluding __pycache__ / .git).

Output layout:
    <OUTPUT_BASE>/<relative_dir>/{<ModelName>_<task_scope>.json(.l)}

Examples:
    python -m epic_eval.run_batch_evaluation
    python -m epic_eval.run_batch_evaluation --dry-run
    python -m epic_eval.run_batch_evaluation --input /path/to/std_format --output /path/to/scores
    python -m epic_eval.run_batch_evaluation --models Qwen3-VL --parallel 4
    python -m epic_eval.run_batch_evaluation --categories Manipulation --task-scope manip_only
    python -m epic_eval.run_batch_evaluation --coord-mode normalized_0_1000
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Default paths
# ============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()  # <repo>/epic_eval
# Repo root: <repo>/epic_eval -> <repo>
EPIC_BENCH_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_BASE = EPIC_BENCH_ROOT / "outputs" / "model_response" / "std_format"
DEFAULT_OUTPUT_BASE = EPIC_BENCH_ROOT / "outputs" / "scores"

# ============================================================================
# Coordinate mode per model family (if not listed, use per-sample field)
# ============================================================================
MODEL_COORD_MODES: Dict[str, str] = {
    "Qwen2.5-VL": "absolute",
    "Qwen3-VL": "normalized_0_1000",
    "Qwen3.5-VL": "normalized_0_1000",
    "InternVL3": "normalized_0_1000",
    "InternVL3.5": "normalized_0_1000",
    "LLaVA": "normalized_0_1",
    "Claude": "absolute",
    "Gemini": "normalized_0_1000",
}
DEFAULT_COORD_MODE = None  # None: do not override; use per-sample field


# ============================================================================
# Helpers
# ============================================================================
def extract_info_from_path(filepath: Path, input_base: Path) -> Tuple[str, str, str]:
    """Return (round_name, model_family, model_name)."""
    try:
        rel = filepath.relative_to(input_base)
        parts = rel.parts
    except ValueError:
        parts = filepath.parts

    round_name = "default"
    m = re.search(r"Round_(\d+)", str(filepath))
    if m:
        round_name = f"Round_{m.group(1)}"

    # model_family: the parent directory name; if parent is input_base, use "default"
    parent = filepath.parent
    if parent == input_base:
        model_family = "default"
    else:
        model_family = parent.name

    name = filepath.name
    # Remove common filename prefixes
    base = name.replace(".jsonl", "")
    base = re.sub(r"^std_format[_-]?", "", base)
    base = re.sub(r"^std_swift_format[_-]?", "", base)
    model_name = base or filepath.stem
    return round_name, model_family, model_name


def find_jsonl_files(
    input_base: Path,
    rounds: Optional[List[str]] = None,
    model_families: Optional[List[str]] = None,
) -> List[Path]:
    if not input_base.exists():
        return []
    # All *.jsonl under input (preserves tree vs. only std_format_*.jsonl naming).
    files = sorted(
        {
            p
            for p in input_base.rglob("*.jsonl")
            if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts
        }
    )

    out: List[Path] = []
    for f in files:
        rn, mf, _ = extract_info_from_path(f, input_base)
        if rounds and rn not in rounds:
            continue
        if model_families and mf not in model_families:
            continue
        out.append(f)
    return out


def resolve_coord_mode(model_family: str, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    return MODEL_COORD_MODES.get(model_family, DEFAULT_COORD_MODE)


def build_command(
    input_file: Path,
    output_path: Path,
    coord_mode: Optional[str],
    categories: Optional[List[str]],
    task_types: Optional[List[str]],
    group: Optional[str],
    expected_counts: Optional[str],
    no_details: bool,
    output_jsonl: bool,
) -> List[str]:
    cmd = [
        sys.executable, "-m", "epic_eval.evaluate",
        str(input_file),
        "--output", str(output_path),
    ]
    if output_jsonl:
        cmd.append("--jsonl")
    if coord_mode:
        cmd += ["--coord-mode", coord_mode]
    if categories:
        cmd += ["--categories", *categories]
    if task_types:
        cmd += ["--task-types", *task_types]
    if group:
        cmd += ["--group", group]
    if expected_counts:
        cmd += ["--expected-counts", expected_counts]
    if no_details:
        cmd += ["--no-details"]
    return cmd


def run_single(
    input_file: Path,
    output_path: Path,
    coord_mode: Optional[str],
    categories: Optional[List[str]],
    task_types: Optional[List[str]],
    group: Optional[str],
    expected_counts: Optional[str],
    no_details: bool,
    output_jsonl: bool,
    cwd: Path,
    dry_run: bool,
    timeout: int,
) -> Tuple[bool, str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        input_file,
        output_path,
        coord_mode,
        categories,
        task_types,
        group,
        expected_counts,
        no_details,
        output_jsonl,
    )
    if dry_run:
        return True, str(input_file), "[DRY-RUN] " + " ".join(cmd)
    try:
        env = os.environ.copy()
        # Ensure the subprocess can import epic_eval (package lives under repo root)
        env["PYTHONPATH"] = str(EPIC_BENCH_ROOT) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # Run from repo root to avoid cwd-dependent path resolution
        result = subprocess.run(
            cmd,
            cwd=str(EPIC_BENCH_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode == 0:
            return True, str(input_file), "Done"
        return False, str(input_file), f"Failed: {(result.stderr or '')[:500]}"
    except subprocess.TimeoutExpired:
        return False, str(input_file), f"Timeout ({timeout}s)"
    except Exception as e:  # pragma: no cover
        return False, str(input_file), f"Execution error: {e}"


# ============================================================================
# main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="EPIC-Bench batch evaluation (std_format)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_BASE), help="std_format input root")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_BASE), help="score output root")
    parser.add_argument("--rounds", nargs="+", default=None, help="filter by round name, e.g. Round_0301")
    parser.add_argument("--models", nargs="+", default=None, help="filter by model family (parent dir name)")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["TargetLocalization", "Navigation", "Manipulation"],
                        help="only evaluate selected task categories")
    parser.add_argument("--task-types", nargs="+", default=None, help="only evaluate selected task types")
    parser.add_argument("--group", default=None,
                        choices=["BasicAttributes", "EmbodiedCompositionalAttributes", "SpatialRelatedAttributes"],
                        help="TargetLocalization only: filter by attribute group")
    parser.add_argument("--coord-mode", default=None,
                        choices=["auto", "absolute", "normalized_0_1", "normalized_0_1000"],
                        help="override coordinate_system for all samples; default follows MODEL_COORD_MODES or per-sample field")
    parser.add_argument("--expected-counts", default=None,
                        help="expected-counts JSON path (overall/by_category/by_type/by_group)")
    parser.add_argument("--no-details", action="store_true", help="do not output per-sample details (summary only)")
    parser.add_argument("--task-scope", default="full", help="suffix for output filename to distinguish eval scopes")
    parser.add_argument("--parallel", type=int, default=1, help="number of worker processes")
    parser.add_argument("--timeout", type=int, default=1800, help="per-file timeout (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="dry-run mode")
    parser.add_argument("--list-models", action="store_true", help="list discovered model families under input")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="write scores as JSONL (1 summary line + 1 line per sample); output extension is .jsonl",
    )
    args = parser.parse_args()

    input_base = Path(args.input).resolve()
    output_base = Path(args.output).resolve()

    if args.list_models:
        files = find_jsonl_files(input_base)
        families = sorted({extract_info_from_path(f, input_base)[1] for f in files})
        print("=" * 50)
        print(f"Input directory: {input_base}")
        print(f"Discovered {len(families)} model families:")
        for fam in families:
            mode = MODEL_COORD_MODES.get(fam, DEFAULT_COORD_MODE)
            tag = f" -> coord_mode={mode}" if mode else " -> coord_mode=(use per-sample field)"
            print(f"  {fam}{tag}")
        return

    files = find_jsonl_files(input_base, rounds=args.rounds, model_families=args.models)
    if not files:
        print(f"No *.jsonl files found under {input_base}")
        return

    print("=" * 60)
    print(f"Input root : {input_base}")
    print(f"Output root: {output_base}")
    print(f"Task scope : {args.task_scope}")
    print(f"Workers    : {args.parallel}")
    print(f"Format     : {'JSONL' if args.jsonl else 'JSON'}")
    if args.rounds:
        print(f"Rounds     : {', '.join(args.rounds)}")
    if args.models:
        print(f"Models     : {', '.join(args.models)}")
    if args.categories:
        print(f"Categories : {', '.join(args.categories)}")
    if args.task_types:
        print(f"Task types : {', '.join(args.task_types)}")
    print("=" * 60)
    print(f"Files      : {len(files)}")

    tasks = []
    for jf in files:
        rn, mf, mn = extract_info_from_path(jf, input_base)
        try:
            rel_dir = jf.parent.relative_to(input_base)
        except ValueError:
            rel_dir = Path(rn) / mf
        ext = ".jsonl" if args.jsonl else ".json"
        out_path = output_base / rel_dir / f"{mn}_{args.task_scope}{ext}"
        coord_mode = resolve_coord_mode(mf, args.coord_mode)
        tasks.append({
            "input_file": jf,
            "output_path": out_path,
            "coord_mode": coord_mode,
            "round_name": rn,
            "model_family": mf,
            "model_name": mn,
        })

    success = fail = 0
    if args.parallel > 1 and not args.dry_run:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(
                    run_single,
                    t["input_file"],
                    t["output_path"],
                    t["coord_mode"],
                    args.categories,
                    args.task_types,
                    args.group,
                    args.expected_counts,
                    args.no_details,
                    args.jsonl,
                    EPIC_BENCH_ROOT,
                    args.dry_run,
                    args.timeout,
                ): t
                for t in tasks
            }
            for fut in as_completed(futures):
                t = futures[fut]
                ok, _, msg = fut.result()
                tag = "[OK]" if ok else "[FAIL]"
                print(f"{tag} {t['model_name']} ({t['round_name']}/{t['model_family']}): {msg}")
                success += int(ok)
                fail += int(not ok)
    else:
        for i, t in enumerate(tasks, 1):
            print("-" * 60)
            print(f"[{i}/{len(tasks)}] {t['input_file']}")
            print(f"  Round  : {t['round_name']}")
            print(f"  Family : {t['model_family']}")
            print(f"  Model  : {t['model_name']}")
            print(f"  Coord  : {t['coord_mode'] or '(use per-sample field)'}")
            print(f"  Output : {t['output_path']}")
            ok, _, msg = run_single(
                t["input_file"], t["output_path"], t["coord_mode"],
                args.categories, args.task_types, args.group,
                args.expected_counts, args.no_details, args.jsonl,
                EPIC_BENCH_ROOT, args.dry_run, args.timeout,
            )
            tag = "[OK]" if ok else "[FAIL]"
            print(f"  {tag} {msg}")
            success += int(ok)
            fail += int(not ok)

    print()
    print("=" * 60)
    print(f"Total: {len(tasks)}  Success: {success}  Failed: {fail}")
    print("=" * 60)


if __name__ == "__main__":
    main()
