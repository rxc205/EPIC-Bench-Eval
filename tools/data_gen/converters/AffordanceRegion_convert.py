#!/usr/bin/env python3
"""
Convert AffordanceRegion annotations to ms-swift inference JSON (two images).

Prompt rule:
- system: first line of system_prompt
- user: remaining system_prompt + response_format + <image><image> +
        Reference Object Description + Task Description
"""

import json
import os
import sys
import argparse
import fnmatch
from pathlib import Path

import prompts.AffordanceRegion as AR


def _abs_path(p: str) -> str:
    p = str(p)
    return os.path.abspath(p) if not os.path.isabs(p) else p


def _first_str(x, field_name: str) -> str:
    if x is None:
        raise ValueError(f"Missing field: {field_name}")
    if isinstance(x, list):
        if not x:
            raise ValueError(f"Empty field: {field_name}")
        return str(x[0])
    return str(x)


def _derive_target_overlay_path(img1_saved: str) -> str:
    if "_image" not in img1_saved:
        raise ValueError(f"Filename does not contain '_image': {img1_saved}")
    base = img1_saved.rsplit("_image", 1)[0]
    return base + "_reference_area_overlay.png"


def split_system_prompt(system_prompt: str):
    """
    system_prompt:
    - head: first line (before the first '\\n'), kept in system role
    - tail: remaining content, moved to user role
    """
    sp = (system_prompt or "").strip()
    if not sp:
        return "", ""
    parts = sp.split("\n", 1)
    head = parts[0].strip()
    tail = parts[1].strip() if len(parts) > 1 else ""
    return head, tail


def build_user_text(
    system_tail: str,
    response_format: str,
    target_en: str,
    task: str,
) -> str:
    """
    USER:
      {system_tail}
      {response_format}
      <image>
      <image>
      Reference Object Description: ...
      Task Description: ...
    """
    chunks = []

    if system_tail:
        chunks.append(system_tail.strip())

    if response_format:
        chunks.append(response_format.strip())

    # Put image placeholders at the end (after tail + format)
    chunks.append(" <image> \n <image> ")

    chunks.append(f"Reference Object Description: {target_en}")
    chunks.append(f"Task Description: {task}")

    return "\n".join(chunks).strip()


def one_sample_to_swift(
    item: dict,
    system_prompt: str,
    response_format: str,
    use_system_prompt: bool,
    check_files_exist: bool,
) -> dict:
    """Extract fields from both new and legacy annotation schemas."""
    # 1) image paths (shared by new/legacy)
    image_paths = item.get("image_path_saved") or []
    if not image_paths:
        raise ValueError("Missing or empty field: image_path_saved")
    img1 = _abs_path(str(image_paths[0]))
    img2 = _abs_path(_derive_target_overlay_path(img1))

    # 2) Extract `task` and `target_en` with schema fallbacks
    text_anno = item.get("text_annotation")
    mask_anno = item.get("mask_annotation") or {}  # present in both schemas

    task = None
    target_en = None

    # Extract task description (task)
    if isinstance(text_anno, dict):
        # New schema: text_annotation
        task_candidates = text_anno.get("text_label") or text_anno.get("task_description_en")
        if task_candidates:
            task = _first_str(task_candidates, "text_annotation.text_label/task_description_en")
    if task is None:
        # Legacy fallback: top-level text_label
        task_candidates_old = item.get("text_label")
        if task_candidates_old:
            task = _first_str(task_candidates_old, "text_label (fallback)")
    if task is None:
        raise ValueError(
            "Failed to extract 'task'. Please check 'text_label' or 'text_annotation.text_label'."
        )

    # Extract reference object description (target_en)
    if isinstance(text_anno, dict):
        # New schema: text_annotation
        target_candidates = text_anno.get("reference_object_description_en")
        if target_candidates:
            target_en = _first_str(target_candidates, "text_annotation.reference_object_description_en")
    if target_en is None:
        # Legacy fallback: mask_annotation
        if isinstance(mask_anno, dict):
            target_candidates_old = mask_anno.get("target_en")
            if target_candidates_old:
                target_en = _first_str(target_candidates_old, "mask_annotation.target_en (fallback)")
    if target_en is None:
        # If missing in both schemas, fail
        raise ValueError(
            "Failed to extract 'target_en'. Please check 'mask_annotation.target_en' or "
            "'text_annotation.reference_object_description_en'."
        )

    # 3) Optional file existence checks
    if check_files_exist:
        missing = [p for p in (img1, img2) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    # 4) Build prompt/messages
    system_head, system_tail = split_system_prompt(system_prompt)

    messages = []
    if use_system_prompt and system_head:
        messages.append({"role": "system", "content": system_head})

    messages.append({
        "role": "user",
        "content": build_user_text(
            system_tail=system_tail,
            response_format=response_format,
            target_en=target_en,
            task=task,
        )
    })

    return {
        "messages": messages,
        "images": [img1, img2],
    }

def iter_json_files(root_dir: str, pattern: str):
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    for p in sorted(root.rglob("*.json")):
        if pattern and not fnmatch.fnmatch(p.name, pattern):
            continue
        yield p


def convert_folder_recursive(
    input_root: str,
    output_path: str,
    system_prompt: str,
    response_format: str,
    pattern: str,
    use_system_prompt: bool,
    skip_bad: bool,
    check_files_exist: bool,
) -> int:
    collected = []
    n_total = 0

    for fp in iter_json_files(input_root, pattern=pattern):
        n_total += 1
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)

            swift_item = one_sample_to_swift(
                data,
                system_prompt=system_prompt,
                response_format=response_format,
                use_system_prompt=use_system_prompt,
                check_files_exist=check_files_exist,
            )
            collected.append(swift_item)

        except Exception as e:
            if skip_bad:
                print(f"Skip {fp}: {e}", file=sys.stderr)
            else:
                raise

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    print(f"JSON files scanned: {n_total}, converted: {len(collected)}", file=sys.stderr)
    return len(collected)


def main():
    parser = argparse.ArgumentParser(description="Recursively convert AffordanceRegion JSON to ms-swift inputs")
    parser.add_argument("input_dir", help="Root directory containing sample JSON files (recursive)")
    parser.add_argument("-o", "--output", default="swift_affordance_input.json")
    parser.add_argument("--pattern", default="*.json")

    parser.add_argument("--system_var", default="system_prompt_base",
                        help="System prompt variable name in prompts/AffordanceRegion.py")
    parser.add_argument("--response_var", default="response_base",
                        help="Response format variable name in prompts/AffordanceRegion.py")

    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--no-skip-bad", action="store_true")
    parser.add_argument("--check-files-exist", action="store_true")

    args = parser.parse_args()

    if not hasattr(AR, args.system_var):
        raise ValueError(f"prompts.AffordanceRegion has no attribute: {args.system_var}")
    if not hasattr(AR, args.response_var):
        raise ValueError(f"prompts.AffordanceRegion has no attribute: {args.response_var}")

    system_prompt = getattr(AR, args.system_var)
    response_format = getattr(AR, args.response_var)

    if not isinstance(system_prompt, str):
        raise TypeError(f"{args.system_var} is not a str")
    if not isinstance(response_format, str):
        raise TypeError(f"{args.response_var} is not a str")

    n = convert_folder_recursive(
        input_root=args.input_dir,
        output_path=args.output,
        system_prompt=system_prompt,
        response_format=response_format,
        pattern=args.pattern,
        use_system_prompt=not args.no_system_prompt,
        skip_bad=not args.no_skip_bad,
        check_files_exist=args.check_files_exist,
    )

    print(f"Converted {n} samples -> {args.output}")


if __name__ == "__main__":
    main()