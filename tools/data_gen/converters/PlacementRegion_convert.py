#!/usr/bin/env python3
"""
Convert PlacementRegion annotations to ms-swift / Qwen-VL inference JSON (two images).

Input requirements:
- image_path_saved: list[str] (use the first)
- reference object description and placement region description:
  - preferred: text_annotation.reference_object_description_en / text_annotation.placement_region_description_en
  - fallback:  mask_annotation.target_en / mask_annotation.placement_region_en

We build two images:
- img1: image_path_saved[0]
- img2: derived from img1 by replacing the last '_image' suffix with '_target_object_mask.png'

Prompt rule:
- system: first line of system_prompt
- user: remaining system_prompt + response_format + <image><image> + descriptions
"""

import json
import os
import sys
import argparse
import fnmatch
from pathlib import Path

import prompts.PlacementRegion as PR


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


def _derive_target_object_mask_path(img1_saved: str) -> str:
    """
    ..._image.jpg -> ..._target_object_mask.png
    """
    if "_image" not in img1_saved:
        raise ValueError(f"Filename does not contain '_image': {img1_saved}")
    base = img1_saved.rsplit("_image", 1)[0]
    return base + "_reference_object_overlay.png"


def split_system_prompt(system_prompt: str):
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
    place_region_en: str,
) -> str:
    """
    USER:
      system_tail
      response_format (append if non-empty)
      <image>
      <image>
      Reference Object Description: ...
      Placement Region Description: ...
    """
    chunks = []

    if system_tail:
        chunks.append(system_tail.strip())

    # Append response_format if non-empty
    if response_format and response_format.strip():
        chunks.append(response_format.strip())

    # Put image placeholders before the description fields
    chunks.append(" <image> \n <image> ")

    chunks.append(f"Reference Object Description: {target_en}")
    chunks.append(f"Placement Region Description: {place_region_en}")

    return "\n".join(chunks).strip()


def one_sample_to_swift(
    item: dict,
    system_prompt: str,
    response_format: str,
    use_system_prompt: bool,
    response_format_in_user: bool,  # legacy flag (ignored)
    check_files_exist: bool,
) -> dict:
    image_paths = item.get("image_path_saved") or []
    if not image_paths:
        raise ValueError("Missing or empty field: image_path_saved")

    img1 = _abs_path(str(image_paths[0]))
    img2 = _abs_path(_derive_target_object_mask_path(img1))

    # Extract fields from both new and legacy schemas
    target_en_value = None
    placement_region_en_value = None

    # New schema: fields under text_annotation
    if "text_annotation" in item and isinstance(item["text_annotation"], dict):
        text_anno = item["text_annotation"]
        target_en_value = text_anno.get("reference_object_description_en")
        placement_region_en_value = text_anno.get("placement_region_description_en")

    # Legacy schema: fields under mask_annotation
    ma = item.get("mask_annotation") or {}
    if target_en_value is None:
        target_en_value = ma.get("target_en")
    if placement_region_en_value is None:
        placement_region_en_value = ma.get("placement_region_en")

    # If both are missing, raise.
    if target_en_value is None:
        raise ValueError(
            "Failed to extract reference object description. Please check "
            "'text_annotation.reference_object_description_en' or 'mask_annotation.target_en'."
        )
    if placement_region_en_value is None:
        raise ValueError(
            "Failed to extract placement region description. Please check "
            "'text_annotation.placement_region_description_en' or 'mask_annotation.placement_region_en'."
        )

    # Use helper to take the first element
    target_en = _first_str(target_en_value, "reference_object_description_en / target_en")
    place_region_en = _first_str(placement_region_en_value, "placement_region_description_en / placement_region_en")
    # End schema compatibility block

    if check_files_exist:
        missing = [p for p in (img1, img2) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

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
            place_region_en=place_region_en,
        ),
    })

    return {"messages": messages, "images": [img1, img2]}


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
    response_format_in_user: bool,
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

            collected.append(
                one_sample_to_swift(
                    item=data,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    use_system_prompt=use_system_prompt,
                    response_format_in_user=response_format_in_user,
                    check_files_exist=check_files_exist,
                )
            )

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
    parser = argparse.ArgumentParser(
        description="Recursively convert PlacementRegion JSON to ms-swift / Qwen-VL inference inputs"
    )
    parser.add_argument("input_dir", help="Root directory containing sample JSON files (recursive)")
    parser.add_argument("-o", "--output", default="swift_place_region_input.json")
    parser.add_argument("--pattern", default="*.json")

    parser.add_argument("--system_var", default="system_prompt_base",
                        help="System prompt variable name in prompts/PlacementRegion.py")
    parser.add_argument("--response_var", default="response_base",
                        help="Response format variable name in prompts/PlacementRegion.py")

    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--no-response-format-in-system", action="store_true")  # legacy flag (ignored)
    parser.add_argument("--response-format-in-user", action="store_true")       # legacy flag (ignored)
    parser.add_argument("--no-skip-bad", action="store_true")
    parser.add_argument("--check-files-exist", action="store_true")

    args = parser.parse_args()

    if not hasattr(PR, args.system_var):
        raise ValueError(f"prompts.PlacementRegion has no attribute: {args.system_var}")
    if not hasattr(PR, args.response_var):
        raise ValueError(f"prompts.PlacementRegion has no attribute: {args.response_var}")

    system_prompt = getattr(PR, args.system_var)
    response_format = getattr(PR, args.response_var)

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
        response_format_in_user=args.response_format_in_user,  # legacy flag (ignored)
        skip_bad=not args.no_skip_bad,
        check_files_exist=args.check_files_exist,
    )

    print(f"Converted {n} samples -> {args.output}")


if __name__ == "__main__":
    main()