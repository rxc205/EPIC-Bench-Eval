#!/usr/bin/env python3
"""Convert EPIC-Bench ContactRelationship annotations (Type 2/3) to ms-swift inference JSON."""

import json
import os
import sys
import argparse
import fnmatch
import re
from pathlib import Path

import prompts.ContactRelationship as CR


def _abs_path(p: str) -> str:
    p = str(p)
    return os.path.abspath(p) if not os.path.isabs(p) else p


def _derive_object_overlay_path(img1_saved: str) -> str:
    """
    .../1085_image.jpg -> .../1085_object_overlay.png
    """
    if "_image" not in img1_saved:
        raise ValueError(f"Filename does not contain '_image': {img1_saved}")
    base = img1_saved.rsplit("_image", 1)[0]
    return base + "_reference_object_overlay.png"


def _first_str(x, field_name: str) -> str:
    """Return the first element if list; otherwise cast to str."""
    if x is None:
        raise ValueError(f"Missing field: {field_name}")
    if isinstance(x, list):
        if not x:
            raise ValueError(f"Empty field: {field_name}")
        return str(x[0])
    return str(x)


def normalize_text_label(s: str) -> str:
    """Normalize text_label by treating repeated '2' as a separator (', ')."""
    s = str(s).strip()
    if not s:
        return s

    s = re.sub(r"2+", ", ", s)            # repeated '2' -> ", "
    s = re.sub(r"\s*,\s*", ", ", s)       # normalize comma spacing
    s = re.sub(r"\s+", " ", s).strip()    # collapse whitespace
    return s


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


def build_user_text(system_tail: str, response_format: str, reference_desc: str) -> str:
    """
    USER:
      {system_tail}
      {response_format}
      <image>
      <image>
      Reference Object Description: {cleaned_text_label}
    """
    chunks = []

    if system_tail:
        chunks.append(system_tail.strip())

    if response_format and response_format.strip():
        chunks.append(response_format.strip())

    chunks.append(" <image> \n <image> ")
    chunks.append(f"Reference Object Description: {reference_desc}")

    return "\n".join(chunks).strip()


def one_sample_to_swift(
    item: dict,
    system_prompt: str,
    response_format: str,
    use_system_prompt: bool,
    response_format_in_system: bool,  # legacy flag (ignored)
    response_format_in_user: bool,    # legacy flag (ignored)
    check_files_exist: bool,
) -> dict:
    image_paths = item.get("image_path_saved") or []
    if not image_paths:
        raise ValueError("Missing or empty field: image_path_saved")

    img1 = _abs_path(str(image_paths[0]))
    img2 = _abs_path(_derive_object_overlay_path(img1))

    # Extract text_label from both new and legacy schemas
    text_label_value = None
    # New schema: text_annotation.text_label
    if "text_annotation" in item and isinstance(item["text_annotation"], dict):
        text_label_value = item["text_annotation"].get("text_label")
    # Legacy schema: top-level text_label
    if text_label_value is None:
        text_label_value = item.get("text_label")
    # If both are missing, raise.
    if text_label_value is None:
        raise ValueError(
            "Failed to extract 'text_label'. Please check 'text_annotation.text_label' "
            "or the top-level 'text_label' field."
        )
    # Use helper to take the first element and apply normalization rules
    reference_desc_raw = _first_str(text_label_value, "text_label")
    reference_desc = normalize_text_label(reference_desc_raw)
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
            reference_desc=reference_desc,
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
    response_format_in_system: bool,
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
                    data,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    use_system_prompt=use_system_prompt,
                    response_format_in_system=response_format_in_system,
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
        description="Recursively convert ContactRelationship JSON to ms-swift/Qwen-VL inputs"
    )
    parser.add_argument("input_dir")
    parser.add_argument("-o", "--output", default="swift_contact_input.json")
    parser.add_argument("--pattern", default="*.json")

    parser.add_argument("--system_var", default="system_prompt_base")
    parser.add_argument("--response_var", default="response_base")

    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--no-response-format-in-system", action="store_true")  # legacy flag (ignored)
    parser.add_argument("--response-format-in-user", action="store_true")       # legacy flag (ignored)
    parser.add_argument("--no-skip-bad", action="store_true")
    parser.add_argument("--check-files-exist", action="store_true")

    args = parser.parse_args()

    if not hasattr(CR, args.system_var):
        raise ValueError(f"prompts.ContactRelationship has no attribute: {args.system_var}")
    if not hasattr(CR, args.response_var):
        raise ValueError(f"prompts.ContactRelationship has no attribute: {args.response_var}")

    system_prompt = getattr(CR, args.system_var)
    response_format = getattr(CR, args.response_var)

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
        response_format_in_system=not args.no_response_format_in_system,
        response_format_in_user=args.response_format_in_user,
        skip_bad=not args.no_skip_bad,
        check_files_exist=args.check_files_exist,
    )

    print(f"Converted {n} samples -> {args.output}")


if __name__ == "__main__":
    main()