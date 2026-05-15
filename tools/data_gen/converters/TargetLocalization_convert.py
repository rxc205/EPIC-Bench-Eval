#!/usr/bin/env python3
"""
Convert EPIC-Bench TargetLocalization annotations to ms-swift / Qwen-VL inference JSON.

Input requirements (per sample JSON):
- image_path_saved: list[str] (use the first)
- text_label: list[str] or text_annotation.text_label

Output (per record):
{
  "messages": [{"role": "system", ...}, {"role": "user", ...}],
  "images": ["/abs/path/to/image.jpg"]
}
"""

import json
import os
import sys
import argparse
import fnmatch
from pathlib import Path

import prompts.TargetLocalization as TL  # canonical prompts namespace


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
    text_label: str,
) -> str:
    """
    USER:
      system_tail
      response_format   (append if non-empty)
      <image>           (appear before Target Object Description)
      Target Object Description: ...
    """
    chunks = []

    if system_tail:
        chunks.append(system_tail.strip())

    # Append response_format if non-empty
    if response_format and response_format.strip():
        chunks.append(response_format.strip())

    # Put image placeholder before the description field
    chunks.append(" <image> ")
    chunks.append(f"Target Object Description: {text_label}")

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

    img = _abs_path(str(image_paths[0]))
    if check_files_exist and (not os.path.isfile(img)):
        raise FileNotFoundError(f"Missing file: {img}")

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
    # Use helper to take the first element
    text_label = _first_str(text_label_value, "text_label")
    # End schema compatibility block

    system_head, system_tail = split_system_prompt(system_prompt)

    messages = []
    if use_system_prompt and system_head:
        messages.append({"role": "system", "content": system_head})

    messages.append({
        "role": "user",
        "content": build_user_text(
            system_tail=system_tail,
            response_format=response_format,
            text_label=text_label,
        ),
    })

    return {"messages": messages, "images": [img]}


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
        description="Recursively convert annotation JSON to ms-swift / Qwen-VL inference inputs"
    )
    parser.add_argument("input_dir", help="Root directory containing sample JSON files (recursive)")
    parser.add_argument("-o", "--output", default="swift_inference_input.json", help="Output JSON path")
    parser.add_argument("--pattern", default="*.json", help="Filename glob pattern (default: *.json)")

    parser.add_argument("--system_var", default="system_prompt_base",
                        help="System prompt variable name in prompts/TargetLocalization.py (e.g., system_prompt_base)")
    parser.add_argument("--response_var", default="response_base",
                        help="Response format variable name in prompts/TargetLocalization.py (e.g., response_base)")

    parser.add_argument("--no-system-prompt", action="store_true",
                        help="Do not add system prompt (omit system head)")

    # Legacy flags (ignored)
    parser.add_argument("--no-response-format-in-system", action="store_true",
                        help="legacy flag (ignored)")
    parser.add_argument("--response-format-in-user", action="store_true",
                        help="legacy flag (ignored)")

    parser.add_argument("--no-skip-bad", action="store_true", help="Do not skip bad samples; fail fast")
    parser.add_argument("--check-files-exist", action="store_true", help="Check that image files exist")

    args = parser.parse_args()

    if not hasattr(TL, args.system_var):
        raise ValueError(f"prompts.TargetLocalization has no attribute: {args.system_var}")
    if not hasattr(TL, args.response_var):
        raise ValueError(f"prompts.TargetLocalization has no attribute: {args.response_var}")

    system_prompt = getattr(TL, args.system_var)
    response_format = getattr(TL, args.response_var)

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