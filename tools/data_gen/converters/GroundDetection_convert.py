#!/usr/bin/env python3
"""
Convert GroundDetection annotations to ms-swift / Qwen-VL inference JSON.

- Single image: images: [img]
- Prompt rule:
  - system: first line of system_prompt
  - user: remaining system_prompt + response_format + <image> (at the end)
"""

import json
import os
import sys
import argparse
import fnmatch
from pathlib import Path

import prompts.GroundDetection as GD


def _abs_path(p: str) -> str:
    p = str(p)
    return os.path.abspath(p) if not os.path.isabs(p) else p


def split_system_prompt(system_prompt: str):
    sp = (system_prompt or "").strip()
    if not sp:
        return "", ""
    parts = sp.split("\n", 1)
    head = parts[0].strip()
    tail = parts[1].strip() if len(parts) > 1 else ""
    return head, tail


def build_user_text(system_tail: str, response_format: str) -> str:
    chunks = []

    if system_tail:
        chunks.append(system_tail.strip())

    # Always append response_format if non-empty
    if response_format and response_format.strip():
        chunks.append(response_format.strip())

    # Put image placeholder at the end
    chunks.append(" <image> ")

    return "\n".join(chunks).strip()


def one_sample_to_swift(
    item: dict,
    system_prompt: str,
    response_format: str,
    use_system_prompt: bool,
    response_format_in_user: bool,  # legacy flag (ignored; always appends response_format)
    check_files_exist: bool,
) -> dict:
    image_paths = item.get("image_path_saved") or []
    if not image_paths:
        raise ValueError("Missing or empty field: image_path_saved")

    img = _abs_path(str(image_paths[0]))

    if check_files_exist and (not os.path.isfile(img)):
        raise FileNotFoundError(f"Missing file: {img}")

    system_head, system_tail = split_system_prompt(system_prompt)

    messages = []
    if use_system_prompt and system_head:
        messages.append({"role": "system", "content": system_head})

    messages.append({
        "role": "user",
        "content": build_user_text(system_tail=system_tail, response_format=response_format),
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

    for fp in iter_json_files(input_root, pattern):
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
    parser = argparse.ArgumentParser(description="Recursively convert GroundDetection JSON to ms-swift inputs")
    parser.add_argument("input_dir")
    parser.add_argument("-o", "--output", default="swift_gd_input.json")
    parser.add_argument("--pattern", default="*.json")

    parser.add_argument("--system_var", default="system_prompt_base")
    parser.add_argument("--response_var", default="response_base")

    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--response-format-in-user", action="store_true")  # legacy flag (ignored)
    parser.add_argument("--no-skip-bad", action="store_true")
    parser.add_argument("--check-files-exist", action="store_true")

    args = parser.parse_args()

    if not hasattr(GD, args.system_var):
        raise ValueError(f"prompts.GroundDetection has no attribute: {args.system_var}")
    if not hasattr(GD, args.response_var):
        raise ValueError(f"prompts.GroundDetection has no attribute: {args.response_var}")

    system_prompt = getattr(GD, args.system_var)
    response_format = getattr(GD, args.response_var)

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
        response_format_in_user=args.response_format_in_user,
        skip_bad=not args.no_skip_bad,
        check_files_exist=args.check_files_exist,
    )

    print(f"Converted {n} samples -> {args.output}")


if __name__ == "__main__":
    main()