#!/usr/bin/env python3
"""
Convert VisualMatching annotations to ms-swift / Qwen-VL inference JSON (three images).

Input requirements:
- image_path_saved: list[str] (at least 2 images)

We build three images:
- img1: image_path_saved[0]
- img2: derived from img1 by replacing '_image' with '_reference_overlay.png'
- img3: image_path_saved[1]

Prompt rule:
- system: first line of system_prompt
- user: remaining system_prompt + response_format + <image><image><image> (at the end)
"""

import json
import os
import sys
import argparse
import fnmatch
from pathlib import Path

import prompts.VisualMatching as VM  # canonical prompts namespace


def _abs_path(p: str) -> str:
    p = str(p)
    return os.path.abspath(p) if not os.path.isabs(p) else p


def _derive_reference_overlay_path(img1_saved: str) -> str:
    """
    ..._image.jpg -> ..._reference_overlay.png
    """
    if "_image" not in img1_saved:
        raise ValueError(f"Filename does not contain '_image': {img1_saved}")
    base = img1_saved.rsplit("_image", 1)[0]
    return base + "_reference_area_overlay.png"


def split_system_prompt(system_prompt: str):
    sp = (system_prompt or "").strip()
    if not sp:
        return "", ""
    parts = sp.split("\n", 1)
    head = parts[0].strip()
    tail = parts[1].strip() if len(parts) > 1 else ""
    return head, tail


def build_user_text(system_tail: str, response_format: str) -> str:
    """
    USER:
      system_tail
      response_format   (append if non-empty)
      <image>
      <image>
      <image>           (at the end)
    """
    chunks = []

    if system_tail and system_tail.strip():
        chunks.append(system_tail.strip())

    # Append response_format if non-empty
    if response_format and response_format.strip():
        chunks.append(response_format.strip())

    # Put image placeholders at the end
    chunks.append(" <image> \n <image> \n <image> ")

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
    if len(image_paths) < 2:
        raise ValueError("image_path_saved has fewer than 2 images")

    img1 = _abs_path(str(image_paths[0]))
    img3 = _abs_path(str(image_paths[1]))
    img2 = _abs_path(_derive_reference_overlay_path(img1))

    if check_files_exist:
        missing = [p for p in (img1, img2, img3) if not os.path.isfile(p)]
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
        ),
    })

    return {
        "messages": messages,
        "images": [img1, img2, img3],
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
                    response_format_in_user=response_format_in_user,  # legacy flag (ignored)
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
        description="Recursively convert VisualMatching JSON to ms-swift / Qwen-VL inference inputs"
    )
    parser.add_argument("input_dir", help="Root directory containing sample JSON files (recursive)")
    parser.add_argument("-o", "--output", default="swift_vm_input.json")
    parser.add_argument("--pattern", default="*.json")

    parser.add_argument("--system_var", default="system_prompt_base")
    parser.add_argument("--response_var", default="response_base")

    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--no-response-format-in-system", action="store_true",
                        help="legacy flag (ignored)")
    parser.add_argument("--response-format-in-user", action="store_true",
                        help="legacy flag (ignored)")
    parser.add_argument("--no-skip-bad", action="store_true")
    parser.add_argument("--check-files-exist", action="store_true")

    args = parser.parse_args()

    if not hasattr(VM, args.system_var):
        raise ValueError(f"prompts.VisualMatching has no attribute: {args.system_var}")
    if not hasattr(VM, args.response_var):
        raise ValueError(f"prompts.VisualMatching has no attribute: {args.response_var}")

    system_prompt = getattr(VM, args.system_var)
    response_format = getattr(VM, args.response_var)

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

