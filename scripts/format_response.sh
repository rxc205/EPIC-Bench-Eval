#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/format_response.sh [--in <PATH>] [--out <PATH>] [--format-type <auto|bbox|point|mask|img>] [--coordinate-system <auto|absolute|normalized_0_1|normalized_0_1000>]

Std-format outputs are JSONL by default (same layout as input; *.json sources -> *.jsonl).

Description:
  Convert all raw ms-swift inference responses under an input directory into the
  EPIC-Bench standard format, while preserving the original directory structure.

Defaults:
  --in   outputs/model_response/swift_format
  --out  outputs/model_response/std_format

Notes:
  - Input files may be *.jsonl or *.json (e.g. swift batch *.json).
  - Std-format outputs are always JSONL (--jsonl): *.json inputs are written as *.jsonl
    next to the same relative path.
EOF
}

# Resolve repo root (walk upwards until dataset/ exists)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ "${REPO_ROOT}" != "/" && ! -d "${REPO_ROOT}/dataset" ]]; do
  REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
done

IN_DIR="${REPO_ROOT}/outputs/model_response/swift_format"
OUT_DIR="${REPO_ROOT}/outputs/model_response/std_format"
FORMAT_TYPE="${FORMAT_TYPE:-auto}"
COORD_SYSTEM="${COORD_SYSTEM:-auto}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in)
      IN_DIR="${2:-}"; shift 2 ;;
    --out)
      OUT_DIR="${2:-}"; shift 2 ;;
    --format-type)
      FORMAT_TYPE="${2:-}"; shift 2 ;;
    --coordinate-system)
      COORD_SYSTEM="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

FORMATTER="${REPO_ROOT}/tools/formatting/format_response.py"
if [[ ! -f "${FORMATTER}" ]]; then
  echo "ERROR: formatter not found: ${FORMATTER}" >&2
  exit 2
fi

if [[ ! -d "${IN_DIR}" ]]; then
  echo "ERROR: input directory not found: ${IN_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "=============================="
echo "EPIC-Bench | Format responses"
echo "Repo root  : ${REPO_ROOT}"
echo "Input dir  : ${IN_DIR}"
echo "Output dir : ${OUT_DIR}"
echo "FormatType : ${FORMAT_TYPE}"
echo "CoordSys   : ${COORD_SYSTEM}"
echo "Output fmt : jsonl (default)"
echo "=============================="

shopt -s nullglob

count=0
fail=0

# Find jsonl/json files and keep relative layout under OUT_DIR
while IFS= read -r -d '' f; do
  rel="${f#${IN_DIR}/}"
  out_path="${OUT_DIR}/${rel}"
  # Default std_format output is JSONL: map *.json -> *.jsonl (keep directory layout).
  if [[ "${out_path}" == *.json ]] && [[ "${out_path}" != *.jsonl ]]; then
    out_path="${out_path%.json}.jsonl"
  fi
  out_parent="$(dirname "${out_path}")"
  mkdir -p "${out_parent}"

  echo ""
  echo ">>> ${rel} -> ${out_path#${OUT_DIR}/}"

  extra_args=(--jsonl)
  [[ -n "${FORMAT_TYPE}" ]] && extra_args+=(--format-type "${FORMAT_TYPE}")
  [[ -n "${COORD_SYSTEM}" ]] && extra_args+=(--coordinate-system "${COORD_SYSTEM}")

  python3 "${FORMATTER}" "${f}" -o "${out_path}" "${extra_args[@]}" || fail=$((fail+1))

  count=$((count+1))
done < <(find "${IN_DIR}" -type f \( -name "*.jsonl" -o -name "*.json" \) -print0)

echo ""
echo "=============================="
echo "Processed: ${count}"
echo "Failed   : ${fail}"
echo "Output   : ${OUT_DIR}"
echo "=============================="

if (( fail > 0 )); then
  exit 1
fi

