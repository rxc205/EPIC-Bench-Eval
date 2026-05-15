#!/usr/bin/env bash
#
# EPIC-Bench: score std-format model responses (batch or single-file).
# - Batch mode: epic_eval.run_batch_evaluation (calls epic_eval.evaluate per JSONL)
# - Single-file mode: epic_eval.evaluate
#
# Default input : outputs/model_response/std_format
# Default output: outputs/scores (preserves input directory layout)
#

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate.sh [--in DIR] [--out DIR] [extra args ...]
  bash scripts/evaluate.sh --single FILE [--single-out PATH] [extra args ...]

Options:
  --in, --input    Std-format response root (recursively scans *.jsonl). Default: outputs/model_response/std_format
  --out, --output  Score output root. Default: outputs/scores
  --single         Evaluate a single std-format file (python -m epic_eval.evaluate)
  --single-out     Output path in single-file mode (default: <out>/<basename>_full.json or ..._full.jsonl with --jsonl)

Any additional arguments are forwarded to:
  - epic_eval.run_batch_evaluation   (batch mode)
  - epic_eval.evaluate               (single-file mode)

JSONL output (one summary line + one line per sample):
  bash scripts/evaluate.sh --jsonl
  bash scripts/evaluate.sh --single YOUR.jsonl --jsonl
  # Or: -o path/to/out.jsonl (infer JSONL from extension)

Examples:
  bash scripts/evaluate.sh
  bash scripts/evaluate.sh --in outputs/model_response/std_format --out outputs/scores --parallel 4
  bash scripts/evaluate.sh --coord-mode normalized_0_1000
  bash scripts/evaluate.sh --jsonl --parallel 4
  bash scripts/evaluate.sh --single outputs/model_response/std_format/Qwen3_VL/Qwen3-VL-8B.jsonl --jsonl
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

abspath() {
  python3 - <<'PY' "$1" "${REPO_ROOT}"
import os, sys
p, base = sys.argv[1], sys.argv[2]
if not os.path.isabs(p):
    p = os.path.join(base, p)
print(os.path.abspath(p))
PY
}

IN_DIR="$(abspath "${REPO_ROOT}/outputs/model_response/std_format")"
OUT_DIR="$(abspath "${REPO_ROOT}/outputs/scores")"
SINGLE=""
SINGLE_OUT=""
JSONL=0

PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --in|--input)
      IN_DIR="$(abspath "${2:-}")"
      shift 2
      ;;
    --out|--output)
      OUT_DIR="$(abspath "${2:-}")"
      shift 2
      ;;
    --single)
      SINGLE="$(abspath "${2:-}")"
      shift 2
      ;;
    --single-out)
      SINGLE_OUT="${2:-}"
      shift 2
      ;;
    --jsonl)
      JSONL=1
      PASSTHROUGH+=("--jsonl")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

if [[ -n "${SINGLE}" ]]; then
  if [[ ! -f "${SINGLE}" ]]; then
    echo "ERROR: --single file not found: ${SINGLE}" >&2
    exit 2
  fi
  base="$(basename "${SINGLE}" .jsonl)"
  base="${base%.json}"
  if [[ -n "${SINGLE_OUT}" ]]; then
    out_json="$(abspath "${SINGLE_OUT}")"
  elif [[ "${JSONL}" == "1" ]]; then
    out_json="${OUT_DIR}/${base}_full.jsonl"
  else
    out_json="${OUT_DIR}/${base}_full.json"
  fi
  mkdir -p "$(dirname "${out_json}")"

  echo "=============================="
  echo "EPIC-Bench | Single-file evaluate"
  echo "Input : ${SINGLE}"
  echo "Output: ${out_json}"
  [[ "${JSONL}" == "1" ]] && echo "Format: JSONL"
  echo "=============================="

  exec python3 -m epic_eval.evaluate "${SINGLE}" -o "${out_json}" "${PASSTHROUGH[@]}"
fi

if [[ ! -d "${IN_DIR}" ]]; then
  echo "ERROR: input directory not found: ${IN_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "=============================="
echo "EPIC-Bench | Batch evaluate"
echo "Input root : ${IN_DIR}"
echo "Output root: ${OUT_DIR}"
echo "=============================="

exec python3 -m epic_eval.run_batch_evaluation \
  --input "${IN_DIR}" \
  --output "${OUT_DIR}" \
  "${PASSTHROUGH[@]}"
