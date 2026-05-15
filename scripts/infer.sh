#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/infer.sh --model <MODEL_FAMILY> [--data <PATH>] [--out <PATH>] [--script <FILE>|--all]

Description:
  Run model inference by calling example scripts under scripts/infer/<MODEL_FAMILY>/.
  The infer scripts are expected to support:
    --val_dataset <PATH>   (directory or file; directory means *.json)
    --result_path <PATH>   (output parent directory)

Options:
  --model <name>    Model family directory name under scripts/infer/ (required)
  --data <path>     Swift-format inference data path.
                   Default: dataset/swift_data/EPIC_Bench
  --out <path>      Output base directory (parent) for model responses.
                   Default: outputs/model_response/swift_format
  --script <file>   Run only one script (file name under scripts/infer/<family>/)
  --all             Run all *.sh scripts under scripts/infer/<family>/
  -h, --help        Show help

Examples:
  bash scripts/infer.sh --model Qwen3_VL
  bash scripts/infer.sh --model InternVL3 --data dataset/swift_data/EPIC_Bench --out outputs/model_response/swift_format
  bash scripts/infer.sh --model Gemma --script infer_gemma_3_4b_it.sh
  bash scripts/infer.sh --model Qwen3.5 --all
EOF
}

# Resolve repo root (walk upwards until dataset/ exists)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ "${REPO_ROOT}" != "/" && ! -d "${REPO_ROOT}/dataset" ]]; do
  REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
done

MODEL=""
DATA_PATH="${REPO_ROOT}/dataset/swift_data/EPIC_Bench"
OUT_BASE="${REPO_ROOT}/outputs/model_response/swift_format"
ONE_SCRIPT=""
RUN_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"; shift 2 ;;
    --family)
      # Backward compatible alias
      MODEL="${2:-}"; shift 2 ;;
    --data)
      DATA_PATH="${2:-}"; shift 2 ;;
    --out)
      OUT_BASE="${2:-}"; shift 2 ;;
    --script)
      ONE_SCRIPT="${2:-}"; shift 2 ;;
    --all)
      RUN_ALL=1; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "ERROR: --model is required." >&2
  echo "" >&2
  echo "Available families:" >&2
  (cd "${REPO_ROOT}" && ls -1 scripts/infer 2>/dev/null || true) >&2
  exit 2
fi

INFER_DIR="${REPO_ROOT}/scripts/infer/${MODEL}"
if [[ ! -d "${INFER_DIR}" ]]; then
  echo "ERROR: infer family dir not found: ${INFER_DIR}" >&2
  echo "" >&2
  echo "Available families:" >&2
  (cd "${REPO_ROOT}" && ls -1 scripts/infer 2>/dev/null || true) >&2
  exit 2
fi

if [[ ! -e "${DATA_PATH}" ]]; then
  echo "ERROR: data path not found: ${DATA_PATH}" >&2
  exit 2
fi

mkdir -p "${OUT_BASE}"

select_scripts() {
  if [[ -n "${ONE_SCRIPT}" ]]; then
    echo "${INFER_DIR}/${ONE_SCRIPT}"
    return 0
  fi

  if [[ "${RUN_ALL}" == "1" ]]; then
    ls -1 "${INFER_DIR}"/*.sh 2>/dev/null || true
    return 0
  fi

  # Default: if only one script exists, run it; otherwise ask user to specify.
  local count
  count="$(ls -1 "${INFER_DIR}"/*.sh 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${count}" == "1" ]]; then
    ls -1 "${INFER_DIR}"/*.sh
    return 0
  fi

  echo "ERROR: multiple infer scripts found under ${INFER_DIR}." >&2
  echo "Please specify one with --script <file> or run all with --all." >&2
  echo "" >&2
  ls -1 "${INFER_DIR}"/*.sh 2>/dev/null | sed 's|.*/||' >&2 || true
  exit 2
}

echo "=============================="
echo "EPIC-Bench | Inference runner"
echo "Repo root : ${REPO_ROOT}"
echo "Model     : ${MODEL}"
echo "Infer dir : ${INFER_DIR}"
echo "Data path : ${DATA_PATH}"
echo "Out base  : ${OUT_BASE}"
echo "=============================="

mapfile -t SCRIPTS < <(select_scripts)
if (( ${#SCRIPTS[@]} == 0 )); then
  echo "ERROR: no infer scripts found in ${INFER_DIR}" >&2
  exit 2
fi

for s in "${SCRIPTS[@]}"; do
  if [[ ! -f "${s}" ]]; then
    echo "ERROR: infer script not found: ${s}" >&2
    exit 2
  fi

  chmod +x "${s}" || true
  name="$(basename "${s}")"
  echo ""
  echo ">>> Running: ${name}"

  # Pass user-defined paths in a way that the example infer scripts understand.
  # NOTE: infer scripts may internally decide output file name based on model_series/model_name,
  # and will place it under --result_path/<model_series>/<model_name>.jsonl.
  bash "${s}" \
    --val_dataset "${DATA_PATH}" \
    --result_path "${OUT_BASE}"
done

echo ""
echo "=============================="
echo "Done. Outputs under: ${OUT_BASE}"
echo "=============================="

