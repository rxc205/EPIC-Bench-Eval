#!/usr/bin/env bash
#
# EPIC-Bench: launch Streamlit visualization (tools/visualization/app.py).
# By default, scans outputs/scores and pre-fills the result JSON/JSONL path.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP="${REPO_ROOT}/tools/visualization/app.py"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/visualization.sh [--help] [streamlit pass-through args ...]

Starts Streamlit with tools/visualization/app.py.

Environment:
  EPIC_VIS_SCORES_ROOT   Directory scanned for a default result file.
                         Default: <repo>/outputs/scores
                         The app picks the first match:
                         *_full.json, *_full.jsonl, *.json, *.jsonl
  EPIC_VIS_RESULTS_JSON  If set to an existing file, used as the default
                         path (overrides EPIC_VIS_SCORES_ROOT scanning).
  PORT                   If set, equivalent to --server.port "$PORT"

Examples:
  bash scripts/visualization.sh
  PORT=8502 bash scripts/visualization.sh
  EPIC_VIS_RESULTS_JSON=/path/to/Qwen3-VL-8B_full.json bash scripts/visualization.sh
  bash scripts/visualization.sh --server.port 8502 --browser.gatherUsageStats false
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${APP}" ]]; then
  echo "ERROR: visualization app not found: ${APP}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export EPIC_VIS_SCORES_ROOT="${EPIC_VIS_SCORES_ROOT:-${REPO_ROOT}/outputs/scores}"

STREAMLIT_EXTRA=()
if [[ -n "${PORT:-}" ]]; then
  STREAMLIT_EXTRA+=(--server.port "${PORT}")
fi

cd "${REPO_ROOT}"
exec streamlit run "${APP}" "${STREAMLIT_EXTRA[@]}" "$@"
