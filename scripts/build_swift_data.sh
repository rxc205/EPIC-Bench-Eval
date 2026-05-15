#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default dataset layout (relative to repo root):
#   dataset/annotation/EPIC_Bench
#   dataset/swift_data/EPIC_Bench

# ANN_ROOT="${ANN_ROOT:-${REPO_ROOT}/examples/annotation_sample/EPIC_Bench}"
# OUT_DIR="${OUT_DIR:-${REPO_ROOT}/examples/swift_data_sample/EPIC_Bench}"

ANN_ROOT="${ANN_ROOT:-${REPO_ROOT}/dataset/annotation/EPIC_Bench}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dataset/swift_data/EPIC_Bench}"

TOOL_ROOT="${TOOL_ROOT:-${REPO_ROOT}/tools/data_gen}"

# Normalize user-provided paths (allow relative paths from repo root)
abspath() {
  python3 - <<'PY' "$1" "${REPO_ROOT}"
import os, sys
p=sys.argv[1]
base=sys.argv[2]
if not os.path.isabs(p):
    p=os.path.join(base, p)
print(os.path.abspath(p))
PY
}

ANN_ROOT="$(abspath "${ANN_ROOT}")"
OUT_DIR="$(abspath "${OUT_DIR}")"
TOOL_ROOT="$(abspath "${TOOL_ROOT}")"
ANN_PARENT="$(cd "$(dirname "${ANN_ROOT}")" && pwd)"
CHECK_FILES_EXIST="${CHECK_FILES_EXIST:-0}"

echo "=============================="
echo "EPIC-Bench | Build swift infer data"
echo "Repo root : ${REPO_ROOT}"
echo "Ann root  : ${ANN_ROOT}"
echo "Ann parent: ${ANN_PARENT}"
echo "Out dir   : ${OUT_DIR}"
echo "Tool root : ${TOOL_ROOT}"
echo "Check img : ${CHECK_FILES_EXIST}"
echo "=============================="

if [[ ! -d "${ANN_ROOT}" ]]; then
  echo "ERROR: annotation root not found: ${ANN_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TOOL_ROOT}" ]]; then
  echo "ERROR: tool root not found: ${TOOL_ROOT}" >&2
  exit 1
fi

# Inputs (derived from ANN_ROOT)
TL_INPUT="${ANN_ROOT}/TargetLocalization"
VM_INPUT="${ANN_ROOT}/Navigation/VisualMatching"
FP_EGO_INPUT="${ANN_ROOT}/Navigation/FeasiblePath/FeasiblePath_Ego"
FP_EXO_INPUT="${ANN_ROOT}/Navigation/FeasiblePath/FeasiblePath_Exo"
GD_INPUT="${ANN_ROOT}/Navigation/GroundDetection"
C_T1_INPUT="${ANN_ROOT}/Manipulation/ContactRelationship/ContactRelationship_TypeOne"
C_T2_INPUT="${ANN_ROOT}/Manipulation/ContactRelationship/ContactRelationship_TypeTwo"
C_T3_INPUT="${ANN_ROOT}/Manipulation/ContactRelationship/ContactRelationship_TypeThree"
PR_INPUT="${ANN_ROOT}/Manipulation/PlacementRegion"
A_INPUT="${ANN_ROOT}/Manipulation/AffordanceRegion"

missing=()
for p in \
  "${TL_INPUT}" "${VM_INPUT}" "${FP_EGO_INPUT}" "${FP_EXO_INPUT}" "${GD_INPUT}" \
  "${C_T1_INPUT}" "${C_T2_INPUT}" "${C_T3_INPUT}" "${PR_INPUT}" "${A_INPUT}"
do
  [[ -d "${p}" ]] || missing+=("${p}")
done
if (( ${#missing[@]} > 0 )); then
  echo "ERROR: missing expected annotation subdirectories:" >&2
  for p in "${missing[@]}"; do
    echo "  - ${p}" >&2
  done
  exit 1
fi

# Prompt variable names (match tools/data_gen/prompts/*.py)
TL_SYSTEM="system_prompt_base"
TL_RESPONSE="response_json"

VM_SYSTEM="system_prompt_base"
VM_RESPONSE="response_json"

FP_EGO_SYSTEM="system_prompt_first_perspective"
FP_EGO_RESPONSE="response_json_1"

FP_EXO_SYSTEM="system_prompt_third_perspective"
FP_EXO_RESPONSE="response_json_3"

GD_SYSTEM="system_prompt_base"
GD_RESPONSE="response_json"

C_T1_SYSTEM="system_prompt_base_1"
C_T1_RESPONSE="response_json_1"

C_T2_SYSTEM="system_prompt_base_2"
C_T2_RESPONSE="response_json_2"

C_T3_SYSTEM="system_prompt_base_3"
C_T3_RESPONSE="response_json_3"

PR_SYSTEM="system_prompt_base"
PR_RESPONSE="response_json"

A_SYSTEM="system_prompt_base"
A_RESPONSE="response_json"

mkdir -p "${OUT_DIR}"

OUT_FILES=(
  "${OUT_DIR}/swift_TargetLocalization.json"
  "${OUT_DIR}/swift_VisualMatching.json"
  "${OUT_DIR}/swift_FeasiblePath_Ego.json"
  "${OUT_DIR}/swift_FeasiblePath_Exo.json"
  "${OUT_DIR}/swift_GroundDetection.json"
  "${OUT_DIR}/swift_ContactRelationship_TypeOne.json"
  "${OUT_DIR}/swift_ContactRelationship_TypeTwo.json"
  "${OUT_DIR}/swift_ContactRelationship_TypeThree.json"
  "${OUT_DIR}/swift_PlacementRegion.json"
  "${OUT_DIR}/swift_AffordanceRegion.json"
)

export PYTHONPATH="${TOOL_ROOT}:${PYTHONPATH:-}"

# IMPORTANT:
# We assume `image_path_saved` in raw JSON is a relative path starting from `EPIC_Bench/...`.
# To resolve it into the user's *actual* absolute path, we must run converters with cwd set to
# `${ANN_PARENT}` (which contains the `EPIC_Bench/` folder).
cd "${ANN_PARENT}"

CHECK_FLAG=()
if [[ "${CHECK_FILES_EXIST}" == "1" ]]; then
  CHECK_FLAG+=(--check-files-exist)
fi

echo ""
echo "[1/10] Target Localization"
python3 "${TOOL_ROOT}/converters/TargetLocalization_convert.py" "${TL_INPUT}" \
  -o "${OUT_FILES[0]}" \
  --system_var "${TL_SYSTEM}" \
  --response_var "${TL_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[2/10] Visual Matching"
python3 "${TOOL_ROOT}/converters/VisualMatching_convert.py" "${VM_INPUT}" \
  -o "${OUT_FILES[1]}" \
  --system_var "${VM_SYSTEM}" \
  --response_var "${VM_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[3/10] Feasible Path (Ego)"
python3 "${TOOL_ROOT}/converters/FeasiblePath_convert.py" "${FP_EGO_INPUT}" \
  -o "${OUT_FILES[2]}" \
  --system_var "${FP_EGO_SYSTEM}" \
  --response_var "${FP_EGO_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[4/10] Feasible Path (Exo)"
python3 "${TOOL_ROOT}/converters/FeasiblePath_convert.py" "${FP_EXO_INPUT}" \
  -o "${OUT_FILES[3]}" \
  --system_var "${FP_EXO_SYSTEM}" \
  --response_var "${FP_EXO_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[5/10] Ground Detection"
python3 "${TOOL_ROOT}/converters/GroundDetection_convert.py" "${GD_INPUT}" \
  -o "${OUT_FILES[4]}" \
  --system_var "${GD_SYSTEM}" \
  --response_var "${GD_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[6/10] Contact Relationship (Type One)"
python3 "${TOOL_ROOT}/converters/ContactRelationship_convert_1.py" "${C_T1_INPUT}" \
  -o "${OUT_FILES[5]}" \
  --system_var "${C_T1_SYSTEM}" \
  --response_var "${C_T1_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[7/10] Contact Relationship (Type Two)"
python3 "${TOOL_ROOT}/converters/ContactRelationship_convert_2_3.py" "${C_T2_INPUT}" \
  -o "${OUT_FILES[6]}" \
  --system_var "${C_T2_SYSTEM}" \
  --response_var "${C_T2_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[8/10] Contact Relationship (Type Three)"
python3 "${TOOL_ROOT}/converters/ContactRelationship_convert_2_3.py" "${C_T3_INPUT}" \
  -o "${OUT_FILES[7]}" \
  --system_var "${C_T3_SYSTEM}" \
  --response_var "${C_T3_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[9/10] Placement Region"
python3 "${TOOL_ROOT}/converters/PlacementRegion_convert.py" "${PR_INPUT}" \
  -o "${OUT_FILES[8]}" \
  --system_var "${PR_SYSTEM}" \
  --response_var "${PR_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "[10/10] Affordance Region"
python3 "${TOOL_ROOT}/converters/AffordanceRegion_convert.py" "${A_INPUT}" \
  -o "${OUT_FILES[9]}" \
  --system_var "${A_SYSTEM}" \
  --response_var "${A_RESPONSE}" \
  "${CHECK_FLAG[@]}"

echo ""
echo "=============================="
echo "Counting samples in outputs"
echo "=============================="

total=0
for file in "${OUT_FILES[@]}"; do
  if [[ -f "${file}" ]]; then
    count="$(python3 -c "import json; print(len(json.load(open('${file}','r',encoding='utf-8'))))" 2>/dev/null || echo 0)"
    printf "%-45s | %d\n" "$(basename "${file}")" "${count}"
    total=$((total + count))
  else
    printf "%-45s | missing\n" "$(basename "${file}")"
  fi
done

echo "--------------------------------------------------"
echo "Total: ${total}"
echo "Saved to: ${OUT_DIR}"
echo "=============================="
