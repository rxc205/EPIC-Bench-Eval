#!/bin/bash
set -ex
export SWIFT_IGNORE_BROKEN_IMAGE=true

# Resolve repo root (walk upwards until dataset/ exists)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
while [[ "${REPO_ROOT}" != "/" && ! -d "${REPO_ROOT}/dataset" ]]; do
  REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
done

# Supports command-line argument passing and is compatible with environment variables passed by the main script (command-line takes precedence)

while [[ $# -gt 0 ]]; do
  case $1 in
    --val_dataset)
      VAL_DATASET="$2"
      shift 2
      ;;
    --result_path)
      RESULT_PATH="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done


model_path="${MODEL_PATH:-${EPIC_MODEL_PATH:-}}"
if [[ -z "${model_path}" ]]; then
  echo "ERROR: MODEL_PATH is required (local path to model weights)." >&2
  exit 1
fi

model_series="Gemma"
model_name="Gemma-3-12B-IT"
RESULT_PATH_PARENT="${RESULT_PATH:-${REPO_ROOT}/examples/results_sample/model_response/swift_format}"
# RESULT_PATH_PARENT="${RESULT_PATH:-${REPO_ROOT}/outputs/model_response/swift_format}"
result_path="${RESULT_PATH_PARENT}/${model_series}/${model_name}.jsonl"

DATA_ROOT_DEFAULT="${REPO_ROOT}/examples/swift_data_sample/EPIC_Bench"
# DATA_ROOT_DEFAULT="${REPO_ROOT}/dataset/swift_data/EPIC_Bench"
if [ -n "$VAL_DATASET" ]; then
  if [ -d "$VAL_DATASET" ]; then
    VAL_DATASET_ARGS=("${VAL_DATASET}"/*.json)
  else
    VAL_DATASET_ARGS=("$VAL_DATASET")
  fi
else
  VAL_DATASET_ARGS=(
    "${DATA_ROOT_DEFAULT}/swift_TargetLocalization.json"
    "${DATA_ROOT_DEFAULT}/swift_GroundDetection.json"
    "${DATA_ROOT_DEFAULT}/swift_FeasiblePath_Ego.json"
    "${DATA_ROOT_DEFAULT}/swift_FeasiblePath_Exo.json"
    "${DATA_ROOT_DEFAULT}/swift_VisualMatching.json"
    "${DATA_ROOT_DEFAULT}/swift_AffordanceRegion.json"
    "${DATA_ROOT_DEFAULT}/swift_ContactRelationship_TypeOne.json"
    "${DATA_ROOT_DEFAULT}/swift_ContactRelationship_TypeTwo.json"
    "${DATA_ROOT_DEFAULT}/swift_ContactRelationship_TypeThree.json"
    "${DATA_ROOT_DEFAULT}/swift_PlacementRegion.json"
  )
fi

gpu_memory_utilization=0.8

tensor_parallel_size=4

max_new_tokens=8192

export FPS_MAX_FRAMES=32
#CUDA_VISIBLE_DEVICES=0 \
swift infer --model $model_path \
  --model_type gemma3_vision \
  --val_dataset "${VAL_DATASET_ARGS[@]}" \
  --infer_backend vllm \
  --result_path $result_path \
  --vllm_gpu_memory_utilization $gpu_memory_utilization \
  --vllm_tensor_parallel_size $tensor_parallel_size \
  --vllm_pipeline_parallel_size 1 \
  --max_new_tokens $max_new_tokens 
