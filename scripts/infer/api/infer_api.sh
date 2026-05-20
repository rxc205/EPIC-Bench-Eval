#!/bin/bash

export JUDGE_OPENAI_API_BASE=https://api.chatanywhere.tech/v1
export JUDGE_OPENAI_API_KEY=sk-xxxxxxxxx

model_version="xxxxx"

main_output_dir="./outputs"
temp_output_dir="${main_output_dir}/temp_${model_version//\//_}"

samples_jsonl_file="${main_output_dir}/${model_version//\//_}.jsonl"
log_file="${main_output_dir}/eval_log_${model_version//\//_}.log"

mkdir -p $main_output_dir $temp_output_dir

if [[ "$model_version" == mcs-* ]]; then
  echo "Using claude-* API"
  export OPENAI_API_KEY="xxxxxxxx"
  export OPENAI_API_BASE="https://kspmas.ksyun.com/v1/"
elif [[ "$model_version" == doubao-* ]]; then
  echo "Using Doubao API"
  export OPENAI_API_KEY="xxxxxxxxxx"
  export OPENAI_API_BASE="https://ark.cn-beijing.volces.com/api/v3"
elif [[ "$model_version" == hunyuan-* ]];then
  echo "Using hunyuan API"
  export OPENAI_API_KEY="sk-xxxxxxxxx"
  export OPENAI_API_BASE="https://api.hunyuan.cloud.tencent.com/v1"
elif [[ "$model_version" == gemini-* ]];then
  echo "Using Gemini API"
  export OPENAI_API_KEY="sk-xxxxxxxxxx"
  export OPENAI_API_BASE="https://api.chatanywhere.tech/v1" 
fi

fps=2
max_num_frames=32
temperature=0.0
max_new_tokens=8192
task="affordance,contact_typeone,contact_typetwo,contact_typethree,feasiblePath_firstperspective,feasiblePath_thirdperspective,grounding_detection,placement_region,target_localization,visual_matching"

api_args="model_version=$model_version,azure_openai=false"
para_args="max_num_frames=$max_num_frames,fps=$fps,temperature=$temperature,max_new_tokens=$max_new_tokens"
cache_save_path="api_cache"
max_workers=8
other_args="continual_mode=True,response_persistent_folder=$cache_save_path,max_workers=$max_workers"
model_args="$api_args,$para_args,$other_args"

echo "To begin evaluation, generate samples_*.json files to a temporary directory: $temp_output_dir"

python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "$model_args" \
    --tasks $task \
    --log_samples \
    --process_with_media \
    --output_path "$temp_output_dir"
1>> "$log_file" 2>&1

if [ $? -ne 0 ]; then
  echo "ERROR: lmms_eval failed to run. Please check the logs. $log_file"
  exit 1
fi

echo "Start merging all samples_*.json files into:$samples_jsonl_file"

cat > "${main_output_dir}/merge_samples_to_jsonl.py" << 'EOF'
import json
import os
import sys
import re

def extract_response_str(response_data):
    """Extract the response from the nested list into a plain string."""
    if isinstance(response_data, list):
        # Unpacking list by layer
        while isinstance(response_data, list) and len(response_data) > 0:
            response_data = response_data[0]

    return str(response_data) if response_data is not None else ""

def main():
    temp_dir = sys.argv[1]
    output_jsonl = sys.argv[2]

    open(output_jsonl, "w", encoding="utf-8").close()

    sample_files = [f for f in os.listdir(temp_dir) if re.match(r".*_samples_.*\.json", f)]
    if not sample_files:
        print(f"Warning: samples_*.json files not found | {temp_dir}")
        sys.exit(0)

    total_samples = 0

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        for fname in sample_files:
            file_path = os.path.join(temp_dir, fname)

            with open(file_path, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample_data = json.loads(line)
                        
                        if "response" in sample_data:
                            sample_data["response"] = extract_response_str(sample_data["response"])
                        
                        out_f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                        total_samples += 1
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skip invalid line | {fname} Line {line_num+1} | {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to process response | {fname} line {line_num+1} | {e}")
                        continue

    print(f"✅ Merger completed!")
    print(f"   Number of sample files processed: {len(sample_files)}")
    print(f"   Total number of samples merged: {total_samples}")
    print(f"   Total number of lines in the JSONL file: {total_samples}")
    print(f"   Output file: {output_jsonl}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 merge_samples_to_jsonl.py <temporary directory> <output JSONL file>")
        sys.exit(1)
    main()
EOF


python3 "${main_output_dir}/merge_samples_to_jsonl.py" "$temp_output_dir" "$samples_jsonl_file"

if [ -f "$samples_jsonl_file" ]; then
    echo -e "\n📝 Sample JSONL file preview (first 2 lines):"
    head -2 "$samples_jsonl_file" | python -m json.tool
    echo -e "\n✅ Final sample JSONL file:$samples_jsonl_file"
    echo "📊 File information: Size =$(du -h $samples_jsonl_file | awk '{print $1}'), Number of lines=$(wc -l < $samples_jsonl_file)"
else
    echo "ERROR: 样本JSONL文件生成失败！"
    exit 1
fi

rm -rf "$temp_output_dir"
rm -f "${main_output_dir}/merge_samples_to_jsonl.py"

echo -e "\n========================================"
echo "All samples_*.json files have been merged into a single JSONL file! "
echo "File path: $samples_jsonl_file"
echo "========================================"