<div align="center">

# 🎯 EPIC-Bench: A Perception-Centric Benchmark for Fine-Grained Embodied Visual Grounding in Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.17070-b31b1b.svg)](https://arxiv.org/abs/2605.17070)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://epic-bench.github.io/EPIC-Bench/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/rxc205/EPIC-Bench)
[![Evaluation Toolkit](https://img.shields.io/badge/⚙️-Evaluation_Toolkit-6366f1.svg)](#-epic-bench-evaluation-toolkit)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

[**Homepage**](https://epic-bench.github.io/EPIC-Bench/) | [**Paper**](https://epic-bench.github.io/EPIC-Bench/) | [**Dataset**](https://huggingface.co/datasets/rxc205/EPIC-Bench) | [**Leaderboard**](https://epic-bench.github.io/EPIC-Bench/#leaderboard-section)

</div>

## 📃 Overview

> This repository is the official [**evaluation toolkit**](#-epic-bench-evaluation-toolkit) for EPIC-Bench.

**EPIC-Bench** is a **Mask-Grounding-based** benchmark that evaluates VLM **Visual Perception** in **Embodied Scenarios**, covering **3 categories** and **23 task types** — without language shortcut exploitation.

<p align="center">
  <img src="./images/teaser.png" alt="EPIC-Bench teaser" width="100%"/>
</p>

Following the realistic **Embodied Workflow**:

- 🎯 **TargetLocalization**: Pinpoint the target object from a natural-language instruction.
- 🧭 **Navigation**: Approach the target by reading key visual cues step by step.
- 🤲 **Manipulation**: Operate on the target via fine-grained, action-oriented grounded perception.

## 📰 News

- [2026.5.19] 🎉 Our [arXiv paper](https://arxiv.org/abs/2605.17070) is now available!
- [2026.5.15] 🚀 [HuggingFace](https://huggingface.co/datasets/rxc205/EPIC-Bench) and [ModelScope](https://www.modelscope.cn/datasets/macarich/EPIC-Bench) Dataset are available!
- [2026.5.15] 🚀 [Project Page](https://epic-bench.github.io/EPIC-Bench/) and [Evaluation Code](https://github.com/rxc205/EPIC-Bench-Eval) are released.


## 🧰 EPIC-Bench Evaluation Toolkit

This repository provides an end-to-end evaluation pipeline for **EPIC-Bench** on both:

- **Open-Source VLMs** via **[ms-swift](https://swift.readthedocs.io/zh-cn/latest/)**
- **Closed-Source / API-Based VLMs** via **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

It covers **Dataset Conversion**, **Inference**, **Response Standardization**, **Scoring**, and a Streamlit-based **Visualization** tool.

## 🚀 Evaluation Guide

### ⚙️ 0) Environment Setup

```bash
conda create -n epicbench python==3.10
conda activate epicbench
```

Suggested dependencies (choose what matches your model stack):

| Model | Environment |
|------|-------------|
| General (Qwen2.5-VL / Qwen3-VL / InternVL / LLaVA-VL / Phi-4 / Gemma / RynnBrain / RoboBrain2) | `pip install uv`<br>`uv pip install 'ms-swift' --torch-backend=auto`<br>`pip install vllm==0.15.1` |
| Qwen3.5 | `pip install vllm==0.17 ms-swift qwen-vl-utils transformers accelerate` |
| Step models | `pip install onnxruntime-gpu tokenizers openai-whisper funasr vllm==0.15.1`<br>`git clone https://github.com/modelscope/ms-swift.git && cd ms-swift && pip install -e .` |
| glm4.6 | `pip install uv`<br>`uv pip install 'ms-swift' --torch-backend=auto`<br>`pip install vllm==0.15.1 transformers==5.2.0` |
| API | `cd lmms-eval && pip install -e .` |

For the visualization tool:

```bash
pip install streamlit pillow numpy pandas pycocotools
```

### 📦 1) Data Preparation

#### 1.1 Download raw annotations

EPIC-Bench contains ~35,000 small files across three task categories. To work around HuggingFace's per-file rate limit (1,000 API requests / 5 min) and to keep download speed reasonable, the annotations are distributed as **three task-level tarballs** instead of raw folders.

Download the tarballs from [HuggingFace](https://huggingface.co/datasets/rxc205/EPIC-Bench) (or [ModelScope](https://www.modelscope.cn/datasets/macarich/EPIC-Bench)):

| Archive | Size | # Files | Contents |
|---------|------|---------|----------|
| `Manipulation.tar.gz`       | 1.70 GB | 7,061   | AffordanceRegion / ContactRelationship / PlacementRegion |
| `Navigation.tar.gz`         | 2.71 GB | 10,868  | FeasiblePath / GroundDetection / VisualMatching |
| `TargetLocalization.tar.gz` | 3.34 GB | 17,665  | BasicAttributes / EmbodiedCompositionalAttributes / SpatialRelatedAttributes |

Place them under `dataset/annotation/` and extract:

```bash
mkdir -p dataset/annotation/EPIC_Bench
cd dataset/annotation/EPIC_Bench

# Option 1: huggingface-cli
hf download rxc205/EPIC-Bench \
    --repo-type dataset --local-dir .

# Option 2: modelscope
modelscope download \
  --dataset macarich/EPIC-Bench \
  --local_dir .


# Extract all three (preserves the original folder layout)
for f in Manipulation.tar.gz Navigation.tar.gz TargetLocalization.tar.gz; do
    tar -xzf "$f" && rm "$f"
done
```

#### 1.2 Build ms-swift inference data

```bash
bash scripts/build_swift_data.sh \
  ANN_ROOT=dataset/annotation/EPIC_Bench \
  OUT_DIR=dataset/swift_data/EPIC_Bench
```

#### 1.3 Customize prompts (optional)

Prompts and response formats can be adjusted in `tools/data_gen/prompts/` and `tools/data_gen/converters/`. We recommend starting from the defaults for best compatibility with the scoring pipeline.

### 🤖 2) Inference

2.1 Open-source model：
Run inference with the unified launcher:
```bash
bash scripts/infer.sh \
  --model Qwen3_VL \
  --data dataset/swift_data/EPIC_Bench \
  --out outputs/model_response/swift_format
```
2.2 Closed-source model：
Modify the model-version and API-key in the script.
```bash
bash scripts/infer/api/infer_api.sh
```

Outputs are saved to `outputs/model_response/swift_format/<model_series>/<model_name>.jsonl`.

For per-model scripts, see `scripts/infer/<MODEL_FAMILY>/`. For closed-source / API models, see `scripts/infer/api/` (configure API keys via environment variables; **do not commit credentials**).

### 🔄 3) Standardize Responses

Convert raw outputs to EPIC-Bench **standard format**:

```bash
bash scripts/format_response.sh \
  --in  outputs/model_response/swift_format \
  --out outputs/model_response/std_format
```

If you use a custom inference framework, ensure its outputs follow the std-format schema produced by `tools/formatting/format_response.py`.

### 📊 4) Scoring

```bash
bash scripts/evaluate.sh \
  --in  outputs/model_response/std_format \
  --out outputs/scores
```

The scorer produces overall / category / task-type breakdowns and per-sample details (pass `--no-details` to skip). Supported formats: **bbox** (most tasks) and **point** (FeasiblePath tasks). Mask-based evaluation is planned.

### 📈 5) Visualization

```bash
bash scripts/visualization.sh
```

Launches a Streamlit app and loads results from `outputs/scores` by default.

## 📋 Todo

- [x] Evaluation code for EPIC-Bench
- [x] The EPIC-Bench datasets
- [ ] Make the evaluation pipeline compatible with mask outputs


## 🏆 Leaderboard and Benchmark

Please refer to the [EPIC-Bench Homepage](https://epic-bench.github.io/EPIC-Bench/) for the full leaderboard, dataset downloads, and data examples.

## 📬 Contact With Us
- Email: xiancong.ren@x-humanoid.com

## 📚 Citation

```BibTeX
@misc{shan2026epicbenchperceptioncentricbenchmarkfinegrained,
      title={EPIC-Bench: A Perception-Centric Benchmark for Fine-Grained Embodied Visual Grounding in Vision-Language Models}, 
      author={Haozhe Shan and Xiancong Ren and Han Dong and Haoyuan Shi and Yingji Zhang and Jiayu Hu and Yi Zhang and Yong Dai and Bin Shen and Lizhen Qu and Zenglin Xu and Xiaozhu Ju},
      year={2026},
      eprint={2605.17070},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.17070}, 
}
```

## 📜 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **ms-swift** for open-source VLM inference: [ms-swift](https://swift.readthedocs.io/zh-cn/latest/)
- **lmms-eval** for API/closed-source evaluation: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
