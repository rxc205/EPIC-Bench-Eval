<div align="center">

# 🎯 EPIC-Bench: A Perception-Centric Benchmark for Fine-Grained Embodied Visual Grounding in Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://epic-bench.github.io/EPIC-Bench/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://epic-bench.github.io/EPIC-Bench/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/rxc205/EPIC-Bench)
[![Evaluation Toolkit](https://img.shields.io/badge/⚙️-Evaluation_Toolkit-6366f1.svg)](#-epic-bench-evaluation-toolkit)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#-license)

[**Homepage**](https://epic-bench.github.io/EPIC-Bench/) | [**Paper**](https://epic-bench.github.io/EPIC-Bench/) | [**Dataset**](https://huggingface.co/datasets/rxc205/EPIC-Bench) | [**Leaderboard**](https://epic-bench.github.io/EPIC-Bench/#leaderboard-section)

</div>

## 📃 Overview

> This repo contains the official evaluation code and dataset for the paper
> **"EPIC-Bench: A Perception-Centric Benchmark for Fine-Grained Embodied Visual Grounding in Vision-Language Models"**

**EPIC-Bench** is a **Mask-Grounding-based** benchmark designed to evaluate a VLM’s **Visual Perception** capability in **Embodied Scenarios**.

<p align="center">
  <img src="./images/teaser.png" alt="EPIC-Bench teaser" width="100%"/>
</p>

📚 EPIC-Bench covers **3 High-Level Categories** and **23 Task Types**, following the realistic **Embodied Workflow**:

- 🎯 **TargetLocalization**: **Pinpoint** the right object in the scene from a natural-language instruction.
- 🧭 **Navigation**: **Approach** the target step by step by reading key visual cues along the way.
- 🤲 **Manipulation**: **Operate** on the target through fine-grained, action-oriented **Grounded Perception**.

The goal is to measure whether models can reliably perceive the critical **Visual** information required throughout the **Embodied Process**.

## ✨ Highlights

-  **Embodied-Scenario** evaluation of VLM **Visual Perception** capability.
-  Focus on **Visual Grounding / Perception** without language shortcut exploitation.
-  **Diverse** and **Fine-Grained** task design.

## 📰 News

- [2026.5.15] 🚀 Huggingface and ModelScope Dataset are available!
- [2026.5.15] 🚀 We released the ArXiv paper and Evaluation Code!

## 📋 Todo

- [x] Evaluation code for EPIC-Bench
- [ ] Make the evaluation pipeline compatible with mask outputs


# 🧰 EPIC-Bench Evaluation Toolkit

This repository provides an end-to-end evaluation pipeline for **EPIC-Bench** on both:

- **Open-Source VLMs** via the **[ms-swift](https://swift.readthedocs.io/zh-cn/latest/)** inference interface
- **Closed-Source / API-Based VLMs** via **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**

It includes **Dataset Conversion** utilities, **Inference Launchers**, **Response Standardization**, **Scoring**, and a Streamlit-based **Visualization** tool.


## 🚀 Evaluation guide

EPIC-Bench evaluation typically consists of the following stages.

### ⚙️ 0) Environment setup

Create a Python environment (example):

```bash
conda create -n epicbench python==3.10
conda activate epicbench
```

Suggested dependencies (reference; choose what matches your model stack):

| Model | Environment |
|------|-------------|
| General environment (compatible with Qwen2.5-VL/Qwen3-VL/InternVL/LLaVA-VL/Phi-4/gemma/RynnBrain/RoboBrain2/) | `pip install uv`<br>`uv pip install 'ms-swift' --torch-backend=auto`<br>`pip install vllm==0.15.1` |
| Qwen3.5 | `pip install vllm==0.17 ms-swift qwen-vl-utils transformers accelerate` |
| Step models | `pip install onnxruntime-gpu tokenizers openai-whisper funasr vllm==0.15.1`<br>`git clone https://github.com/modelscope/ms-swift.git`<br>`cd ms-swift`<br>`pip install -e .` |
| glm4.6 | `pip install uv`<br>`uv pip install 'ms-swift' --torch-backend=auto`<br>`pip install vllm==0.15.1 transformers==5.2.0` |
| API | `cd lmms-eval`<br>`pip install -e .` |

For the visualization tool:

```bash
pip install streamlit pillow numpy pandas pycocotools
```

### 📦 1) Data preparation

#### 1.1 Download raw annotations

Download EPIC-Bench raw annotation data (and the referenced images) from the official release page (e.g., Hugging Face / ModelScope) and place them under:

```
dataset/annotation/EPIC_Bench
```

#### 1.2 Build ms-swift inference data (swift format)

Generate ms-swift compatible inference JSON files from raw annotations:

```bash
bash scripts/build_swift_data.sh \
  ANN_ROOT=dataset/annotation/EPIC_Bench \
  OUT_DIR=dataset/swift_data/EPIC_Bench
```

Outputs will be written to:

```
dataset/swift_data/EPIC_Bench
```

#### 1.3 Customize prompts (optional)

You can customize prompts and response formats in:

- `tools/data_gen/prompts/`
- `tools/data_gen/converters/`

For best compatibility with the scoring pipeline, we recommend starting with the default settings in this repo.

### 🤖 2) Inference

Run inference using either:

- example per-model scripts under `scripts/infer/<MODEL_FAMILY>/`, or
- the unified launcher `scripts/infer.sh`

Recommended (unified launcher):

```bash
bash scripts/infer.sh \
  --model Qwen3_VL \
  --data dataset/swift_data/EPIC_Bench \
  --out outputs/model_response/swift_format
```

By default, raw ms-swift outputs are organized under:

```
outputs/model_response/swift_format/<model_series>/<model_name>.jsonl
```

Closed-source / API inference (optional):

- `scripts/infer/api/` contains an example script for `lmms-eval`.
- You must configure API keys via environment variables and **must not commit credentials** to GitHub.

### 🔄 3) Standardize responses (std_format)

Convert raw ms-swift outputs into EPIC-Bench **standard format** while preserving directory structure:

```bash
bash scripts/format_response.sh \
  --in  outputs/model_response/swift_format \
  --out outputs/model_response/std_format
```

If you evaluate a custom model/framework outside this repo, please ensure your outputs follow the **same std-format schema produced by** `tools/formatting/format_response.py`.

### 📊 4) Scoring

After obtaining standardized responses, compute detailed scores:

```bash
bash scripts/evaluate.sh \
  --in  outputs/model_response/std_format \
  --out outputs/scores
```

The scorer produces:

- overall / category / type breakdowns
- per-sample details (unless you pass `--no-details`)

Supported formats:

- **bbox** (most tasks)
- **point** (FeasiblePath tasks)

Mask-based evaluation is planned (releasing soon).


```bash
bash scripts/evaluate.sh 
```

### 📈 5) Visualization

Launch the Streamlit visualization tool and default-load results from `outputs/scores`:

```bash
bash scripts/visualization.sh
```


## 🏆 Leaderboard and Benchmark

Please refer to the [EPIC-Bench Homepage](https://epic-bench.github.io/EPIC-Bench/) for:

- Leaderboard
- Full dataset downloads
- EPIC-Bench data examples

## 📚 Citation

```BibTeX
@article{EPIC-Bench,
  title={EPIC-Bench: A Perception-Centric Benchmark for Fine-Grained Embodied Visual Grounding in Vision-Language Models},
  author={XXX, XXX, XXX},
  journal={},
  year={2026}
}
```

## 📜 License

Please add an explicit `LICENSE` file before open-sourcing. If EPIC-Bench annotations or images have redistribution constraints, publish them separately (e.g., Hugging Face / ModelScope) and keep this repo code-only + small examples.

## 🙏 Acknowledgements

- **ms-swift** for open-source VLM inference: [ms-swift](https://swift.readthedocs.io/zh-cn/latest/)
- **lmms-eval** for API/closed-source evaluation: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
