# EPIC-Bench Visualization Tool

This Streamlit app visualizes EPIC-Bench detailed evaluation results produced by `epic_eval`.
It supports both `*_full.json` and `*_full.jsonl` outputs.

## Quick start

```bash
pip install streamlit pillow numpy pandas pycocotools
bash scripts/visualization.sh
```

After launching, use the sidebar to upload a file or paste a local path to a result file.

## Environment variables

The launcher script `scripts/visualization.sh` supports:

- `EPIC_VIS_RESULTS_JSON`: absolute/relative path to a `*_full.json` or `*_full.jsonl` file (highest priority)
- `EPIC_VIS_SCORES_ROOT`: directory to scan for a default result file (default: `outputs/scores`)
- `PORT`: Streamlit server port

## Result loading

- The app expects detailed results containing per-sample entries (equivalent to `details.by_samples`).
- For most samples, only `ground_truth.gt_json_path` is stored in the result file; the app loads
  the raw GT JSON from that path (see `epic_eval/config.py` and `epic_eval/utils/gt_loader.py`).

## What you can do

- Filter by task_category / task_type / format_type / validity / score range
- Search by text/path (gt_json_path and key annotation fields)
- Sort, page, and inspect per-sample overlays (GT masks and model predictions)
