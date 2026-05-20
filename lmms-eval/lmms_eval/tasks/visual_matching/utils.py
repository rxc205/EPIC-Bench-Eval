import os
from pathlib import Path
import numpy as np
import sys
import re
import ast
from pycocotools import mask as mask_utils
import json

# Ensure the local task utilities can be imported
sys.path.append(os.getcwd())
# print(f"Current working directory: {os.getcwd()}")

# from metrics.iou_calculator import (
#     bbox_to_mask, 
#     compute_iou_for_boxes, 
#     compute_accuracy_stats,
#     IOUResult
# )

import yaml
from PIL import Image
from lmms_eval.cot_utils import remove_think


with open(Path(__file__).parent / "visual_matching.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))

# Pass in video path here
# Can only work correctly with video llm
def visual_matching_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    visual = [Image.open(image_path) for image_path in doc["images"]]
    return visual


# This is the place where you format your question
def visual_matching_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    # Format question with choices
    messages = doc.get("messages")
    user_content = next((msg["content"] for msg in messages if msg.get("role") == "user"), "")
    system_content = next((msg["content"] for msg in messages if msg.get("role") == "system"), "")
    question = system_content + user_content
    # question = system_content

    return question

def extract_bbox_and_count(pred_input):
    """
    Extract bounding boxes and object count from a prediction (string or dict).

    Args:
        pred_input: either a string containing bounding_boxes and number_of_objects,
            or a dict with those keys.
    Returns:
        bbox_str: string form of bounding box list, e.g. "[(160, 257, 191, 384), ...]"
        count: extracted object count (int)
    """
    # Initialize outputs
    bbox_str = None
    count = None
    
    # Dict input
    if isinstance(pred_input, dict):
        # Convert bbox list to the expected string representation
        bboxes = pred_input.get("bounding_boxes", [])
        # Convert inner lists to tuples for stable string formatting
        bbox_str = str([tuple(bbox) for bbox in bboxes])
        # Extract object count
        count = pred_input.get("number_of_objects", 0)
        return bbox_str, count
    
    # String input (legacy compatibility)
    if isinstance(pred_input, str):
        # Regex 1: match list after bounding_boxes
        bbox_pattern = r'\*\*bounding_boxes\*\*:\s*(\[.*?\])'
        # Regex 2: match number after number_of_objects
        count_pattern = r'\*\*number_of_objects\*\*:\s*(\d+)'
        
        # Extract bbox string
        bbox_match = re.search(bbox_pattern, pred_input)
        if bbox_match:
            bbox_str = bbox_match.group(1)
            # Optional: validate list-likeness for robustness
            try:
                ast.literal_eval(bbox_str)
            except (SyntaxError, ValueError):
                bbox_str = None
        
        # Extract object count
        count_match = re.search(count_pattern, pred_input)
        count = int(count_match.group(1)) if count_match else None
        
        return bbox_str, count
    
    # Other input types
    return bbox_str, count

def decode_masklet(masklet_all):
    """
    Decode an RLE-encoded masklet list into a NumPy array.

    Args:
        masklet_all: list containing RLE dict(s) with "counts" and "size".
    Returns:
        mask_np: decoded mask as a NumPy array of shape (H, W).
    """
    # Guard against empty input
    if not masklet_all or len(masklet_all) == 0:
        print("masklet_all is empty; cannot decode")
        return None
    
    # Take the first masklet RLE (EPIC-Bench uses one per sample here)
    rle_data = masklet_all[0]
    counts = rle_data["counts"]
    size = rle_data["size"]  # [width, height] (note the order)
    
    # Build RLE dict for pycocotools
    rle = {
        "counts": counts,
        "size": size  # pycocotools expects [width, height]
    }
    
    # Decode to a binary mask
    mask = mask_utils.decode(rle)
    
    # Convert to uint8 for visualization/processing
    mask_np = mask.astype(np.uint8) * 255
    
    return mask_np



def visual_matching_doc_to_answer(doc):
    # masklet_all = doc.get("additional").get("mask_annotation").get("masklet_all")
    # if masklet_all and len(masklet_all) > 0:
    #     size = masklet_all[0].get("size")  # read size from the first entry
    #     h, w = size[0], size[1]
    #     image_shape = (h, w)
    # else:
    #     image_shape = None
    # gt_mask = decode_masklet(masklet_all)
    # gt_count = doc.get("additional").get("mask_annotation").get("number_of_instance")[0]
    # precise_count = doc.get("additional").get("mask_annotation").get("precise_count")[0]
    # 
    imgs = []
    images = doc.get("images",[])
    for path in images:
        imgs.append(
            {
                "bytes": None,
                "path": path
            }
        )

    return imgs

# Process result for mcq answer generation
def visual_matching_process_results_generation(doc, result):
    resps = remove_think(result[0])
    imgs = visual_matching_doc_to_answer(doc)
    messages = doc.get("messages",[])
    for msg in messages:
        # Add a "loss" field with null (Python None)
        msg["loss"] = None
    messages.append(
        {
            "role": "assistant",
            "content": resps
        }
    )

    # score = 0.0
    return {"messages": messages, "images": imgs}


def visual_matching_aggregate_score(results, args):
    return np.mean(results)