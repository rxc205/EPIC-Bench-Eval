# -*- coding: utf-8 -*-
"""
Mask utilities:

- Decode GT compressed RLE list into a binary mask (union across instances)
- Convert a list of bboxes into a binary mask (union)
- Decode model response mask field (supports compressed RLE list / single RLE dict / uncompressed RLE)
- Compute IoU between two binary masks
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from pycocotools import mask as mask_utils  # type: ignore
    _HAS_PYCOCO = True
except ImportError:  # pragma: no cover
    _HAS_PYCOCO = False

Bbox = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Single RLE decode (counts: bytes / str / list)
# ---------------------------------------------------------------------------
def _decode_single_rle(rle: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(rle, dict):
        return None
    size = rle.get("size")
    counts = rle.get("counts")
    if not size or counts is None:
        return None
    try:
        h, w = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError):
        return None

    # Compressed RLE: counts is str/bytes
    if isinstance(counts, (str, bytes)):
        if not _HAS_PYCOCO:
            return None
        try:
            rle_obj = {"size": [h, w], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}
            m = mask_utils.decode(rle_obj)
            if m is None or m.size == 0:
                return None
            return (m > 0).astype(np.uint8)
        except Exception:
            return None

    # Uncompressed RLE: counts is list[int]
    if isinstance(counts, list):
        if not _HAS_PYCOCO:
            return _decode_uncompressed_rle(h, w, counts)
        try:
            # pycocotools.frPyObjects converts uncompressed RLE to compressed RLE
            rle_obj = mask_utils.frPyObjects([{"size": [h, w], "counts": counts}], h, w)
            m = mask_utils.decode(rle_obj)
            if m is None or m.size == 0:
                return None
            # frPyObjects returns (h, w, n)
            if m.ndim == 3:
                m = m[..., 0]
            return (m > 0).astype(np.uint8)
        except Exception:
            return _decode_uncompressed_rle(h, w, counts)
    return None


def _decode_uncompressed_rle(h: int, w: int, counts: Sequence[int]) -> Optional[np.ndarray]:
    """Decode COCO uncompressed RLE (column-major). Returns uint8 mask with shape (h, w)."""
    try:
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0
        for n in counts:
            n = int(n)
            if idx + n > h * w:
                n = h * w - idx
                if n <= 0:
                    break
            if val == 1:
                flat[idx : idx + n] = 1
            idx += n
            val = 1 - val
            if idx >= h * w:
                break
        # Column-major reshape
        return flat.reshape((w, h)).T.astype(np.uint8)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decode GT mask field (typically RLE list); union across instances.
# ---------------------------------------------------------------------------
def decode_rle_list_to_mask(rle_list: Any) -> Optional[np.ndarray]:
    """RLE list or single RLE dict -> binary mask (union). Returns None on failure."""
    if rle_list is None:
        return None
    items: List[Dict[str, Any]] = []
    if isinstance(rle_list, list):
        items = [x for x in rle_list if isinstance(x, dict)]
    elif isinstance(rle_list, dict):
        items = [rle_list]
    if not items:
        return None

    masks: List[np.ndarray] = []
    h, w = None, None
    for it in items:
        m = _decode_single_rle(it)
        if m is None:
            continue
        masks.append(m)
        if h is None:
            h, w = m.shape[0], m.shape[1]
    if not masks or h is None:
        return None
    out = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m.shape == (h, w):
            np.maximum(out, m, out=out)
    return out


# ---------------------------------------------------------------------------
# Model response mask (formatted_response_localization when format_type=mask)
# Supported: compressed RLE list / single compressed RLE dict / uncompressed RLE
# ---------------------------------------------------------------------------
def decode_response_mask(value: Any) -> Optional[np.ndarray]:
    """
    Decode formatted_response_localization (mask mode) into a binary mask.
    Unions multiple RLEs. Returns None on failure or if value is not mask-like.
    """
    if value is None:
        return None
    # Delegate to RLE list decoding
    if isinstance(value, list):
        # All dicts -> treat as RLE list; otherwise try single dict
        if all(isinstance(x, dict) for x in value):
            return decode_rle_list_to_mask(value)
        # Keep tolerant for nested / future extensions
        return decode_rle_list_to_mask([x for x in value if isinstance(x, dict)])
    if isinstance(value, dict):
        return decode_rle_list_to_mask([value])
    return None


# ---------------------------------------------------------------------------
# bbox -> mask (union across boxes)
# ---------------------------------------------------------------------------
def bboxes_to_mask(
    bboxes: Sequence[Bbox],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in bboxes:
        if not b or len(b) < 4:
            continue
        x1, y1, x2, y2 = (max(0, int(b[0])), max(0, int(b[1])), min(width, int(b[2])), min(height, int(b[3])))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------
def mask_iou(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> float:
    """Binary mask IoU. Both empty => 1.0; one empty => 0.0."""
    if pred is None or gt is None:
        return 0.0
    if pred.shape != gt.shape:
        # Size mismatch: cannot compare; treat as 0
        return 0.0
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)
