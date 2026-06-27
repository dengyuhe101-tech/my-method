"""Rotated NMS helpers for V3."""

from __future__ import annotations

import numpy as np

from orsdet_geometry.geometry import hbb_from_obb, rotated_iou
from orsdet_candidates.decode import decode_obb_boxes


FIRST_IOU_THRESHOLDS = np.asarray([0.5, 0.4, 0.3, 0.2], dtype=np.float64)
FIRST_OBJ_THRESHOLDS = np.asarray([1.0, 0.7, 0.5, 0.3], dtype=np.float64)
SECOND_IOU_THRESHOLD = 0.25


def decode_rows_obb(rows, lims):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    centers_x = 0.5 * (rows[:, 0] + rows[:, 2])
    centers_y = 0.5 * (rows[:, 1] + rows[:, 3])
    return decode_obb_boxes(
        centers_x,
        centers_y,
        rows[:, 8],
        rows[:, 9],
        rows[:, 10:12],
        lims,
    )


def _hbb_overlap_mask(current_hbb, remaining_hbb):
    return (
        (np.minimum(current_hbb[2], remaining_hbb[:, 2]) > np.maximum(current_hbb[0], remaining_hbb[:, 0]))
        & (np.minimum(current_hbb[3], remaining_hbb[:, 3]) > np.maximum(current_hbb[1], remaining_hbb[:, 1]))
    )


def _suppression_mask(
    current_box,
    remaining_boxes,
    remaining_obj,
    iou_thresholds,
    obj_thresholds,
    current_hbb,
    remaining_hbb,
):
    if remaining_boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    candidate = (remaining_obj < obj_thresholds[0]) & _hbb_overlap_mask(current_hbb, remaining_hbb)
    out = np.zeros((remaining_boxes.shape[0],), dtype=bool)
    if not np.any(candidate):
        return out

    cand_idx = np.flatnonzero(candidate)
    ious = np.asarray(rotated_iou(current_box, remaining_boxes[cand_idx]), dtype=np.float64)
    out[cand_idx] = (
        ((ious > iou_thresholds[0]) & (remaining_obj[cand_idx] < obj_thresholds[0]))
        | ((ious > iou_thresholds[1]) & (remaining_obj[cand_idx] < obj_thresholds[1]))
        | ((ious > iou_thresholds[2]) & (remaining_obj[cand_idx] < obj_thresholds[2]))
        | ((ious > iou_thresholds[3]) & (remaining_obj[cand_idx] < obj_thresholds[3]))
    )
    return out


def local_nms(rows, boxes, iou_thresholds=FIRST_IOU_THRESHOLDS, obj_thresholds=FIRST_OBJ_THRESHOLDS):
    rows = np.asarray(rows, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    if rows.shape[0] == 0:
        return rows, boxes

    order = np.argsort(rows[:, 5])[::-1]
    rows = rows[order]
    boxes = boxes[order]
    hbbs = hbb_from_obb(boxes)
    active = np.ones(rows.shape[0], dtype=bool)
    keep = []

    for idx in range(rows.shape[0]):
        if not active[idx]:
            continue
        keep.append(idx)
        rem = np.flatnonzero(active[idx + 1 :]) + idx + 1
        if rem.size == 0:
            continue
        sup = _suppression_mask(
            boxes[idx],
            boxes[rem],
            rows[rem, 5],
            iou_thresholds,
            obj_thresholds,
            hbbs[idx],
            hbbs[rem],
        )
        active[rem[sup]] = False

    keep = np.asarray(keep, dtype=np.int64)
    return rows[keep], boxes[keep]


def _shift_tile(rows, boxes, direction, patch_shift):
    rows = np.asarray(rows, dtype=np.float64).copy()
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    rows[:, 0] += direction[0] * patch_shift
    rows[:, 2] += direction[0] * patch_shift
    rows[:, 1] += direction[1] * patch_shift
    rows[:, 3] += direction[1] * patch_shift
    if boxes.shape[0] > 0:
        boxes[:, 0] += direction[0] * patch_shift
        boxes[:, 1] += direction[1] * patch_shift
    return rows, boxes


def merge_nms(rows, boxes, comp_rows, comp_boxes, direction, threshold, patch_shift, overlap):
    rows = np.asarray(rows, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    comp_rows = np.asarray(comp_rows, dtype=np.float64)
    comp_boxes = np.asarray(comp_boxes, dtype=np.float64)
    if rows.shape[0] == 0:
        return rows, boxes
    if comp_rows.shape[0] == 0:
        return rows, boxes

    keep_mask = (
        (rows[:, 0] > overlap)
        & (rows[:, 2] < patch_shift)
        & (rows[:, 1] > overlap)
        & (rows[:, 3] < patch_shift)
    )
    remain_mask = ~keep_mask
    keep_rows = [rows[keep_mask]]
    keep_boxes = [boxes[keep_mask]]

    shifted_rows, shifted_boxes = _shift_tile(comp_rows, comp_boxes, direction, patch_shift)
    shifted_hbb = hbb_from_obb(shifted_boxes)
    current_hbb = hbb_from_obb(boxes)

    for idx in np.flatnonzero(remain_mask):
        current_box = boxes[idx]
        current_obj = rows[idx, 5]
        candidate = (current_obj < shifted_rows[:, 5]) & _hbb_overlap_mask(current_hbb[idx], shifted_hbb)
        suppress = False
        if np.any(candidate):
            ious = np.asarray(rotated_iou(current_box, shifted_boxes[np.flatnonzero(candidate)]), dtype=np.float64)
            suppress = bool(np.any(ious > threshold))
        if not suppress:
            keep_rows.append(rows[idx : idx + 1])
            keep_boxes.append(boxes[idx : idx + 1])

    if len(keep_rows) == 1:
        return keep_rows[0], keep_boxes[0]
    return np.concatenate(keep_rows, axis=0), np.concatenate(keep_boxes, axis=0)
