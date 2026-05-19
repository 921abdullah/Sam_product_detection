from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from app.utils.mask_utils import mask_iou, resize_masks_to_image_shape


@dataclass
class MatchResult:
    matched_pairs: list[dict]
    removed_indices: list[int]
    new_indices: list[int]
    before_masks_resized: np.ndarray | None = None
    after_masks_resized: np.ndarray | None = None


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def _greedy_match(
    candidate_matches: list[tuple[float, int, int]],
    num_before: int,
    num_after: int,
    score_key: str,
) -> MatchResult:
    candidate_matches.sort(reverse=True, key=lambda x: x[0])

    matched_before: set[int] = set()
    matched_after: set[int] = set()
    final_matches: list[dict] = []

    for score, before_idx, after_idx in candidate_matches:
        if before_idx in matched_before or after_idx in matched_after:
            continue

        matched_before.add(before_idx)
        matched_after.add(after_idx)
        final_matches.append(
            {
                "before_idx": before_idx,
                "after_idx": after_idx,
                score_key: score,
            }
        )

    removed_indices = [
        idx for idx in range(num_before) if idx not in matched_before
    ]
    new_indices = [
        idx for idx in range(num_after) if idx not in matched_after
    ]

    return MatchResult(
        matched_pairs=final_matches,
        removed_indices=removed_indices,
        new_indices=new_indices,
    )


class BBoxMatcher:
    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold

    def match(self, before_boxes: np.ndarray, after_boxes: np.ndarray) -> MatchResult:
        num_before = len(before_boxes)
        num_after = len(after_boxes)
        candidate_matches: list[tuple[float, int, int]] = []

        for before_idx in range(num_before):
            for after_idx in range(num_after):
                score = bbox_iou(before_boxes[before_idx], after_boxes[after_idx])
                if score >= self.iou_threshold:
                    candidate_matches.append((score, before_idx, after_idx))

        return _greedy_match(
            candidate_matches, num_before, num_after, score_key="bbox_iou"
        )


class MaskMatcher:
    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold

    def match(
        self,
        before_masks: np.ndarray,
        after_masks: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> MatchResult:
        before_masks = resize_masks_to_image_shape(before_masks, target_h, target_w)
        after_masks = resize_masks_to_image_shape(after_masks, target_h, target_w)

        num_before = len(before_masks)
        num_after = len(after_masks)
        candidate_matches: list[tuple[float, int, int]] = []

        for before_idx in range(num_before):
            for after_idx in range(num_after):
                score = mask_iou(before_masks[before_idx], after_masks[after_idx])
                if score >= self.iou_threshold:
                    candidate_matches.append((score, before_idx, after_idx))

        result = _greedy_match(
            candidate_matches, num_before, num_after, score_key="mask_iou"
        )
        return replace(
            result,
            before_masks_resized=before_masks,
            after_masks_resized=after_masks,
        )


def get_matcher(
    mode: Literal["bbox", "mask"],
    bbox_threshold: float,
    mask_threshold: float,
):
    if mode == "bbox":
        return BBoxMatcher(iou_threshold=bbox_threshold)
    if mode == "mask":
        return MaskMatcher(iou_threshold=mask_threshold)
    raise ValueError(f"Unsupported matching mode: {mode}")
