from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from app.core.config import PipelineConfig
from app.services.dino_service import (
    build_mask_to_embedding_index_map,
    compute_embedding_similarity_matrix,
    lookup_pair_dino_similarity,
)
from app.services.dino_service import DinoService
from app.services.verification_service import VerificationService
from app.utils.mask_utils import (
    deduplicate_masks,
    mask_iou,
    mask_overlap_metrics,
    resize_masks_to_image_shape,
)


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
    candidate_matches: list[tuple],
    num_before: int,
    num_after: int,
    score_key: str,
    extra_fields: tuple[str, ...] = (),
) -> MatchResult:
    candidate_matches.sort(reverse=True, key=lambda x: x[0])

    matched_before: set[int] = set()
    matched_after: set[int] = set()
    final_matches: list[dict] = []

    for entry in candidate_matches:
        score = entry[0]
        before_idx = entry[1]
        after_idx = entry[2]
        extras = entry[3:]

        if before_idx in matched_before or after_idx in matched_after:
            continue

        matched_before.add(before_idx)
        matched_after.add(after_idx)

        match_dict = {
            "before_idx": before_idx,
            "after_idx": after_idx,
            score_key: score,
        }
        for field_name, value in zip(extra_fields, extras):
            match_dict[field_name] = value

        final_matches.append(match_dict)

    removed_indices = [idx for idx in range(num_before) if idx not in matched_before]
    new_indices = [idx for idx in range(num_after) if idx not in matched_after]

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
    """Simple mask IoU matching (legacy bbox-style threshold only)."""

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


class AdvancedMaskMatcher:
    """
    Multi-stage mask matching from light.py:
    Stage 1: spatial overlap + DINO validation
    Stage 2: local DINO rescue
    Stage 2B: one-sided DINO verification on tentative changes
    Stage 3: crop-based local LightGlue + RANSAC verification
    """

    def __init__(
        self,
        dino_service: DinoService,
        verification_service: VerificationService,
        output_dir: Path | None = None,
    ):
        self.dino_service = dino_service
        self.verification_service = verification_service
        self.output_dir = output_dir

    def match(
        self,
        before_masks: np.ndarray,
        after_masks: np.ndarray,
        target_h: int,
        target_w: int,
        before_rgb: np.ndarray,
        aligned_after_rgb: np.ndarray,
        config: PipelineConfig,
    ) -> MatchResult:
        before_masks = resize_masks_to_image_shape(before_masks, target_h, target_w)
        after_masks = resize_masks_to_image_shape(after_masks, target_h, target_w)

        before_masks, _ = deduplicate_masks(
            before_masks,
            duplicate_iou_threshold=config.duplicate_mask_iou_threshold,
        )
        after_masks, _ = deduplicate_masks(
            after_masks,
            duplicate_iou_threshold=config.duplicate_mask_iou_threshold,
        )

        save_debug = config.save_intermediate and self.output_dir is not None
        debug_root = self.output_dir if save_debug else None

        before_embeddings, before_embedding_indices = (
            self.dino_service.extract_embeddings_from_masks(
                before_rgb,
                before_masks,
                config,
                save_debug_crops=save_debug,
                debug_crop_dir=debug_root / "dino_before_crops" if debug_root else None,
            )
        )
        after_embeddings, after_embedding_indices = (
            self.dino_service.extract_embeddings_from_masks(
                aligned_after_rgb,
                after_masks,
                config,
                save_debug_crops=save_debug,
                debug_crop_dir=debug_root / "dino_after_crops" if debug_root else None,
            )
        )

        embedding_similarity_matrix = compute_embedding_similarity_matrix(
            before_embeddings,
            after_embeddings,
        )
        before_mask_to_embedding_idx = build_mask_to_embedding_index_map(
            before_embedding_indices
        )
        after_mask_to_embedding_idx = build_mask_to_embedding_index_map(
            after_embedding_indices
        )

        num_before = len(before_masks)
        num_after = len(after_masks)

        matched_before, matched_after, final_matches = self._stage1_spatial_dino(
            before_masks,
            after_masks,
            num_before,
            num_after,
            before_mask_to_embedding_idx,
            after_mask_to_embedding_idx,
            embedding_similarity_matrix,
            config,
        )

        self._stage2_dino_rescue(
            before_masks,
            after_masks,
            num_before,
            num_after,
            matched_before,
            matched_after,
            final_matches,
            before_mask_to_embedding_idx,
            after_mask_to_embedding_idx,
            embedding_similarity_matrix,
            config,
        )

        tentative_removed = [
            idx for idx in range(num_before) if idx not in matched_before
        ]
        tentative_new = [idx for idx in range(num_after) if idx not in matched_after]

        removed_indices = self._stage2b_onesided_removed(
            tentative_removed,
            before_masks,
            before_embeddings,
            before_mask_to_embedding_idx,
            before_rgb,
            aligned_after_rgb,
            config,
            debug_root,
            save_debug,
        )
        new_indices = self._stage2b_onesided_new(
            tentative_new,
            after_masks,
            after_embeddings,
            after_mask_to_embedding_idx,
            before_rgb,
            aligned_after_rgb,
            config,
            debug_root,
            save_debug,
        )

        removed_indices = self._stage3_local_ransac_removed(
            removed_indices,
            before_masks,
            before_rgb,
            aligned_after_rgb,
            config,
            debug_root,
            save_debug,
        )
        new_indices = self._stage3_local_ransac_new(
            new_indices,
            after_masks,
            before_rgb,
            aligned_after_rgb,
            config,
            debug_root,
            save_debug,
        )

        return MatchResult(
            matched_pairs=final_matches,
            removed_indices=removed_indices,
            new_indices=new_indices,
            before_masks_resized=before_masks,
            after_masks_resized=after_masks,
        )

    def _stage1_spatial_dino(
        self,
        before_masks: np.ndarray,
        after_masks: np.ndarray,
        num_before: int,
        num_after: int,
        before_mask_to_embedding_idx: dict[int, int],
        after_mask_to_embedding_idx: dict[int, int],
        embedding_similarity_matrix: np.ndarray,
        config: PipelineConfig,
    ) -> tuple[set[int], set[int], list[dict]]:
        spatial_candidate_matches: list[tuple] = []

        for before_idx in range(num_before):
            for after_idx in range(num_after):
                iou_score, containment_score = mask_overlap_metrics(
                    before_masks[before_idx],
                    after_masks[after_idx],
                )

                is_same_area_match = (
                    iou_score >= config.mask_iou_threshold
                    or containment_score >= config.mask_containment_threshold
                )
                if not is_same_area_match:
                    continue

                dino_similarity = lookup_pair_dino_similarity(
                    before_idx,
                    after_idx,
                    before_mask_to_embedding_idx,
                    after_mask_to_embedding_idx,
                    embedding_similarity_matrix,
                )
                if dino_similarity is None or dino_similarity < config.dino_similarity_threshold:
                    continue

                match_score = max(iou_score, containment_score)
                spatial_candidate_matches.append(
                    (
                        match_score,
                        before_idx,
                        after_idx,
                        iou_score,
                        containment_score,
                        dino_similarity,
                        "spatial_dino_validated",
                    )
                )

        spatial_candidate_matches.sort(reverse=True, key=lambda x: x[0])
        matched_before: set[int] = set()
        matched_after: set[int] = set()
        final_matches: list[dict] = []

        for entry in spatial_candidate_matches:
            _, before_idx, after_idx, iou_score, containment_score, dino_similarity, match_type = entry
            if before_idx in matched_before or after_idx in matched_after:
                continue

            matched_before.add(before_idx)
            matched_after.add(after_idx)
            final_matches.append(
                {
                    "before_idx": before_idx,
                    "after_idx": after_idx,
                    "match_type": match_type,
                    "match_score": float(entry[0]),
                    "mask_iou": float(iou_score),
                    "containment": float(containment_score),
                    "dino_similarity": float(dino_similarity),
                }
            )

        return matched_before, matched_after, final_matches

    def _stage2_dino_rescue(
        self,
        before_masks: np.ndarray,
        after_masks: np.ndarray,
        num_before: int,
        num_after: int,
        matched_before: set[int],
        matched_after: set[int],
        final_matches: list[dict],
        before_mask_to_embedding_idx: dict[int, int],
        after_mask_to_embedding_idx: dict[int, int],
        embedding_similarity_matrix: np.ndarray,
        config: PipelineConfig,
    ) -> None:
        dino_candidate_matches: list[tuple] = []

        for before_idx in range(num_before):
            if before_idx in matched_before or before_idx not in before_mask_to_embedding_idx:
                continue

            before_embedding_idx = before_mask_to_embedding_idx[before_idx]

            for after_idx in range(num_after):
                if after_idx in matched_after or after_idx not in after_mask_to_embedding_idx:
                    continue

                after_embedding_idx = after_mask_to_embedding_idx[after_idx]
                iou_score, containment_score = mask_overlap_metrics(
                    before_masks[before_idx],
                    after_masks[after_idx],
                )

                is_local_candidate = (
                    iou_score >= config.local_dino_min_iou
                    or containment_score >= config.local_dino_min_containment
                )
                if not is_local_candidate:
                    continue

                dino_similarity = float(
                    embedding_similarity_matrix[before_embedding_idx, after_embedding_idx]
                )
                if dino_similarity >= config.dino_similarity_threshold:
                    dino_candidate_matches.append(
                        (dino_similarity, before_idx, after_idx, iou_score, containment_score)
                    )

        dino_candidate_matches.sort(reverse=True, key=lambda x: x[0])

        for dino_similarity, before_idx, after_idx, iou_score, containment_score in dino_candidate_matches:
            if before_idx in matched_before or after_idx in matched_after:
                continue

            matched_before.add(before_idx)
            matched_after.add(after_idx)
            final_matches.append(
                {
                    "before_idx": before_idx,
                    "after_idx": after_idx,
                    "match_type": "local_dino_rescue",
                    "mask_iou": float(iou_score),
                    "containment": float(containment_score),
                    "dino_similarity": float(dino_similarity),
                }
            )

    def _stage2b_onesided_removed(
        self,
        tentative_removed: list[int],
        before_masks: np.ndarray,
        before_embeddings: np.ndarray,
        before_mask_to_embedding_idx: dict[int, int],
        before_rgb: np.ndarray,
        aligned_after_rgb: np.ndarray,
        config: PipelineConfig,
        debug_root: Path | None,
        save_debug: bool,
    ) -> list[int]:
        removed_indices: list[int] = []
        onesided_debug_dir = (
            debug_root / "dino_onesided_verification" if debug_root else None
        )

        for before_idx in tentative_removed:
            if before_idx not in before_mask_to_embedding_idx:
                removed_indices.append(before_idx)
                continue

            before_embedding_idx = before_mask_to_embedding_idx[before_idx]
            before_reference_embedding = before_embeddings[before_embedding_idx]

            result = self.dino_service.verify_onesided_removed_candidate(
                before_idx=before_idx,
                before_mask=before_masks[before_idx],
                before_reference_embedding=before_reference_embedding,
                before_image_rgb=before_rgb,
                aligned_after_image_rgb=aligned_after_rgb,
                config=config,
                save_debug_crops=save_debug,
                debug_dir=onesided_debug_dir,
            )

            if not result["suppress"]:
                removed_indices.append(before_idx)

        return removed_indices

    def _stage2b_onesided_new(
        self,
        tentative_new: list[int],
        after_masks: np.ndarray,
        after_embeddings: np.ndarray,
        after_mask_to_embedding_idx: dict[int, int],
        before_rgb: np.ndarray,
        aligned_after_rgb: np.ndarray,
        config: PipelineConfig,
        debug_root: Path | None,
        save_debug: bool,
    ) -> list[int]:
        new_indices: list[int] = []
        onesided_debug_dir = (
            debug_root / "dino_onesided_verification" if debug_root else None
        )

        for after_idx in tentative_new:
            if after_idx not in after_mask_to_embedding_idx:
                new_indices.append(after_idx)
                continue

            after_embedding_idx = after_mask_to_embedding_idx[after_idx]
            after_reference_embedding = after_embeddings[after_embedding_idx]

            result = self.dino_service.verify_onesided_new_candidate(
                after_idx=after_idx,
                after_mask=after_masks[after_idx],
                after_reference_embedding=after_reference_embedding,
                before_image_rgb=before_rgb,
                aligned_after_image_rgb=aligned_after_rgb,
                config=config,
                save_debug_crops=save_debug,
                debug_dir=onesided_debug_dir,
            )

            if not result["suppress"]:
                new_indices.append(after_idx)

        return new_indices

    def _stage3_local_ransac_removed(
        self,
        removed_indices: list[int],
        before_masks: np.ndarray,
        before_rgb: np.ndarray,
        aligned_after_rgb: np.ndarray,
        config: PipelineConfig,
        debug_root: Path | None,
        save_debug: bool,
    ) -> list[int]:
        verified: list[int] = []
        ransac_debug_dir = debug_root / "local_ransac_debug_crops" if debug_root else None

        for before_idx in removed_indices:
            support = self.verification_service.local_crop_ransac_support_for_mask(
                candidate_mask=before_masks[before_idx],
                before_image_rgb=before_rgb,
                aligned_after_image_rgb=aligned_after_rgb,
                config=config,
                debug_name=f"removed_candidate_{before_idx}" if save_debug else None,
                debug_dir=ransac_debug_dir,
            )
            if not support["has_local_support"]:
                verified.append(before_idx)

        return verified

    def _stage3_local_ransac_new(
        self,
        new_indices: list[int],
        after_masks: np.ndarray,
        before_rgb: np.ndarray,
        aligned_after_rgb: np.ndarray,
        config: PipelineConfig,
        debug_root: Path | None,
        save_debug: bool,
    ) -> list[int]:
        verified: list[int] = []
        ransac_debug_dir = debug_root / "local_ransac_debug_crops" if debug_root else None

        for after_idx in new_indices:
            support = self.verification_service.local_crop_ransac_support_for_mask(
                candidate_mask=after_masks[after_idx],
                before_image_rgb=before_rgb,
                aligned_after_image_rgb=aligned_after_rgb,
                config=config,
                debug_name=f"new_candidate_{after_idx}" if save_debug else None,
                debug_dir=ransac_debug_dir,
            )
            if not support["has_local_support"]:
                verified.append(after_idx)

        return verified


def get_matcher(
    mode: Literal["bbox", "mask"],
    config: PipelineConfig,
    dino_service: DinoService | None = None,
    verification_service: VerificationService | None = None,
    output_dir: Path | None = None,
):
    if mode == "bbox":
        return BBoxMatcher(iou_threshold=config.bbox_iou_threshold)
    if mode == "mask":
        if dino_service is not None and verification_service is not None:
            return AdvancedMaskMatcher(
                dino_service=dino_service,
                verification_service=verification_service,
                output_dir=output_dir,
            )
        return MaskMatcher(iou_threshold=config.mask_iou_threshold)
    raise ValueError(f"Unsupported matching mode: {mode}")
