from dataclasses import replace
from pathlib import Path

import numpy as np

from app.core.config import PipelineConfig
from app.services.embedding_service import EmbeddingService
from app.services.local_verification_service import LocalVerificationService
from app.services.matching_service import MatchResult
from app.utils.mask_utils import (
    deduplicate_masks,
    mask_overlap_metrics,
    resize_masks_to_image_shape,
)


class MaskChangeMatcher:
    """Mask matching pipeline from light.py: dedupe, overlap, DINO rescue, local RANSAC."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None,
        local_verification_service: LocalVerificationService,
    ):
        self.embedding_service = embedding_service
        self.local_verification_service = local_verification_service

    def match(
        self,
        before_masks: np.ndarray,
        after_masks: np.ndarray,
        before_rgb: np.ndarray,
        aligned_after_rgb: np.ndarray,
        config: PipelineConfig,
        output_dir: Path | None = None,
    ) -> MatchResult:
        target_h, target_w = aligned_after_rgb.shape[:2]

        before_masks_resized = resize_masks_to_image_shape(
            before_masks, target_h, target_w
        )
        after_masks_resized = resize_masks_to_image_shape(
            after_masks, target_h, target_w
        )

        before_masks_resized, _ = deduplicate_masks(
            before_masks_resized,
            duplicate_iou_threshold=config.duplicate_mask_iou_threshold,
        )
        after_masks_resized, _ = deduplicate_masks(
            after_masks_resized,
            duplicate_iou_threshold=config.duplicate_mask_iou_threshold,
        )

        num_before = len(before_masks_resized)
        num_after = len(after_masks_resized)

        matched_before: set[int] = set()
        matched_after: set[int] = set()
        final_matches: list[dict] = []

        spatial_candidate_matches: list[tuple] = []
        for before_idx in range(num_before):
            for after_idx in range(num_after):
                iou_score, containment_score = mask_overlap_metrics(
                    before_masks_resized[before_idx],
                    after_masks_resized[after_idx],
                )
                is_same_area_match = (
                    iou_score >= config.mask_iou_threshold
                    or containment_score >= config.mask_containment_threshold
                )
                if is_same_area_match:
                    match_score = max(iou_score, containment_score)
                    spatial_candidate_matches.append(
                        (
                            match_score,
                            before_idx,
                            after_idx,
                            iou_score,
                            containment_score,
                        )
                    )

        spatial_candidate_matches.sort(reverse=True, key=lambda x: x[0])

        for (
            match_score,
            before_idx,
            after_idx,
            iou_score,
            containment_score,
        ) in spatial_candidate_matches:
            if before_idx in matched_before or after_idx in matched_after:
                continue
            matched_before.add(before_idx)
            matched_after.add(after_idx)
            final_matches.append(
                {
                    "before_idx": before_idx,
                    "after_idx": after_idx,
                    "match_type": "same_area_overlap",
                    "match_score": float(match_score),
                    "mask_iou": float(iou_score),
                    "containment": float(containment_score),
                    "dino_similarity": None,
                }
            )

        if self.embedding_service is not None:
            dino_debug_before = (
                output_dir / "dino_before_crops" if output_dir else None
            )
            dino_debug_after = (
                output_dir / "dino_after_crops" if output_dir else None
            )

            before_embeddings, before_embedding_indices = (
                self.embedding_service.extract_from_masks(
                    before_rgb,
                    before_masks_resized,
                    config,
                    debug_crop_dir=dino_debug_before,
                )
            )
            after_embeddings, after_embedding_indices = (
                self.embedding_service.extract_from_masks(
                    aligned_after_rgb,
                    after_masks_resized,
                    config,
                    debug_crop_dir=dino_debug_after,
                )
            )

            embedding_similarity_matrix = EmbeddingService.similarity_matrix(
                before_embeddings,
                after_embeddings,
            )

            before_mask_to_embedding_idx = {
                original_mask_idx: embedding_idx
                for embedding_idx, original_mask_idx in enumerate(
                    before_embedding_indices
                )
            }
            after_mask_to_embedding_idx = {
                original_mask_idx: embedding_idx
                for embedding_idx, original_mask_idx in enumerate(
                    after_embedding_indices
                )
            }

            dino_candidate_matches: list[tuple] = []
            for before_idx in range(num_before):
                if before_idx in matched_before:
                    continue
                if before_idx not in before_mask_to_embedding_idx:
                    continue
                before_embedding_idx = before_mask_to_embedding_idx[before_idx]

                for after_idx in range(num_after):
                    if after_idx in matched_after:
                        continue
                    if after_idx not in after_mask_to_embedding_idx:
                        continue
                    after_embedding_idx = after_mask_to_embedding_idx[after_idx]

                    iou_score, containment_score = mask_overlap_metrics(
                        before_masks_resized[before_idx],
                        after_masks_resized[after_idx],
                    )
                    is_local_candidate = (
                        iou_score >= config.local_dino_min_iou
                        or containment_score >= config.local_dino_min_containment
                    )
                    if not is_local_candidate:
                        continue

                    dino_similarity = float(
                        embedding_similarity_matrix[
                            before_embedding_idx,
                            after_embedding_idx,
                        ]
                    )
                    if dino_similarity >= config.dino_similarity_threshold:
                        dino_candidate_matches.append(
                            (
                                dino_similarity,
                                before_idx,
                                after_idx,
                                iou_score,
                                containment_score,
                            )
                        )

            dino_candidate_matches.sort(reverse=True, key=lambda x: x[0])

            for (
                dino_similarity,
                before_idx,
                after_idx,
                iou_score,
                containment_score,
            ) in dino_candidate_matches:
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

        removed_indices = [
            idx for idx in range(num_before) if idx not in matched_before
        ]
        new_indices = [
            idx for idx in range(num_after) if idx not in matched_after
        ]

        ransac_debug_dir = (
            output_dir / "local_ransac_debug_crops" if output_dir else None
        )

        verified_removed_indices: list[int] = []
        for before_idx in removed_indices:
            support = self.local_verification_service.crop_ransac_support_for_mask(
                candidate_mask=before_masks_resized[before_idx],
                before_image_rgb=before_rgb,
                aligned_after_image_rgb=aligned_after_rgb,
                config=config,
                debug_name=f"removed_candidate_{before_idx}",
                debug_dir=ransac_debug_dir,
            )
            if not support["has_local_support"]:
                verified_removed_indices.append(before_idx)

        verified_new_indices: list[int] = []
        for after_idx in new_indices:
            support = self.local_verification_service.crop_ransac_support_for_mask(
                candidate_mask=after_masks_resized[after_idx],
                before_image_rgb=before_rgb,
                aligned_after_image_rgb=aligned_after_rgb,
                config=config,
                debug_name=f"new_candidate_{after_idx}",
                debug_dir=ransac_debug_dir,
            )
            if not support["has_local_support"]:
                verified_new_indices.append(after_idx)

        return replace(
            MatchResult(
                matched_pairs=final_matches,
                removed_indices=verified_removed_indices,
                new_indices=verified_new_indices,
            ),
            before_masks_resized=before_masks_resized,
            after_masks_resized=after_masks_resized,
        )
