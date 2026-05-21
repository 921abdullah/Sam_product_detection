from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import PipelineConfig
from app.services.alignment_service import AlignmentResult, AlignmentService
from app.services.matching_service import MatchResult, get_matcher
from app.services.mask_change_matcher import MaskChangeMatcher
from app.services.rendering_service import RenderingService
from app.services.segmentation_service import SegmentationResult, SegmentationService
from app.utils.file_utils import unique_output_path
from app.utils.image_utils import save_rgb_as_bgr


@dataclass
class PipelineResult:
    alignment: AlignmentResult
    before_detection: SegmentationResult
    after_detection: SegmentationResult
    match_result: MatchResult
    output_image: np.ndarray
    output_image_path: Path | None = None
    aligned_after_path: Path | None = None
    overlay_path: Path | None = None


class DrawerChangePipeline:
    def __init__(
        self,
        alignment_service: AlignmentService,
        segmentation_service: SegmentationService,
        rendering_service: RenderingService,
        mask_change_matcher: MaskChangeMatcher,
        output_dir: str | Path,
    ):
        self.alignment_service = alignment_service
        self.segmentation_service = segmentation_service
        self.rendering_service = rendering_service
        self.mask_change_matcher = mask_change_matcher
        self.output_dir = Path(output_dir)

    def run(
        self,
        before_path: str,
        after_path: str,
        config: PipelineConfig,
    ) -> PipelineResult:
        alignment = self.alignment_service.align(
            before_path,
            after_path,
            config=config,
        )

        aligned_after_path = unique_output_path(
            self.output_dir, "aligned_after"
        )
        save_rgb_as_bgr(str(aligned_after_path), alignment.aligned_after_rgb)

        overlay_path = None
        if config.save_intermediate:
            overlay_path = unique_output_path(self.output_dir, "alignment_overlay")
            save_rgb_as_bgr(str(overlay_path), alignment.overlay_rgb)

        before_detection = self.segmentation_service.segment_image(
            before_path,
            config.sam_prompts,
        )

        after_detection = self.segmentation_service.segment_image(
            str(aligned_after_path),
            config.sam_prompts,
        )

        if config.matching_mode == "bbox":
            matcher = get_matcher(
                mode="bbox",
                bbox_threshold=config.bbox_iou_threshold,
                mask_threshold=config.mask_iou_threshold,
            )
            match_result = matcher.match(
                before_detection.boxes,
                after_detection.boxes,
            )
        else:
            match_result = self.mask_change_matcher.match(
                before_detection.masks,
                after_detection.masks,
                before_rgb=alignment.before_rgb,
                aligned_after_rgb=alignment.aligned_after_rgb,
                config=config,
                output_dir=self.output_dir if config.save_intermediate else None,
            )

        output_image = self.rendering_service.render(
            mode=config.matching_mode,
            aligned_after_rgb=alignment.aligned_after_rgb,
            before_detection=before_detection,
            after_detection=after_detection,
            match_result=match_result,
            before_masks_resized=match_result.before_masks_resized,
            after_masks_resized=match_result.after_masks_resized,
        )

        prefix = (
            "after_new_removed_masks"
            if config.matching_mode == "mask"
            else "after_new_removed_boxes"
        )
        output_image_path = unique_output_path(self.output_dir, prefix)
        self.rendering_service.save_image(output_image_path, output_image)

        return PipelineResult(
            alignment=alignment,
            before_detection=before_detection,
            after_detection=after_detection,
            match_result=match_result,
            output_image=output_image,
            output_image_path=output_image_path,
            aligned_after_path=aligned_after_path,
            overlay_path=overlay_path,
        )
