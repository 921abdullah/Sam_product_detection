from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import PipelineConfig
from app.services.alignment_service import AlignmentResult, AlignmentService
from app.services.dino_service import DinoService
from app.services.matching_service import MatchResult, get_matcher
from app.services.rendering_service import RenderingService
from app.services.segmentation_service import SegmentationResult, SegmentationService
from app.services.verification_service import VerificationService
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
        dino_service: DinoService | None,
        verification_service: VerificationService | None,
        output_dir: str | Path,
    ):
        self.alignment_service = alignment_service
        self.segmentation_service = segmentation_service
        self.rendering_service = rendering_service
        self.dino_service = dino_service
        self.verification_service = verification_service
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

        matcher = get_matcher(
            mode=config.matching_mode,
            config=config,
            dino_service=self.dino_service,
            verification_service=self.verification_service,
            output_dir=self.output_dir if config.matching_mode == "mask" else None,
        )

        target_h, target_w = alignment.aligned_after_rgb.shape[:2]

        if config.matching_mode == "bbox":
            match_result = matcher.match(
                before_detection.boxes,
                after_detection.boxes,
            )
        else:
            match_result = matcher.match(
                before_detection.masks,
                after_detection.masks,
                target_h,
                target_w,
                alignment.before_rgb,
                alignment.aligned_after_rgb,
                config,
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
