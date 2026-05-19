from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from app.services.matching_service import MatchResult
from app.services.segmentation_service import SegmentationResult
from app.utils.image_utils import draw_boxes_only, save_rgb_as_bgr
from app.utils.mask_utils import (
    draw_mask_contour,
    overlay_colored_mask,
    put_label_near_mask,
    resize_masks_to_image_shape,
)


GREEN = (0, 255, 0)
RED = (255, 0, 0)


class RenderingService:
    def render(
        self,
        mode: Literal["bbox", "mask"],
        aligned_after_rgb: np.ndarray,
        before_detection: SegmentationResult,
        after_detection: SegmentationResult,
        match_result: MatchResult,
        before_masks_resized: np.ndarray | None = None,
        after_masks_resized: np.ndarray | None = None,
    ) -> np.ndarray:
        if mode == "bbox":
            return self.render_bbox_changes(
                aligned_after_rgb,
                before_detection,
                after_detection,
                match_result,
            )

        return self.render_mask_changes(
            aligned_after_rgb,
            before_detection,
            after_detection,
            match_result,
            before_masks_resized=before_masks_resized,
            after_masks_resized=after_masks_resized,
        )

    def render_bbox_changes(
        self,
        aligned_after_rgb: np.ndarray,
        before_detection: SegmentationResult,
        after_detection: SegmentationResult,
        match_result: MatchResult,
    ) -> np.ndarray:
        output = aligned_after_rgb.copy()
        before_boxes = before_detection.boxes
        after_boxes = after_detection.boxes

        for after_idx in match_result.new_indices:
            x1, y1, x2, y2 = after_boxes[after_idx].tolist()
            cv2.rectangle(output, (x1, y1), (x2, y2), GREEN, 4)
            cv2.putText(
                output,
                "NEW",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                GREEN,
                3,
            )

        for before_idx in match_result.removed_indices:
            x1, y1, x2, y2 = before_boxes[before_idx].tolist()
            cv2.rectangle(output, (x1, y1), (x2, y2), RED, 4)
            cv2.putText(
                output,
                "REMOVED",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                RED,
                3,
            )

        return output

    def render_mask_changes(
        self,
        aligned_after_rgb: np.ndarray,
        before_detection: SegmentationResult,
        after_detection: SegmentationResult,
        match_result: MatchResult,
        before_masks_resized: np.ndarray | None = None,
        after_masks_resized: np.ndarray | None = None,
    ) -> np.ndarray:
        target_h, target_w = aligned_after_rgb.shape[:2]

        if before_masks_resized is None:
            before_masks_resized = resize_masks_to_image_shape(
                before_detection.masks, target_h, target_w
            )
        if after_masks_resized is None:
            after_masks_resized = resize_masks_to_image_shape(
                after_detection.masks, target_h, target_w
            )

        output = aligned_after_rgb.copy()

        for after_idx in match_result.new_indices:
            new_mask = after_masks_resized[after_idx]
            output = overlay_colored_mask(output, new_mask, GREEN, alpha=0.45)
            output = draw_mask_contour(output, new_mask, GREEN, thickness=3)
            output = put_label_near_mask(output, new_mask, "NEW", GREEN)

        for before_idx in match_result.removed_indices:
            removed_mask = before_masks_resized[before_idx]
            output = overlay_colored_mask(output, removed_mask, RED, alpha=0.45)
            output = draw_mask_contour(output, removed_mask, RED, thickness=3)
            output = put_label_near_mask(output, removed_mask, "REMOVED", RED)

        return output

    def render_detection_boxes(
        self,
        image_rgb: np.ndarray,
        detection: SegmentationResult,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        return draw_boxes_only(image_rgb, detection.boxes, color=color, thickness=3)

    def save_image(self, path: Path, image_rgb: np.ndarray) -> Path:
        save_rgb_as_bgr(str(path), image_rgb)
        return path
