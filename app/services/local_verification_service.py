from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue.utils import rbd

from app.core.config import PipelineConfig
from app.utils.image_utils import rgb_numpy_to_lightglue_tensor, save_rgb_as_bgr
from app.utils.mask_utils import (
    dilate_binary_mask,
    mask_to_padded_bbox,
    points_inside_mask,
)


class LocalVerificationService:
    def __init__(self, extractor, matcher, device: torch.device):
        self.extractor = extractor
        self.matcher = matcher
        self.device = device

    def run_local_lightglue_matching(
        self,
        before_crop_rgb: np.ndarray,
        after_crop_rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        before_crop_tensor = rgb_numpy_to_lightglue_tensor(before_crop_rgb, self.device)
        after_crop_tensor = rgb_numpy_to_lightglue_tensor(after_crop_rgb, self.device)

        with torch.inference_mode():
            local_feats0 = self.extractor.extract(before_crop_tensor, resize=None)
            local_feats1 = self.extractor.extract(after_crop_tensor, resize=None)
            local_matches01 = self.matcher(
                {"image0": local_feats0, "image1": local_feats1}
            )

        local_feats0, local_feats1, local_matches01 = [
            rbd(x) for x in [local_feats0, local_feats1, local_matches01]
        ]

        local_matches = local_matches01["matches"]
        if local_matches is None or len(local_matches) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        local_before_points = (
            local_feats0["keypoints"][local_matches[..., 0]]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        local_after_points = (
            local_feats1["keypoints"][local_matches[..., 1]]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return local_before_points, local_after_points

    def crop_ransac_support_for_mask(
        self,
        candidate_mask: np.ndarray,
        before_image_rgb: np.ndarray,
        aligned_after_image_rgb: np.ndarray,
        config: PipelineConfig,
        debug_name: str,
        debug_dir: Path | None = None,
    ) -> dict:
        image_h, image_w = before_image_rgb.shape[:2]

        crop_bbox = mask_to_padded_bbox(
            candidate_mask,
            padding_px=config.local_ransac_crop_padding_px,
            image_h=image_h,
            image_w=image_w,
        )
        if crop_bbox is None:
            return self._support_result(False, 0, 0, 0, 0.0, "empty_mask")

        x1, y1, x2, y2 = crop_bbox
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < config.local_ransac_min_crop_side or crop_h < config.local_ransac_min_crop_side:
            return self._support_result(False, 0, 0, 0, 0.0, "crop_too_small")

        before_crop_rgb = before_image_rgb[y1:y2, x1:x2].copy()
        after_crop_rgb = aligned_after_image_rgb[y1:y2, x1:x2].copy()
        candidate_crop_mask = dilate_binary_mask(
            candidate_mask[y1:y2, x1:x2].copy(),
            dilation_px=config.local_ransac_mask_dilation_px,
        )

        if config.save_local_ransac_debug_crops and debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_rgb_as_bgr(str(debug_dir / f"{debug_name}_before_crop.jpg"), before_crop_rgb)
            save_rgb_as_bgr(str(debug_dir / f"{debug_name}_after_crop.jpg"), after_crop_rgb)
            cv2.imwrite(
                str(debug_dir / f"{debug_name}_candidate_mask.jpg"),
                candidate_crop_mask * 255,
            )

        local_before_points, local_after_points = self.run_local_lightglue_matching(
            before_crop_rgb,
            after_crop_rgb,
        )
        num_raw_matches = len(local_before_points)
        if num_raw_matches == 0:
            return self._support_result(False, 0, 0, 0, 0.0, "no_local_matches")

        before_inside = points_inside_mask(local_before_points, candidate_crop_mask)
        after_inside = points_inside_mask(local_after_points, candidate_crop_mask)
        region_keep = before_inside & after_inside

        region_before_points = local_before_points[region_keep]
        region_after_points = local_after_points[region_keep]
        num_region_matches = len(region_before_points)

        if num_region_matches < config.local_ransac_min_matches:
            return self._support_result(
                False,
                num_raw_matches,
                num_region_matches,
                0,
                0.0,
                "not_enough_region_matches",
            )

        local_affine, local_inliers = cv2.estimateAffinePartial2D(
            region_after_points.astype(np.float32),
            region_before_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=config.local_ransac_reproj_threshold,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )

        if local_affine is None or local_inliers is None:
            return self._support_result(
                False,
                num_raw_matches,
                num_region_matches,
                0,
                0.0,
                "local_ransac_failed",
            )

        num_inliers = int(local_inliers.sum())
        inlier_ratio = num_inliers / max(num_region_matches, 1)
        has_local_support = (
            num_inliers >= config.local_ransac_min_inliers
            and inlier_ratio >= config.local_ransac_min_inlier_ratio
        )

        return self._support_result(
            has_local_support,
            num_raw_matches,
            num_region_matches,
            num_inliers,
            float(inlier_ratio),
            "supported" if has_local_support else "weak_local_support",
        )

    @staticmethod
    def _support_result(
        has_local_support: bool,
        num_raw_matches: int,
        num_region_matches: int,
        num_inliers: int,
        inlier_ratio: float,
        reason: str,
    ) -> dict:
        return {
            "has_local_support": has_local_support,
            "num_raw_matches": num_raw_matches,
            "num_region_matches": num_region_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": inlier_ratio,
            "reason": reason,
        }
