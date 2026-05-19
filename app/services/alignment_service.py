from dataclasses import dataclass

import cv2
import numpy as np
import torch
from lightglue.utils import load_image, rbd

from app.core.config import PipelineConfig
from app.utils.image_utils import torch_image_to_rgb_numpy


@dataclass
class AlignmentResult:
    before_rgb: np.ndarray
    after_rgb: np.ndarray
    aligned_after_rgb: np.ndarray
    overlay_rgb: np.ndarray
    homography: np.ndarray
    inlier_ratio: float
    num_matches: int
    num_inliers: int


class AlignmentService:
    def __init__(self, extractor, matcher, device: torch.device):
        self.extractor = extractor
        self.matcher = matcher
        self.device = device

    def align(
        self,
        before_path: str,
        after_path: str,
        config: PipelineConfig | None = None,
    ) -> AlignmentResult:
        config = config or PipelineConfig()

        image0 = load_image(before_path).to(self.device)
        image1 = load_image(after_path).to(self.device)

        feats0 = self.extractor.extract(image0, resize=None)
        feats1 = self.extractor.extract(image1, resize=None)

        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]

        matches = matches01["matches"]
        points0 = feats0["keypoints"][matches[..., 0]]
        points1 = feats1["keypoints"][matches[..., 1]]

        num_matches = len(matches)
        if num_matches < 4:
            raise RuntimeError("Need at least 4 matches to compute homography.")

        pts0_np = points0.detach().cpu().numpy().astype(np.float32)
        pts1_np = points1.detach().cpu().numpy().astype(np.float32)

        H, inlier_mask = cv2.findHomography(
            pts1_np,
            pts0_np,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_threshold,
        )

        if H is None:
            raise RuntimeError("Homography estimation failed.")

        inlier_mask = inlier_mask.ravel().astype(bool)
        num_inliers = int(inlier_mask.sum())
        inlier_ratio = num_inliers / len(inlier_mask) if len(inlier_mask) else 0.0

        image0_np = torch_image_to_rgb_numpy(image0)
        image1_np = torch_image_to_rgb_numpy(image1)
        h0, w0 = image0_np.shape[:2]

        aligned_after_rgb = cv2.warpPerspective(image1_np, H, (w0, h0))
        overlay_rgb = cv2.addWeighted(image0_np, 0.5, aligned_after_rgb, 0.5, 0)

        return AlignmentResult(
            before_rgb=image0_np,
            after_rgb=image1_np,
            aligned_after_rgb=aligned_after_rgb,
            overlay_rgb=overlay_rgb,
            homography=H,
            inlier_ratio=inlier_ratio,
            num_matches=num_matches,
            num_inliers=num_inliers,
        )
