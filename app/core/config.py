from dataclasses import dataclass, field
from typing import Literal


DEFAULT_SAM_PROMPTS = [
    "item",
    "hydraulic pressure test ports",
]


@dataclass
class PipelineConfig:
    sam_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_SAM_PROMPTS))
    sam_conf: float = 0.50
    matching_mode: Literal["bbox", "mask"] = "mask"
    bbox_iou_threshold: float = 0.15
    mask_iou_threshold: float = 0.35
    mask_containment_threshold: float = 0.65
    duplicate_mask_iou_threshold: float = 0.80
    max_num_keypoints: int = 2048
    ransac_threshold: float = 5.0
    save_intermediate: bool = True

    # DINOv2 rescue (stage 2)
    dino_model_name: str = "facebook/dinov2-base"
    dino_batch_size: int = 16
    dino_crop_padding: int = 0
    dino_background_value: int = 127
    dino_similarity_threshold: float = 0.85
    local_dino_min_iou: float = 0.05
    local_dino_min_containment: float = 0.25
    save_dino_debug_crops: bool = False

    # Local crop LightGlue + RANSAC verification (stage 3)
    local_ransac_crop_padding_px: int = 18
    local_ransac_mask_dilation_px: int = 8
    local_ransac_min_crop_side: int = 24
    local_ransac_min_matches: int = 4
    local_ransac_min_inliers: int = 4
    local_ransac_min_inlier_ratio: float = 0.50
    local_ransac_reproj_threshold: float = 4.0
    save_local_ransac_debug_crops: bool = False
