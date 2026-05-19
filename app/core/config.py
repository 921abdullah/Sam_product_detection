from dataclasses import dataclass, field
from typing import Literal


DEFAULT_SAM_PROMPTS = [
    "item",
    "gold bolt",
    "gold nut",
    "blue bolt",
    "blue nut",
]


@dataclass
class PipelineConfig:
    sam_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_SAM_PROMPTS))
    sam_conf: float = 0.50
    matching_mode: Literal["bbox", "mask"] = "mask"
    bbox_iou_threshold: float = 0.15
    mask_iou_threshold: float = 0.50
    max_num_keypoints: int = 2048
    ransac_threshold: float = 5.0
    save_intermediate: bool = True
