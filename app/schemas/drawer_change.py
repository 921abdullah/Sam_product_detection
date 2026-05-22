from typing import Literal

from pydantic import BaseModel, Field


class AlignmentStats(BaseModel):
    num_matches: int
    num_inliers: int
    inlier_ratio: float


class ChangeStats(BaseModel):
    new_items: int
    removed_items: int
    matched_items: int


class PipelineParameters(BaseModel):
    """Thresholds used for this run (echoed from request / defaults)."""

    sam_conf: float = Field(
        description="SAM3 detection confidence threshold"
    )
    mask_iou_threshold: float = Field(
        description="Minimum mask IoU"
    )
    mask_containment_threshold: float = Field(
        description="Minimum containment score for same-area matching"
    )
    dino_similarity_threshold: float = Field(
        description="Minimum DINOv2 cosine similarity to accept a visual match"
    )


class ObjectMatchStats(BaseModel):
    before_idx: int
    after_idx: int
    match_type: str | None = None
    mask_iou: float | None = None
    containment: float | None = None
    dino_similarity: float | None = None


class DrawerChangeResponse(BaseModel):
    mode: Literal["bbox", "mask"]
    status: str = "success"
    alignment: AlignmentStats
    changes: ChangeStats
    parameters: PipelineParameters
    matches: list[ObjectMatchStats] = Field(
        default_factory=list,
        description="Per-object match metrics (IoU, containment, DINO where applicable)",
    )
    output_image: str
