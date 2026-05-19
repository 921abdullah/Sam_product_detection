from typing import Literal

from pydantic import BaseModel


class AlignmentStats(BaseModel):
    num_matches: int
    num_inliers: int
    inlier_ratio: float


class ChangeStats(BaseModel):
    new_items: int
    removed_items: int
    matched_items: int


class DrawerChangeResponse(BaseModel):
    mode: Literal["bbox", "mask"]
    status: str = "success"
    alignment: AlignmentStats
    changes: ChangeStats
    output_image: str
