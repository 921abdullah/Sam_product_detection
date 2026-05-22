from typing import Literal

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.core.config import DEFAULT_SAM_PROMPTS, PipelineConfig
from app.schemas.drawer_change import (
    AlignmentStats,
    ChangeStats,
    DrawerChangeResponse,
    PipelineParameters,
)
from app.utils.file_utils import absolute_output_url, public_base_url, save_upload_to_temp

router = APIRouter(tags=["product_change"])


def _parse_prompts(prompts: str) -> list[str]:
    return [p.strip() for p in prompts.split(",") if p.strip()]


def _parameters_from_config(config: PipelineConfig) -> PipelineParameters:
    return PipelineParameters(
        sam_conf=config.sam_conf,
        mask_iou_threshold=config.mask_iou_threshold,
        mask_containment_threshold=config.mask_containment_threshold,
        dino_similarity_threshold=config.dino_similarity_threshold,
    )


@router.post("/product_change", response_model=DrawerChangeResponse)
async def analyze_product_change(
    request: Request,
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    mode: Literal["bbox", "mask"] = Form("mask"),
    prompts: str = Form(",".join(DEFAULT_SAM_PROMPTS)),
    sam_conf: float = Form(0.40, description="SAM3 detection confidence"),
    mask_iou_threshold: float = Form(
        0.80,
        description="Minimum mask IoU for spatial matching",
    ),
    mask_containment_threshold: float = Form(
        0.70,
        description="Minimum mask containment for spatial matching",
    ),
    dino_similarity_threshold: float = Form(
        0.85,
        description="Minimum DINOv2 cosine similarity for visual matching",
    ),
):
    pipeline = request.app.state.pipeline
    outputs_root = request.app.state.outputs_root
    temp_dir = request.app.state.temp_dir

    before_bytes = await before_image.read()
    after_bytes = await after_image.read()

    before_path = save_upload_to_temp(before_bytes, temp_dir, "before")
    after_path = save_upload_to_temp(after_bytes, temp_dir, "after")

    config = PipelineConfig(
        sam_prompts=_parse_prompts(prompts),
        sam_conf=sam_conf,
        matching_mode=mode,
        mask_iou_threshold=mask_iou_threshold,
        mask_containment_threshold=mask_containment_threshold,
        dino_similarity_threshold=dino_similarity_threshold,
    )

    result = pipeline.run(
        before_path=str(before_path),
        after_path=str(after_path),
        config=config,
    )

    return DrawerChangeResponse(
        mode=mode,
        status="success",
        alignment=AlignmentStats(
            num_matches=result.alignment.num_matches,
            num_inliers=result.alignment.num_inliers,
            inlier_ratio=round(result.alignment.inlier_ratio, 4),
        ),
        changes=ChangeStats(
            new_items=len(result.match_result.new_indices),
            removed_items=len(result.match_result.removed_indices),
            matched_items=len(result.match_result.matched_pairs),
        ),
        parameters=_parameters_from_config(config),
        output_image=absolute_output_url(
            result.output_image_path,
            outputs_root,
            public_base_url(request),
        ),
    )
