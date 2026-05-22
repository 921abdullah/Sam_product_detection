"""Run the drawer change pipeline locally without starting the API."""

import argparse
from pathlib import Path

from app.core.config import DEFAULT_SAM_PROMPTS, PipelineConfig
from app.core.model_registry import ModelRegistry
from app.main import OUTPUT_DIR, _build_pipeline
from app.utils.file_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Run drawer change detection pipeline")
    parser.add_argument("--before", required=True, help="Path to before image")
    parser.add_argument("--after", required=True, help="Path to after image")
    parser.add_argument(
        "--mode",
        choices=["bbox", "mask"],
        default="mask",
        help="Matching mode",
    )
    parser.add_argument(
        "--prompts",
        default=",".join(DEFAULT_SAM_PROMPTS),
        help="Comma-separated SAM prompts",
    )
    parser.add_argument("--mask-iou-threshold", type=float, default=0.80)
    parser.add_argument("--mask-containment-threshold", type=float, default=0.70)
    parser.add_argument("--dino-similarity-threshold", type=float, default=0.85)
    parser.add_argument("--sam-conf", type=float, default=0.40)
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    config = PipelineConfig(
        sam_prompts=[p.strip() for p in args.prompts.split(",") if p.strip()],
        matching_mode=args.mode,
        sam_conf=args.sam_conf,
        mask_iou_threshold=args.mask_iou_threshold,
        mask_containment_threshold=args.mask_containment_threshold,
        dino_similarity_threshold=args.dino_similarity_threshold,
    )

    print(f"Loading models on device...")
    models = ModelRegistry()
    models.load_models(config)

    pipeline = _build_pipeline(models)
    print(f"Running pipeline (mode={config.matching_mode})...")

    result = pipeline.run(
        before_path=args.before,
        after_path=args.after,
        config=config,
    )

    print("\nAlignment:")
    print(f"  matches: {result.alignment.num_matches}")
    print(f"  inliers: {result.alignment.num_inliers}")
    print(f"  inlier ratio: {result.alignment.inlier_ratio:.2%}")

    print("\nChanges:")
    print(f"  new: {len(result.match_result.new_indices)}")
    print(f"  removed: {len(result.match_result.removed_indices)}")
    print(f"  matched: {len(result.match_result.matched_pairs)}")

    print("\nOutputs:")
    print(f"  {result.output_image_path}")
    if result.aligned_after_path:
        print(f"  {result.aligned_after_path}")
    if result.overlay_path:
        print(f"  {result.overlay_path}")


if __name__ == "__main__":
    main()
