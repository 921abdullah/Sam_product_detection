from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from app.core.config import PipelineConfig
from app.utils.image_utils import save_rgb_as_bgr


def create_masked_object_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    padding: int,
    background_value: int,
) -> np.ndarray | None:
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    image_h, image_w = image_rgb.shape[:2]
    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, image_w)
    y2 = min(int(ys.max()) + padding + 1, image_h)

    crop_rgb = image_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].astype(bool)

    masked_crop = np.full_like(crop_rgb, fill_value=background_value, dtype=np.uint8)
    masked_crop[crop_mask] = crop_rgb[crop_mask]
    return masked_crop


def compute_embedding_similarity_matrix(
    before_embeddings: np.ndarray,
    after_embeddings: np.ndarray,
) -> np.ndarray:
    if len(before_embeddings) == 0 or len(after_embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return before_embeddings @ after_embeddings.T


def build_mask_to_embedding_index_map(embedding_indices: list[int]) -> dict[int, int]:
    return {
        original_mask_idx: embedding_idx
        for embedding_idx, original_mask_idx in enumerate(embedding_indices)
    }


def lookup_pair_dino_similarity(
    before_idx: int,
    after_idx: int,
    before_mask_to_embedding_idx: dict[int, int],
    after_mask_to_embedding_idx: dict[int, int],
    embedding_similarity_matrix: np.ndarray,
) -> float | None:
    if before_idx not in before_mask_to_embedding_idx:
        return None
    if after_idx not in after_mask_to_embedding_idx:
        return None

    before_embedding_idx = before_mask_to_embedding_idx[before_idx]
    after_embedding_idx = after_mask_to_embedding_idx[after_idx]
    return float(
        embedding_similarity_matrix[before_embedding_idx, after_embedding_idx]
    )


def embedding_cosine_similarity(embedding_a: np.ndarray | None, embedding_b: np.ndarray | None) -> float:
    if embedding_a is None or embedding_b is None:
        return 0.0
    return float(np.dot(embedding_a, embedding_b))


class DinoService:
    def __init__(
        self,
        processor: AutoImageProcessor,
        model: AutoModel,
        device: torch.device,
    ):
        self.processor = processor
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name: str, device: torch.device) -> "DinoService":
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        return cls(processor, model, device)

    def extract_embeddings_from_masks(
        self,
        image_rgb: np.ndarray,
        masks: np.ndarray,
        config: PipelineConfig,
        save_debug_crops: bool = False,
        debug_crop_dir: str | Path | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        pil_crops: list[Image.Image] = []
        valid_mask_indices: list[int] = []

        if save_debug_crops and debug_crop_dir is not None:
            Path(debug_crop_dir).mkdir(parents=True, exist_ok=True)

        for mask_idx, mask in enumerate(masks):
            crop = create_masked_object_crop(
                image_rgb=image_rgb,
                mask=mask,
                padding=config.dino_crop_padding,
                background_value=config.dino_background_value,
            )
            if crop is None:
                continue

            pil_crops.append(Image.fromarray(crop).convert("RGB"))
            valid_mask_indices.append(mask_idx)

            if save_debug_crops and debug_crop_dir is not None:
                save_rgb_as_bgr(
                    str(Path(debug_crop_dir) / f"object_{mask_idx}.jpg"),
                    crop,
                )

        if len(pil_crops) == 0:
            return np.empty((0, 0), dtype=np.float32), []

        all_embeddings: list[torch.Tensor] = []
        batch_size = config.dino_batch_size

        for start_idx in range(0, len(pil_crops), batch_size):
            batch_crops = pil_crops[start_idx : start_idx + batch_size]
            inputs = self.processor(images=batch_crops, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.detach().cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
        return embeddings, valid_mask_indices

    def compute_embedding_for_rgb_crop(self, crop_rgb: np.ndarray | None) -> np.ndarray | None:
        if crop_rgb is None:
            return None

        pil_crop = Image.fromarray(crop_rgb).convert("RGB")
        inputs = self.processor(images=[pil_crop], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.detach().cpu().numpy()[0].astype(np.float32)

    def verify_onesided_removed_candidate(
        self,
        before_idx: int,
        before_mask: np.ndarray,
        before_reference_embedding: np.ndarray,
        before_image_rgb: np.ndarray,
        aligned_after_image_rgb: np.ndarray,
        config: PipelineConfig,
        save_debug_crops: bool = False,
        debug_dir: str | Path | None = None,
    ) -> dict:
        if before_reference_embedding is None:
            return {"suppress": False, "similarity": 0.0, "reason": "no_before_embedding"}

        reference_crop_rgb = create_masked_object_crop(
            before_image_rgb,
            before_mask,
            config.dino_crop_padding,
            config.dino_background_value,
        )
        opposite_crop_rgb = create_masked_object_crop(
            aligned_after_image_rgb,
            before_mask,
            config.dino_crop_padding,
            config.dino_background_value,
        )

        if opposite_crop_rgb is None:
            return {"suppress": False, "similarity": 0.0, "reason": "empty_opposite_crop"}

        opposite_embedding = self.compute_embedding_for_rgb_crop(opposite_crop_rgb)
        similarity = embedding_cosine_similarity(
            before_reference_embedding,
            opposite_embedding,
        )

        if save_debug_crops and debug_dir is not None:
            self._save_onesided_debug_crops(
                Path(debug_dir) / "removed_candidates",
                f"before_{before_idx}",
                reference_crop_rgb,
                opposite_crop_rgb,
            )

        suppress = similarity >= config.dino_similarity_threshold
        return {
            "suppress": suppress,
            "similarity": similarity,
            "reason": "high_opposite_similarity" if suppress else "low_opposite_similarity",
        }

    def verify_onesided_new_candidate(
        self,
        after_idx: int,
        after_mask: np.ndarray,
        after_reference_embedding: np.ndarray,
        before_image_rgb: np.ndarray,
        aligned_after_image_rgb: np.ndarray,
        config: PipelineConfig,
        save_debug_crops: bool = False,
        debug_dir: str | Path | None = None,
    ) -> dict:
        if after_reference_embedding is None:
            return {"suppress": False, "similarity": 0.0, "reason": "no_after_embedding"}

        reference_crop_rgb = create_masked_object_crop(
            aligned_after_image_rgb,
            after_mask,
            config.dino_crop_padding,
            config.dino_background_value,
        )
        opposite_crop_rgb = create_masked_object_crop(
            before_image_rgb,
            after_mask,
            config.dino_crop_padding,
            config.dino_background_value,
        )

        if opposite_crop_rgb is None:
            return {"suppress": False, "similarity": 0.0, "reason": "empty_opposite_crop"}

        opposite_embedding = self.compute_embedding_for_rgb_crop(opposite_crop_rgb)
        similarity = embedding_cosine_similarity(
            after_reference_embedding,
            opposite_embedding,
        )

        if save_debug_crops and debug_dir is not None:
            self._save_onesided_debug_crops(
                Path(debug_dir) / "new_candidates",
                f"after_{after_idx}",
                reference_crop_rgb,
                opposite_crop_rgb,
            )

        suppress = similarity >= config.dino_similarity_threshold
        return {
            "suppress": suppress,
            "similarity": similarity,
            "reason": "high_opposite_similarity" if suppress else "low_opposite_similarity",
        }

    @staticmethod
    def _save_onesided_debug_crops(
        debug_dir: Path,
        prefix: str,
        reference_crop_rgb: np.ndarray | None,
        opposite_crop_rgb: np.ndarray | None,
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        if reference_crop_rgb is not None:
            save_rgb_as_bgr(str(debug_dir / f"{prefix}_reference.jpg"), reference_crop_rgb)
        if opposite_crop_rgb is not None:
            save_rgb_as_bgr(
                str(debug_dir / f"{prefix}_opposite_region.jpg"),
                opposite_crop_rgb,
            )
