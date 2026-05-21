from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import PipelineConfig


class EmbeddingService:
    def __init__(self, processor, model, device: torch.device):
        self.processor = processor
        self.model = model
        self.device = device

    def create_masked_object_crop(
        self,
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

    def extract_from_masks(
        self,
        image_rgb: np.ndarray,
        masks: np.ndarray,
        config: PipelineConfig,
        debug_crop_dir: Path | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        pil_crops: list[Image.Image] = []
        valid_mask_indices: list[int] = []

        save_debug = config.save_dino_debug_crops and debug_crop_dir is not None
        if save_debug:
            debug_crop_dir.mkdir(parents=True, exist_ok=True)

        for mask_idx, mask in enumerate(masks):
            crop = self.create_masked_object_crop(
                image_rgb=image_rgb,
                mask=mask,
                padding=config.dino_crop_padding,
                background_value=config.dino_background_value,
            )
            if crop is None:
                continue

            pil_crops.append(Image.fromarray(crop).convert("RGB"))
            valid_mask_indices.append(mask_idx)

            if save_debug:
                crop_path = debug_crop_dir / f"object_{mask_idx}.jpg"
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

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

    @staticmethod
    def similarity_matrix(
        before_embeddings: np.ndarray,
        after_embeddings: np.ndarray,
    ) -> np.ndarray:
        if len(before_embeddings) == 0 or len(after_embeddings) == 0:
            return np.empty((0, 0), dtype=np.float32)
        return before_embeddings @ after_embeddings.T
