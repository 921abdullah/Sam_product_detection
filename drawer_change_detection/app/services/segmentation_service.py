from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass
class SegmentationResult:
    boxes: np.ndarray
    masks: np.ndarray


class SegmentationService:
    def __init__(self, predictor, predictor_lock: Lock):
        self.predictor = predictor
        self.lock = predictor_lock

    def segment_image(
        self,
        image_path: str,
        prompts: list[str],
    ) -> SegmentationResult:
        with self.lock:
            if hasattr(self.predictor, "reset_image"):
                self.predictor.reset_image()

            self.predictor.set_image(image_path)
            results = self.predictor(text=prompts)

        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = (
                result.boxes.xyxy.detach().cpu().numpy().astype(np.int32)
            )
        else:
            boxes = np.empty((0, 4), dtype=np.int32)

        if result.masks is not None and result.masks.data is not None:
            masks = (
                result.masks.data.detach().cpu().numpy().astype(np.uint8)
            )
        else:
            masks = np.empty((0,), dtype=np.uint8)

        return SegmentationResult(boxes=boxes, masks=masks)
