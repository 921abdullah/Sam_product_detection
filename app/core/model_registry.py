import torch
from threading import Lock

from lightglue import LightGlue, SuperPoint
from ultralytics.models.sam import SAM3SemanticPredictor

from app.core.config import PipelineConfig


class ModelRegistry:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = None
        self.matcher = None
        self.sam_predictor = None
        self.sam_lock = Lock()

    def load_models(self, config: PipelineConfig | None = None):
        config = config or PipelineConfig()

        self.extractor = SuperPoint(
            max_num_keypoints=config.max_num_keypoints,
        ).eval().to(self.device)

        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        sam_overrides = dict(
            conf=config.sam_conf,
            task="segment",
            mode="predict",
            model="sam3.pt",
            half=True,
            save=False,
            show_labels=False,
            show_conf=False,
            show_boxes=True,
        )

        self.sam_predictor = SAM3SemanticPredictor(overrides=sam_overrides)
