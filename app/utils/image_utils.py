import cv2
import numpy as np
import torch


def rgb_numpy_to_lightglue_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    image_rgb = np.ascontiguousarray(image_rgb)
    tensor = torch.from_numpy(image_rgb)
    tensor = tensor.permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def torch_image_to_rgb_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert LightGlue tensor (3, H, W) in [0, 1] to RGB uint8 (H, W, 3)."""
    return (
        image_tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        * 255
    ).astype(np.uint8)


def draw_boxes_only(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw xyxy bounding boxes on an RGB image."""
    output = image_rgb.copy()

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

    return output


def save_rgb_as_bgr(path: str, image_rgb: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
