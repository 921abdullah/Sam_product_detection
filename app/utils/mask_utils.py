import cv2
import numpy as np


def resize_masks_to_image_shape(
    masks: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize masks to (target_h, target_w) with nearest-neighbor interpolation."""
    if masks is None or not isinstance(masks, np.ndarray) or masks.size == 0:
        return np.empty((0, target_h, target_w), dtype=np.uint8)

    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]

    resized_masks = []
    for mask in masks:
        if mask.shape != (target_h, target_w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = (mask > 0).astype(np.uint8)
        resized_masks.append(mask)

    return np.stack(resized_masks, axis=0)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def overlay_colored_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    output = image_rgb.copy()
    mask_bool = mask.astype(bool)

    color_layer = np.zeros_like(output, dtype=np.uint8)
    color_layer[:, :] = color

    output[mask_bool] = cv2.addWeighted(
        output[mask_bool],
        1 - alpha,
        color_layer[mask_bool],
        alpha,
        0,
    )

    return output


def draw_mask_contour(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> np.ndarray:
    output = image_rgb.copy()
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(output, contours, -1, color, thickness)
    return output


def put_label_near_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    output = image_rgb.copy()
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return output

    x = int(xs.min())
    y = int(ys.min())

    cv2.putText(
        output,
        label,
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        3,
    )

    return output
