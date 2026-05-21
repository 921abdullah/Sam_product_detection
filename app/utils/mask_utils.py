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


def mask_overlap_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    """Return (iou, containment) for two binary masks."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0 or min(area_a, area_b) == 0:
        return 0.0, 0.0

    iou = float(intersection / union)
    containment = float(intersection / min(area_a, area_b))
    return iou, containment


def deduplicate_masks(
    masks: np.ndarray,
    duplicate_iou_threshold: float = 0.80,
) -> tuple[np.ndarray, list[int]]:
    """Remove near-duplicate masks within one image (larger masks kept first)."""
    if masks is None or len(masks) == 0:
        return masks, []

    mask_areas = [int(mask.sum()) for mask in masks]
    sorted_indices = sorted(
        range(len(masks)),
        key=lambda idx: mask_areas[idx],
        reverse=True,
    )

    kept_indices: list[int] = []
    for idx in sorted_indices:
        current_mask = masks[idx]
        is_duplicate = False
        for kept_idx in kept_indices:
            if mask_iou(current_mask, masks[kept_idx]) >= duplicate_iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_indices.append(idx)

    kept_indices = sorted(kept_indices)
    return masks[kept_indices], kept_indices


def dilate_binary_mask(mask: np.ndarray, dilation_px: int = 8) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    kernel_size = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask_uint8, kernel, iterations=1)


def points_inside_mask(points_xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=bool)

    h, w = mask.shape[:2]
    x = np.round(points_xy[:, 0]).astype(int)
    y = np.round(points_xy[:, 1]).astype(int)

    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    inside = np.zeros(len(points_xy), dtype=bool)
    inside[valid] = mask[y[valid], x[valid]] > 0
    return inside


def mask_to_padded_bbox(
    mask: np.ndarray,
    padding_px: int,
    image_h: int,
    image_w: int,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = max(int(xs.min()) - padding_px, 0)
    y1 = max(int(ys.min()) - padding_px, 0)
    x2 = min(int(xs.max()) + padding_px + 1, image_w)
    y2 = min(int(ys.max()) + padding_px + 1, image_h)
    return x1, y1, x2, y2


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
