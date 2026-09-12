import cv2
import numpy as np
from typing import Dict, Any, Optional

def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    bin_mask = (binary_mask > 127).astype(np.uint8) * 255
    skel = np.zeros(bin_mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_mask = bin_mask.copy()

    while True:
        eroded = cv2.erode(temp_mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_mask = eroded.copy()
        if cv2.countNonZero(temp_mask) == 0:
            break

    return skel

def extract_crack_features(
    binary_mask: np.ndarray,
    px_to_mm_ratio: Optional[float] = None
) -> Dict[str, Any]:
    bin_mask = (binary_mask > 127).astype(np.uint8) * 255
    total_pixels = bin_mask.size
    crack_pixels = int(np.sum(bin_mask > 0))
    area_pct = float((crack_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0

    if crack_pixels == 0:
        return {
            'has_crack': False,
            'crack_pixel_count': 0,
            'affected_area_pct': 0.0,
            'length_px': 0,
            'max_width_px': 0.0,
            'mean_width_px': 0.0,
            'component_count': 0,
            'branch_points': 0,
            'unit': 'px',
            'calibrated': False,
            'length_calibrated': None,
            'max_width_calibrated': None,
            'skeleton_mask': np.zeros_like(bin_mask),
            'distance_map': np.zeros_like(bin_mask, dtype=np.float32)
        }

    skel = skeletonize_mask(bin_mask)
    length_px = int(np.sum(skel > 0))

    dist_map = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    skel_distances = dist_map[skel > 0]

    if len(skel_distances) > 0:
        max_width_px = float(2.0 * np.max(skel_distances))
        mean_width_px = float(2.0 * np.mean(skel_distances))
    else:
        max_width_px = 0.0
        mean_width_px = 0.0

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    component_count = max(0, num_labels - 1)

    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D((skel > 0).astype(np.uint8), -1, neighbor_kernel)
    branch_points = int(np.sum((skel > 0) & (neighbors >= 3)))

    calibrated = False
    length_calibrated = None
    max_width_calibrated = None
    unit = 'px'

    if px_to_mm_ratio is not None and px_to_mm_ratio > 0:
        calibrated = True
        unit = 'mm'
        length_calibrated = round(length_px * px_to_mm_ratio, 2)
        max_width_calibrated = round(max_width_px * px_to_mm_ratio, 2)

    return {
        'has_crack': True,
        'crack_pixel_count': crack_pixels,
        'affected_area_pct': round(area_pct, 3),
        'length_px': length_px,
        'max_width_px': round(max_width_px, 2),
        'mean_width_px': round(mean_width_px, 2),
        'component_count': component_count,
        'branch_points': branch_points,
        'unit': unit,
        'calibrated': calibrated,
        'length_calibrated': length_calibrated,
        'max_width_calibrated': max_width_calibrated,
        'skeleton_mask': skel,
        'distance_map': dist_map
    }

def create_inspection_overlay(
    image_rgb: np.ndarray,
    binary_mask: np.ndarray,
    skeleton_mask: Optional[np.ndarray] = None,
    alpha: float = 0.45
) -> np.ndarray:
    overlay = image_rgb.copy()
    mask_bool = binary_mask > 127
    overlay[mask_bool] = [235, 45, 45]

    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, overlay, alpha, 0)

    if skeleton_mask is not None:
        skel_bool = skeleton_mask > 0
        blended[skel_bool] = [255, 235, 59]

    return blended

if __name__ == '__main__':
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.line(dummy_mask, (30, 30), (200, 200), 255, 6)
    feats = extract_crack_features(dummy_mask, px_to_mm_ratio=0.1)
    print('Dummy mask features:')
    for k, v in feats.items():
        if not isinstance(v, np.ndarray):
            print(f'  {k}: {v}')
