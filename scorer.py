import cv2
import numpy as np


def compute_fov_mask(image, threshold=10):
    """
    Compute endoscopic field-of-view mask.
    Exclude outer black background.

    Returns:
        fov_mask: boolean mask of valid image region
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Basic threshold to separate valid region from black border
    mask = gray > threshold

    # Morphological cleanup
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fov_mask = labels == largest_label

    return fov_mask


def compute_blur_value(image):
    """
    Compute Laplacian variance as a blur indicator.
    Higher value usually means sharper image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_value = laplacian.var()
    return blur_value


def normalize_blur_score(blur_value, min_blur=420, max_blur=700):
    """
    Convert blur_value into a score between 0 and 1.
    """
    score = (blur_value - min_blur) / (max_blur - min_blur)
    score = max(0.0, min(1.0, score))
    return score


def compute_exposure_score(image):
    """
    Old weak feature: global mean brightness.
    Keep it for reference only, not as a main quality metric.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_norm = gray / 255.0
    mean_val = gray_norm.mean()

    ideal_low = 0.3
    ideal_high = 0.7

    if mean_val < ideal_low:
        score = mean_val / ideal_low
    elif mean_val > ideal_high:
        score = (1 - mean_val) / (1 - ideal_high)
    else:
        score = 1.0

    score = max(0.0, min(1.0, score))
    return score, mean_val


def compute_dark_visibility_score(
    image,
    dark_threshold=85,
    detail_ref=18.0,
    kernel_size=9
):
    """
    Improved dark visibility using stable dark-region analysis.

    Returns:
        score
        dark_ratio
        bad_dark_ratio
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fov_mask = compute_fov_mask(image)

    if fov_mask.sum() == 0:
        return 0.0, 0.0, 1.0

    # ---- Dark mask ----
    dark_mask = (gray < dark_threshold) & fov_mask
    dark_ratio = float(dark_mask.sum() / fov_mask.sum())

    # ---- Stabilize dark region (not too small, not too noisy) ----
    dark_uint8 = dark_mask.astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # erode slightly (remove boundary)
    dark_eroded = cv2.erode(dark_uint8, kernel, iterations=1)

    # dilate back (recover region size)
    dark_stable = cv2.dilate(dark_eroded, kernel, iterations=1)

    dark_stable_mask = dark_stable > 0

    # fallback
    if dark_stable_mask.sum() == 0:
        dark_stable_mask = dark_mask

    # ---- Gradient ----
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    detail_score = grad_mag / detail_ref
    detail_score = np.clip(detail_score, 0.0, 1.0)

    # ---- Darkness ----
    darkness = 1.0 - gray / 255.0
    darkness = np.clip(darkness, 0.0, 1.0)

    # ---- Soft badness ----
    badness = (darkness ** 1.2) * (1.0 - detail_score)

    if dark_stable_mask.sum() > 0:
        bad_dark_ratio = float(badness[dark_stable_mask].mean())
    else:
        bad_dark_ratio = 0.0

    # ---- Final score ----
    penalty = bad_dark_ratio * 1.5
    score = 1.0 - penalty
    score = max(0.0, min(1.0, score))

    return score, dark_ratio, bad_dark_ratio