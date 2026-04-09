import cv2
import numpy as np
from scorer import compute_fov_mask


def find_dark_regions(image, fov_mask, threshold=70, min_area=300):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    masked = gray.copy()
    masked[fov_mask == 0] = 255

    _, binary = cv2.threshold(masked, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        mask = (labels == i).astype(np.uint8)
        regions.append(mask)

    return regions


def compute_patch_score(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean = gray.mean()
    std = gray.std()

    darkness = np.clip((80 - mean) / 80, 0, 1)
    detail = np.clip(std / 40, 0, 1)

    score = darkness * (1 - detail)
    return float(score)


def extract_patches_from_regions(image, regions, patch_size=64, sample_step=16):
    """
    Sample candidate patches from dark regions using a coarse grid.
    This avoids exploding patch count.
    """
    h, w, _ = image.shape
    patches = []

    for region in regions:
        ys, xs = np.where(region > 0)

        if len(xs) == 0:
            continue

        # bounding box of this region
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # grid sampling instead of every pixel
        for y in range(y_min, y_max + 1, sample_step):
            for x in range(x_min, x_max + 1, sample_step):
                if region[y, x] == 0:
                    continue

                y0 = max(0, y - patch_size // 2)
                x0 = max(0, x - patch_size // 2)

                if y0 + patch_size > h or x0 + patch_size > w:
                    continue

                patch = image[y0:y0 + patch_size, x0:x0 + patch_size]
                score = compute_patch_score(patch)

                patches.append((score, x0, y0))

    return patches


def nms_patches(patches, patch_size=64, iou_thresh=0.3):
    if not patches:
        return []

    boxes = []
    scores = []

    for score, x, y in patches:
        boxes.append([x, y, x + patch_size, y + patch_size])
        scores.append(score)

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    indices = scores.argsort()[::-1]
    keep = []

    while len(indices) > 0:
        i = indices[0]
        keep.append(i)

        x1, y1, x2, y2 = boxes[i]
        rest = indices[1:]
        new_rest = []

        for j in rest:
            xx1 = max(x1, boxes[j][0])
            yy1 = max(y1, boxes[j][1])
            xx2 = min(x2, boxes[j][2])
            yy2 = min(y2, boxes[j][3])

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

            area1 = (x2 - x1) * (y2 - y1)
            area2 = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])

            iou = inter / (area1 + area2 - inter + 1e-6)

            if iou < iou_thresh:
                new_rest.append(j)

        indices = np.array(new_rest, dtype=np.int32)

    return [patches[i] for i in keep]


def get_worst_patches(image, max_patches=20):
    fov_mask = compute_fov_mask(image)

    regions = find_dark_regions(image, fov_mask)

    if len(regions) == 0:
        return []

    patches = extract_patches_from_regions(
        image,
        regions,
        patch_size=64,
        sample_step=16
    )

    patches = sorted(patches, key=lambda x: x[0], reverse=True)

    patches = nms_patches(patches, patch_size=64, iou_thresh=0.3)

    return patches[:max_patches]


def compute_patch_based_score(image, max_patches=20):
    """
    Compatibility helper for batch calibration scripts.
    Returns:
      final_score: 0..1 where higher means better visibility in selected patches
      worst_patches: selected patch tuples (badness_score, x, y)
    """
    worst_patches = get_worst_patches(image, max_patches=max_patches)
    if not worst_patches:
        return 1.0, []

    mean_badness = float(np.mean([p[0] for p in worst_patches]))
    final_score = float(np.clip(1.0 - mean_badness, 0.0, 1.0))
    return final_score, worst_patches
