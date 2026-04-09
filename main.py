import cv2
import os
import matplotlib.pyplot as plt
from scorer import (
    compute_blur_value,
    normalize_blur_score,
    compute_exposure_score,
    compute_dark_visibility_score,
    compute_fov_mask
)
from patch_scorer import get_worst_patches


def main():
    # -------------------------
    # 1. Load image
    # -------------------------
    image_path = os.path.join("data", "raw", "test.jpg")
    print("Trying to load image from:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("Error: failed to load image.")
        return

    print("Image loaded successfully.")
    print("Image shape:", image.shape)

    # -------------------------
    # 2. Global metrics
    # -------------------------
    blur_value = compute_blur_value(image)
    blur_score = normalize_blur_score(blur_value)

    exposure_score, mean_val = compute_exposure_score(image)

    dark_visibility_score, dark_ratio, bad_dark_ratio = compute_dark_visibility_score(image)

    fov_mask = compute_fov_mask(image)

    print(f"Blur value: {blur_value:.2f}")
    print(f"Blur score: {blur_score:.3f}")

    print(f"Exposure mean: {mean_val:.3f}")
    print(f"Exposure score (weak reference): {exposure_score:.3f}")

    print(f"Dark ratio (inside FOV): {dark_ratio:.3f}")
    print(f"Bad dark ratio: {bad_dark_ratio:.3f}")
    print(f"Dark visibility score: {dark_visibility_score:.3f}")

    # -------------------------
    # 3. Worst patches
    # -------------------------
    worst_patches = get_worst_patches(image)
    print(f"Number of worst patches selected: {len(worst_patches)}")

    # -------------------------
    # 4. Convert for display
    # -------------------------
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -------------------------
    # 5. Draw worst patches
    # -------------------------
    vis = image_rgb.copy()

    patch_size = 64

    for score, x, y in worst_patches:
        cv2.rectangle(
            vis,
            (x, y),
            (x + patch_size, y + patch_size),
            (255, 0, 0),
            2
        )

    # -------------------------
    # 6. Show worst patches
    # -------------------------
    plt.figure(figsize=(7, 7))
    plt.imshow(vis)
    plt.title("Worst patches")
    plt.axis("off")
    plt.show()

    # -------------------------
    # 7. Show original + FOV mask
    # -------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(
        f"Original\n"
        f"Blur={blur_score:.3f} | DarkVis={dark_visibility_score:.3f}\n"
        f"DarkRatio={dark_ratio:.3f} | BadDark={bad_dark_ratio:.3f}"
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(fov_mask, cmap="gray")
    plt.title("FOV Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()