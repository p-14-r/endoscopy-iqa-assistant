import cv2
import os
from patch_scorer import compute_patch_based_score

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def process_folder(folder_path):
    results = []  # (filename, final_score)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(IMG_EXT):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        final_score, _ = compute_patch_based_score(image)
        results.append((filename, final_score))

    return results


def summary(results, label):
    print(f"\n--- {label} ---")

    if not results:
        print("No valid images.")
        return

    scores = sorted([r[1] for r in results])
    n = len(scores)

    def p(values, q):
        return values[int(q * (n - 1))]

    print(f"Count: {n}")
    print(f"Min: {scores[0]:.3f}")
    print(f"P10: {p(scores, 0.10):.3f}")
    print(f"P25: {p(scores, 0.25):.3f}")
    print(f"Median: {p(scores, 0.50):.3f}")
    print(f"P75: {p(scores, 0.75):.3f}")
    print(f"P90: {p(scores, 0.90):.3f}")
    print(f"Max: {scores[-1]:.3f}")


def main():
    base = os.path.join("data", "calibration")

    good = process_folder(os.path.join(base, "good"))
    medium = process_folder(os.path.join(base, "medium"))
    bad = process_folder(os.path.join(base, "bad"))

    summary(good, "GOOD")
    summary(medium, "MEDIUM")
    summary(bad, "BAD")


if __name__ == "__main__":
    main()