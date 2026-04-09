import cv2
import os
from scorer import compute_exposure_score

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def process_folder(folder_path):
    results = []  # (filename, mean_val, exposure_score)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(IMG_EXT):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        exposure_score, mean_val = compute_exposure_score(image)
        results.append((filename, mean_val, exposure_score))

    return results


def summary(results, label):
    print(f"\n--- {label} ---")

    if not results:
        print("No valid images.")
        return

    mean_vals = [r[1] for r in results]
    scores = [r[2] for r in results]

    mean_vals_sorted = sorted(mean_vals)
    scores_sorted = sorted(scores)
    n = len(mean_vals_sorted)

    def p(values, q):
        return values[int(q * (n - 1))]

    print(f"Count: {n}")

    print("Mean brightness statistics:")
    print(f"  Min: {mean_vals_sorted[0]:.3f}")
    print(f"  P10: {p(mean_vals_sorted, 0.10):.3f}")
    print(f"  P25: {p(mean_vals_sorted, 0.25):.3f}")
    print(f"  Median: {p(mean_vals_sorted, 0.50):.3f}")
    print(f"  P75: {p(mean_vals_sorted, 0.75):.3f}")
    print(f"  P90: {p(mean_vals_sorted, 0.90):.3f}")
    print(f"  Max: {mean_vals_sorted[-1]:.3f}")

    print("Exposure score statistics:")
    print(f"  Min: {scores_sorted[0]:.3f}")
    print(f"  P10: {p(scores_sorted, 0.10):.3f}")
    print(f"  P25: {p(scores_sorted, 0.25):.3f}")
    print(f"  Median: {p(scores_sorted, 0.50):.3f}")
    print(f"  P75: {p(scores_sorted, 0.75):.3f}")
    print(f"  P90: {p(scores_sorted, 0.90):.3f}")
    print(f"  Max: {scores_sorted[-1]:.3f}")


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