import cv2
import os
from scorer import compute_dark_visibility_score

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def process_folder(folder_path):
    results = []  # (filename, dark_ratio, bad_dark_ratio, dark_visibility_score)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(IMG_EXT):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        dark_visibility_score, dark_ratio, bad_dark_ratio = compute_dark_visibility_score(image)
        results.append((filename, dark_ratio, bad_dark_ratio, dark_visibility_score))

    return results


def summary(results, label):
    print(f"\n--- {label} ---")

    if not results:
        print("No valid images.")
        return

    dark_ratios = sorted([r[1] for r in results])
    bad_dark_ratios = sorted([r[2] for r in results])
    scores = sorted([r[3] for r in results])

    n = len(scores)

    def p(values, q):
        return values[int(q * (n - 1))]

    print(f"Count: {n}")

    print("Dark ratio statistics:")
    print(f"  Min: {dark_ratios[0]:.3f}")
    print(f"  P10: {p(dark_ratios, 0.10):.3f}")
    print(f"  P25: {p(dark_ratios, 0.25):.3f}")
    print(f"  Median: {p(dark_ratios, 0.50):.3f}")
    print(f"  P75: {p(dark_ratios, 0.75):.3f}")
    print(f"  P90: {p(dark_ratios, 0.90):.3f}")
    print(f"  Max: {dark_ratios[-1]:.3f}")

    print("Bad dark ratio statistics:")
    print(f"  Min: {bad_dark_ratios[0]:.3f}")
    print(f"  P10: {p(bad_dark_ratios, 0.10):.3f}")
    print(f"  P25: {p(bad_dark_ratios, 0.25):.3f}")
    print(f"  Median: {p(bad_dark_ratios, 0.50):.3f}")
    print(f"  P75: {p(bad_dark_ratios, 0.75):.3f}")
    print(f"  P90: {p(bad_dark_ratios, 0.90):.3f}")
    print(f"  Max: {bad_dark_ratios[-1]:.3f}")

    print("Dark visibility score statistics:")
    print(f"  Min: {scores[0]:.3f}")
    print(f"  P10: {p(scores, 0.10):.3f}")
    print(f"  P25: {p(scores, 0.25):.3f}")
    print(f"  Median: {p(scores, 0.50):.3f}")
    print(f"  P75: {p(scores, 0.75):.3f}")
    print(f"  P90: {p(scores, 0.90):.3f}")
    print(f"  Max: {scores[-1]:.3f}")


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