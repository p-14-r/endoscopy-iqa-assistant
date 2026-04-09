import cv2
import os
from scorer import compute_blur_value


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def process_folder(folder_path):
    results = []  # (filename, blur_value)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(IMG_EXT):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load: {image_path}")
            continue

        blur_value = compute_blur_value(image)
        results.append((filename, blur_value))

    return results


def summary(results, label):
    print(f"\n--- {label} ---")

    if not results:
        print("No valid images.")
        return

    values = [v for _, v in results]

    values_sorted = sorted(values)
    n = len(values_sorted)

    def p(q):
        return values_sorted[int(q * (n - 1))]

    print(f"Count: {n}")
    print(f"Min: {values_sorted[0]:.2f}")
    print(f"P10: {p(0.10):.2f}")
    print(f"P25: {p(0.25):.2f}")
    print(f"Median: {p(0.50):.2f}")
    print(f"P75: {p(0.75):.2f}")
    print(f"P90: {p(0.90):.2f}")
    print(f"Max: {values_sorted[-1]:.2f}")


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