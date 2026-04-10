import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from artifacts.null_artifact_model import NullArtifactModel
from enhancers.zero_dce_enhancer import ZeroDCEEnhancer
from models.fusion_iqa import FusionIQAModel
from models.heuristic_iqa import HeuristicIQAModel
from models.pyiqa_model import PyIQAModel
from scorer import compute_fov_mask


def build_iqa_model(args):
    heuristic_model = HeuristicIQAModel(max_patches=args.max_patches)

    if args.iqa_mode == "heuristic":
        return heuristic_model

    if args.iqa_mode == "pyiqa":
        return PyIQAModel(metric_name=args.pyiqa_metric, device=args.device)

    if args.iqa_mode == "fusion":
        neural_model = PyIQAModel(metric_name=args.pyiqa_metric, device=args.device)
        return FusionIQAModel(
            heuristic_model=heuristic_model,
            neural_model=neural_model,
            w_heuristic=args.fusion_w1,
            w_neural=args.fusion_w2,
        )

    raise ValueError(f"Unsupported iqa_mode: {args.iqa_mode}")


def maybe_enhance(image, args):
    if args.enhancer == "none":
        return image, {"name": "none", "enabled": False}

    if args.enhancer == "zero_dce":
        enhancer = ZeroDCEEnhancer(
            checkpoint_path=args.zero_dce_checkpoint,
            device=args.device,
            strict=False,
        )
        result = enhancer.enhance(image)
        meta = {"name": result.name, "enabled": True, **(result.metadata or {})}
        return result.enhanced_image, meta

    raise ValueError(f"Unsupported enhancer: {args.enhancer}")


def save_visualizations(image_bgr, fov_mask, worst_patches, output_dir: Path, patch_size: int = 64):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    vis = image_rgb.copy()

    for score, x, y in worst_patches:
        cv2.rectangle(vis, (x, y), (x + patch_size, y + patch_size), (255, 0, 0), 2)

    # Worst patch overlay
    plt.figure(figsize=(7, 7))
    plt.imshow(vis)
    plt.title("Worst patches")
    plt.axis("off")
    worst_path = output_dir / "worst_patches.png"
    plt.tight_layout()
    plt.savefig(worst_path, dpi=150)
    plt.close()

    # Original + FOV mask
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(fov_mask, cmap="gray")
    plt.title("FOV Mask")
    plt.axis("off")

    compare_path = output_dir / "original_and_fov.png"
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150)
    plt.close()

    return {"worst_patches": str(worst_path), "original_and_fov": str(compare_path)}


def parse_args():
    parser = argparse.ArgumentParser(description="Endoscopy IQA assistant Phase 1 runner")
    parser.add_argument(
        "--image",
        default="data/calibration/good/202511241219G061.jpg",
        help="Path to input image",
    )
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for output artifacts")

    parser.add_argument(
        "--iqa-mode",
        choices=["heuristic", "pyiqa", "fusion"],
        default="heuristic",
        help="IQA backend: heuristic baseline, pyiqa neural baseline, or weighted fusion",
    )
    parser.add_argument("--pyiqa-metric", default="hyperiqa", help="Metric name for pyiqa mode")
    parser.add_argument("--max-patches", type=int, default=20, help="Max worst patches for heuristic mode")
    parser.add_argument("--fusion-w1", type=float, default=0.7, help="Fusion weight for heuristic score")
    parser.add_argument("--fusion-w2", type=float, default=0.3, help="Fusion weight for neural score")

    parser.add_argument(
        "--enhancer",
        choices=["none", "zero_dce"],
        default="none",
        help="Optional enhancer backend",
    )
    parser.add_argument(
        "--zero-dce-checkpoint",
        default=None,
        help="Path to Zero-DCE checkpoint (optional). If missing, image is left unchanged.",
    )

    parser.add_argument("--device", default=None, help="Torch device override, e.g., cpu/cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    image_for_iqa, enhancement_meta = maybe_enhance(image, args)

    iqa_model = build_iqa_model(args)
    iqa_result = iqa_model.predict(image_for_iqa)

    artifact_result = NullArtifactModel().predict(image_for_iqa)

    fov_mask = compute_fov_mask(image_for_iqa)
    worst_patches = []
    if iqa_result.patch_scores:
        worst_patches = [(p["score"], int(p["x"]), int(p["y"])) for p in iqa_result.patch_scores]

    vis_paths = save_visualizations(
        image_for_iqa,
        fov_mask,
        worst_patches,
        output_dir=output_dir,
        patch_size=64,
    )

    if args.enhancer != "none":
        enhanced_path = output_dir / "enhanced_image.png"
        cv2.imwrite(str(enhanced_path), image_for_iqa)
        vis_paths["enhanced_image"] = str(enhanced_path)

    report = {
        "input_image": str(image_path),
        "iqa_mode": args.iqa_mode,
        "iqa_result": {
            "name": iqa_result.name,
            "score": iqa_result.score,
            "metadata": iqa_result.metadata,
            "num_patches": len(iqa_result.patch_scores or []),
        },
        "artifact_result": {
            "name": artifact_result.name,
            "artifact_score": artifact_result.artifact_score,
            "metadata": artifact_result.metadata,
        },
        "enhancement": enhancement_meta,
        "outputs": vis_paths,
    }

    if args.iqa_mode == "fusion" and iqa_result.metadata:
        report["fusion_breakdown"] = {
            "fusion_score": iqa_result.metadata.get("fusion_score", iqa_result.score),
            "components": iqa_result.metadata.get("components", {}),
        }

    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
