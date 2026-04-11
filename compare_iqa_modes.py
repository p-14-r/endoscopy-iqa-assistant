import argparse
import json
from pathlib import Path

import cv2
from main import build_iqa_model, maybe_enhance


class _Args:
    def __init__(
        self,
        iqa_mode,
        pyiqa_metric,
        device,
        max_patches,
        enhancer,
        zero_dce_checkpoint,
        fusion_w1,
        fusion_w2,
    ):
        self.iqa_mode = iqa_mode
        self.pyiqa_metric = pyiqa_metric
        self.device = device
        self.max_patches = max_patches
        self.enhancer = enhancer
        self.zero_dce_checkpoint = zero_dce_checkpoint
        self.fusion_w1 = fusion_w1
        self.fusion_w2 = fusion_w2


def parse_args():
    p = argparse.ArgumentParser(description="Compare heuristic vs neural vs fusion IQA on same image")
    p.add_argument("--image", default="data/calibration/good/202511241219G061.jpg")
    p.add_argument("--output", default="outputs/compare_iqa.json")
    p.add_argument("--pyiqa-metric", default="hyperiqa")
    p.add_argument("--device", default=None)
    p.add_argument("--max-patches", type=int, default=20)
    p.add_argument("--enhancer", choices=["none", "zero_dce"], default="none")
    p.add_argument("--zero-dce-checkpoint", default=None)
    p.add_argument("--fusion-w1", type=float, default=0.7)
    p.add_argument("--fusion-w2", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {args.image}")

    common = {
        "pyiqa_metric": args.pyiqa_metric,
        "device": args.device,
        "max_patches": args.max_patches,
        "enhancer": args.enhancer,
        "zero_dce_checkpoint": args.zero_dce_checkpoint,
        "fusion_w1": args.fusion_w1,
        "fusion_w2": args.fusion_w2,
    }

    out = {
        "image": args.image,
        "fusion_weights": {"w1": args.fusion_w1, "w2": args.fusion_w2},
        "runs": [],
    }

    for mode in ["heuristic", "pyiqa", "fusion"]:
        run_args = _Args(iqa_mode=mode, **common)
        image_for_iqa, enhancement_meta = maybe_enhance(image, run_args)
        model = build_iqa_model(run_args)
        result = model.predict(image_for_iqa)

        out["runs"].append(
            {
                "iqa_mode": mode,
                "name": result.name,
                "score": result.score,
                "metadata": result.metadata,
                "num_patches": len(result.patch_scores or []),
                "enhancement": enhancement_meta,
                "fusion_breakdown": {
                    "fusion_score": (result.metadata or {}).get("fusion_score"),
                    "components": (result.metadata or {}).get("components"),
                }
                if mode == "fusion"
                else None,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
