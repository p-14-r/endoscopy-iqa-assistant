import argparse
import json
from pathlib import Path

import cv2

from main import build_iqa_model, maybe_enhance


def parse_args():
    p = argparse.ArgumentParser(description="Small-subset smoke test for Phase 1 pipelines")
    p.add_argument("--output", default="outputs/phase1_smoke/smoke_summary.json")
    p.add_argument("--max-images", type=int, default=3, help="Use only a small subset")
    p.add_argument("--pyiqa-metric", default="hyperiqa")
    p.add_argument("--device", default=None)
    p.add_argument("--max-patches", type=int, default=20)
    p.add_argument("--enhancer", choices=["none", "zero_dce"], default="none")
    p.add_argument("--zero-dce-checkpoint", default=None)
    p.add_argument("--fusion-w1", type=float, default=0.7)
    p.add_argument("--fusion-w2", type=float, default=0.3)
    return p.parse_args()


def collect_subset(max_images: int):
    paths = sorted(Path("data/calibration").glob("*/*.jpg"))
    return [str(p) for p in paths[:max_images]]


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


def run_one(image_path: str, mode: str, cfg):
    image = cv2.imread(image_path)
    if image is None:
        return {"image": image_path, "iqa_mode": mode, "error": "load_failed"}

    args = _Args(
        iqa_mode=mode,
        pyiqa_metric=cfg.pyiqa_metric,
        device=cfg.device,
        max_patches=cfg.max_patches,
        enhancer=cfg.enhancer,
        zero_dce_checkpoint=cfg.zero_dce_checkpoint,
        fusion_w1=cfg.fusion_w1,
        fusion_w2=cfg.fusion_w2,
    )

    image_for_iqa, enhancement_meta = maybe_enhance(image, args)
    model = build_iqa_model(args)
    result = model.predict(image_for_iqa)

    return {
        "image": image_path,
        "iqa_mode": mode,
        "score": result.score,
        "name": result.name,
        "num_patches": len(result.patch_scores or []),
        "enhancement": enhancement_meta,
    }


def main():
    args = parse_args()
    subset = collect_subset(args.max_images)

    out = {"subset_size": len(subset), "images": subset, "results": []}

    for image_path in subset:
        out["results"].append(run_one(image_path, "heuristic", args))
        try:
            out["results"].append(run_one(image_path, "pyiqa", args))
        except Exception as exc:  # optional neural path may be unavailable
            out["results"].append(
                {
                    "image": image_path,
                    "iqa_mode": "pyiqa",
                    "error": str(exc),
                }
            )
        try:
            out["results"].append(run_one(image_path, "fusion", args))
        except Exception as exc:  # optional neural path may be unavailable
            out["results"].append(
                {
                    "image": image_path,
                    "iqa_mode": "fusion",
                    "error": str(exc),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
