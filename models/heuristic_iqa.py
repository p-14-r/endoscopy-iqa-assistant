from __future__ import annotations

from typing import Dict, List

import numpy as np

from models.interfaces import IQAModel, IQAResult
from patch_scorer import get_worst_patches
from scorer import (
    compute_blur_value,
    compute_dark_visibility_score,
    compute_exposure_score,
    normalize_blur_score,
)


class HeuristicIQAModel(IQAModel):
    """Current prototype heuristic IQA baseline (kept for A/B comparison)."""

    def __init__(self, patch_size: int = 64, max_patches: int = 20):
        self.patch_size = patch_size
        self.max_patches = max_patches

    def predict(self, image: np.ndarray) -> IQAResult:
        blur_value = compute_blur_value(image)
        blur_score = normalize_blur_score(blur_value)

        exposure_score, mean_val = compute_exposure_score(image)
        dark_visibility_score, dark_ratio, bad_dark_ratio = compute_dark_visibility_score(image)

        worst_patches = get_worst_patches(image, max_patches=self.max_patches)
        patch_scores: List[Dict[str, float]] = [
            {"score": float(score), "x": float(x), "y": float(y)} for score, x, y in worst_patches
        ]

        # Keep weighting simple in this phase; this remains heuristic-only baseline.
        final_score = float(0.45 * blur_score + 0.10 * exposure_score + 0.45 * dark_visibility_score)

        return IQAResult(
            name="heuristic_baseline",
            score=final_score,
            patch_scores=patch_scores,
            metadata={
                "blur_value": float(blur_value),
                "blur_score": float(blur_score),
                "exposure_mean": float(mean_val),
                "exposure_score": float(exposure_score),
                "dark_visibility_score": float(dark_visibility_score),
                "dark_ratio": float(dark_ratio),
                "bad_dark_ratio": float(bad_dark_ratio),
                "patch_size": self.patch_size,
            },
        )
