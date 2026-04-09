from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from models.interfaces import IQAModel, IQAResult


class PyIQAModel(IQAModel):
    """Optional neural IQA baseline backed by pyiqa (HyperIQA by default)."""

    def __init__(self, metric_name: str = "hyperiqa", device: Optional[str] = None):
        try:
            import pyiqa  # type: ignore
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyIQA is not installed. Install optional deps (torch, pyiqa) to use neural IQA baseline."
            ) from exc

        self._pyiqa = pyiqa
        self._torch = torch
        self.metric_name = metric_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = pyiqa.create_metric(metric_name, device=self.device)

    def predict(self, image: np.ndarray) -> IQAResult:
        # BGR uint8 -> RGB float tensor in [0,1], shape [1,3,H,W]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            score_tensor = self.metric(tensor)

        score = float(score_tensor.item())

        return IQAResult(
            name=f"pyiqa_{self.metric_name}",
            score=score,
            patch_scores=None,
            metadata={"metric_name": self.metric_name, "device": self.device},
        )
