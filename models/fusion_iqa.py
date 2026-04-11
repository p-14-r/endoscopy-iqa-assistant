from __future__ import annotations

import numpy as np

from models.heuristic_iqa import HeuristicIQAModel
from models.interfaces import IQAModel, IQAResult
from models.pyiqa_model import PyIQAModel


class FusionIQAModel(IQAModel):
    """
    Weighted fusion of heuristic + neural IQA:
        fusion_score = w1 * heuristic + w2 * neural
    """

    def __init__(
        self,
        heuristic_model: HeuristicIQAModel,
        neural_model: PyIQAModel,
        w_heuristic: float = 0.7,
        w_neural: float = 0.3,
    ):
        self.heuristic_model = heuristic_model
        self.neural_model = neural_model
        self.w_heuristic = float(w_heuristic)
        self.w_neural = float(w_neural)

    def predict(self, image: np.ndarray) -> IQAResult:
        heuristic = self.heuristic_model.predict(image)
        neural = self.neural_model.predict(image)

        fusion_score = float(self.w_heuristic * heuristic.score + self.w_neural * neural.score)

        return IQAResult(
            name="fusion_iqa",
            score=fusion_score,
            patch_scores=heuristic.patch_scores,  # patch only from heuristic
            metadata={
                "fusion_score": fusion_score,
                "weights": {
                    "w_heuristic": self.w_heuristic,
                    "w_neural": self.w_neural,
                },
                "components": {
                    "heuristic": float(heuristic.score),
                    "neural": float(neural.score),
                },
                "heuristic_metadata": heuristic.metadata,
                "neural_metadata": neural.metadata,
            },
        )
