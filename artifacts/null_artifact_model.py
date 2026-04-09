from __future__ import annotations

import numpy as np

from models.interfaces import ArtifactModel, ArtifactResult


class NullArtifactModel(ArtifactModel):
    """Phase 1 placeholder: keeps interface stable while artifact model is integrated later."""

    def predict(self, image: np.ndarray) -> ArtifactResult:
        _ = image
        return ArtifactResult(
            name="null_artifact_model",
            artifact_score=0.0,
            artifact_mask=None,
            metadata={"note": "Artifact model not integrated in Phase 1."},
        )
