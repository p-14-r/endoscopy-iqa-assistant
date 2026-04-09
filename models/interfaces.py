from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class IQAResult:
    name: str
    score: float
    patch_scores: Optional[List[Dict[str, float]]] = None
    metadata: Optional[Dict[str, Any]] = None


class IQAModel(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> IQAResult:
        """Return a scalar quality score and optional patch-level details."""


@dataclass
class ArtifactResult:
    name: str
    artifact_score: float
    artifact_mask: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class ArtifactModel(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> ArtifactResult:
        """Return artifact confidence and optional mask."""


@dataclass
class EnhancementResult:
    name: str
    enhanced_image: np.ndarray
    metadata: Optional[Dict[str, Any]] = None


class Enhancer(ABC):
    @abstractmethod
    def enhance(self, image: np.ndarray) -> EnhancementResult:
        """Return enhanced image and metadata."""
