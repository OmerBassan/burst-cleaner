# core/embeddings_core.py
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class EmbeddingBackend(ABC):
    """
    Abstract interface for visual embedding computation.

    image_id:
        - בדסקטופ: path לקובץ ("C:/.../IMG_001.jpg")
        - באנדרואיד: URI / מזהה לוגי שה־Loader יודע לפרש
    """

    @abstractmethod
    def embed_single(self, image_id: str) -> np.ndarray:
        """
        Compute embedding for a single image identifier.
        Must return a 1D numpy array (feature vector).
        """
        pass

    def embed_batch(self, image_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Default batch implementation — can be overridden.
        Returns dict: {image_id: embedding}.
        """
        return {image_id: self.embed_single(image_id) for image_id in image_ids}
