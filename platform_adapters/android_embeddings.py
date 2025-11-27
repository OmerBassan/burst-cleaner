# platform/android_embeddings.py
from typing import List, Dict
import numpy as np
from burst_cleaner.core.embeddings_core import EmbeddingBackend


class AndroidEmbeddingBackend(EmbeddingBackend):
    """
    Android implementation placeholder.

    הרעיון:
    - בצד ה־Android (Kotlin/Java) יש מודול TFLite / Torch Mobile
    - הוא מחשב embeddings ומחזיר אותם לפייתון (או כולה רץ ב־Kotlin ומחזיר numpy-ים לפייתון אם יש גשר)
    """

    def __init__(self, bridge):
        """
        bridge:
            אובייקט/פונקציה שהאפליקציה מספקת, שמסוגל:
            - לקבל image_id (URI)
            - להחזיר embedding כ־list[float] או np.ndarray
        """
        self.bridge = bridge

    def embed_single(self, image_id: str) -> np.ndarray:
        """
        Delegates the actual computation to the Android-side bridge.
        """
        vec = self.bridge.compute_embedding(image_id)  # expected: list[float] or np.ndarray
        if isinstance(vec, np.ndarray):
            return vec
        return np.array(vec, dtype="float32")

    def embed_batch(self, image_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        אופציונלי: אם ה־bridge יודע לעשות batch, אפשר לנצל.
        אחרת נ fallback ל־super().
        """
        if hasattr(self.bridge, "compute_embeddings_batch"):
            batch = self.bridge.compute_embeddings_batch(image_ids)  # {id: list[float]}
            return {
                image_id: (np.array(vec, dtype="float32")
                           if not isinstance(vec, np.ndarray) else vec)
                for image_id, vec in batch.items()
            }
        # fallback ל־מימוש הדיפולטי
        return super().embed_batch(image_ids)
