# core/similarity.py
from typing import List
import numpy as np


def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute centroid (mean vector) of a list of embeddings.
    """
    if not embeddings:
        raise ValueError("Cannot compute centroid of empty list")
    stacked = np.stack(embeddings, axis=0)  # shape: (N, D)
    return stacked.mean(axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def cosine_similarities(
    embeddings: List[np.ndarray],
    centroid: np.ndarray
) -> List[float]:
    """
    Compute cosine similarity of each embedding vs centroid.
    """
    return [cosine_similarity(e, centroid) for e in embeddings]


def select_best_image(
    embeddings: List[np.ndarray],
    image_ids: List[str]
) -> str:
    """
    Select the image id with highest cosine similarity to the centroid.
    """
    if not embeddings or not image_ids:
        raise ValueError("embeddings and image_ids must be non-empty and same length")

    centroid = compute_centroid(embeddings)
    sims = cosine_similarities(embeddings, centroid)
    best_idx = int(np.argmax(sims))
    return image_ids[best_idx]
