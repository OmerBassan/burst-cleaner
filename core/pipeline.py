# burst_cleaner/core/pipeline.py

from typing import List, Dict
import numpy as np

from .loader_core import ImageLoaderInterface
from .embeddings_core import EmbeddingBackend
from .clustering import cluster_by_time_gaps
from .similarity import compute_centroid, cosine_similarities, select_best_image


# ---------------------------------------------------------
# 1) Simple time-based pipeline (your original version)
# ---------------------------------------------------------
def pipeline_detect_time_based_bursts(
    loader: ImageLoaderInterface,
    folder_path: str,
    time_gap_max: float,
    min_burst_len: int
) -> Dict:
    """
    Basic version: time-based clustering only (no embeddings).
    """

    image_ids = loader.scan_folder(folder_path)

    timestamps = [
        loader.extract_timestamp(image_id)
        for image_id in image_ids
    ]

    bursts = cluster_by_time_gaps(
        timestamps=timestamps,
        time_gap_max=time_gap_max,
        min_cluster_size=min_burst_len
    )

    bursts_mapped = [
        [image_ids[i] for i in burst]
        for burst in bursts
    ]

    return {
        "folder": folder_path,
        "num_images": len(image_ids),
        "num_bursts": len(bursts),
        "bursts": bursts_mapped
    }


# ---------------------------------------------------------
# 2) Full pipeline with embeddings + similarity filtering
# ---------------------------------------------------------
def pipeline_bursts_with_similarity(
    loader: ImageLoaderInterface,
    strict_embedder: EmbeddingBackend,
    loose_embedder: EmbeddingBackend,
    folder_path: str,
    time_gap_max: float,
    min_burst_len: int,
    similarity_threshold: float,
) -> Dict:
    """
    Full version (Hybrid Strict):
    - scan folder
    - extract timestamps
    - cluster by time
    - for each burst: compute embeddings ONLY for its images
    - strict decision by ResNet50, soft support by ResNet18
    """

    # 1) Load folder image identifiers
    image_ids = loader.scan_folder(folder_path)

    # 2) Extract timestamps and sort by time
    pairs = [
        (image_id, loader.extract_timestamp(image_id))
        for image_id in image_ids
    ]
    pairs.sort(key=lambda x: x[1])

    image_ids = [p[0] for p in pairs]
    timestamps = [p[1] for p in pairs]

    # 3) Time-based clustering
    bursts_idx = cluster_by_time_gaps(
        timestamps=timestamps,
        time_gap_max=time_gap_max,
        min_cluster_size=min_burst_len,
    )

    result_bursts: List[Dict] = []
    burst_counter = 1

    # Thresholds: strict (ResNet50) + loose (ResNet18)
    T_strict = similarity_threshold          
    T_loose  = similarity_threshold

    # 4) Embeddings only inside each burst
    for idx_list in bursts_idx:
        burst_image_ids = [image_ids[i] for i in idx_list]

        # (A) compute embeddings ONLY for current burst
        emb50_dict = strict_embedder.embed_batch(burst_image_ids)
        emb18_dict = loose_embedder.embed_batch(burst_image_ids)

        emb50_list = [emb50_dict[iid] for iid in burst_image_ids]
        emb18_list = [emb18_dict[iid] for iid in burst_image_ids]

        # (B) centroids per model
        centroid50 = compute_centroid(emb50_list)
        centroid18 = compute_centroid(emb18_list)

        sims50 = cosine_similarities(emb50_list, centroid50)
        sims18 = cosine_similarities(emb18_list, centroid18)

        avg50 = float(np.mean(sims50))
        avg18 = float(np.mean(sims18))

        # Hybrid Strict rule:
        # אם גם ResNet50 נמוך וגם ResNet18 נמוך -> לדלג על ה-burst
        if (avg50 < T_strict) and (avg18 < T_loose):
            del emb50_dict, emb18_dict, emb50_list, emb18_list
            continue

        # (C) recommended_keep לפי ResNet50 (strict)
        best_idx = int(np.argmax(sims50))
        best_image = burst_image_ids[best_idx]

        result_bursts.append({
            "burst_id": burst_counter,
            "num_images": len(burst_image_ids),
            "image_ids": burst_image_ids,
            "avg_similarity_strict": avg50,
            "avg_similarity_loose": avg18,
            "recommended_keep": best_image,
        })

        burst_counter += 1

        # (D) cleanup RAM immediately for this burst
        del emb50_dict, emb18_dict, emb50_list, emb18_list

    return {
        "folder": folder_path,
        "num_images": len(image_ids),
        "num_bursts": len(result_bursts),
        "bursts": result_bursts,
    }

