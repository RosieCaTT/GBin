"""Dispatch latent clustering to the configured backend.

* ``auto``   -> Leiden if the RAPIDS stack imports, else medoid.
* ``leiden`` -> kNN graph (cuML/torch) + community detection (cuGraph Leiden,
                or NumPy label propagation if cuGraph is absent).
* ``medoid`` -> pure-PyTorch iterative medoid.
"""

from __future__ import annotations

import numpy as np

from ..config import GBinConfig
from ..utils import logger, timed
from .community import (
    labelprop_communities,
    leiden_communities,
    rapids_available,
)
from .knn_graph import build_knn_graph
from .medoid_torch import cluster_medoid


def cluster_latent(
    latent: np.ndarray,
    lengths: np.ndarray,
    cfg: GBinConfig,
    device,
) -> np.ndarray:
    """Return an integer bin label per contig."""
    method = cfg.cluster_method
    if method == "auto":
        # Medoid (pure-torch, like VAMB) is the robust default: it works on any
        # CUDA GPU with no RAPIDS/numba surprises and no cold-start. Leiden is an
        # opt-in acceleration for healthy RAPIDS setups and very large data.
        method = "medoid"
        logger.info(
            "Cluster method: medoid (default). Pass --cluster leiden to use "
            "cuGraph Leiden if your RAPIDS/GPU stack supports it."
        )

    with timed(f"Clustering ({method})"):
        if method == "medoid":
            labels = cluster_medoid(latent, lengths, device, seed=cfg.seed)
        elif method == "leiden":
            labels = _cluster_leiden(latent, lengths, cfg, device)
        else:
            raise ValueError(f"Unknown cluster method: {method}")

    n_bins = len(np.unique(labels))
    logger.info(f"Clustering produced {n_bins} raw bins from {len(latent)} contigs")
    return labels


def _cluster_leiden(latent, lengths, cfg, device) -> np.ndarray:
    """kNN graph + community detection, with a medoid fallback on any failure.

    The RAPIDS stack on bleeding-edge GPUs (e.g. Blackwell) can fail in the
    cudf->host conversion or kernel JIT; rather than abort a long run, we warn and
    fall back to the pure-torch medoid clusterer.
    """
    try:
        src, dst, w = build_knn_graph(latent, cfg.knn_k, device)
        logger.debug(f"kNN graph: {len(src)} directed edges")
        if rapids_available():
            return leiden_communities(src, dst, w, len(latent), seed=cfg.seed)
        logger.warning(
            "RAPIDS/cuGraph not found; using NumPy label propagation on the kNN "
            "graph (CPU, slower). Install RAPIDS or use --cluster medoid for "
            "large datasets."
        )
        return labelprop_communities(src, dst, w, len(latent), seed=cfg.seed)
    except Exception as e:
        logger.warning(
            f"Leiden/graph clustering failed ({type(e).__name__}: {e}); used the "
            "pure-torch medoid clusterer instead. On new GPUs (e.g. Blackwell) the "
            "RAPIDS/numba stack is often not ready -- medoid is the recommended "
            "default there."
        )
        logger.opt(exception=True).debug("Leiden failure traceback")
        return cluster_medoid(latent, lengths, device, seed=cfg.seed)
