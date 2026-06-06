"""Build a k-nearest-neighbour similarity graph over the latent space.

Primary backend is cuML's GPU NearestNeighbors (cosine); a chunked pure-PyTorch
brute force serves as the fallback and keeps the path testable without RAPIDS.
Returns a directed edge list (src, dst, weight) with self-edges removed and
weights = max(cosine_similarity, 0). The community-detection step treats it as
undirected.
"""

from __future__ import annotations

import numpy as np

from ..utils import logger


def _knn_cuml(latent: np.ndarray, k: int):
    from cuml.neighbors import NearestNeighbors

    from .community import _to_host  # cupy-based device->host (avoids numba)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(latent)
    dist, idx = nn.kneighbors(latent)  # includes self at distance ~0
    dist = _to_host(dist)
    idx = _to_host(idx)
    n = latent.shape[0]
    src = np.repeat(np.arange(n), k + 1)
    dst = idx.reshape(-1)
    sim = (1.0 - dist).reshape(-1)
    keep = src != dst
    return src[keep], dst[keep], np.clip(sim[keep], 0.0, None).astype(np.float32)


def _knn_torch(latent: np.ndarray, k: int, device, chunk: int = 8192):
    import torch

    x = torch.as_tensor(latent, dtype=torch.float32, device=device)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    n = x.shape[0]
    kk = min(k + 1, n)
    src_parts, dst_parts, w_parts = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = x[start:stop] @ x.T  # (b, n) cosine similarity
        topv, topi = sims.topk(kk, dim=1)
        rows = torch.arange(start, stop, device=device).unsqueeze(1).expand_as(topi)
        self_mask = topi != rows  # drop self-edges
        src_parts.append(rows[self_mask])
        dst_parts.append(topi[self_mask])
        w_parts.append(topv[self_mask].clamp_(min=0.0))
    src = torch.cat(src_parts).cpu().numpy()
    dst = torch.cat(dst_parts).cpu().numpy()
    w = torch.cat(w_parts).cpu().numpy().astype(np.float32)
    return src, dst, w


def build_knn_graph(latent: np.ndarray, k: int, device, prefer_cuml: bool = True):
    """Return (src, dst, weight) edge arrays for the kNN similarity graph."""
    n = latent.shape[0]
    if prefer_cuml:
        try:
            from cuml.neighbors import NearestNeighbors  # noqa: F401

            logger.debug("Building kNN graph with cuML")
            return _knn_cuml(latent, min(k, n - 1))
        except Exception as e:  # pragma: no cover - depends on RAPIDS
            logger.debug(f"cuML kNN unavailable ({e}); using torch brute force")
    return _knn_torch(latent, k, device)
