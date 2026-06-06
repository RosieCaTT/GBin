"""Community detection on the kNN similarity graph.

Primary backend: cuGraph Leiden (GPU). Fallback: weighted label propagation in
NumPy, which keeps the graph clustering path working and testable without
RAPIDS. Both take a directed edge list and return one integer community label
per node (0..K-1).
"""

from __future__ import annotations

import numpy as np

from ..utils import logger


def rapids_available() -> bool:
    """True if both cuML and cuGraph import (the full GPU clustering stack)."""
    try:
        import cugraph  # noqa: F401
        import cuml  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# cuGraph Leiden (GPU)
# --------------------------------------------------------------------------- #
def _to_host(series) -> np.ndarray:
    """cudf Series/cupy/numpy -> numpy, going via cupy (NOT numba).

    cudf's ``.to_numpy()`` routes the device->host copy through numba_cuda, which
    fails to enumerate brand-new GPUs (e.g. Blackwell / sm_120) and raises
    IndexError. cupy's host copy uses the CUDA runtime directly and avoids this.
    """
    try:
        import cupy

        if isinstance(series, cupy.ndarray):
            return cupy.asnumpy(series)
        if hasattr(series, "to_cupy"):  # cudf Series/Column
            return cupy.asnumpy(series.to_cupy())
    except ImportError:
        pass
    return np.asarray(series)


def leiden_communities(src, dst, weight, n_nodes, resolution: float = 1.0, seed: int = 0):
    """Leiden community detection via cuGraph. Returns labels (n_nodes,)."""
    import cudf
    import cugraph

    df = cudf.DataFrame(
        {"src": np.asarray(src), "dst": np.asarray(dst), "w": np.asarray(weight)}
    )
    g = cugraph.Graph()
    g.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="w", renumber=True)
    parts, _modularity = cugraph.leiden(g, resolution=resolution, random_state=seed)
    parts = parts.sort_values("vertex")
    labels = np.full(n_nodes, -1, dtype=np.int64)
    v = _to_host(parts["vertex"]).astype(np.int64)
    p = _to_host(parts["partition"]).astype(np.int64)
    labels[v] = p
    # Compact labels and assign singletons to any unreferenced nodes.
    return _compact_labels(labels)


# --------------------------------------------------------------------------- #
# Label propagation (NumPy fallback)
# --------------------------------------------------------------------------- #
def _undirected_csr(src, dst, weight, n):
    s = np.concatenate([np.asarray(src), np.asarray(dst)])
    d = np.concatenate([np.asarray(dst), np.asarray(src)])
    w = np.concatenate([np.asarray(weight, dtype=np.float64)] * 2)
    order = np.argsort(s, kind="stable")
    s, d, w = s[order], d[order], w[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    counts = np.bincount(s, minlength=n)
    indptr[1:] = np.cumsum(counts)
    return indptr, d, w


def labelprop_communities(src, dst, weight, n_nodes, iters: int = 50, seed: int = 0):
    """Weighted asynchronous label propagation. Returns labels (n_nodes,)."""
    indptr, nbr, w = _undirected_csr(src, dst, weight, n_nodes)
    labels = np.arange(n_nodes, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for _ in range(iters):
        changed = 0
        for node in rng.permutation(n_nodes):
            a, b = indptr[node], indptr[node + 1]
            if a == b:
                continue
            votes: dict[int, float] = {}
            for j in range(a, b):
                lab = labels[nbr[j]]
                votes[lab] = votes.get(lab, 0.0) + w[j]
            # Max weight, ties broken by smallest label (deterministic).
            best = min((-v, lab) for lab, v in votes.items())[1]
            if best != labels[node]:
                labels[node] = best
                changed += 1
        if changed == 0:
            break
    return _compact_labels(labels)


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary labels (incl. -1 singletons) to a dense 0..K-1 range."""
    labels = labels.copy()
    # Give each unassigned node (-1) its own fresh label.
    unassigned = np.nonzero(labels < 0)[0]
    if len(unassigned):
        nxt = labels.max() + 1 if labels.max() >= 0 else 0
        labels[unassigned] = np.arange(nxt, nxt + len(unassigned))
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.int64)
