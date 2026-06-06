"""SCG-based decontamination of bins.

A bin whose single-copy genes appear (on median) two or more times almost
certainly merges multiple genomes. We split such bins with length-weighted
KMeans on the latent space, using contigs that carry the duplicated SCGs as the
initial centroids (so each split is seeded toward a distinct genome). Mirrors the
reclustering in VAMB/SemiBin, but the KMeans runs in pure PyTorch (GPU-capable,
no sklearn/cuML dependency).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..markers.scg import Markers
from ..utils import logger


def _weighted_kmeans(X, init, weights, iters: int = 100):
    """Length-weighted KMeans with fixed init. Returns hard assignments (m,)."""
    import torch

    centers = init.clone()
    assign = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for _ in range(iters):
        d = torch.cdist(X, centers)  # (m, k)
        new_assign = d.argmin(dim=1)
        new_centers = centers.clone()
        for k in range(centers.shape[0]):
            mask = new_assign == k
            if mask.any():
                w = weights[mask].unsqueeze(1)
                new_centers[k] = (X[mask] * w).sum(0) / w.sum()
        if torch.equal(new_assign, assign) and torch.allclose(new_centers, centers):
            break
        centers, assign = new_centers, new_assign
    return assign


def _kmeans_seeds(contigs, markers: Markers, lengths, counts, median: int):
    """Pick `median` seed contigs, one per copy of a representative SCG.

    Among SCGs occurring exactly `median` times, choose the one whose carrying
    contigs are longest (most reliable), and return those contigs as seeds.
    """
    considered = {i for i, c in enumerate(counts) if c == median}
    by_marker: dict[int, list[int]] = defaultdict(list)
    for c in contigs:
        m = markers.markers[c]
        if m is None:
            continue
        for mid in m:
            if mid in considered:
                by_marker[int(mid)].append(c)
    if not by_marker:
        return None
    best = max(by_marker.values(), key=lambda cs: min(lengths[i] for i in cs))
    return best  # exactly `median` contigs


def refine_bins(
    labels: np.ndarray,
    latent: np.ndarray,
    lengths: np.ndarray,
    markers: Markers,
    device,
    seed: int = 0,
) -> np.ndarray:
    """Split contaminated bins; return new contiguous labels."""
    import torch

    latent_t = torch.as_tensor(latent, dtype=torch.float32, device=device)
    lengths_t = torch.as_tensor(lengths, dtype=torch.float32, device=device)

    members: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        members[int(lab)].append(i)

    new_labels = np.full(len(labels), -1, dtype=np.int64)
    next_label = 0
    n_split = 0

    for lab, idxs in members.items():
        if len(idxs) == 1:
            new_labels[idxs[0]] = next_label
            next_label += 1
            continue

        counts = markers.counts(idxs)
        median = int(np.sort(counts)[len(counts) // 2])
        if median < 2:
            for i in idxs:
                new_labels[i] = next_label
            next_label += 1
            continue

        seeds = _kmeans_seeds(idxs, markers, lengths, counts, median)
        if seeds is None or len(seeds) < 2:
            for i in idxs:
                new_labels[i] = next_label
            next_label += 1
            continue

        idx_t = torch.as_tensor(idxs, device=device)
        assign = _weighted_kmeans(
            latent_t[idx_t],
            latent_t[torch.as_tensor(seeds, device=device)],
            lengths_t[idx_t],
        ).cpu().numpy()
        for sub in np.unique(assign):
            for i, a in zip(idxs, assign):
                if a == sub:
                    new_labels[i] = next_label
            next_label += 1
        n_split += 1

    if n_split:
        logger.info(f"Refinement split {n_split} contaminated bins")
    # Labels are already contiguous 0..next_label-1 by construction.
    return new_labels
