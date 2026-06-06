"""Iterative medoid clustering of the latent space, in pure PyTorch.

A faithful port of VAMB's clusterer (cosine distance on row-normalized latent;
adaptive peak/valley threshold detection), with one improvement: the
length-weighted distance histogram is built on the GPU via ``bucketize`` +
``index_add`` instead of being copied to the CPU each iteration. Only the tiny
(~60-bin) peak/valley scan runs on the CPU.

This is the always-available clusterer (no RAPIDS needed) and the fallback for
the Leiden path. It repeatedly finds a dense medoid, emits the points within an
adaptively chosen radius as a bin, and removes them; the ``peak_valley_ratio``
relaxes when clean valleys stop appearing, guaranteeing termination.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from ..utils import logger

# Distance scale is [0, 1] (normalized cosine). Constants match VAMB.
_DEFAULT_RADIUS = 0.06
_MEDOID_RADIUS = 0.05
_DELTA_X = 0.005
_XMAX = 0.3

# Discretized N(0, 0.01) PDF (31 points spaced _DELTA_X apart) used to smooth the
# distance histogram so peaks/valleys are robust to bin noise.
_NORMALPDF = _DELTA_X * np.array(
    [2.43432053e-11, 9.13472041e-10, 2.66955661e-08, 6.07588285e-07, 1.07697600e-05,
     1.48671951e-04, 1.59837411e-03, 1.33830226e-02, 8.72682695e-02, 4.43184841e-01,
     1.75283005e00, 5.39909665e00, 1.29517596e01, 2.41970725e01, 3.52065327e01,
     3.98942280e01, 3.52065327e01, 2.41970725e01, 1.29517596e01, 5.39909665e00,
     1.75283005e00, 4.43184841e-01, 8.72682695e-02, 1.33830226e-02, 1.59837411e-03,
     1.48671951e-04, 1.07697600e-05, 6.07588285e-07, 2.66955661e-08, 9.13472041e-10,
     2.43432053e-11],
    dtype=np.float64,
)


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Scale rows so cosine distance becomes ``0.5 - matrix @ matrix[i]`` in [0, 1]."""
    matrix = matrix.clone()
    zero = (matrix == 0).all(dim=1)
    if torch.any(zero):
        matrix[zero] = 1.0 / matrix.shape[1]
    norm = matrix.norm(dim=1, keepdim=True) * (2 ** 0.5)
    return matrix / norm


def _distances(matrix: torch.Tensor, index: int) -> torch.Tensor:
    d = 0.5 - matrix @ matrix[index]
    d[index] = 0.0
    return d


def _weighted_histogram(values, weights, n_bins, vmax):
    """Length-weighted histogram over [0, vmax], computed on-device.

    Uniform left-closed bins: bin = floor(value / vmax * n_bins), clamped into
    range (matches numpy.histogram semantics, unlike bucketize at exact edges).
    """
    idx = (values * (n_bins / vmax)).long().clamp_(0, n_bins - 1)
    hist = torch.zeros(n_bins, device=values.device, dtype=weights.dtype)
    hist.index_add_(0, idx, weights)
    return hist


def _find_threshold(hist_cpu: np.ndarray, peak_valley_ratio: float):
    """Find a density valley in the smoothed histogram; return float or None."""
    pdf_len = len(_NORMALPDF)
    densities = np.zeros(len(hist_cpu) + pdf_len - 1)
    for i in range(len(hist_cpu)):
        densities[i : i + pdf_len] += _NORMALPDF * hist_cpu[i]
    densities = densities[15:-15]

    delta_x = _XMAX / len(hist_cpu)
    peak_density = 0.0
    peak_over = False
    density_at_min = 0.0
    threshold = None
    x = 0.0
    for density in densities:
        if not peak_over and density > peak_density:
            if x > 0.1:
                return None
            peak_density = density
        if not peak_over and density < 0.6 * peak_density:
            peak_over = True
            density_at_min = density
        if peak_over and density > 1.5 * density_at_min:
            break
        if peak_over and density < density_at_min:
            density_at_min = density
            if density < peak_valley_ratio * peak_density:
                threshold = x
        x += delta_x

    if threshold is None or threshold > 0.2 + peak_valley_ratio:
        return None
    return float(threshold)


def cluster_medoid(
    latent: np.ndarray,
    lengths: np.ndarray,
    device,
    *,
    seed: int = 0,
    windowsize: int = 200,
    minsuccesses: int = 15,
    max_steps: int = 25,
) -> np.ndarray:
    """Cluster ``latent`` (N, D) into bins; return an int label per contig.

    Contigs are weighted by length. ``peak_valley_ratio`` adapts upward when
    clean valleys stop appearing, so the loop always terminates.
    """
    n = int(latent.shape[0])
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    matrix = _normalize_rows(torch.as_tensor(latent, dtype=torch.float32, device=device))
    lengths_t = torch.as_tensor(lengths, dtype=torch.float32, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    labels = np.full(n, -1, dtype=np.int64)

    order = np.argsort(lengths)[::-1].copy()  # longest contigs make the best seeds
    rng = random.Random(seed)
    n_bins = round(_XMAX / _DELTA_X)

    state = {"pos": 0, "pvr": 0.1, "successes": 0}
    attempts: deque[bool] = deque()
    n_clusters = 0
    remaining = n

    def next_seed() -> int:
        while True:
            i = int(order[state["pos"] % n])
            state["pos"] += 1
            if active[i]:
                return i

    def local_density(medoid: int):
        d = _distances(matrix, medoid)
        within = (d <= _MEDOID_RADIUS) & active
        closeness = _MEDOID_RADIUS - d[within]
        density = (lengths_t[within] * closeness).sum().item()
        return d, within, density

    def record_attempt(success: bool) -> None:
        # Manual window bookkeeping (don't rely on deque.maxlen so we can keep an
        # incremental success count, as VAMB does).
        if len(attempts) == windowsize:
            state["successes"] -= attempts.popleft()
        attempts.append(success)
        state["successes"] += success
        if len(attempts) == windowsize and state["successes"] < minsuccesses:
            state["pvr"] += 0.1
            attempts.clear()
            state["successes"] = 0
            state["pos"] = 0  # restart from the best seeds under the relaxed criteria

    while remaining > 0:
        medoid = next_seed()
        d, within, density = local_density(medoid)

        # Wander toward a denser medoid in the neighbourhood.
        tried = {medoid}
        cands = [int(i) for i in torch.nonzero(within).flatten().tolist() if i not in tried]
        rng.shuffle(cands)
        cands = cands[:max_steps]
        k = 0
        while k < len(cands):
            cand = cands[k]
            tried.add(cand)
            d2, within2, density2 = local_density(cand)
            if density2 > density:
                medoid, d, within, density = cand, d2, within2, density2
                cands = [int(i) for i in torch.nonzero(within).flatten().tolist()
                         if i not in tried]
                rng.shuffle(cands)
                cands = cands[:max_steps]
                k = 0
            else:
                k += 1

        # Decide the cluster around this medoid.
        n_close = int(((d < _MEDOID_RADIUS) & active).sum().item())
        if n_close <= 1:
            members = torch.nonzero((d < _MEDOID_RADIUS) & active).flatten()
            counts_success = None
        else:
            below = (d <= _XMAX) & active
            hist = _weighted_histogram(d[below], lengths_t[below], n_bins, _XMAX)
            thr = _find_threshold(hist.cpu().numpy(), state["pvr"])
            if thr is None:
                if state["pvr"] > 0.55:
                    members = torch.nonzero((d <= _DEFAULT_RADIUS) & active).flatten()
                    counts_success = None
                else:
                    record_attempt(False)
                    continue
            else:
                members = torch.nonzero((d <= thr) & active).flatten()
                counts_success = True

        idx = members.cpu().numpy()
        labels[idx] = n_clusters
        active[members] = False
        remaining -= len(idx)
        n_clusters += 1
        if counts_success and state["pvr"] < 0.55:
            record_attempt(True)

    logger.info(f"Medoid clustering produced {n_clusters} bins")
    return labels
