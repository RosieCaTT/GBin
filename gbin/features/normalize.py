"""Normalize composition + abundance into VAE inputs (on the GPU).

Replicates VAMB's ``make_dataloader`` preprocessing as torch ops:

* per-sample depth normalization (so samples contribute equally),
* per-contig abundance normalized to sum to 1 (a categorical distribution),
* total abundance -> log -> z-score (one extra scalar feature),
* TNF z-scored per feature,
* per-contig loss weights from log length.

Returns plain float32 numpy arrays; the training dataloader (M3) turns them into
tensors and streams batches back to the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizedFeatures:
    depths: np.ndarray          # (N, S) rows sum to 1
    tnf: np.ndarray             # (N, 103) z-scored per column
    total_abundance: np.ndarray  # (N, 1) log + z-scored
    weights: np.ndarray         # (N, 1) length-based loss weights

    @property
    def nsamples(self) -> int:
        return self.depths.shape[1]


def _zscore_cols(x):
    import torch

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (x - mean) / std


def normalize_features(
    abundance: np.ndarray,
    tnf: np.ndarray,
    lengths: np.ndarray,
    device,
) -> NormalizedFeatures:
    """Compute VAE-ready features on ``device`` (see module docstring)."""
    import torch

    ab = torch.as_tensor(np.asarray(abundance, dtype=np.float32), device=device).clone()
    tn = torch.as_tensor(np.asarray(tnf, dtype=np.float32), device=device).clone()
    ln = torch.as_tensor(np.asarray(lengths, dtype=np.float32), device=device)
    n, nsamples = ab.shape

    # Per-sample depth normalization to a common scale.
    sample_sum = ab.sum(dim=0)  # (S,)
    if torch.any(sample_sum == 0):
        raise ValueError(
            "One or more samples have zero total coverage across all contigs; "
            "such a sample carries no signal and cannot be depth-normalized."
        )
    ab = ab * (1_000_000.0 / sample_sum)
    total_abundance = ab.sum(dim=1)  # (N,)

    # Per-contig: normalize to a distribution that sums to 1. Contigs with zero
    # coverage everywhere get a uniform distribution.
    zero = total_abundance == 0
    if torch.any(zero):
        ab[zero] = 1.0 / nsamples
    denom = total_abundance.clone()
    denom[zero] = 1.0
    ab = ab / denom.reshape(-1, 1)

    # Total abundance as an extra feature: log then z-score.
    ta = torch.log(total_abundance.clamp(min=0.001))
    ta = (ta - ta.mean()) / (ta.std(unbiased=False).clamp(min=1e-8))
    ta = ta.reshape(-1, 1)

    tn = _zscore_cols(tn)

    # Loss weights: longer contigs are more trustworthy. Matches VAMB.
    w = torch.log(ln) - 5.0
    w = torch.clamp(w, min=2.0)
    w = w * (w.numel() / w.sum())
    w = w.reshape(-1, 1)

    return NormalizedFeatures(
        depths=ab.cpu().numpy().astype(np.float32),
        tnf=tn.cpu().numpy().astype(np.float32),
        total_abundance=ta.cpu().numpy().astype(np.float32),
        weights=w.cpu().numpy().astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# NumPy reference (test oracle)
# --------------------------------------------------------------------------- #
def normalize_features_reference(
    abundance: np.ndarray, tnf: np.ndarray, lengths: np.ndarray
) -> NormalizedFeatures:
    ab = abundance.astype(np.float32).copy()
    tn = tnf.astype(np.float32).copy()
    n, nsamples = ab.shape

    sample_sum = ab.sum(axis=0)
    ab *= 1_000_000.0 / sample_sum
    total = ab.sum(axis=1)

    zero = total == 0
    ab[zero] = 1.0 / nsamples
    denom = total.copy()
    denom[zero] = 1.0
    ab /= denom.reshape(-1, 1)

    ta = np.log(np.clip(total, 0.001, None))
    ta = (ta - ta.mean()) / max(ta.std(), 1e-8)
    ta = ta.reshape(-1, 1).astype(np.float32)

    mean = tn.mean(axis=0)
    std = tn.std(axis=0)
    std[std == 0] = 1.0
    tn = (tn - mean) / std

    w = np.log(lengths.astype(np.float32)) - 5.0
    w[w < 2.0] = 2.0
    w *= len(w) / w.sum()
    w = w.reshape(-1, 1).astype(np.float32)

    return NormalizedFeatures(ab.astype(np.float32), tn.astype(np.float32), ta, w)
