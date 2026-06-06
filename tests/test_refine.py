"""Tests for SCG-based decontamination (bin splitting)."""

import numpy as np
import torch

from gbin.markers.scg import Markers
from gbin.cluster.refine import refine_bins, _weighted_kmeans


def test_weighted_kmeans_separates_two_blobs():
    X = torch.tensor([[0, 0], [0.1, 0], [5, 5], [5.1, 5]], dtype=torch.float32)
    init = torch.tensor([[0, 0], [5, 5]], dtype=torch.float32)
    a = _weighted_kmeans(X, init, torch.ones(4))
    assert a[0] == a[1] and a[2] == a[3] and a[0] != a[2]


def _two_genome_bin(seed=0, n_markers=10):
    rng = np.random.default_rng(seed)
    a = np.array([1, 0, 0, 0], np.float32) + 0.02 * rng.normal(size=(10, 4))
    b = np.array([0, 1, 0, 0], np.float32) + 0.02 * rng.normal(size=(10, 4))
    latent = np.vstack([a, b]).astype(np.float32)
    # Each genome carries all n_markers once -> merged bin duplicates them all.
    rows = [[i % n_markers] for i in range(10)] * 2
    markers = Markers(
        [np.array(r, dtype=np.uint8) for r in rows],
        [[str(i)] for i in range(n_markers)],
        b"x" * 32,
    )
    return latent, markers


def test_refine_splits_contaminated_bin():
    latent, markers = _two_genome_bin()
    labels = np.zeros(20, dtype=np.int64)  # everything merged into one bin
    lengths = np.full(20, 5000, dtype=np.int32)
    new = refine_bins(labels, latent, lengths, markers, torch.device("cpu"))
    assert len(np.unique(new)) == 2
    assert len(set(new[:10].tolist())) == 1   # genome A stays together
    assert len(set(new[10:].tolist())) == 1   # genome B stays together
    assert new[0] != new[10]                  # and they are separated


def test_refine_keeps_clean_bin():
    rng = np.random.default_rng(0)
    latent = (np.array([1, 0], np.float32) + 0.02 * rng.normal(size=(10, 2))).astype(np.float32)
    # One genome, each SCG present once -> median count < 2 -> no split.
    markers = Markers(
        [np.array([i], dtype=np.uint8) for i in range(10)],
        [[str(i)] for i in range(10)],
        b"x" * 32,
    )
    labels = np.zeros(10, dtype=np.int64)
    lengths = np.full(10, 5000, dtype=np.int32)
    new = refine_bins(labels, latent, lengths, markers, torch.device("cpu"))
    assert len(np.unique(new)) == 1


def test_refine_relabels_contiguously():
    latent, markers = _two_genome_bin()
    # Two input bins (already split) -> output labels should be 0..K-1.
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    lengths = np.full(20, 5000, dtype=np.int32)
    new = refine_bins(labels, latent, lengths, markers, torch.device("cpu"))
    assert set(new.tolist()) == set(range(len(np.unique(new))))
