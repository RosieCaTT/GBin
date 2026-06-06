"""Derive cannot-link constraints from single-copy marker genes.

If two contigs both carry the same single-copy gene, they almost certainly come
from different genomes (a genome has one copy). Such pairs become cannot-link
constraints that the VAE's hybrid loss uses to push the contigs apart in latent
space. This is the marker signal feeding *training* (decontamination uses the
same markers at the *refinement* stage).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .scg import Markers


def cannot_link_pairs(
    markers: Markers, max_pairs: int = 2_000_000, seed: int = 0
) -> np.ndarray:
    """Return an (P, 2) int array of contig index pairs that must not co-bin."""
    by_marker: dict[int, list[int]] = defaultdict(list)
    for i, m in enumerate(markers.markers):
        if m is None:
            continue
        for mid in m.tolist():
            by_marker[mid].append(i)

    pairs: list[tuple[int, int]] = []
    for contigs in by_marker.values():
        for a in range(len(contigs)):
            for b in range(a + 1, len(contigs)):
                pairs.append((contigs[a], contigs[b]))

    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)

    arr = np.unique(np.array(pairs, dtype=np.int64), axis=0)
    if len(arr) > max_pairs:
        rng = np.random.default_rng(seed)
        arr = arr[rng.choice(len(arr), max_pairs, replace=False)]
    return arr
