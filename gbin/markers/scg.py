"""Single-copy marker gene (SCG) bookkeeping and bin quality scoring.

A :class:`Markers` object stores, for each contig, the set of single-copy marker
genes found on it (deduplicated within a contig). From these we estimate a bin's
completeness (fraction of distinct SCGs present) and contamination (excess copies
of SCGs, signalling multiple genomes), following VAMB/CheckM conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from ..utils import RefHasher


@dataclass
class Markers:
    markers: list[Optional[np.ndarray]]   # per contig: uint8 SCG ids, or None
    marker_names: list[list[str]]          # SCG id -> list of equivalent names
    refhash: bytes

    @property
    def n_markers(self) -> int:
        return len(self.marker_names)

    @property
    def n_seqs(self) -> int:
        return len(self.markers)

    def counts(self, indices: Iterable[int]) -> np.ndarray:
        """Vector of per-SCG counts across the given contig indices."""
        counts = np.zeros(self.n_markers, dtype=np.int32)
        for i in indices:
            m = self.markers[i]
            if m is not None:
                counts[m] += 1
        return counts

    def score_bin(self, indices: Iterable[int]) -> tuple[float, float]:
        """Return (completeness, contamination) in [0, 1+] for a set of contigs."""
        counts = self.counts(indices)
        n_unique = int((counts > 0).sum())
        completeness = n_unique / self.n_markers
        contamination = (int(counts.sum()) - n_unique) / self.n_markers
        return completeness, contamination

    # ------------------------------------------------------------------ #
    def save(self, path: Union[Path, str]) -> None:
        rep = {
            "markers": [None if m is None else m.tolist() for m in self.markers],
            "marker_names": self.marker_names,
            "refhash": self.refhash.hex(),
        }
        with open(path, "w") as f:
            json.dump(rep, f)

    @classmethod
    def load(cls, path: Union[Path, str], refhash: Optional[bytes] = None) -> "Markers":
        with open(path) as f:
            rep = json.load(f)
        observed = bytes.fromhex(rep["refhash"])
        if refhash is not None:
            RefHasher.verify(observed, refhash, "markers")
        markers = [None if m is None else np.array(m, dtype=np.uint8)
                   for m in rep["markers"]]
        return cls(markers, rep["marker_names"], observed)
