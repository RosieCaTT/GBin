"""GPU tetranucleotide-frequency (TNF) composition features.

Pipeline per chunk of contigs (all on the GPU):
  raw ASCII bytes --LUT--> base codes 0..3 (4 = invalid)
  sliding 4-mer codes (windows not crossing contig boundaries, skipping any
      window containing an invalid base)
  segmented bincount --> per-contig 256-dim 4-mer counts
  normalize to frequencies, zero-center, project 256 -> 103 via VAMB's kernel

The 103-dim projection (kernel.npz, reused from VAMB) folds reverse-complement
pairs together and decorrelates the features. Output matches VAMB's
``parsecontigs`` (un-z-scored); z-scoring happens when the training dataloader is
assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..io.fasta import iter_fasta_chunks
from ..utils import RefHasher, data_path, logger, timed

NTNF = 103  # projected composition dimensionality
_DEFAULT_MAX_BASES = 32_000_000  # per GPU chunk; ~1-2 GB of working tensors


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class Composition:
    """Projected TNF matrix plus contig metadata."""

    identifiers: np.ndarray  # dtype=object, str ids
    lengths: np.ndarray      # int32, (N,)
    tnf: np.ndarray          # float32, (N, 103)
    refhash: bytes

    @property
    def n_contigs(self) -> int:
        return len(self.identifiers)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            identifiers=self.identifiers,
            lengths=self.lengths,
            tnf=self.tnf,
            refhash=np.frombuffer(self.refhash, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: Path) -> "Composition":
        d = np.load(path, allow_pickle=True)
        return cls(
            identifiers=d["identifiers"],
            lengths=d["lengths"],
            tnf=d["tnf"],
            refhash=d["refhash"].tobytes(),
        )


# --------------------------------------------------------------------------- #
# Lookup table & kernel
# --------------------------------------------------------------------------- #
def build_lut() -> np.ndarray:
    """256-entry ASCII -> base-code table. A/C/G/T(/U)->0..3, everything else->4."""
    lut = np.full(256, 4, dtype=np.uint8)
    for code, base in enumerate("ACGT"):
        lut[ord(base)] = code
        lut[ord(base.lower())] = code
    lut[ord("U")] = 3  # treat uracil as thymine (RNA safety)
    lut[ord("u")] = 3
    return lut


def load_kernel() -> np.ndarray:
    """Load the bundled 256x103 TNF projection kernel."""
    arr = np.load(data_path("kernel.npz"))["arr_0"]
    assert arr.shape == (256, NTNF), f"unexpected kernel shape {arr.shape}"
    return arr.astype(np.float32)


# --------------------------------------------------------------------------- #
# GPU counting
# --------------------------------------------------------------------------- #
def count_4mers(seq_bytes: np.ndarray, lengths: np.ndarray, device, lut_dev):
    """Return a (n_contigs, 256) float tensor of 4-mer counts on ``device``.

    ``seq_bytes`` is the concatenation of all contig sequences (uint8 ASCII);
    ``lengths`` gives the per-contig split. Windows never cross contig
    boundaries, and any window containing a non-ACGT base is skipped.
    """
    import torch

    n = int(lengths.shape[0])
    lengths_t = torch.as_tensor(lengths, device=device, dtype=torch.int64)
    seq = torch.as_tensor(seq_bytes, device=device)  # uint8 (T,)
    total = int(seq.numel())
    if total < 4:
        return torch.zeros((n, 256), device=device)

    codes = lut_dev[seq.long()].to(torch.int32)  # (T,) values 0..4
    del seq

    # Per-base contig id and within-contig position.
    contig_id = torch.repeat_interleave(
        torch.arange(n, device=device, dtype=torch.int64), lengths_t
    )  # (T,)
    starts = torch.cumsum(lengths_t, 0) - lengths_t  # exclusive prefix sum (n,)
    pos_in_contig = torch.arange(total, device=device, dtype=torch.int64) - starts[contig_id]
    length_per_base = lengths_t[contig_id]
    # A window may start here iff 4 bases fit before the contig ends.
    is_start = pos_in_contig <= (length_per_base - 4)  # (T,) bool

    c0, c1, c2, c3 = codes[:-3], codes[1:-2], codes[2:-1], codes[3:]
    code = c0 * 64 + c1 * 16 + c2 * 4 + c3  # (T-3,)
    base_valid = (c0 < 4) & (c1 < 4) & (c2 < 4) & (c3 < 4)
    valid = base_valid & is_start[:-3]

    win_contig = contig_id[:-3]
    target = (win_contig * 256 + code)[valid]
    counts = torch.bincount(target, minlength=n * 256).reshape(n, 256).float()
    return counts


def project(counts, kernel_dev):
    """Normalize 256-dim counts to zero-centered frequencies and project to 103."""
    import torch

    s = counts.sum(dim=1, keepdim=True)
    s = torch.where(s == 0, torch.ones_like(s), s)
    freqs = counts / s
    freqs = freqs - (1.0 / 256.0)
    return freqs @ kernel_dev  # (n, 103)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_composition(
    fasta: Path,
    device,
    min_length: int = 2000,
    max_bases_per_chunk: int = _DEFAULT_MAX_BASES,
) -> Composition:
    """Compute projected TNF for all contigs >= ``min_length`` in ``fasta``."""
    import torch

    lut_dev = torch.as_tensor(build_lut(), device=device)
    kernel_dev = torch.as_tensor(load_kernel(), device=device)

    identifiers: list[str] = []
    lengths_parts: list[np.ndarray] = []
    tnf_parts: list[np.ndarray] = []
    hasher = RefHasher()

    with timed("Computing composition (TNF)"):
        for chunk in iter_fasta_chunks(fasta, min_length, max_bases_per_chunk):
            counts = count_4mers(chunk.seq_bytes, chunk.lengths, device, lut_dev)
            tnf = project(counts, kernel_dev)  # (n, 103)
            tnf_parts.append(tnf.cpu().numpy().astype(np.float32))
            lengths_parts.append(chunk.lengths)
            identifiers.extend(chunk.names)
            for name in chunk.names:
                hasher.add(name)
            logger.debug(f"  processed {len(identifiers)} contigs")

    if not identifiers:
        raise ValueError(
            f"No contigs >= {min_length} bp found in {fasta}. "
            "Lower --min-contig-len or check the input."
        )

    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "Contig identifiers are not unique (gbin uses the header up to the "
            "first whitespace). De-duplicate your FASTA headers."
        )

    comp = Composition(
        identifiers=np.array(identifiers, dtype=object),
        lengths=np.concatenate(lengths_parts).astype(np.int32),
        tnf=np.concatenate(tnf_parts, axis=0),
        refhash=hasher.digest(),
    )
    logger.info(f"Composition: {comp.n_contigs} contigs x {NTNF} TNF features")
    return comp


# --------------------------------------------------------------------------- #
# CPU reference (used by tests to validate the GPU path)
# --------------------------------------------------------------------------- #
def count_4mers_reference(seq: str) -> np.ndarray:
    """Plain-Python 256-dim 4-mer count for one sequence (test oracle)."""
    lut = build_lut()
    counts = np.zeros(256, dtype=np.float64)
    arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    codes = lut[arr]
    for i in range(len(codes) - 3):
        a, b, c, d = codes[i], codes[i + 1], codes[i + 2], codes[i + 3]
        if a < 4 and b < 4 and c < 4 and d < 4:
            counts[a * 64 + b * 16 + c * 4 + d] += 1
    return counts


def project_reference(counts_2d: np.ndarray) -> np.ndarray:
    """NumPy version of :func:`project` for tests."""
    kernel = load_kernel()
    s = counts_2d.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    freqs = counts_2d / s - (1.0 / 256.0)
    return (freqs @ kernel).astype(np.float32)
