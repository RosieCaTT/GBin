"""Per-sample abundance (coverage) features.

Two input routes, both producing an (N_contigs x N_samples) matrix aligned to the
composition's contig order:

* strobealign ``--aemb`` TSVs: one 2-column file (contig, coverage) per sample,
  or a single pre-merged file with a ``contigname`` header and one column per
  sample. This is the fast, recommended route.
* sorted BAM files via ``pycoverm`` (trimmed-mean coverage, like VAMB/CoverM).

Alignment is by contig *name*, so the abundance source may list contigs in any
order and may include short contigs that the composition step filtered out.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..utils import RefHasher, logger


@dataclass
class Abundance:
    """Coverage matrix aligned to the composition contig order."""

    matrix: np.ndarray          # float32, (N, S)
    samplenames: list[str]
    refhash: bytes              # hash of the contig identifiers it is aligned to

    @property
    def nseqs(self) -> int:
        return self.matrix.shape[0]

    @property
    def nsamples(self) -> int:
        return self.matrix.shape[1]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            matrix=self.matrix,
            samplenames=np.array(self.samplenames, dtype=object),
            refhash=np.frombuffer(self.refhash, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: Path, refhash: Optional[bytes] = None) -> "Abundance":
        d = np.load(path, allow_pickle=True)
        ab = cls(
            matrix=d["matrix"].astype(np.float32),
            samplenames=list(d["samplenames"]),
            refhash=d["refhash"].tobytes(),
        )
        if refhash is not None:
            RefHasher.verify(ab.refhash, refhash, "abundance")
        return ab


# --------------------------------------------------------------------------- #
# aemb / TSV parsing
# --------------------------------------------------------------------------- #
def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _looks_like_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _read_two_column(path: Path) -> dict[str, float]:
    """Read a (contig, value) TSV into a dict, tolerating an optional header."""
    out: dict[str, float] = {}
    with _open_text(path) as f:
        for lineno, line in enumerate(f):
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno + 1}: expected 2 tab-separated columns")
            name, value = parts[0], parts[1]
            if not _looks_like_float(value):
                if lineno == 0:
                    continue  # header line
                raise ValueError(f"{path}:{lineno + 1}: non-numeric coverage '{value}'")
            out[name] = float(value)
    return out


def _read_merged(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Read a merged TSV: header 'contigname<TAB>s1<TAB>s2...' + one row per contig."""
    with _open_text(path) as f:
        header = f.readline().rstrip("\n\r").split("\t")
        if len(header) < 2:
            raise ValueError(f"{path}: merged abundance TSV needs >= 2 columns")
        samples = header[1:]
        rows: dict[str, np.ndarray] = {}
        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(header):
                raise ValueError(
                    f"{path}:{lineno}: expected {len(header)} columns, got {len(parts)}"
                )
            rows[parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return samples, rows


def _is_merged(path: Path) -> bool:
    """A single file is 'merged' if its first line has > 2 tab-separated fields."""
    with _open_text(path) as f:
        first = f.readline().rstrip("\n\r")
    return len(first.split("\t")) > 2


def from_tsv(
    paths: Sequence[Path],
    identifiers: Sequence[str],
    refhash: bytes,
) -> Abundance:
    """Build an Abundance from aemb TSV(s), aligned to ``identifiers``."""
    paths = [Path(p) for p in paths]
    index = {name: i for i, name in enumerate(identifiers)}
    n = len(identifiers)

    if len(paths) == 1 and _is_merged(paths[0]):
        samples, rows = _read_merged(paths[0])
        matrix = np.zeros((n, len(samples)), dtype=np.float32)
        _fill_from_rows(matrix, rows, index, paths[0])
    else:
        samples = [p.name.split(".")[0] for p in paths]
        matrix = np.zeros((n, len(paths)), dtype=np.float32)
        for s, path in enumerate(paths):
            d = _read_two_column(path)
            missing = _fill_column(matrix, s, d, index)
            if missing:
                raise ValueError(
                    f"{path}: {missing} of {n} contigs have no coverage value. "
                    "Make sure abundances were computed against the same FASTA."
                )

    logger.info(f"Abundance: {n} contigs x {len(samples)} samples")
    return Abundance(matrix=matrix, samplenames=list(samples), refhash=refhash)


def _fill_column(matrix, s, name_to_value, index) -> int:
    seen = np.zeros(matrix.shape[0], dtype=bool)
    for name, value in name_to_value.items():
        i = index.get(name)
        if i is not None:
            matrix[i, s] = value
            seen[i] = True
    return int((~seen).sum())


def _fill_from_rows(matrix, rows, index, path) -> None:
    seen = np.zeros(matrix.shape[0], dtype=bool)
    for name, vec in rows.items():
        i = index.get(name)
        if i is not None:
            matrix[i] = vec
            seen[i] = True
    missing = int((~seen).sum())
    if missing:
        raise ValueError(
            f"{path}: {missing} contigs missing from the merged abundance TSV."
        )


# --------------------------------------------------------------------------- #
# BAM parsing (pycoverm)
# --------------------------------------------------------------------------- #
def from_bam(
    bam_paths: Sequence[Path],
    identifiers: Sequence[str],
    refhash: bytes,
    minid: float = 0.0,
    threads: int = 8,
) -> Abundance:
    """Build an Abundance from sorted BAM files via pycoverm (trimmed mean)."""
    try:
        import pycoverm
    except ImportError as e:
        raise ImportError(
            "pycoverm is required for --bamdir. Install with `pip install pycoverm`, "
            "or use strobealign --aemb TSVs via -a instead."
        ) from e

    bam_paths = [str(p) for p in bam_paths]
    # pycoverm bug workaround (issue #7): minid must be slightly > 0.
    minid = max(minid, 0.001)
    headers, coverage = pycoverm.get_coverages_from_bam(
        bam_paths,
        threads=min(threads, 16),
        min_identity=minid,
        trim_upper=0.1,
        trim_lower=0.1,
    )
    coverage = np.asarray(coverage, dtype=np.float32)  # (n_bam_contigs, n_bam)

    index = {name: i for i, name in enumerate(identifiers)}
    n = len(identifiers)
    matrix = np.zeros((n, len(bam_paths)), dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    for row, name in enumerate(headers):
        i = index.get(name)
        if i is not None:
            matrix[i] = coverage[row]
            seen[i] = True
    missing = int((~seen).sum())
    if missing:
        raise ValueError(
            f"{missing} of {n} contigs were absent from the BAM headers. "
            "BAMs must be mapped against the same FASTA used for composition."
        )

    samplenames = [Path(p).name for p in bam_paths]
    logger.info(f"Abundance: {n} contigs x {len(bam_paths)} samples (from BAM)")
    return Abundance(matrix=matrix, samplenames=samplenames, refhash=refhash)
