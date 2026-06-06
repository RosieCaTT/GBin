"""Streaming FASTA reader.

Parsing is deliberately separated from feature extraction: this module turns a
(possibly gzipped) FASTA file into chunks of *raw sequence bytes* plus lengths
and identifiers. Mapping bytes to nucleotide codes and counting k-mers happens
on the GPU in :mod:`gbin.features.composition`.

Chunks are bounded by total base count (not contig count) so that the GPU
working set per chunk is predictable regardless of contig-length distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils import logger


@dataclass
class FastaChunk:
    """A batch of contigs as concatenated raw ASCII bytes.

    Attributes
    ----------
    names: identifiers (header up to first whitespace), one per contig
    lengths: int32 array of sequence lengths, one per contig
    seq_bytes: uint8 array of all sequences concatenated, length == lengths.sum()
    """

    names: list[str]
    lengths: np.ndarray
    seq_bytes: np.ndarray

    @property
    def n_contigs(self) -> int:
        return len(self.names)


def _open_fastx(path: Path):
    """Return a pyfastx.Fastx iterator yielding (name, seq) string pairs."""
    try:
        import pyfastx
    except ImportError as e:  # pragma: no cover - exercised only without pyfastx
        raise ImportError(
            "pyfastx is required to read FASTA files. Install with `pip install pyfastx`."
        ) from e
    # Fastx is the index-free streaming reader; it transparently handles gzip.
    return pyfastx.Fastx(str(path))


def iter_fasta_chunks(
    path: Path,
    min_length: int = 2000,
    max_bases_per_chunk: int = 64_000_000,
) -> Iterator[FastaChunk]:
    """Yield :class:`FastaChunk` objects, each with up to ``max_bases_per_chunk`` bp.

    Contigs shorter than ``min_length`` are skipped. Identifiers are taken as the
    header up to the first whitespace and must be unique across the file (checked
    by the caller via the reference hash).
    """
    if min_length < 4:
        raise ValueError(f"min_length must be >= 4 for tetranucleotides, not {min_length}")

    names: list[str] = []
    lengths: list[int] = []
    buffers: list[np.ndarray] = []
    bases_in_chunk = 0

    for name, seq in _open_fastx(path):
        if len(seq) < min_length:
            continue
        identifier = name.split(None, 1)[0]
        arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
        names.append(identifier)
        lengths.append(len(arr))
        buffers.append(arr)
        bases_in_chunk += len(arr)

        if bases_in_chunk >= max_bases_per_chunk:
            yield FastaChunk(
                names=names,
                lengths=np.asarray(lengths, dtype=np.int32),
                seq_bytes=np.concatenate(buffers),
            )
            names, lengths, buffers, bases_in_chunk = [], [], [], 0

    if names:
        yield FastaChunk(
            names=names,
            lengths=np.asarray(lengths, dtype=np.int32),
            seq_bytes=np.concatenate(buffers),
        )


def count_contigs(path: Path, min_length: int = 2000) -> int:
    """Count contigs passing the length filter (a quick pre-pass if needed)."""
    n = 0
    for _name, seq in _open_fastx(path):
        if len(seq) >= min_length:
            n += 1
    return n


def iter_sequences(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (identifier, sequence) for every record (no length filter).

    Used by the bin writer to stream sequences back out without holding the whole
    assembly in memory.
    """
    for name, seq in _open_fastx(path):
        yield name.split(None, 1)[0], seq
