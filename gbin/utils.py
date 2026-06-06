"""Shared utilities: logging, reference hashing, chunking, timing.

These helpers carry no heavy dependencies (only numpy + loguru) so they can be
imported anywhere without pulling in torch.
"""

from __future__ import annotations

import hashlib
import sys
import time
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from loguru import logger

__all__ = [
    "logger",
    "setup_logging",
    "RefHasher",
    "hash_identifiers",
    "chunked",
    "timed",
    "human_bytes",
    "n50",
    "data_path",
]


def data_path(name: str) -> Path:
    """Return the filesystem path to a bundled data file (e.g. 'kernel.npz').

    Works both from an installed wheel and from a source checkout.
    """
    return Path(str(resources.files("gbin").joinpath("data", name)))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool = False, logfile: Path | None = None) -> None:
    """Configure loguru's default sink.

    Removes the default handler and installs one stderr sink (INFO, or DEBUG if
    ``verbose``), optionally mirroring everything to ``logfile``.
    """
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    fmt = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True, enqueue=True)
    if logfile is not None:
        logger.add(str(logfile), level="DEBUG", format=fmt, enqueue=True)


# --------------------------------------------------------------------------- #
# Reference hashing
# --------------------------------------------------------------------------- #
class RefHasher:
    """Order-sensitive hash of contig identifiers.

    A binning run threads several artifacts together (composition, abundance,
    markers, latent). They are only mutually consistent if they were all built
    from the same contigs in the same order. We record a hash of the identifier
    list with each artifact and verify it on load, mirroring VAMB's refhash
    safety check.
    """

    def __init__(self) -> None:
        self._h = hashlib.sha256()

    def add(self, name: str) -> "RefHasher":
        # Length-prefix so that ("ab", "c") and ("a", "bc") hash differently.
        encoded = name.encode("utf-8")
        self._h.update(len(encoded).to_bytes(8, "little"))
        self._h.update(encoded)
        return self

    def digest(self) -> bytes:
        return self._h.digest()

    @classmethod
    def hash(cls, names: Iterable[str]) -> bytes:
        h = cls()
        for name in names:
            h.add(name)
        return h.digest()

    @staticmethod
    def verify(observed: bytes, expected: bytes, what: str = "artifact") -> None:
        if observed != expected:
            raise ValueError(
                f"Reference hash mismatch for {what}: this {what} was built from "
                "a different set/order of contigs than the one it is being combined "
                "with. Re-run the upstream step on the same FASTA, or clear the cache."
            )


def hash_identifiers(names: Sequence[str]) -> bytes:
    """Convenience wrapper around :meth:`RefHasher.hash`."""
    return RefHasher.hash(names)


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #
def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    """Yield consecutive slices of ``seq`` of length ``size`` (last may be short)."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, not {size}")
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


@contextmanager
def timed(message: str) -> Iterator[None]:
    """Log ``message`` and the wall-clock time the enclosed block took."""
    logger.info(f"{message} ...")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{message} done in {elapsed:.1f}s")


def human_bytes(n: int | float) -> str:
    """Render a byte count as a human-readable string (e.g. '1.5 GiB')."""
    step = 1024.0
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


def n50(lengths: np.ndarray) -> int:
    """Compute the N50 of a set of sequence lengths."""
    if len(lengths) == 0:
        return 0
    ordered = np.sort(np.asarray(lengths))[::-1]
    half = ordered.sum() / 2.0
    cumulative = np.cumsum(ordered)
    idx = int(np.searchsorted(cumulative, half))
    return int(ordered[min(idx, len(ordered) - 1)])
