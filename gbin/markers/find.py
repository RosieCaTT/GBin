"""Predict single-copy marker genes per contig with pyrodigal + pyhmmer.

For each contig: pyrodigal (metagenomic mode) calls genes, the proteins are
searched against the bundled SCG HMM set with pyhmmer, and hits passing each
HMM's trusted cutoff are recorded (deduplicated per contig). Both tools are
pure-Python wheels (no external binaries), so this works the same on Windows and
Linux.

Single-process mode streams the FASTA in chunks; multi-process mode splits the
contigs into per-worker temp files (each worker loads the HMMs itself, avoiding
the cost/pitfalls of pickling HMM objects across processes).
"""

from __future__ import annotations

import itertools
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..io.fasta import iter_sequences
from ..utils import RefHasher, data_path, logger, timed
from .scg import Markers

# Markers that are biologically the same SCG under different accessions.
NORMALIZE_MARKER_TRANS_DICT = {
    "TIGR00388": "TIGR00389",
    "TIGR00471": "TIGR00472",
    "TIGR00408": "TIGR00409",
    "TIGR02386": "TIGR02387",
}

_CHUNK = 2048


def _as_str(x) -> str:
    """pyhmmer returns names as bytes (<=0.10) or str (>=0.12); normalize to str."""
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _load_hmms(hmm_path: Path):
    import pyhmmer

    with open(hmm_path, "rb") as f:
        return list(pyhmmer.plan7.HMMFile(f))


def _name_to_id(hmms):
    """Map HMM names to compact integer ids, merging equivalent markers."""
    name_to_id: dict[str, int] = {}
    for hmm in hmms:
        name = _as_str(hmm.name)
        if name in NORMALIZE_MARKER_TRANS_DICT:
            continue
        name_to_id[name] = len(name_to_id)
    for old, new in NORMALIZE_MARKER_TRANS_DICT.items():
        if new in name_to_id:
            name_to_id[old] = name_to_id[new]
    if len(set(name_to_id.values())) > 256:
        raise ValueError("At most 256 distinct markers are supported (uint8 ids)")
    id_to_names: dict[int, list[str]] = defaultdict(list)
    for name, i in name_to_id.items():
        id_to_names[i].append(name)
    marker_names = [id_to_names[i] for i in range(len(id_to_names))]
    return name_to_id, marker_names


def _process_records(records: list[tuple[str, str]], hmms, name_to_id):
    """Run gene calling + HMM search on a batch of (name, seq). Returns dict."""
    import pyhmmer
    import pyrodigal

    finder = pyrodigal.GeneFinder(meta=True)
    alphabet = pyhmmer.easel.Alphabet.amino()
    digitized = []
    for name, seq in records:
        for gene in finder.find_genes(seq):
            ts = pyhmmer.easel.TextSequence(
                name=name.encode(), sequence=gene.translate()
            )
            digitized.append(ts.digitize(alphabet))

    found: dict[str, set[int]] = defaultdict(set)
    if not digitized:
        return found
    for hmm, top_hits in zip(hmms, pyhmmer.hmmsearch(hmms, digitized, cpus=1)):
        name = _as_str(hmm.name)
        marker_id = name_to_id.get(name)
        if marker_id is None:
            continue
        cutoff = hmm.cutoffs.trusted1  # remove HMMER false positives
        if cutoff is None:
            continue
        for hit in top_hits:
            if hit.score >= cutoff:
                found[_as_str(hit.name)].add(marker_id)
    return found


def _worker(args):
    """Multiprocessing worker: process one temp FASTA file."""
    file_path, hmm_path = args
    hmms = _load_hmms(hmm_path)
    name_to_id, _ = _name_to_id(hmms)
    result: list[tuple[str, np.ndarray]] = []
    batch: list[tuple[str, str]] = []
    for name, seq in iter_sequences(Path(file_path)):
        batch.append((name, seq))
        if len(batch) == _CHUNK:
            for n, ids in _process_records(batch, hmms, name_to_id).items():
                result.append((n, np.array(sorted(ids), dtype=np.uint8)))
            batch.clear()
    for n, ids in _process_records(batch, hmms, name_to_id).items():
        result.append((n, np.array(sorted(ids), dtype=np.uint8)))
    return result


def _split_fasta(fasta: Path, keep: set[str], tmpdir: Path, n: int) -> list[Path]:
    paths = [tmpdir / f"part{i}.fna" for i in range(n)]
    handles = [open(p, "w") for p in paths]
    try:
        for i, (name, seq) in enumerate(
            (ns for ns in iter_sequences(fasta) if ns[0] in keep)
        ):
            h = handles[i % n]
            h.write(f">{name}\n{seq}\n")
    finally:
        for h in handles:
            h.close()
    return paths


def compute_markers(
    fasta: Path,
    identifiers: Iterable[str],
    refhash: bytes,
    hmm_path: Optional[Path] = None,
    n_processes: int = 1,
) -> Markers:
    """Compute per-contig SCGs for the contigs in ``identifiers``."""
    hmm_path = Path(hmm_path) if hmm_path else data_path("marker.hmm")
    identifiers = [str(x) for x in identifiers]
    keep = set(identifiers)
    index = {name: i for i, name in enumerate(identifiers)}
    hmms = _load_hmms(hmm_path)
    name_to_id, marker_names = _name_to_id(hmms)
    marker_list: list[Optional[np.ndarray]] = [None] * len(identifiers)

    n_processes = max(1, min(n_processes, 64))
    with timed(f"Finding single-copy markers ({n_processes} proc)"):
        if n_processes == 1:
            batch: list[tuple[str, str]] = []

            def flush(b):
                for name, ids in _process_records(b, hmms, name_to_id).items():
                    marker_list[index[name]] = np.array(sorted(ids), dtype=np.uint8)

            for name, seq in iter_sequences(fasta):
                if name not in keep:
                    continue
                batch.append((name, seq))
                if len(batch) == _CHUNK:
                    flush(batch)
                    batch.clear()
            flush(batch)
        else:
            from multiprocessing.pool import Pool

            tmpdir = Path(tempfile.mkdtemp(prefix="gbin_markers_"))
            try:
                paths = _split_fasta(fasta, keep, tmpdir, n_processes)
                with Pool(n_processes) as pool:
                    for sub in pool.imap_unordered(
                        _worker, [(str(p), str(hmm_path)) for p in paths]
                    ):
                        for name, ids in sub:
                            marker_list[index[name]] = ids
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    n_with = sum(m is not None for m in marker_list)
    logger.info(
        f"Markers: {len(marker_names)} SCGs; {n_with}/{len(identifiers)} contigs "
        "carry at least one"
    )
    return Markers(marker_list, marker_names, refhash)
