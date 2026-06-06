"""Write bins to FASTA and emit per-bin / per-contig reports.

Sequences are streamed back from the input FASTA (not held in memory). Bins
whose total length is below ``min_bin_size`` are dropped. An optional quality map
(completeness/contamination from single-copy markers) is included in the report
when available.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .fasta import iter_sequences
from ..utils import logger, n50


def _bin_name(i: int) -> str:
    return f"bin{i:05d}"


def write_bins(
    outdir: Path,
    identifiers: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    fasta: Path,
    min_bin_size: int = 200_000,
    quality: Optional[dict[int, tuple[float, float]]] = None,
    compress: bool = False,
) -> dict:
    """Write kept bins and reports. Returns a summary dict.

    quality: optional {raw_label: (completeness, contamination)} in [0, 1].
    """
    outdir = Path(outdir)
    bins_dir = outdir / "bins"
    bins_dir.mkdir(parents=True, exist_ok=True)

    members: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if int(lab) < 0:
            continue  # negative label = unbinned (e.g. on a re-write after refine)
        members[int(lab)].append(i)

    # Keep only large-enough bins; assign clean sequential names.
    kept: dict[int, str] = {}
    ext = ".fna.gz" if compress else ".fna"
    bin_index = 0
    contig_to_file: dict[str, Path] = {}
    rows = []
    for lab in sorted(members):
        idxs = members[lab]
        size = int(lengths[idxs].sum())
        if size < min_bin_size:
            continue
        name = _bin_name(bin_index)
        path = bins_dir / f"{name}{ext}"
        kept[lab] = name
        bin_index += 1
        bin_n50 = n50(lengths[idxs])
        comp, cont = (quality or {}).get(lab, (None, None))
        rows.append({
            "bin": name, "n_contigs": len(idxs), "size_bp": size, "n50": bin_n50,
            "completeness": comp, "contamination": cont,
        })
        for i in idxs:
            contig_to_file[str(identifiers[i])] = path

    # Stream sequences into per-bin files (handles kept open; bin counts are
    # typically well under the OS limit after the size filter).
    handles: dict[Path, object] = {}
    opener = gzip.open if compress else open
    try:
        for name, seq in iter_sequences(fasta):
            path = contig_to_file.get(name)
            if path is None:
                continue
            fh = handles.get(path)
            if fh is None:
                fh = opener(path, "wt")
                handles[path] = fh
            fh.write(f">{name}\n{seq}\n")
    finally:
        for fh in handles.values():
            fh.close()

    _write_reports(outdir, identifiers, labels, kept, rows)
    summary = {
        "n_bins": len(kept),
        "n_contigs_binned": sum(r["n_contigs"] for r in rows),
        "total_bp": sum(r["size_bp"] for r in rows),
        "kept": dict(kept),  # {raw_label: bin_name} — lets callers map names back to labels
    }
    logger.info(
        f"Wrote {summary['n_bins']} bins "
        f"({summary['n_contigs_binned']} contigs, {summary['total_bp']:,} bp) to {bins_dir}"
    )
    return summary


def _write_reports(outdir, identifiers, labels, kept, rows) -> None:
    # Per-bin info.
    info = outdir / "bins_info.tsv"
    cols = ["bin", "n_contigs", "size_bp", "n50", "completeness", "contamination"]
    with open(info, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: r["bin"]):
            vals = []
            for c in cols:
                v = r[c]
                if v is None:
                    vals.append("NA")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")

    # Per-contig assignment (only contigs in kept bins).
    mapping = outdir / "contig_bins.tsv"
    with open(mapping, "w") as f:
        f.write("contig\tbin\n")
        for i, lab in enumerate(labels):
            name = kept.get(int(lab))
            if name is not None:
                f.write(f"{identifiers[i]}\t{name}\n")
