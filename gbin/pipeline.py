"""End-to-end orchestration and per-stage entry points.

Each stage caches its artifact under ``cfg.cache_dir`` with a small JSON
fingerprint, so re-running with changed model/clustering settings reuses the
expensive feature computation. Stages not yet implemented raise
``NotImplementedError`` with a pointer to the milestone that adds them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .config import GBinConfig, resolve_device
from .features.composition import Composition, compute_composition
from .io.abundance import Abundance, from_bam, from_tsv
from .utils import logger


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _fasta_fingerprint(fasta: Path, **extra) -> dict:
    st = os.stat(fasta)
    fp = {
        "fasta": os.path.abspath(fasta),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }
    fp.update(extra)
    return fp


def _cache_is_valid(meta_path: Path, fingerprint: dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            return json.load(f) == fingerprint
    except (OSError, ValueError):
        return False


def _write_meta(meta_path: Path, fingerprint: dict) -> None:
    with open(meta_path, "w") as f:
        json.dump(fingerprint, f, sort_keys=True)


# --------------------------------------------------------------------------- #
# Composition stage
# --------------------------------------------------------------------------- #
def load_or_compute_composition(cfg: GBinConfig, device) -> Composition:
    """Return cached composition if the fingerprint matches, else recompute."""
    cache = cfg.cache_dir / "composition.npz"
    meta = cfg.cache_dir / "composition.meta.json"
    fp = _fasta_fingerprint(cfg.fasta, min_length=cfg.min_contig_len, kind="composition")

    if _cache_is_valid(meta, fp) and cache.exists():
        logger.info(f"Loading cached composition from {cache}")
        return Composition.load(cache)

    comp = compute_composition(cfg.fasta, device, min_length=cfg.min_contig_len)
    comp.save(cache)
    _write_meta(meta, fp)
    logger.info(f"Cached composition to {cache}")
    return comp


def run_composition(cfg: GBinConfig) -> None:
    if cfg.fasta is None:
        raise ValueError("--fasta is required")
    device = resolve_device(cfg.device)
    comp = load_or_compute_composition(cfg, device)
    logger.info(
        f"Done: {comp.n_contigs} contigs, "
        f"total {int(comp.lengths.sum()):,} bp, "
        f"min length {int(comp.lengths.min())} bp"
    )


# --------------------------------------------------------------------------- #
# Abundance stage
# --------------------------------------------------------------------------- #
def map_reads_to_tsvs(cfg: GBinConfig, comp: Composition) -> list[Path]:
    """Map each sample's reads to the contigs, returning per-sample coverage TSVs.

    Per-sample results are cached (keyed on reads + mapper + contig refhash) so
    re-runs skip the expensive mapping for unchanged samples.
    """
    from .io.mapping import collect_samples, map_one_sample

    samples = collect_samples(cfg.reads, cfg.reads_tsv)
    out_dir = cfg.cache_dir / "abundance_tsv"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsvs: list[Path] = []
    for s in samples:
        tsv = out_dir / f"{s.name}.tsv"
        meta = out_dir / f"{s.name}.meta.json"
        fp = {
            "mapper": cfg.mapper, "preset": cfg.mapper_preset,
            "refhash": comp.refhash.hex(),
            "reads": [
                {"path": os.path.abspath(r), "size": os.stat(r).st_size,
                 "mtime_ns": os.stat(r).st_mtime_ns}
                for r in s.reads()
            ],
        }
        if _cache_is_valid(meta, fp) and tsv.exists():
            logger.info(f"Reusing cached mapping for sample '{s.name}'")
        else:
            map_one_sample(cfg.fasta, s, tsv, mapper=cfg.mapper,
                           threads=cfg.num_threads, preset=cfg.mapper_preset)
            _write_meta(meta, fp)
        tsvs.append(tsv)
    return tsvs


def _abundance_inputs(cfg: GBinConfig, comp: Composition) -> tuple[str, list[Path]]:
    """Resolve the abundance source to (kind, paths), mapping reads if needed."""
    if cfg.reads or cfg.reads_tsv:
        return "tsv", map_reads_to_tsvs(cfg, comp)
    if cfg.bamdir is not None:
        paths = sorted(Path(cfg.bamdir).glob("*.bam"))
        if not paths:
            raise ValueError(f"No .bam files found in {cfg.bamdir}")
        return "bam", paths
    if cfg.abundance_tsv:
        return "tsv", [Path(p) for p in cfg.abundance_tsv]
    raise ValueError(
        "Provide abundance via --reads/--reads-tsv (map reads), --bamdir, or "
        "-a/--abundance-tsv"
    )


def load_or_compute_abundance(
    cfg: GBinConfig, comp: Composition
) -> Abundance:
    """Return cached abundance if fingerprint matches, else recompute & align."""
    kind, paths = _abundance_inputs(cfg, comp)
    cache = cfg.cache_dir / "abundance.npz"
    meta = cfg.cache_dir / "abundance.meta.json"
    fp = {
        "kind": kind,
        "refhash": comp.refhash.hex(),
        "sources": [
            {"path": os.path.abspath(p), "size": os.stat(p).st_size,
             "mtime_ns": os.stat(p).st_mtime_ns}
            for p in paths
        ],
    }
    if _cache_is_valid(meta, fp) and cache.exists():
        logger.info(f"Loading cached abundance from {cache}")
        return Abundance.load(cache, refhash=comp.refhash)

    ids = [str(x) for x in comp.identifiers]
    if kind == "bam":
        ab = from_bam(paths, ids, comp.refhash, threads=cfg.num_threads)
    else:
        ab = from_tsv(paths, ids, comp.refhash)
    ab.save(cache)
    _write_meta(meta, fp)
    logger.info(f"Cached abundance to {cache}")
    return ab


def run_abundance(cfg: GBinConfig) -> None:
    if cfg.fasta is None:
        raise ValueError("--fasta is required")
    device = resolve_device(cfg.device)
    comp = load_or_compute_composition(cfg, device)
    ab = load_or_compute_abundance(cfg, comp)
    logger.info(f"Done: abundance {ab.nseqs} contigs x {ab.nsamples} samples")


# --------------------------------------------------------------------------- #
# Markers stage
# --------------------------------------------------------------------------- #
def load_or_compute_markers(cfg: GBinConfig, comp: Composition):
    """Return cached SCG markers if fingerprint matches, else recompute."""
    from .markers.find import compute_markers
    from .markers.scg import Markers

    cache = cfg.cache_dir / "markers.json"
    meta = cfg.cache_dir / "markers.meta.json"
    fp = _fasta_fingerprint(
        cfg.fasta, min_length=cfg.min_contig_len, kind="markers",
        refhash=comp.refhash.hex(),
    )
    if _cache_is_valid(meta, fp) and cache.exists():
        logger.info(f"Loading cached markers from {cache}")
        return Markers.load(cache, refhash=comp.refhash)

    markers = compute_markers(
        cfg.fasta, comp.identifiers, comp.refhash, n_processes=cfg.num_threads
    )
    markers.save(cache)
    _write_meta(meta, fp)
    logger.info(f"Cached markers to {cache}")
    return markers


def run_markers(cfg: GBinConfig) -> None:
    if cfg.fasta is None:
        raise ValueError("--fasta is required")
    device = resolve_device(cfg.device)
    comp = load_or_compute_composition(cfg, device)
    markers = load_or_compute_markers(cfg, comp)
    comp_, cont_ = markers.score_bin(range(comp.n_contigs))
    logger.info(
        f"Done: {sum(m is not None for m in markers.markers)} contigs carry SCGs "
        f"(whole-assembly completeness={comp_:.1%}, contamination={cont_:.1%})"
    )


# --------------------------------------------------------------------------- #
# Embedding stage
# --------------------------------------------------------------------------- #
def train_embedding(cfg, comp, ab, device, markers=None):
    """Normalize features, train the (optionally marker-guided) VAE, cache latent."""
    from .features.normalize import normalize_features
    from .model.train import train_vae

    cfg.resolve_model_defaults(ab.nsamples)
    feats = normalize_features(ab.matrix, comp.tnf, comp.lengths, device)
    cannot_link = None
    if markers is not None and cfg.marker_loss_weight > 0:
        from .markers.constraints import cannot_link_pairs

        cannot_link = cannot_link_pairs(markers, seed=cfg.seed)
    _model, latent = train_vae(feats, cfg, device, cannot_link=cannot_link)
    np.save(cfg.cache_dir / "latent.npy", latent)
    _model.save(cfg.cache_dir / "vae.pt")
    logger.info(f"Cached latent {latent.shape} and model to {cfg.cache_dir}")
    return latent


# --------------------------------------------------------------------------- #
# Clustering + output
# --------------------------------------------------------------------------- #
def _internal_scg_quality(labels, markers) -> Optional[dict]:
    """{label: (completeness, contamination)} from the internal SCG scorer, or None."""
    from collections import defaultdict

    if markers is None:
        return None
    members: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if int(lab) < 0:
            continue
        members[int(lab)].append(i)
    return {lab: markers.score_bin(idxs) for lab, idxs in members.items()}


def cluster_and_write(cfg, comp, latent, device, markers=None) -> dict:
    from .cluster.clusterer import cluster_latent
    from .io.write import write_bins

    labels = cluster_latent(latent, comp.lengths, cfg, device)

    if markers is not None and cfg.refine:
        from .cluster.refine import refine_bins

        labels = refine_bins(labels, latent, comp.lengths, markers, device, seed=cfg.seed)

    quality = _internal_scg_quality(labels, markers)
    if quality is not None:
        n_hq = sum(1 for c, k in quality.values() if c >= 0.9 and k <= 0.05)
        logger.info(f"High-quality bins (>=90% complete, <=5% contam): {n_hq}")

    summary = write_bins(
        cfg.outdir, comp.identifiers, comp.lengths, labels, cfg.fasta,
        min_bin_size=cfg.min_bin_size, quality=quality,
    )
    if summary["n_bins"] == 0:
        return summary

    if cfg.checkm2_refine and markers is not None:
        return _checkm2_refine_and_rewrite(
            cfg, comp, latent, labels, markers, device, summary["kept"]
        )
    if cfg.checkm2_refine and markers is None:
        logger.warning(
            "--checkm2-refine needs single-copy markers to seed splits but markers "
            "are off (--no-markers); doing a plain CheckM2 QC pass instead."
        )
    if cfg.checkm2 or cfg.checkm2_refine:
        from .qc.checkm2 import run_and_merge

        # Binning already succeeded and bins are written; never let a CheckM2
        # setup issue (e.g. a missing/misplaced database) fail the whole run.
        try:
            run_and_merge(cfg.outdir, threads=cfg.num_threads, db_path=cfg.checkm2_db,
                          checkm2_bin=cfg.checkm2_bin)
        except Exception as e:
            logger.warning(
                f"CheckM2 step failed ({type(e).__name__}: {e}). The bins are still "
                f"written to {cfg.outdir / 'bins'}. Add CheckM2 later with:\n"
                f"  gbin qc -o {cfg.outdir} --checkm2-bin {cfg.checkm2_bin} "
                "--checkm2-db /path/to/uniref100.KO.1.dmnd"
            )
    return summary


def _checkm2_refine_and_rewrite(cfg, comp, latent, labels, markers, device, kept) -> dict:
    """CheckM2-guided refine -> re-write bins -> merge CheckM2 quality. Returns summary.

    ``kept`` is the {label: bin_name} map for the bins currently on disk. On any
    failure the already-written bins are left untouched.
    """
    from .io.write import write_bins
    from .qc.checkm2 import merge_into_bins_info, write_quality_report
    from .qc.checkm2_refine import checkm2_guided_refine

    try:
        new_labels, label_rows = checkm2_guided_refine(
            cfg, comp, latent, labels, markers, device,
            initial_bins_dir=cfg.outdir / "bins", initial_kept=kept,
        )
    except Exception as e:
        logger.warning(
            f"CheckM2-guided refine failed ({type(e).__name__}: {e}). The initial "
            f"bins remain in {cfg.outdir / 'bins'}; rerun later with "
            f"`gbin qc -o {cfg.outdir} --refine`."
        )
        return {"n_bins": len(kept)}

    # Clear the first-pass FASTAs so a smaller refined set leaves no stale bins,
    # then re-write with the refined labels + internal SCG quality columns.
    bins_dir = cfg.outdir / "bins"
    for old in list(bins_dir.glob("*.fna")) + list(bins_dir.glob("*.fna.gz")):
        old.unlink()
    summary = write_bins(
        cfg.outdir, comp.identifiers, comp.lengths, new_labels, cfg.fasta,
        min_bin_size=cfg.min_bin_size, quality=_internal_scg_quality(new_labels, markers),
    )
    kept_final = summary["kept"]  # {label: bin_name}

    # Merge CheckM2 numbers onto the freshly written bin names, and rewrite the
    # native CheckM2 report so its Name column matches the FINAL bins (not the
    # pre-split round-0 snapshot).
    name_cc: dict[str, tuple[float, float]] = {}
    final_rows = []
    for lab, row in label_rows.items():
        name = kept_final.get(lab)
        if name is None:
            continue
        name_cc[name] = (float(row["Completeness"]), float(row["Contamination"]))
        final_rows.append({**row, "Name": name})
    n_hq = merge_into_bins_info(cfg.outdir / "bins_info.tsv", name_cc)
    write_quality_report(
        cfg.outdir / "checkm2" / "quality_report.tsv",
        sorted(final_rows, key=lambda r: r["Name"]),
    )
    logger.info(
        f"CheckM2-guided refine: {summary['n_bins']} bins, {n_hq} high-quality "
        "(>=90% complete, <=5% contam)."
    )
    return summary


def run_qc(cfg: GBinConfig) -> None:
    """CheckM2 on an existing output's bins/; merge results (and optionally refine)."""
    bins_dir = cfg.outdir / "bins"
    if not bins_dir.exists():
        raise FileNotFoundError(
            f"No bins found at {bins_dir}. Run `gbin bin` first (same --outdir)."
        )

    if cfg.checkm2_refine:
        _run_qc_refine(cfg)
        return

    from .qc.checkm2 import run_and_merge

    run_and_merge(cfg.outdir, threads=cfg.num_threads, db_path=cfg.checkm2_db,
                  checkm2_bin=cfg.checkm2_bin)
    logger.info(f"CheckM2 QC merged into {cfg.outdir / 'bins_info.tsv'}")


def _load_labels_from_output(cfg, comp) -> tuple[np.ndarray, dict[int, str]]:
    """Rebuild (labels aligned to comp order, {label: bin_name}) from contig_bins.tsv."""
    name_to_idx = {str(c): i for i, c in enumerate(comp.identifiers)}
    labels = np.full(comp.n_contigs, -1, dtype=np.int64)
    bin_to_label: dict[str, int] = {}
    with open(cfg.outdir / "contig_bins.tsv") as f:
        next(f, None)  # header: contig\tbin
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            contig, binname = parts
            idx = name_to_idx.get(contig)
            if idx is None:
                continue
            lab = bin_to_label.setdefault(binname, len(bin_to_label))
            labels[idx] = lab
    return labels, {lab: name for name, lab in bin_to_label.items()}


def _run_qc_refine(cfg: GBinConfig) -> None:
    """CheckM2-guided refinement on an existing output (reload caches, refine, rewrite)."""
    if cfg.fasta is None:
        raise ValueError(
            "`gbin qc --refine` needs -i/--fasta (the original contigs) to re-write "
            "bin sequences. Pass the same FASTA you gave `gbin bin`."
        )
    device = resolve_device(cfg.device)
    comp = Composition.load(cfg.cache_dir / "composition.npz")
    latent_path = cfg.cache_dir / "latent.npy"
    markers_path = cfg.cache_dir / "markers.json"
    missing = [str(p) for p in (latent_path, markers_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "CheckM2-guided refine needs the cached latent and markers from the "
            f"original run, but these are missing: {', '.join(missing)}. Rerun "
            "`gbin bin` (same --outdir), or use `gbin qc` without --refine."
        )
    latent = np.load(latent_path)
    from .markers.scg import Markers

    markers = Markers.load(markers_path, refhash=comp.refhash)
    labels, kept = _load_labels_from_output(cfg, comp)
    if not kept:
        raise ValueError(f"No bins parsed from {cfg.outdir / 'contig_bins.tsv'}")
    _checkm2_refine_and_rewrite(cfg, comp, latent, labels, markers, device, kept)
    logger.info(f"CheckM2-guided refine merged into {cfg.outdir / 'bins_info.tsv'}")


def run_cluster(cfg: GBinConfig) -> None:
    if cfg.fasta is None:
        raise ValueError("--fasta is required (to write bin sequences)")
    device = resolve_device(cfg.device)
    comp = Composition.load(cfg.cache_dir / "composition.npz")
    latent_path = cfg.cache_dir / "latent.npy"
    if not latent_path.exists():
        raise FileNotFoundError(f"No cached latent at {latent_path}. Run `gbin bin` first.")
    latent = np.load(latent_path)
    markers = None
    if cfg.use_markers and (cfg.cache_dir / "markers.json").exists():
        from .markers.scg import Markers

        markers = Markers.load(cfg.cache_dir / "markers.json", refhash=comp.refhash)
    summary = cluster_and_write(cfg, comp, latent, device, markers)
    logger.info(f"Done: {summary['n_bins']} bins")


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def run_bin(cfg: GBinConfig) -> None:
    if cfg.fasta is None:
        raise ValueError("--fasta is required")
    device = resolve_device(cfg.device)
    comp = load_or_compute_composition(cfg, device)
    ab = load_or_compute_abundance(cfg, comp)
    markers = load_or_compute_markers(cfg, comp) if cfg.use_markers else None
    latent = train_embedding(cfg, comp, ab, device, markers)
    summary = cluster_and_write(cfg, comp, latent, device, markers)
    logger.info(f"Binning complete: {summary['n_bins']} bins in {cfg.outdir / 'bins'}")
