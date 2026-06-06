"""Command-line interface for gbin.

Subcommands:
  bin           end-to-end binning (composition -> abundance -> [markers] ->
                VAE -> GPU clustering -> [refine] -> bins)
  composition   compute & cache TNF composition features only
  abundance     compute & cache abundance features only
  markers       compute & cache single-copy marker genes only
  cluster       cluster a previously trained latent representation

Feature modules are imported lazily inside each handler so that ``gbin --help``
works without torch / a GPU present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .config import GBinConfig
from .utils import setup_logging, logger


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("common")
    g.add_argument("-o", "--outdir", type=Path, required=True, help="output directory")
    g.add_argument(
        "--device",
        default="auto",
        help="auto | cuda | cuda:0 | cpu (default: auto)",
    )
    g.add_argument(
        "-t",
        "--threads",
        type=int,
        default=None,
        help="CPU threads for I/O / markers (default: all cores)",
    )
    g.add_argument("--cache-dir", type=Path, default=None, help="override cache dir")
    g.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    g.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def _add_inputs(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("inputs")
    g.add_argument("-i", "--fasta", type=Path, required=True, help="contigs FASTA(.gz)")
    g.add_argument(
        "--min-contig-len",
        type=int,
        default=2000,
        help="ignore contigs shorter than this (default: 2000)",
    )


def _add_abundance(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("abundance (choose one source)")
    g.add_argument(
        "--reads",
        action="append",
        metavar="R1[,R2]",
        help="reads for one sample (comma-separates paired files); repeat per "
        "sample. gbin maps them to the contigs to compute coverage.",
    )
    g.add_argument(
        "--reads-tsv",
        type=Path,
        help="reads manifest, one sample per line: name<TAB>R1[<TAB>R2]",
    )
    g.add_argument(
        "--mapper",
        choices=["strobealign", "minimap2"],
        default="strobealign",
        help="read mapper (default: strobealign for short reads; minimap2 for long)",
    )
    g.add_argument(
        "--mapper-preset",
        default="sr",
        help="minimap2 preset: sr | map-ont | map-pb (default: sr)",
    )
    g.add_argument("--bamdir", type=Path, help="directory of sorted BAM files")
    g.add_argument(
        "-a",
        "--abundance-tsv",
        type=Path,
        nargs="+",
        help="precomputed strobealign --aemb TSV file(s)",
    )


def _add_model(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model / training")
    g.add_argument("--latent", type=int, default=32, help="latent dim (default: 32)")
    g.add_argument("--epochs", type=int, default=300, help="epochs (default: 300)")
    g.add_argument("--batch-size", type=int, default=256, help="start batch size")
    g.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="training precision (default: bf16; Blackwell-friendly)",
    )


def _add_cluster(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("clustering")
    g.add_argument(
        "--cluster",
        dest="cluster_method",
        choices=["auto", "leiden", "medoid"],
        default="auto",
        help="auto/medoid: pure-torch iterative medoid (robust default, works on "
        "any CUDA GPU); leiden: cuGraph Leiden (opt-in, needs a healthy RAPIDS "
        "stack; falls back to medoid on failure)",
    )
    g.add_argument("--knn-k", type=int, default=15, help="kNN neighbours (default: 15)")
    g.add_argument(
        "--min-bin-size",
        type=int,
        default=200_000,
        help="drop bins smaller than this many bp (default: 200000)",
    )


def _add_markers(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("markers / refinement")
    g.add_argument(
        "--no-markers",
        action="store_true",
        help="skip single-copy marker genes (pure self-supervised VAE)",
    )
    g.add_argument(
        "--marker-loss-weight",
        type=float,
        default=0.0,
        help="weight of the SCG cannot-link loss term (0 = off)",
    )
    g.add_argument(
        "--no-refine",
        action="store_true",
        help="skip SCG-based decontamination of output bins",
    )
    g.add_argument(
        "--checkm2",
        action="store_true",
        help="run CheckM2 on the output bins for accurate completeness/contamination",
    )
    g.add_argument(
        "--checkm2-db",
        type=Path,
        default=None,
        help="CheckM2 database path (default: CheckM2's configured/default location)",
    )
    g.add_argument(
        "--checkm2-bin",
        default="checkm2",
        help="path to the checkm2 executable (point this at a separate checkm2 "
        "conda env, e.g. ~/miniforge3/envs/checkm2/bin/checkm2)",
    )
    g.add_argument(
        "--checkm2-refine",
        action="store_true",
        help="use CheckM2 to re-refine bins: split contaminated bins (on the GPU) "
        "and keep a split only if CheckM2 confirms it improves quality (implies "
        "--checkm2)",
    )


def _add_checkm2_refine_tuning(p: argparse.ArgumentParser) -> None:
    """Shared CheckM2-guided-refine thresholds (used by both `bin` and `qc`)."""
    g = p.add_argument_group("checkm2-guided refine tuning")
    g.add_argument(
        "--checkm2-refine-iters",
        type=int,
        default=1,
        help="guided-refine rounds; each round is one extra CheckM2 run (default: 1)",
    )
    g.add_argument(
        "--checkm2-refine-min-contamination",
        type=float,
        default=10.0,
        help="only split bins with at least this CheckM2 contamination %% (default: 10)",
    )
    g.add_argument(
        "--checkm2-refine-min-completeness",
        type=float,
        default=50.0,
        help="only split bins with at least this CheckM2 completeness %% (default: 50)",
    )
    g.add_argument(
        "--checkm2-refine-score-weight",
        type=float,
        default=5.0,
        help="DAS-Tool score weight: completeness - w*contamination (default: 5)",
    )
    g.add_argument(
        "--checkm2-refine-min-gain",
        type=float,
        default=10.0,
        help="min DAS-score gain to accept a split; rejects cosmetic 'peel one "
        "contig' splits that don't reduce contamination (default: 10)",
    )
    g.add_argument(
        "--checkm2-refine-max-k",
        type=int,
        default=3,
        help="max number of sub-bins to try when splitting a bin (default: 3)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gbin",
        description="GPU-accelerated metagenomic binning (hybrid VAE + markers).",
    )
    parser.add_argument("--version", action="version", version=f"gbin {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # bin: the whole pipeline
    p_bin = sub.add_parser("bin", help="end-to-end binning")
    _add_inputs(p_bin)
    _add_abundance(p_bin)
    _add_model(p_bin)
    _add_cluster(p_bin)
    _add_markers(p_bin)
    _add_checkm2_refine_tuning(p_bin)
    _add_common(p_bin)
    p_bin.set_defaults(func=_cmd_bin)

    # composition
    p_comp = sub.add_parser("composition", help="compute & cache TNF only")
    _add_inputs(p_comp)
    _add_common(p_comp)
    p_comp.set_defaults(func=_cmd_composition)

    # abundance
    p_ab = sub.add_parser("abundance", help="compute & cache abundance only")
    _add_inputs(p_ab)
    _add_abundance(p_ab)
    _add_common(p_ab)
    p_ab.set_defaults(func=_cmd_abundance)

    # markers
    p_mk = sub.add_parser("markers", help="compute & cache marker genes only")
    _add_inputs(p_mk)
    _add_common(p_mk)
    p_mk.set_defaults(func=_cmd_markers)

    # cluster
    p_cl = sub.add_parser("cluster", help="cluster a cached latent representation")
    _add_inputs(p_cl)  # --fasta needed to write bin sequences
    _add_cluster(p_cl)
    _add_common(p_cl)
    p_cl.set_defaults(func=_cmd_cluster)

    # qc: run CheckM2 on an existing gbin output directory
    p_qc = sub.add_parser("qc", help="run CheckM2 QC on an existing output's bins/")
    p_qc.add_argument(
        "--checkm2-db", type=Path, default=None, help="CheckM2 database path"
    )
    p_qc.add_argument(
        "--checkm2-bin", default="checkm2",
        help="path to the checkm2 executable (e.g. a separate checkm2 conda env)",
    )
    p_qc.add_argument(
        "--refine",
        dest="checkm2_refine",
        action="store_true",
        help="CheckM2-guided refinement: split contaminated bins (GPU) and keep a "
        "split only if CheckM2 confirms it improves quality (reuses the cached "
        "latent/markers from the original run)",
    )
    p_qc.add_argument(
        "-i", "--fasta", type=Path, default=None,
        help="contigs FASTA, required with --refine (to re-write bin sequences); "
        "use the same file passed to `gbin bin`",
    )
    _add_checkm2_refine_tuning(p_qc)
    _add_common(p_qc)
    p_qc.set_defaults(func=_cmd_qc)

    return parser


def _config_from_args(args: argparse.Namespace) -> GBinConfig:
    cfg = GBinConfig(outdir=args.outdir)
    cfg.fasta = getattr(args, "fasta", None)
    cfg.bamdir = getattr(args, "bamdir", None)
    cfg.abundance_tsv = list(getattr(args, "abundance_tsv", None) or [])
    cfg.reads = list(getattr(args, "reads", None) or [])
    cfg.reads_tsv = getattr(args, "reads_tsv", None)
    cfg.mapper = getattr(args, "mapper", cfg.mapper)
    cfg.mapper_preset = getattr(args, "mapper_preset", cfg.mapper_preset)
    cfg.cache_dir = getattr(args, "cache_dir", None)
    cfg.min_contig_len = getattr(args, "min_contig_len", cfg.min_contig_len)
    cfg.min_bin_size = getattr(args, "min_bin_size", cfg.min_bin_size)
    cfg.latent = getattr(args, "latent", cfg.latent)
    cfg.epochs = getattr(args, "epochs", cfg.epochs)
    cfg.batch_size = getattr(args, "batch_size", cfg.batch_size)
    cfg.precision = getattr(args, "precision", cfg.precision)
    cfg.cluster_method = getattr(args, "cluster_method", cfg.cluster_method)
    cfg.knn_k = getattr(args, "knn_k", cfg.knn_k)
    cfg.use_markers = not getattr(args, "no_markers", False)
    cfg.marker_loss_weight = getattr(args, "marker_loss_weight", cfg.marker_loss_weight)
    cfg.refine = not getattr(args, "no_refine", False)
    cfg.checkm2 = getattr(args, "checkm2", False)
    cfg.checkm2_db = getattr(args, "checkm2_db", None)
    cfg.checkm2_bin = getattr(args, "checkm2_bin", cfg.checkm2_bin)
    cfg.checkm2_refine = getattr(args, "checkm2_refine", False)
    cfg.checkm2_refine_iters = getattr(args, "checkm2_refine_iters", cfg.checkm2_refine_iters)
    cfg.checkm2_refine_min_contamination = getattr(
        args, "checkm2_refine_min_contamination", cfg.checkm2_refine_min_contamination
    )
    cfg.checkm2_refine_min_completeness = getattr(
        args, "checkm2_refine_min_completeness", cfg.checkm2_refine_min_completeness
    )
    cfg.checkm2_refine_score_weight = getattr(
        args, "checkm2_refine_score_weight", cfg.checkm2_refine_score_weight
    )
    cfg.checkm2_refine_min_gain = getattr(
        args, "checkm2_refine_min_gain", cfg.checkm2_refine_min_gain
    )
    cfg.checkm2_refine_max_k = getattr(
        args, "checkm2_refine_max_k", cfg.checkm2_refine_max_k
    )
    cfg.device = args.device
    if args.threads is not None:
        cfg.num_threads = args.threads
    cfg.seed = args.seed
    cfg.verbose = args.verbose
    cfg.resolve_paths()
    return cfg


# --------------------------------------------------------------------------- #
# Handlers (lazy imports keep CLI startup fast and torch-free)
# --------------------------------------------------------------------------- #
def _cmd_bin(args: argparse.Namespace) -> int:
    from .pipeline import run_bin

    cfg = _config_from_args(args)
    run_bin(cfg)
    return 0


def _cmd_composition(args: argparse.Namespace) -> int:
    from .pipeline import run_composition

    cfg = _config_from_args(args)
    run_composition(cfg)
    return 0


def _cmd_abundance(args: argparse.Namespace) -> int:
    from .pipeline import run_abundance

    cfg = _config_from_args(args)
    run_abundance(cfg)
    return 0


def _cmd_markers(args: argparse.Namespace) -> int:
    from .pipeline import run_markers

    cfg = _config_from_args(args)
    run_markers(cfg)
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    from .pipeline import run_cluster

    cfg = _config_from_args(args)
    run_cluster(cfg)
    return 0


def _cmd_qc(args: argparse.Namespace) -> int:
    from .pipeline import run_qc

    cfg = _config_from_args(args)
    cfg.checkm2 = True
    run_qc(cfg)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose, logfile=args.outdir / "gbin.log")
    logger.info(f"gbin {__version__}")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
