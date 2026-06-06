"""Run configuration and device/VRAM handling.

The heavy ``torch`` import is deferred to the functions that need it so that
``gbin --help`` and unit tests for pure-CPU helpers stay light.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .utils import logger

ClusterMethod = Literal["auto", "leiden", "medoid"]
Precision = Literal["bf16", "fp16", "fp32"]


@dataclass
class GBinConfig:
    """All knobs for a binning run, populated from CLI arguments.

    Defaults follow VAMB/SemiBin where they are well established, adapted for a
    16 GB GPU. ``nsamples`` and several model defaults are resolved later, once
    the abundance matrix is known (see :meth:`resolve_model_defaults`).
    """

    # --- I/O ---
    fasta: Optional[Path] = None
    outdir: Optional[Path] = None
    bamdir: Optional[Path] = None
    abundance_tsv: list[Path] = field(default_factory=list)
    cache_dir: Optional[Path] = None  # defaults to outdir/cache

    # --- abundance from reads (mapping) ---
    reads: list[str] = field(default_factory=list)  # each "R1[,R2]" = one sample
    reads_tsv: Optional[Path] = None                # manifest: name<TAB>R1[<TAB>R2]
    mapper: str = "strobealign"                     # strobealign | minimap2
    mapper_preset: str = "sr"                       # minimap2 preset (sr|map-ont|map-pb)

    # --- filtering ---
    min_contig_len: int = 2000
    min_bin_size: int = 200_000  # bp; bins smaller than this are dropped

    # --- model ---
    latent: int = 32
    nhiddens: Optional[list[int]] = None  # auto from nsamples if None
    dropout: Optional[float] = None
    beta: float = 200.0
    alpha: Optional[float] = None
    epochs: int = 300
    batch_size: int = 256
    batchsteps: list[int] = field(default_factory=lambda: [25, 75, 150, 225])
    learning_rate: float = 1e-3
    precision: Precision = "bf16"

    # --- markers / hybrid ---
    use_markers: bool = True
    marker_loss_weight: float = 0.0  # lambda for the cannot-link term; 0 = off
    marker_margin: float = 1.0       # latent margin to separate cannot-link pairs
    refine: bool = True

    # --- final QC ---
    checkm2: bool = False             # run CheckM2 on the output bins
    checkm2_db: Optional[Path] = None  # CheckM2 database path (else default/env)
    checkm2_bin: str = "checkm2"      # checkm2 executable (use a path for a separate env)

    # --- CheckM2-guided refinement (opt-in; decontaminates bins using CheckM2's
    # accurate completeness/contamination instead of only the coarse SCG scorer) ---
    checkm2_refine: bool = False              # split contaminated bins, validated by CheckM2
    checkm2_refine_iters: int = 2             # guided rounds (recurse into still-dirty children)
    checkm2_refine_min_contamination: float = 10.0  # only split bins with >= this contamination %
    checkm2_refine_min_completeness: float = 50.0   # ... and >= this completeness % (worth saving)
    checkm2_refine_score_weight: float = 5.0  # DAS-Tool score = completeness - w * contamination
    checkm2_refine_min_gain: float = 10.0     # min DAS-score gain to accept a split (rejects
    #                                           cosmetic "peel one contig" splits, which gain ~0)
    checkm2_refine_max_k: int = 3             # max sub-bins to try when splitting a bin

    # --- clustering ---
    cluster_method: ClusterMethod = "auto"
    knn_k: int = 15

    # --- runtime ---
    device: str = "auto"  # "auto" | "cuda" | "cuda:0" | "cpu"
    num_threads: int = field(default_factory=lambda: os.cpu_count() or 8)
    max_gpu_mem_gb: Optional[float] = None  # cap for chunk sizing; None = autodetect
    seed: int = 0
    verbose: bool = False

    # --------------------------------------------------------------------- #
    def resolve_paths(self) -> None:
        """Fill in derived paths and create the output directory."""
        if self.outdir is None:
            raise ValueError("outdir must be set")
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir is None:
            self.cache_dir = self.outdir / "cache"
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_model_defaults(self, nsamples: int) -> None:
        """Set model hyperparameters that depend on the number of samples.

        Mirrors VAMB: with a single sample there is little abundance signal, so
        the network is smaller, dropout is off, and composition is weighted more
        heavily (higher alpha).
        """
        multi = nsamples > 1
        if self.nhiddens is None:
            self.nhiddens = [512, 512] if multi else [256, 256]
        if self.dropout is None:
            self.dropout = 0.2 if multi else 0.0
        if self.alpha is None:
            self.alpha = 0.15 if multi else 0.50


def resolve_device(spec: str = "auto") -> "object":
    """Return a ``torch.device`` for the given spec, logging the choice.

    ``auto`` picks CUDA when available, else CPU. An explicit ``cuda`` request on
    a machine without CUDA raises, so misconfiguration fails loudly rather than
    silently running 100x slower on the CPU.
    """
    import torch

    if spec == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("No CUDA device found; falling back to CPU (will be slow).")
            device = torch.device("cpu")
    elif spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{spec}' requested but torch.cuda.is_available() is False. "
                "Check your CUDA/driver install (Blackwell needs CUDA 12.8+ and a "
                "matching PyTorch build), or pass --device cpu."
            )
        device = torch.device(spec)
    else:
        device = torch.device(spec)

    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        cap = torch.cuda.get_device_capability(idx)
        logger.info(f"Using GPU {idx}: {name} ({total:.1f} GiB, sm_{cap[0]}{cap[1]})")
    else:
        logger.info("Using CPU")
    return device


def available_gpu_mem_gb(device: "object", cap_gb: Optional[float] = None) -> float:
    """Best-effort estimate of usable free VRAM in GiB for chunk sizing.

    Returns ``cap_gb`` when given; otherwise queries the device. On CPU returns a
    conservative constant so chunked code paths still behave.
    """
    import torch

    if cap_gb is not None:
        return cap_gb
    if getattr(device, "type", "cpu") != "cuda":
        return 4.0
    idx = device.index or 0
    free, _total = torch.cuda.mem_get_info(idx)
    return free / 1024**3


def supports_bf16(device: "object") -> bool:
    """Whether bf16 autocast is worthwhile on this device."""
    import torch

    if getattr(device, "type", "cpu") != "cuda":
        return False
    return torch.cuda.is_bf16_supported()
