"""Generate per-sample abundance by mapping reads back to the contigs.

Two backends, both producing a 2-column (contig, coverage) TSV per sample that
the existing abundance loader consumes:

* strobealign --aemb  -> short Illumina reads (fast, low memory; the recommended
  route, used by VAMB and SemiBin). Writes the abundance TSV directly.
* minimap2 | samtools sort | samtools coverage  -> long reads (Nanopore/PacBio)
  or when strobealign is unavailable. We parse the ``meandepth`` column.

For the best bins, map *every* sample's reads to the assembly (multi-sample
abundance) -- co-abundance across samples is the strongest binning signal.

External tools are invoked via subprocess; command construction is factored into
pure functions so it can be unit-tested without the binaries installed.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..utils import logger

_FASTQ_EXTS = (".fastq.gz", ".fq.gz", ".fastq", ".fq", ".gz")


@dataclass
class Sample:
    name: str
    r1: Path
    r2: Optional[Path] = None

    def reads(self) -> list[str]:
        return [str(self.r1)] + ([str(self.r2)] if self.r2 else [])


# --------------------------------------------------------------------------- #
# Sample specification
# --------------------------------------------------------------------------- #
def _strip_fastq_ext(path: Path) -> str:
    name = path.name
    for ext in _FASTQ_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def _sample_from_reads_arg(arg: str) -> Sample:
    parts = [p for p in arg.split(",") if p]
    if not parts or len(parts) > 2:
        raise ValueError(
            f"--reads expects 'R1' or 'R1,R2', got '{arg}'"
        )
    r1 = Path(parts[0])
    r2 = Path(parts[1]) if len(parts) == 2 else None
    stem = _strip_fastq_ext(r1)
    for suffix in ("_R1", "_1", ".R1", ".1"):  # drop common forward-read tags
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return Sample(name=stem or r1.stem, r1=r1, r2=r2)


def parse_reads_manifest(path: Path) -> list[Sample]:
    """Parse a manifest TSV: ``name<TAB>R1[<TAB>R2]`` per line (# = comment)."""
    samples: list[Sample] = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                raise ValueError(f"{path}:{lineno}: expected name<TAB>R1[<TAB>R2]")
            name, r1 = cols[0], Path(cols[1])
            r2 = Path(cols[2]) if len(cols) >= 3 and cols[2] else None
            samples.append(Sample(name=name, r1=r1, r2=r2))
    return samples


def collect_samples(
    reads_args: Sequence[str], reads_tsv: Optional[Path]
) -> list[Sample]:
    """Merge --reads occurrences and an optional manifest into a sample list."""
    samples: list[Sample] = []
    if reads_tsv is not None:
        samples.extend(parse_reads_manifest(Path(reads_tsv)))
    for arg in reads_args or []:
        samples.append(_sample_from_reads_arg(arg))
    if not samples:
        raise ValueError("No reads provided (use --reads and/or --reads-tsv)")
    names = [s.name for s in samples]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate sample names among reads: {names}")
    for s in samples:
        for r in s.reads():
            if not Path(r).exists():
                raise FileNotFoundError(f"Reads file not found: {r}")
    return samples


# --------------------------------------------------------------------------- #
# Command construction (pure -> unit-testable)
# --------------------------------------------------------------------------- #
def strobealign_cmd(contigs: Path, sample: Sample, threads: int) -> list[str]:
    return ["strobealign", "--aemb", "-t", str(threads),
            str(contigs), *sample.reads()]


def minimap2_cmd(contigs: Path, sample: Sample, threads: int, preset: str) -> list[str]:
    return ["minimap2", "-ax", preset, "-t", str(threads),
            str(contigs), *sample.reads()]


def samtools_sort_cmd(out_bam: Path, threads: int) -> list[str]:
    return ["samtools", "sort", "-@", str(threads), "-o", str(out_bam), "-"]


def samtools_coverage_cmd(bam: Path) -> list[str]:
    return ["samtools", "coverage", str(bam)]


def parse_samtools_coverage(text: str) -> dict[str, float]:
    """Parse `samtools coverage` output -> {contig: meandepth}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 7:
            out[f[0]] = float(f[6])  # meandepth column
    return out


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise FileNotFoundError(
            f"'{tool}' not found on PATH. Install it (e.g. `conda install -c bioconda "
            f"{tool}`) or choose another --mapper."
        )


def map_one_sample(
    contigs: Path,
    sample: Sample,
    out_tsv: Path,
    mapper: str = "strobealign",
    threads: int = 8,
    preset: str = "sr",
) -> Path:
    """Map one sample's reads and write a (contig, coverage) TSV to ``out_tsv``."""
    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Mapping sample '{sample.name}' with {mapper}")

    if mapper == "strobealign":
        _require("strobealign")
        with open(out_tsv, "w") as fh:
            subprocess.run(
                strobealign_cmd(contigs, sample, threads),
                stdout=fh, check=True,
            )
    elif mapper == "minimap2":
        _require("minimap2")
        _require("samtools")
        bam = out_tsv.with_suffix(".bam")
        mm = subprocess.Popen(
            minimap2_cmd(contigs, sample, threads, preset), stdout=subprocess.PIPE
        )
        sort = subprocess.Popen(
            samtools_sort_cmd(bam, threads), stdin=mm.stdout
        )
        mm.stdout.close()  # allow mm to receive SIGPIPE
        if sort.wait() != 0 or mm.wait() != 0:
            raise RuntimeError(f"minimap2|samtools sort failed for {sample.name}")
        cov = subprocess.run(
            samtools_coverage_cmd(bam), check=True, capture_output=True, text=True
        )
        depths = parse_samtools_coverage(cov.stdout)
        with open(out_tsv, "w") as fh:
            for contig, d in depths.items():
                fh.write(f"{contig}\t{d}\n")
        bam.unlink(missing_ok=True)
    else:
        raise ValueError(f"Unknown mapper '{mapper}' (use strobealign or minimap2)")
    return out_tsv
