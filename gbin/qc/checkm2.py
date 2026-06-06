"""Optional final quality control with CheckM2.

The internal single-copy-marker scorer (``markers/scg.py``) is fast and runs
inside the decontamination loop, but it is a coarse universal-marker heuristic.
CheckM2 is a far more accurate, ML-based estimator of completeness /
contamination, so we offer it as an opt-in *final* QC pass over the written bins:
run ``checkm2 predict`` on the bin FASTAs, then merge its numbers into
``bins_info.tsv`` (keeping the internal SCG columns alongside for comparison).

CheckM2 is an external tool (needs DIAMOND + a ~3 GB database). The command
builder and the report parser are pure functions so they can be unit-tested
without CheckM2 or its database installed.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..utils import logger


def checkm2_available(checkm2_bin: str = "checkm2") -> bool:
    """Whether the checkm2 executable is found (accepts a name or a full path)."""
    return shutil.which(checkm2_bin) is not None


def predict_cmd(
    bins_dir: Path,
    out_dir: Path,
    threads: int,
    extension: str = "fna",
    db_path: Optional[Path] = None,
    checkm2_bin: str = "checkm2",
) -> list[str]:
    cmd = [
        checkm2_bin, "predict",
        "--threads", str(threads),
        "-x", extension,
        "--input", str(bins_dir),
        "--output-directory", str(out_dir),
        "--force",
    ]
    if db_path is not None:
        cmd += ["--database_path", str(db_path)]
    return cmd


def parse_quality_rows(path: Path) -> dict[str, dict]:
    """Parse CheckM2 quality_report.tsv -> {bin_name: full row dict} (all columns)."""
    with open(path) as f:
        return {row["Name"]: row for row in csv.DictReader(f, delimiter="\t")}


def parse_quality_report(path: Path) -> dict[str, tuple[float, float]]:
    """Parse CheckM2 quality_report.tsv -> {bin_name: (completeness%, contamination%)}.

    Values are CheckM2's native 0-100 percentages.
    """
    return {
        name: (float(row["Completeness"]), float(row["Contamination"]))
        for name, row in parse_quality_rows(path).items()
    }


def write_quality_report(path: Path, rows: list[dict]) -> None:
    """Write a CheckM2-style quality_report.tsv from full row dicts.

    Used after CheckM2-guided refinement to leave a native report whose ``Name``
    column matches the final (post-split) bins on disk, instead of a stale
    pre-split snapshot. Column set follows the first row (CheckM2's own columns).
    """
    rows = list(rows)
    fieldnames = list(rows[0].keys()) if rows else ["Name", "Completeness", "Contamination"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_checkm2(
    bins_dir: Path,
    out_dir: Path,
    threads: int = 8,
    extension: str = "fna",
    db_path: Optional[Path] = None,
    checkm2_bin: str = "checkm2",
) -> Path:
    """Run ``checkm2 predict``; return the path to quality_report.tsv."""
    if not checkm2_available(checkm2_bin):
        raise FileNotFoundError(
            f"'{checkm2_bin}' not found. CheckM2 needs Python >=3.12 and has heavy "
            "deps, so install it in its OWN conda env, then point gbin at it:\n"
            "  conda create -n checkm2 -c bioconda -c conda-forge checkm2\n"
            "  conda run -n checkm2 checkm2 database --download\n"
            "  gbin ... --checkm2 --checkm2-bin $(conda run -n checkm2 which checkm2)\n"
            "Or omit --checkm2."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running CheckM2 on {bins_dir}")
    if db_path is None:
        logger.info(
            "No --checkm2-db given; relying on CheckM2's configured database "
            "(CHECKM2DB env var or `checkm2 database --setdblocation`)."
        )

    # CheckM2 shells out to diamond and prodigal. When checkm2 lives in its own
    # conda env (the recommended setup), those tools are next to the checkm2
    # executable but NOT on gbin's PATH. Prepend the executable's bin dir so they
    # are found without the user having to activate the checkm2 env.
    env = os.environ.copy()
    resolved = shutil.which(checkm2_bin)
    if resolved:
        bin_dir = os.path.dirname(os.path.abspath(resolved))
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

    # CheckM2's completeness "Neural Network" model is TensorFlow. When CheckM2
    # lives in the SAME env as gbin (the single-env setup), a GPU TensorFlow would
    # try to grab VRAM and can clash with torch over CUDA libraries. CheckM2's NN
    # is tiny (the DIAMOND search is the real cost) so we force it onto the CPU,
    # leaving the whole GPU to gbin. Harmless in the separate-env setup too.
    env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        subprocess.run(
            predict_cmd(bins_dir, out_dir, threads, extension, db_path, checkm2_bin),
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"CheckM2 exited with status {e.returncode}. Common causes: the "
            "DIAMOND database (point gbin at it with --checkm2-db "
            "/path/to/uniref100.KO.1.dmnd), or diamond/prodigal not being found "
            "(gbin adds the checkm2 executable's bin dir to PATH automatically; "
            "make sure diamond is installed in that same env)."
        ) from e
    report = out_dir / "quality_report.tsv"
    if not report.exists():
        raise RuntimeError(f"CheckM2 finished but {report} is missing")
    return report


def merge_into_bins_info(
    bins_info: Path, quality: dict[str, tuple[float, float]]
) -> int:
    """Add checkm2 completeness/contamination columns (as 0-1) to bins_info.tsv.

    Returns the number of MIMAG high-quality bins (>=90% complete, <=5% contam).
    """
    with open(bins_info) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fieldnames = (list(rows[0].keys()) if rows else
                  ["bin", "n_contigs", "size_bp", "n50",
                   "completeness", "contamination"])
    for col in ("checkm2_completeness", "checkm2_contamination"):
        if col not in fieldnames:
            fieldnames.append(col)

    n_hq = 0
    for row in rows:
        comp_pct, cont_pct = quality.get(row["bin"], (None, None))
        if comp_pct is None:
            row["checkm2_completeness"] = "NA"
            row["checkm2_contamination"] = "NA"
        else:
            row["checkm2_completeness"] = f"{comp_pct / 100:.4f}"
            row["checkm2_contamination"] = f"{cont_pct / 100:.4f}"
            if comp_pct >= 90.0 and cont_pct <= 5.0:
                n_hq += 1

    with open(bins_info, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return n_hq


def run_and_merge(
    outdir: Path,
    threads: int = 8,
    db_path: Optional[Path] = None,
    extension: str = "fna",
    checkm2_bin: str = "checkm2",
) -> None:
    """Run CheckM2 on ``outdir/bins`` and merge results into bins_info.tsv."""
    outdir = Path(outdir)
    bins_dir = outdir / "bins"
    bin_files = list(bins_dir.glob(f"*.{extension}"))
    if not bin_files:
        logger.warning(f"No *.{extension} bins in {bins_dir}; skipping CheckM2")
        return
    report = run_checkm2(
        bins_dir, outdir / "checkm2", threads, extension, db_path, checkm2_bin
    )
    quality = parse_quality_report(report)
    n_hq = merge_into_bins_info(outdir / "bins_info.tsv", quality)
    logger.info(
        f"CheckM2: {n_hq}/{len(bin_files)} high-quality bins "
        "(>=90% complete, <=5% contamination). "
        f"Full report: {report}"
    )
