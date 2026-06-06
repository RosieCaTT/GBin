"""Tests for CheckM2-guided refinement (CheckM2 itself is mocked).

The split *math* is exercised on CPU; CheckM2 is replaced by fakes that score a
bin from the genome(s) its contigs belong to (a bin mixing genomes A and B is
"contaminated"), so the propose -> score -> accept/reject loop is fully tested
without the external tool or its database.
"""

import csv
from pathlib import Path

import numpy as np
import torch

from gbin.config import GBinConfig
from gbin.features.composition import Composition
from gbin.io.write import write_bins
from gbin.markers.scg import Markers
from gbin.qc import checkm2_refine
from gbin.qc.checkm2_refine import (
    checkm2_guided_refine,
    propose_splits,
    quality_score,
)

REFHASH = b"x" * 32
IDS = [f"A{i}" for i in range(10)] + [f"B{i}" for i in range(10)]


# --------------------------------------------------------------------------- #
# Synthetic two-genome bin (genome A latent != genome B latent; each genome
# carries every SCG once, so the merged bin duplicates them all).
# --------------------------------------------------------------------------- #
def _latent(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([1, 0, 0, 0], np.float32) + 0.02 * rng.normal(size=(10, 4))
    b = np.array([0, 1, 0, 0], np.float32) + 0.02 * rng.normal(size=(10, 4))
    return np.vstack([a, b]).astype(np.float32)


def _markers() -> Markers:
    rows = [np.array([i], dtype=np.uint8) for i in range(10)] * 2
    return Markers(rows, [[str(i)] for i in range(10)], REFHASH)


def _write_fasta(path: Path) -> None:
    path.write_text("".join(f">{c}\nACGTACGTAC\n" for c in IDS))


def _cfg(tmp_path: Path) -> GBinConfig:
    cfg = GBinConfig(outdir=tmp_path / "out")
    cfg.fasta = tmp_path / "contigs.fna"
    _write_fasta(cfg.fasta)
    cfg.num_threads = 1
    cfg.resolve_paths()
    return cfg


def _comp() -> Composition:
    return Composition(
        np.array(IDS, dtype=object),
        np.full(len(IDS), 30_000, dtype=np.int32),
        np.zeros((len(IDS), 103), dtype=np.float32),
        REFHASH,
    )


def _cc(row: dict) -> tuple[float, float]:
    return float(row["Completeness"]), float(row["Contamination"])


def _write_report(out_dir, rows):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "quality_report.tsv"
    report.write_text("Name\tCompleteness\tContamination\n" + "".join(rows))
    return report


def _fake_checkm2_by_genome(bins_dir, out_dir, *a, **k):
    """A bin is contaminated iff it mixes genome A and B contigs (by id prefix)."""
    rows = []
    for fa in sorted(Path(bins_dir).glob("*.fna")):
        genomes = {ln[1] for ln in fa.read_text().splitlines() if ln.startswith(">")}
        cont = 0.5 if len(genomes) <= 1 else 60.0
        rows.append(f"{fa.stem}\t95.0\t{cont}\n")
    return _write_report(out_dir, rows)


def _initial_merged_bin(cfg, comp):
    """Write one merged (A+B) bin and return its {label: name} map."""
    labels = np.zeros(len(IDS), dtype=np.int64)
    summary = write_bins(
        cfg.outdir, comp.identifiers, comp.lengths, labels, cfg.fasta, min_bin_size=0
    )
    return labels, summary["kept"]


# --------------------------------------------------------------------------- #
def test_quality_score():
    assert quality_score(90.0, 0.0) == 90.0
    assert quality_score(90.0, 4.0, weight=5.0) == 70.0
    assert quality_score(50.0, 10.0, weight=5.0) == 0.0


def test_propose_splits_separates_two_genomes():
    latent_t = torch.as_tensor(_latent())
    lengths = np.full(len(IDS), 30_000, dtype=np.int32)
    lengths_t = torch.as_tensor(lengths, dtype=torch.float32)
    parts = propose_splits(
        list(range(len(IDS))), latent_t, lengths_t, lengths, _markers(),
        torch.device("cpu"),
    )
    assert len(parts) >= 1
    halves = {frozenset(range(10)), frozenset(range(10, 20))}
    # at least one proposed partition cleanly separates the two genomes
    assert any({frozenset(g) for g in p} == halves for p in parts)


def test_guided_refine_accepts_clean_split(tmp_path, monkeypatch):
    monkeypatch.setattr(checkm2_refine, "run_checkm2", _fake_checkm2_by_genome)
    cfg, comp = _cfg(tmp_path), _comp()
    labels, kept = _initial_merged_bin(cfg, comp)
    new_labels, label_rows = checkm2_guided_refine(
        cfg, comp, _latent(), labels, _markers(), torch.device("cpu"),
        initial_bins_dir=cfg.outdir / "bins", initial_kept=kept,
    )
    # Contaminated merged bin is split back into the two genomes.
    assert len(set(new_labels.tolist())) == 2
    assert len(set(new_labels[:10].tolist())) == 1
    assert len(set(new_labels[10:].tolist())) == 1
    assert new_labels[0] != new_labels[10]
    # Every surviving bin now has a CheckM2 row (clean: 95 / 0.5).
    assert set(label_rows) == set(new_labels.tolist())
    assert all(_cc(r) == (95.0, 0.5) for r in label_rows.values())


def test_guided_refine_rejects_low_completeness_split(tmp_path, monkeypatch):
    # Round 0 flags the parent (95/11); the children together don't beat the
    # parent's DAS score by the min-gain margin, so the split is reverted.
    calls = {"n": 0}

    def fake(bins_dir, out_dir, *a, **k):
        first = calls["n"] == 0
        calls["n"] += 1
        rows = [
            f"{fa.stem}\t95.0\t11.0\n" if first else f"{fa.stem}\t20.0\t0.0\n"
            for fa in sorted(Path(bins_dir).glob("*.fna"))
        ]
        return _write_report(out_dir, rows)

    monkeypatch.setattr(checkm2_refine, "run_checkm2", fake)
    cfg, comp = _cfg(tmp_path), _comp()
    labels, kept = _initial_merged_bin(cfg, comp)
    new_labels, label_rows = checkm2_guided_refine(
        cfg, comp, _latent(), labels, _markers(), torch.device("cpu"),
        initial_bins_dir=cfg.outdir / "bins", initial_kept=kept,
    )
    assert len(set(new_labels.tolist())) == 1
    assert set(label_rows) == {0}
    assert _cc(label_rows[0]) == (95.0, 11.0)


def test_guided_refine_rejects_cosmetic_peel(tmp_path, monkeypatch):
    # The "split" leaves one child as dirty as the parent and peels a near-empty
    # fragment -> DAS gain ~0 -> min_gain rejects it (the v2 behaviour we want).
    calls = {"n": 0}

    def fake(bins_dir, out_dir, *a, **k):
        bins = sorted(Path(bins_dir).glob("*.fna"))
        if calls["n"] == 0:
            rows = [f"{fa.stem}\t93.81\t73.22\n" for fa in bins]
        else:
            rows = [
                (f"{fa.stem}\t93.0\t73.0\n" if i == 0 else f"{fa.stem}\t1.0\t0.0\n")
                for i, fa in enumerate(bins)
            ]
        calls["n"] += 1
        return _write_report(out_dir, rows)

    monkeypatch.setattr(checkm2_refine, "run_checkm2", fake)
    cfg, comp = _cfg(tmp_path), _comp()
    labels, kept = _initial_merged_bin(cfg, comp)
    new_labels, _ = checkm2_guided_refine(
        cfg, comp, _latent(), labels, _markers(), torch.device("cpu"),
        initial_bins_dir=cfg.outdir / "bins", initial_kept=kept,
    )
    assert len(set(new_labels.tolist())) == 1  # cosmetic split rejected; bin kept whole


def test_gbin_qc_refine_cli_and_aligned_report(tmp_path, monkeypatch):
    from gbin.cli import main

    monkeypatch.setattr(checkm2_refine, "run_checkm2", _fake_checkm2_by_genome)
    comp = _comp()
    out = tmp_path / "out"
    cache = out / "cache"
    cache.mkdir(parents=True)
    comp.save(cache / "composition.npz")
    np.save(cache / "latent.npy", _latent())
    _markers().save(cache / "markers.json")
    fasta = tmp_path / "contigs.fna"
    _write_fasta(fasta)

    # Pretend a prior `gbin bin` produced one merged (contaminated) bin.
    write_bins(out, comp.identifiers, comp.lengths, np.zeros(len(IDS), np.int64),
               fasta, min_bin_size=0)

    rc = main(["qc", "-o", str(out), "--refine", "-i", str(fasta), "--device", "cpu"])
    assert rc == 0

    bins = sorted((out / "bins").glob("*.fna"))
    assert len(bins) == 2  # split into the two genomes
    rows = list(csv.DictReader(open(out / "bins_info.tsv"), delimiter="\t"))
    for row in rows:
        assert row["checkm2_completeness"] == "0.9500"
        assert row["checkm2_contamination"] == "0.0050"
    # The native report's Name column matches the FINAL bins (not a stale snapshot).
    report = list(csv.DictReader(open(out / "checkm2" / "quality_report.tsv"),
                                 delimiter="\t"))
    assert {r["Name"] for r in report} == {b.stem for b in bins}
