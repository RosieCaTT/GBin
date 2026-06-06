"""End-to-end CLI test: `gbin bin` should recover the synthetic genomes."""

import csv
from collections import Counter, defaultdict

from gbin.cli import main
from make_synthetic import make_synthetic


def test_gbin_bin_recovers_pure_bins(tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "out"
    meta = make_synthetic(data, n_genomes=8, n_samples=5,
                          contigs_per_genome=6, seed=0)

    rc = main([
        "bin",
        "-i", str(meta["fasta"]),
        "-a", *[str(p) for p in meta["tsvs"]],
        "-o", str(out),
        "--device", "cpu",
        "--cluster", "medoid",
        "--epochs", "80",
        "--latent", "24",
        "--min-contig-len", "2000",
        "--min-bin-size", "5000",
    ])
    assert rc == 0

    # Output artifacts exist.
    assert (out / "bins_info.tsv").exists()
    assert (out / "contig_bins.tsv").exists()
    bin_files = list((out / "bins").glob("*.fna"))
    assert len(bin_files) >= 7  # ~8 genomes, allow mild fragmentation

    # Every bin should be (near) pure: dominated by a single genome.
    rows = list(csv.DictReader(open(out / "contig_bins.tsv"), delimiter="\t"))
    assert len(rows) == 48  # all contigs binned
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[r["bin"]].append(r["contig"].split("_")[0])
    purities = []
    for genomes in by_bin.values():
        c = Counter(genomes)
        purities.append(c.most_common(1)[0][1] / sum(c.values()))
    assert min(purities) >= 0.95, f"impure bin found: {purities}"


def test_gbin_bin_full_pipeline_with_checkm2(tmp_path, monkeypatch):
    # Everything together: composition + abundance + markers + VAE + medoid +
    # refine + CheckM2 (mocked subprocess) + write/merge.
    import csv

    data = tmp_path / "data"
    out = tmp_path / "out"
    meta = make_synthetic(data, n_genomes=6, n_samples=4,
                          contigs_per_genome=6, seed=0)

    def fake_run(bins_dir, out_dir, threads=8, extension="fna", db_path=None,
                 checkm2_bin="checkm2"):
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rep = Path(out_dir) / "quality_report.tsv"
        lines = ["Name\tCompleteness\tContamination"]
        for f in sorted(Path(bins_dir).glob("*.fna")):
            lines.append(f"{f.stem}\t93.0\t2.0")
        rep.write_text("\n".join(lines) + "\n")
        return rep

    monkeypatch.setattr("gbin.qc.checkm2.run_checkm2", fake_run)

    rc = main([
        "bin", "-i", str(meta["fasta"]), "-a", *[str(p) for p in meta["tsvs"]],
        "-o", str(out), "--device", "cpu", "--cluster", "medoid",
        "--epochs", "60", "--min-bin-size", "5000",
        "--checkm2", "--checkm2-bin", "checkm2",
    ])
    assert rc == 0

    # CheckM2 ran and its numbers were merged into the report.
    assert (out / "checkm2" / "quality_report.tsv").exists()
    rows = list(csv.DictReader(open(out / "bins_info.tsv"), delimiter="\t"))
    assert len(rows) >= 1
    assert all("checkm2_completeness" in r for r in rows)
    assert rows[0]["checkm2_completeness"] == "0.9300"
    assert rows[0]["checkm2_contamination"] == "0.0200"
    # Internal SCG columns are still there alongside.
    assert "completeness" in rows[0]


def test_checkm2_failure_does_not_fail_binning(tmp_path, monkeypatch):
    # A CheckM2 problem (e.g. missing DIAMOND DB) must not throw away a finished
    # binning run: bins stay written and the command still succeeds.
    import csv

    meta = make_synthetic(tmp_path / "data", n_genomes=4, n_samples=4,
                          contigs_per_genome=5, seed=0)

    def boom(*a, **k):
        raise RuntimeError("simulated: DIAMOND database not found")

    monkeypatch.setattr("gbin.qc.checkm2.run_checkm2", boom)
    out = tmp_path / "out"
    rc = main([
        "bin", "-i", str(meta["fasta"]), "-a", *[str(p) for p in meta["tsvs"]],
        "-o", str(out), "--device", "cpu", "--cluster", "medoid",
        "--epochs", "30", "--min-bin-size", "5000",
        "--checkm2", "--checkm2-bin", "checkm2",
    ])
    assert rc == 0  # binning succeeded despite CheckM2 failing
    assert len(list((out / "bins").glob("*.fna"))) >= 1
    rows = list(csv.DictReader(open(out / "bins_info.tsv"), delimiter="\t"))
    assert "checkm2_completeness" not in rows[0]  # merge was skipped, no crash


def test_gbin_bin_is_cached(tmp_path):
    # A second run should reuse cached composition + abundance.
    data = tmp_path / "data"
    out = tmp_path / "out"
    meta = make_synthetic(data, n_genomes=4, n_samples=4,
                          contigs_per_genome=5, seed=1)
    args = [
        "bin", "-i", str(meta["fasta"]), "-a", *[str(p) for p in meta["tsvs"]],
        "-o", str(out), "--device", "cpu", "--cluster", "medoid",
        "--epochs", "30", "--latent", "16", "--min-bin-size", "5000",
    ]
    assert main(args) == 0
    assert (out / "cache" / "composition.npz").exists()
    assert (out / "cache" / "abundance.npz").exists()
    assert (out / "cache" / "latent.npy").exists()
    # Re-run cluster-only from cache (no re-training needed).
    assert main([
        "cluster", "-i", str(meta["fasta"]), "-o", str(out),
        "--device", "cpu", "--cluster", "medoid", "--min-bin-size", "5000",
    ]) == 0
