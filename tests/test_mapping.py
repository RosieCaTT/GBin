"""Tests for read-mapping -> abundance.

The external mappers (strobealign/minimap2) are not invoked: command builders are
pure functions and the pipeline wiring is exercised with a mocked mapper.
"""

import numpy as np
import pytest

from gbin.io import mapping as mp
from gbin.io.mapping import Sample


def test_sample_name_strips_read_tag(tmp_path):
    (tmp_path / "s1_R1.fastq.gz").write_text("")
    (tmp_path / "s1_R2.fastq.gz").write_text("")
    s = mp._sample_from_reads_arg(f"{tmp_path/'s1_R1.fastq.gz'},{tmp_path/'s1_R2.fastq.gz'}")
    assert s.name == "s1"
    assert s.r2 is not None


def test_collect_samples_from_args_and_manifest(tmp_path):
    for n in ["a_1.fq", "a_2.fq", "b_R1.fq", "b_R2.fq"]:
        (tmp_path / n).write_text("")
    manifest = tmp_path / "m.tsv"
    manifest.write_text(f"# comment\nbX\t{tmp_path/'b_R1.fq'}\t{tmp_path/'b_R2.fq'}\n")
    samples = mp.collect_samples([f"{tmp_path/'a_1.fq'},{tmp_path/'a_2.fq'}"], manifest)
    names = sorted(s.name for s in samples)
    assert names == ["a", "bX"]


def test_collect_samples_rejects_duplicates(tmp_path):
    (tmp_path / "x_R1.fq").write_text("")
    with pytest.raises(ValueError, match="Duplicate"):
        mp.collect_samples([f"{tmp_path/'x_R1.fq'}", f"{tmp_path/'x_R1.fq'}"], None)


def test_collect_samples_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        mp.collect_samples([str(tmp_path / "nope.fq")], None)


def test_strobealign_cmd():
    s = Sample("s1", "r1.fq", "r2.fq")
    cmd = mp.strobealign_cmd("contigs.fna", s, threads=8)
    assert cmd[:3] == ["strobealign", "--aemb", "-t"]
    assert cmd[-3:] == ["contigs.fna", "r1.fq", "r2.fq"]


def test_minimap2_cmd_single_end():
    s = Sample("s1", "r1.fq", None)
    cmd = mp.minimap2_cmd("contigs.fna", s, threads=4, preset="map-ont")
    assert "map-ont" in cmd
    assert cmd[-2:] == ["contigs.fna", "r1.fq"]


def test_parse_samtools_coverage():
    text = (
        "#rname\tstartpos\tendpos\tnumreads\tcovbases\tcoverage\tmeandepth\tmbq\tmmq\n"
        "c1\t1\t1000\t50\t1000\t100.0\t12.5\t30\t40\n"
        "c2\t1\t2000\t10\t1500\t75.0\t3.2\t30\t40\n"
    )
    d = mp.parse_samtools_coverage(text)
    assert d == {"c1": 12.5, "c2": 3.2}


def test_reads_to_abundance_with_mocked_mapper(tmp_path, monkeypatch):
    # Full wiring: --reads -> (mocked) mapping -> TSV -> aligned abundance matrix.
    from make_synthetic import make_synthetic
    from gbin.cli import main
    from gbin.features.composition import Composition
    from gbin.io.abundance import Abundance

    meta = make_synthetic(tmp_path / "data", n_genomes=4, n_samples=1,
                          contigs_per_genome=4, seed=0)
    contig_names = meta["contig_names"]
    r1 = tmp_path / "s1_R1.fq"
    r1.write_text("@x\nACGT\n+\nIIII\n")

    def fake_map(contigs, sample, out_tsv, mapper="strobealign", threads=8, preset="sr"):
        # coverage = position-in-list + 1, written in a scrambled order
        with open(out_tsv, "w") as f:
            for i, n in reversed(list(enumerate(contig_names))):
                f.write(f"{n}\t{float(i + 1)}\n")
        return out_tsv

    monkeypatch.setattr("gbin.io.mapping.map_one_sample", fake_map)

    out = tmp_path / "out"
    rc = main(["abundance", "-i", str(meta["fasta"]), "--reads", str(r1),
               "-o", str(out), "--device", "cpu", "--min-contig-len", "2000"])
    assert rc == 0

    comp = Composition.load(out / "cache" / "composition.npz")
    ab = Abundance.load(out / "cache" / "abundance.npz", refhash=comp.refhash)
    assert ab.nsamples == 1
    # The mapped coverage must be correctly aligned to composition order.
    idx = {n: i for i, n in enumerate(contig_names)}
    for ci, name in enumerate(comp.identifiers):
        assert ab.matrix[ci, 0] == idx[str(name)] + 1

    # Re-running reuses the cached mapping (fake_map would overwrite identically;
    # assert the per-sample meta + tsv exist).
    assert (out / "cache" / "abundance_tsv" / "s1.tsv").exists()
    assert (out / "cache" / "abundance_tsv" / "s1.meta.json").exists()
