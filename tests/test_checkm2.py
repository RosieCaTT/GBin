"""Tests for the CheckM2 QC integration (CheckM2 itself is mocked)."""

import csv
import os
from pathlib import Path

from gbin.qc import checkm2


def test_predict_cmd_with_db():
    cmd = checkm2.predict_cmd("bins", "out", threads=8, extension="fna", db_path="db")
    assert cmd[:2] == ["checkm2", "predict"]
    assert "--threads" in cmd and "8" in cmd
    assert "-x" in cmd and "fna" in cmd
    assert "--input" in cmd and "bins" in cmd
    assert "--output-directory" in cmd and "out" in cmd
    assert "--database_path" in cmd and "db" in cmd
    assert "--force" in cmd


def test_predict_cmd_without_db():
    assert "--database_path" not in checkm2.predict_cmd("bins", "out", 8)


def test_predict_cmd_custom_executable():
    # --checkm2-bin lets gbin call checkm2 from a separate conda env by path.
    cmd = checkm2.predict_cmd("bins", "out", 8,
                              checkm2_bin="/envs/checkm2/bin/checkm2")
    assert cmd[0] == "/envs/checkm2/bin/checkm2"
    assert cmd[1] == "predict"


def test_parse_quality_report(tmp_path):
    p = tmp_path / "quality_report.tsv"
    p.write_text(
        "Name\tCompleteness\tContamination\tCompleteness_Model_Used\n"
        "bin00000\t99.04\t0.50\tGradient Boost\n"
        "bin00001\t45.00\t12.30\tNeural Network\n"
    )
    q = checkm2.parse_quality_report(p)
    assert q["bin00000"] == (99.04, 0.50)
    assert q["bin00001"] == (45.00, 12.30)


def test_merge_into_bins_info(tmp_path):
    info = tmp_path / "bins_info.tsv"
    info.write_text(
        "bin\tn_contigs\tsize_bp\tn50\tcompleteness\tcontamination\n"
        "bin00000\t5\t100000\t2000\t0.9000\t0.0000\n"
        "bin00001\t3\t50000\t1500\t0.4000\t0.1000\n"
    )
    quality = {"bin00000": (99.0, 2.0), "bin00001": (45.0, 12.0)}
    n_hq = checkm2.merge_into_bins_info(info, quality)
    assert n_hq == 1  # only bin00000 is >=90% complete and <=5% contam

    rows = list(csv.DictReader(open(info), delimiter="\t"))
    assert rows[0]["checkm2_completeness"] == "0.9900"  # 99% -> 0.99
    assert rows[0]["checkm2_contamination"] == "0.0200"
    assert rows[1]["checkm2_completeness"] == "0.4500"
    # Internal SCG columns are preserved alongside.
    assert rows[0]["completeness"] == "0.9000"


def test_merge_marks_missing_bins_na(tmp_path):
    info = tmp_path / "bins_info.tsv"
    info.write_text(
        "bin\tn_contigs\tsize_bp\tn50\tcompleteness\tcontamination\n"
        "bin00000\t5\t100000\t2000\t0.9\t0.0\n"
    )
    checkm2.merge_into_bins_info(info, {})  # CheckM2 returned nothing for it
    row = next(csv.DictReader(open(info), delimiter="\t"))
    assert row["checkm2_completeness"] == "NA"


def test_run_and_merge_mocked(tmp_path, monkeypatch):
    (tmp_path / "bins").mkdir()
    (tmp_path / "bins" / "bin00000.fna").write_text(">c\nACGT\n")
    (tmp_path / "bins_info.tsv").write_text(
        "bin\tn_contigs\tsize_bp\tn50\tcompleteness\tcontamination\n"
        "bin00000\t1\t4\t4\t0.9\t0.0\n"
    )

    def fake_run(bins_dir, out_dir, threads=8, extension="fna", db_path=None,
                 checkm2_bin="checkm2"):
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rep = Path(out_dir) / "quality_report.tsv"
        rep.write_text("Name\tCompleteness\tContamination\nbin00000\t95.0\t1.0\n")
        return rep

    monkeypatch.setattr("gbin.qc.checkm2.run_checkm2", fake_run)
    checkm2.run_and_merge(tmp_path, threads=4)
    row = next(csv.DictReader(open(tmp_path / "bins_info.tsv"), delimiter="\t"))
    assert row["checkm2_completeness"] == "0.9500"


def test_run_checkm2_prepends_executable_bin_dir_to_path(tmp_path, monkeypatch):
    # checkm2 in its own conda env: gbin must add that env's bin/ to PATH so the
    # sibling diamond/prodigal tools are found by the checkm2 subprocess.
    bindir = tmp_path / "envs" / "checkm2" / "bin"
    bindir.mkdir(parents=True)
    fake = bindir / "checkm2"
    fake.write_text("")

    monkeypatch.setattr("gbin.qc.checkm2.shutil.which", lambda x: str(fake))

    captured = {}

    def fake_run(cmd, check=True, env=None, **kw):
        captured["env"] = env
        od = Path(cmd[cmd.index("--output-directory") + 1])
        od.mkdir(parents=True, exist_ok=True)
        (od / "quality_report.tsv").write_text("Name\tCompleteness\tContamination\n")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr("gbin.qc.checkm2.subprocess.run", fake_run)
    checkm2.run_checkm2(tmp_path / "bins", tmp_path / "out", checkm2_bin=str(fake))
    assert str(bindir) in captured["env"]["PATH"].split(os.pathsep)
    # In a single env, CheckM2's TensorFlow must stay off the GPU (leave it to gbin).
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_gbin_qc_cli_mocked(tmp_path, monkeypatch):
    from gbin.cli import main

    out = tmp_path / "out"
    (out / "bins").mkdir(parents=True)
    (out / "bins" / "bin00000.fna").write_text(">c\nACGT\n")
    (out / "bins_info.tsv").write_text(
        "bin\tn_contigs\tsize_bp\tn50\tcompleteness\tcontamination\n"
        "bin00000\t1\t4\t4\t0.9\t0.0\n"
    )

    def fake_run(bins_dir, out_dir, threads=8, extension="fna", db_path=None,
                 checkm2_bin="checkm2"):
        from pathlib import Path

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rep = Path(out_dir) / "quality_report.tsv"
        rep.write_text("Name\tCompleteness\tContamination\nbin00000\t92.0\t3.0\n")
        return rep

    monkeypatch.setattr("gbin.qc.checkm2.run_checkm2", fake_run)
    assert main(["qc", "-o", str(out), "--device", "cpu"]) == 0
    row = next(csv.DictReader(open(out / "bins_info.tsv"), delimiter="\t"))
    assert row["checkm2_completeness"] == "0.9200"
    assert row["checkm2_contamination"] == "0.0300"
