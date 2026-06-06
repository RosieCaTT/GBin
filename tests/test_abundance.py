import numpy as np
import pytest

from gbin.io import abundance as ab
from gbin.utils import hash_identifiers


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n")


def test_two_column_alignment_and_order(tmp_path):
    ids = ["c1", "c2", "c3"]
    rh = hash_identifiers(ids)
    # Two samples; rows deliberately out of order and with an extra short contig.
    _write(tmp_path / "s1.tsv", ["c3\t3.0", "c1\t1.0", "cShort\t9.9", "c2\t2.0"])
    _write(tmp_path / "s2.tsv", ["c1\t10.0", "c2\t20.0", "c3\t30.0"])
    a = ab.from_tsv([tmp_path / "s1.tsv", tmp_path / "s2.tsv"], ids, rh)
    assert a.matrix.shape == (3, 2)
    np.testing.assert_array_equal(a.matrix[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(a.matrix[:, 1], [10.0, 20.0, 30.0])
    assert a.samplenames == ["s1", "s2"]


def test_header_is_tolerated(tmp_path):
    ids = ["c1", "c2"]
    rh = hash_identifiers(ids)
    _write(tmp_path / "s1.tsv", ["contig\tcoverage", "c1\t1.5", "c2\t2.5"])
    a = ab.from_tsv([tmp_path / "s1.tsv"], ids, rh)
    np.testing.assert_array_equal(a.matrix[:, 0], [1.5, 2.5])


def test_missing_contig_errors(tmp_path):
    ids = ["c1", "c2", "c3"]
    rh = hash_identifiers(ids)
    _write(tmp_path / "s1.tsv", ["c1\t1.0", "c2\t2.0"])  # c3 missing
    with pytest.raises(ValueError, match="no coverage value"):
        ab.from_tsv([tmp_path / "s1.tsv"], ids, rh)


def test_merged_tsv(tmp_path):
    ids = ["c1", "c2"]
    rh = hash_identifiers(ids)
    _write(
        tmp_path / "merged.tsv",
        ["contigname\tsampleA\tsampleB", "c1\t1\t2", "c2\t3\t4"],
    )
    a = ab.from_tsv([tmp_path / "merged.tsv"], ids, rh)
    assert a.matrix.shape == (2, 2)
    np.testing.assert_array_equal(a.matrix, [[1, 2], [3, 4]])
    assert a.samplenames == ["sampleA", "sampleB"]


def test_save_load_roundtrip(tmp_path):
    ids = ["c1", "c2"]
    rh = hash_identifiers(ids)
    _write(tmp_path / "s1.tsv", ["c1\t1.0", "c2\t2.0"])
    a = ab.from_tsv([tmp_path / "s1.tsv"], ids, rh)
    a.save(tmp_path / "abundance.npz")
    b = ab.Abundance.load(tmp_path / "abundance.npz", refhash=rh)
    np.testing.assert_array_equal(a.matrix, b.matrix)
    assert b.samplenames == a.samplenames
