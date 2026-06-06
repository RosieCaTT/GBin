"""Unit tests for SCG bookkeeping and completeness/contamination scoring."""

import numpy as np

from gbin.markers.scg import Markers


def _markers(rows, n_markers, refhash=b"x" * 32):
    arrs = [None if r is None else np.array(r, dtype=np.uint8) for r in rows]
    names = [[str(i)] for i in range(n_markers)]
    return Markers(arrs, names, refhash)


def test_complete_and_clean():
    m = _markers([[0, 1, 2], [3, 4]], n_markers=5)
    comp, cont = m.score_bin([0, 1])
    assert comp == 1.0
    assert cont == 0.0


def test_contamination_from_duplicates():
    # Two contigs each carrying SCGs 0,1,2 -> every present SCG is duplicated.
    m = _markers([[0, 1, 2], [0, 1, 2]], n_markers=5)
    comp, cont = m.score_bin([0, 1])
    assert comp == 3 / 5
    assert abs(cont - 3 / 5) < 1e-9


def test_partial_completeness():
    m = _markers([[0, 1]], n_markers=4)
    comp, cont = m.score_bin([0])
    assert comp == 0.5
    assert cont == 0.0


def test_none_contigs_ignored():
    m = _markers([None, [0, 1]], n_markers=4)
    comp, _ = m.score_bin([0, 1])
    assert comp == 0.5


def test_counts_vector():
    m = _markers([[0, 1], [1, 2]], n_markers=4)
    np.testing.assert_array_equal(m.counts([0, 1]), [1, 2, 1, 0])


def test_save_load_roundtrip(tmp_path):
    m = _markers([[0, 1], None, [2]], n_markers=3)
    m.save(tmp_path / "m.json")
    loaded = Markers.load(tmp_path / "m.json", refhash=m.refhash)
    assert loaded.n_markers == 3
    assert loaded.markers[1] is None
    np.testing.assert_array_equal(loaded.markers[0], [0, 1])
    assert loaded.markers[0].dtype == np.uint8
