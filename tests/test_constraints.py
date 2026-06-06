"""Tests for deriving cannot-link constraints from shared single-copy markers."""

import numpy as np

from gbin.markers.scg import Markers
from gbin.markers.constraints import cannot_link_pairs


def _markers(rows, n_markers):
    arrs = [None if r is None else np.array(r, dtype=np.uint8) for r in rows]
    return Markers(arrs, [[str(i)] for i in range(n_markers)], b"x" * 32)


def test_pairs_from_shared_marker():
    # Three contigs carry the same SCG -> all three pairs are cannot-link.
    p = cannot_link_pairs(_markers([[0], [0], [0]], 1))
    assert {tuple(x) for x in p} == {(0, 1), (0, 2), (1, 2)}


def test_no_pairs_when_markers_unique():
    # Each SCG appears once -> nothing to separate.
    assert len(cannot_link_pairs(_markers([[0], [1], [2]], 3))) == 0


def test_none_contigs_excluded():
    p = cannot_link_pairs(_markers([[0], None, [0]], 1))
    assert {tuple(x) for x in p} == {(0, 2)}


def test_max_pairs_cap_is_respected():
    # 100 contigs share one SCG -> 4950 pairs, capped to 50.
    p = cannot_link_pairs(_markers([[0]] * 100, 1), max_pairs=50, seed=0)
    assert len(p) == 50
    assert p.shape[1] == 2
