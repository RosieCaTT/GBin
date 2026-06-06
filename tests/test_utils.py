import numpy as np

from gbin.utils import RefHasher, hash_identifiers, n50, chunked, human_bytes


def test_refhash_order_sensitive():
    a = hash_identifiers(["c1", "c2", "c3"])
    b = hash_identifiers(["c1", "c3", "c2"])
    assert a != b
    assert a == hash_identifiers(["c1", "c2", "c3"])


def test_refhash_no_collision_on_boundary():
    # Length-prefixing prevents ("ab","c") from hashing like ("a","bc").
    assert hash_identifiers(["ab", "c"]) != hash_identifiers(["a", "bc"])


def test_refhash_verify():
    h = hash_identifiers(["x", "y"])
    RefHasher.verify(h, h)  # no raise
    try:
        RefHasher.verify(h, hash_identifiers(["x", "z"]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatch")


def test_n50():
    # lengths 2,3,4,5,6 -> total 20, half 10; sorted desc 6,5,4 cumsum 6,11 -> N50=5
    assert n50(np.array([2, 3, 4, 5, 6])) == 5
    assert n50(np.array([])) == 0
    assert n50(np.array([100])) == 100


def test_chunked():
    assert [list(c) for c in chunked([1, 2, 3, 4, 5], 2)] == [[1, 2], [3, 4], [5]]


def test_human_bytes():
    assert human_bytes(0) == "0.0 B"
    assert human_bytes(1024) == "1.0 KiB"
    assert human_bytes(1024**3) == "1.0 GiB"
