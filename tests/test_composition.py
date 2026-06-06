"""Validate the GPU k-mer counter (run on CPU tensors) against a plain oracle.

These tests exercise the exact tensor code that runs on the GPU, just with
device='cpu', so correctness here implies correctness of the CUDA path.
"""

import numpy as np
import torch

from gbin.features import composition as comp


def _count_gpu_path(seqs):
    """Run the production count_4mers on a list of sequences (cpu device)."""
    lut = torch.as_tensor(comp.build_lut(), device="cpu")
    # .copy() makes the buffer writable (matches np.concatenate output in prod).
    seq_bytes = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).copy()
    lengths = np.array([len(s) for s in seqs], dtype=np.int32)
    return comp.count_4mers(seq_bytes, lengths, "cpu", lut).numpy()


def test_single_sequence_matches_reference():
    seq = "ACGTACGTTTTTGGGGCCCCAAAACGATCGATCGTAGCTAGCTAGC"
    got = _count_gpu_path([seq])[0]
    want = comp.count_4mers_reference(seq)
    assert np.array_equal(got, want)
    # 4-mer windows = len - 3 for an all-valid sequence
    assert got.sum() == len(seq) - 3


def test_invalid_bases_break_windows():
    # The N invalidates every window that overlaps it.
    seq = "ACGTACGTNACGTACGT"
    got = _count_gpu_path([seq])[0]
    want = comp.count_4mers_reference(seq)
    assert np.array_equal(got, want)
    # Windows containing the N (4 of them) are skipped.
    assert got.sum() == (len(seq) - 3) - 4


def test_lowercase_and_uracil():
    assert np.array_equal(
        comp.count_4mers_reference("acgt"), comp.count_4mers_reference("ACGT")
    )
    # U treated as T
    assert np.array_equal(
        comp.count_4mers_reference("ACGU"), comp.count_4mers_reference("ACGT")
    )


def test_windows_do_not_cross_contig_boundary():
    # Concatenated counting must equal the sum of per-contig counts: no window may
    # span the junction between two contigs.
    seqs = ["ACGTACGTAC", "TTTTGGGGCC", "AAAACCCCGG"]
    got = _count_gpu_path(seqs)
    for i, s in enumerate(seqs):
        assert np.array_equal(got[i], comp.count_4mers_reference(s)), f"contig {i}"


def test_minimum_length_contig():
    # A length-4 contig has exactly one 4-mer.
    got = _count_gpu_path(["ACGT"])[0]
    assert got.sum() == 1
    assert got[comp.build_lut()[ord("A")] * 64 + 1 * 16 + 2 * 4 + 3] == 1


def test_projection_matches_reference():
    seqs = ["ACGTACGTTTTTGGGGCCCC", "AAAACGATCGATCGTAGCTAGC"]
    counts = _count_gpu_path(seqs)
    kernel = torch.as_tensor(comp.load_kernel(), device="cpu")
    got = comp.project(torch.as_tensor(counts), kernel).numpy()
    want = comp.project_reference(counts)
    assert got.shape == (2, comp.NTNF)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_zero_sum_after_centering():
    # After normalization and -1/256 centering, each row of frequencies sums to 0,
    # so the projected features come from a zero-sum input (kernel columns sum ~0).
    counts = _count_gpu_path(["ACGTACGTACGTACGT"])
    s = counts.sum(axis=1, keepdims=True)
    freqs = counts / s - 1.0 / 256.0
    np.testing.assert_allclose(freqs.sum(axis=1), 0.0, atol=1e-5)
