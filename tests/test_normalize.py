import numpy as np

from gbin.features.normalize import normalize_features, normalize_features_reference


def _data(n=50, s=4, seed=0):
    rng = np.random.default_rng(seed)
    abundance = rng.exponential(5.0, size=(n, s)).astype(np.float32)
    tnf = rng.normal(0, 1, size=(n, 103)).astype(np.float32)
    lengths = rng.integers(2000, 50000, size=n).astype(np.int32)
    return abundance, tnf, lengths


def test_matches_reference_multisample():
    abundance, tnf, lengths = _data()
    got = normalize_features(abundance, tnf, lengths, "cpu")
    want = normalize_features_reference(abundance, tnf, lengths)
    np.testing.assert_allclose(got.depths, want.depths, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got.tnf, want.tnf, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got.total_abundance, want.total_abundance, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got.weights, want.weights, rtol=1e-5, atol=1e-6)


def test_depths_sum_to_one():
    abundance, tnf, lengths = _data()
    got = normalize_features(abundance, tnf, lengths, "cpu")
    np.testing.assert_allclose(got.depths.sum(axis=1), 1.0, atol=1e-5)


def test_tnf_zscored():
    abundance, tnf, lengths = _data()
    got = normalize_features(abundance, tnf, lengths, "cpu")
    np.testing.assert_allclose(got.tnf.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(got.tnf.std(axis=0), 1.0, atol=1e-3)


def test_single_sample_depths_are_one():
    # With one sample, every contig's depth distribution is trivially [1.0].
    abundance, tnf, lengths = _data(s=1)
    got = normalize_features(abundance, tnf, lengths, "cpu")
    np.testing.assert_allclose(got.depths, 1.0, atol=1e-5)


def test_zero_coverage_contig_gets_uniform():
    abundance, tnf, lengths = _data(s=3)
    abundance[7] = 0.0
    got = normalize_features(abundance, tnf, lengths, "cpu")
    np.testing.assert_allclose(got.depths[7], 1.0 / 3.0, atol=1e-5)


def test_weights_lower_bounded():
    abundance, tnf, lengths = _data()
    lengths[:] = 1000  # log(1000)-5 < 2 -> clamped to 2 before rescaling
    got = normalize_features(abundance, tnf, lengths, "cpu")
    # All equal lengths -> all weights equal to 1 after rescale by N/sum.
    np.testing.assert_allclose(got.weights, 1.0, atol=1e-5)
