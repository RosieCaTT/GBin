"""Integration test: the VAE latent should separate genomes on synthetic data."""

import numpy as np
import torch

from gbin.config import GBinConfig
from gbin.features.composition import compute_composition
from gbin.features.normalize import normalize_features
from gbin.io.abundance import from_tsv
from gbin.model.train import train_vae
from gbin.model.vae import VAE, VAEHyperParams

from make_synthetic import make_synthetic


def _nn_purity(latent, labels):
    """Fraction of contigs whose nearest neighbour (cosine) shares its genome.

    This is the signal graph clustering relies on, and is robust in high
    dimensions (unlike a raw distance ratio). Random baseline here is ~1/9.
    """
    xn = latent / (np.linalg.norm(latent, axis=1, keepdims=True) + 1e-9)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -2.0)
    return float((labels[sim.argmax(1)] == labels).mean())


def _prep(tmp_path, seed=0):
    meta = make_synthetic(tmp_path, n_genomes=8, n_samples=5,
                          contigs_per_genome=8, seed=seed)
    comp = compute_composition(meta["fasta"], "cpu", min_length=2000)
    ids = [str(x) for x in comp.identifiers]
    ab = from_tsv(meta["tsvs"], ids, comp.refhash)
    feats = normalize_features(ab.matrix, comp.tnf, comp.lengths, "cpu")
    # Map contig order -> genome label.
    name_to_genome = {n: g for n, g in zip(meta["contig_names"], meta["contig_genome"])}
    labels = np.array([name_to_genome[i] for i in ids])
    return feats, labels


def _config(tmp_path):
    cfg = GBinConfig(outdir=tmp_path)
    cfg.resolve_paths()
    cfg.resolve_model_defaults(nsamples=5)
    cfg.nhiddens = [256, 256]
    cfg.dropout = 0.0
    cfg.epochs = 120
    cfg.latent = 24
    cfg.batch_size = 64
    cfg.batchsteps = []
    cfg.precision = "fp32"
    cfg.seed = 0
    return cfg


def test_vae_forward_shapes():
    hp = VAEHyperParams(nsamples=5, nhiddens=[64, 64], nlatent=16,
                        dropout=0.0, alpha=0.15, beta=200.0)
    model = VAE(hp)
    model.eval()
    d = torch.rand(10, 5)
    d = d / d.sum(1, keepdim=True)
    t = torch.randn(10, 103)
    a = torch.randn(10, 1)
    d_out, t_out, a_out, mu = model(d, t, a)
    assert d_out.shape == (10, 5)
    assert torch.allclose(d_out.sum(1), torch.ones(10), atol=1e-5)  # softmax
    assert t_out.shape == (10, 103)
    assert a_out.shape == (10, 1)
    assert mu.shape == (10, 16)


def test_latent_separates_genomes(tmp_path):
    feats, labels = _prep(tmp_path)
    cfg = _config(tmp_path)
    _model, latent = train_vae(feats, cfg, torch.device("cpu"))
    assert latent.shape == (len(labels), 24)
    purity = _nn_purity(latent, labels)
    # Both modalities encode genome identity; the fused latent should be highly
    # clusterable. Random baseline ~0.11; we require near-perfect.
    assert purity > 0.85, f"latent did not separate genomes (1-NN purity={purity:.2f})"


def test_encode_is_deterministic(tmp_path):
    feats, _labels = _prep(tmp_path)
    cfg = _config(tmp_path)
    cfg.epochs = 5
    model, latent1 = train_vae(feats, cfg, torch.device("cpu"))
    from gbin.model.train import encode_latent

    latent2 = encode_latent(model, feats, torch.device("cpu"))
    np.testing.assert_allclose(latent1, latent2, atol=1e-5)


def test_hybrid_cannot_link_training_runs(tmp_path):
    # The hybrid term should run end-to-end and not degrade separation.
    feats, labels = _prep(tmp_path)
    cfg = _config(tmp_path)
    cfg.epochs = 60
    cfg.marker_loss_weight = 0.5
    cfg.marker_margin = 2.0

    by_genome: dict[int, list[int]] = {}
    for i, g in enumerate(labels):
        by_genome.setdefault(int(g), []).append(i)
    genomes = list(by_genome)
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(300):  # cross-genome cannot-link pairs
        ga, gb = rng.choice(len(genomes), 2, replace=False)
        a = rng.choice(by_genome[genomes[ga]])
        b = rng.choice(by_genome[genomes[gb]])
        pairs.append((a, b))
    pairs = np.array(pairs, dtype=np.int64)

    _model, latent = train_vae(feats, cfg, torch.device("cpu"), cannot_link=pairs)
    assert latent.shape == (len(labels), 24)
    assert _nn_purity(latent, labels) > 0.85
