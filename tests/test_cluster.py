"""Tests for the pure-PyTorch iterative medoid clusterer."""

import numpy as np
import torch

from gbin.cluster.medoid_torch import cluster_medoid, _weighted_histogram
from gbin.cluster.knn_graph import build_knn_graph
from gbin.cluster.community import labelprop_communities, _compact_labels


def _angular_blobs(K=8, per=25, dim=24, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(K, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts, labels = [], []
    for k in range(K):
        for _ in range(per):
            pts.append(dirs[k] + noise * rng.normal(size=dim))
            labels.append(k)
    return np.array(pts, dtype=np.float32), np.array(labels)


def _purity(pred, true):
    total = 0
    for c in np.unique(pred):
        members = true[pred == c]
        if len(members):
            total += np.bincount(members).max()
    return total / len(true)


def _completeness(pred, true):
    # Each true cluster should map mostly into one predicted cluster.
    total = 0
    for c in np.unique(true):
        members = pred[true == c]
        total += np.bincount(members - members.min()).max() if len(members) else 0
    return total / len(true)


def test_weighted_histogram_matches_numpy():
    vals = torch.tensor([0.0, 0.05, 0.05, 0.29, 0.5])
    wts = torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0])
    hist = _weighted_histogram(vals, wts, n_bins=60, vmax=0.3).numpy()
    # 0.5 is out of range -> bucketize clamps into the last bin; the in-range mass
    # for the first bin (0..0.005) is weight 1.0.
    assert hist[0] == 1.0
    # 0.05 falls in bin index 10 (0.05/0.005); both contribute -> 5.0
    assert hist[10] == 5.0


def test_medoid_recovers_angular_blobs():
    rng = np.random.default_rng(0)
    K, per, dim = 8, 25, 24
    dirs = rng.normal(size=(K, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts, labels = [], []
    for k in range(K):
        for _ in range(per):
            pts.append(dirs[k] + 0.02 * rng.normal(size=dim))
            labels.append(k)
    latent = np.array(pts, dtype=np.float32)
    labels = np.array(labels)
    lengths = np.full(len(latent), 5000, dtype=np.int32)

    pred = cluster_medoid(latent, lengths, torch.device("cpu"), seed=0)
    assert _purity(pred, labels) > 0.95
    assert _completeness(pred, labels) > 0.9
    # Should recover roughly K clusters (allow mild fragmentation).
    assert K <= len(np.unique(pred)) <= 2 * K


def test_medoid_single_point():
    pred = cluster_medoid(np.array([[1.0, 2.0]], dtype=np.float32),
                          np.array([1000], dtype=np.int32), torch.device("cpu"))
    assert pred.tolist() == [0]


def test_medoid_all_points_assigned():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(60, 16)).astype(np.float32)
    lengths = rng.integers(2000, 9000, size=60).astype(np.int32)
    pred = cluster_medoid(latent, lengths, torch.device("cpu"))
    assert (pred >= 0).all()  # every contig lands in some bin
    assert len(pred) == 60


def test_knn_graph_torch_neighbours_same_cluster():
    latent, labels = _angular_blobs()
    src, dst, w = build_knn_graph(latent, k=10, device=torch.device("cpu"),
                                  prefer_cuml=False)
    assert (src != dst).all()  # no self edges
    assert (w >= 0).all()
    # The vast majority of kNN edges should connect same-genome contigs.
    same = (labels[src] == labels[dst]).mean()
    assert same > 0.9, f"only {same:.2f} of kNN edges are intra-cluster"


def test_labelprop_recovers_blobs():
    latent, labels = _angular_blobs()
    src, dst, w = build_knn_graph(latent, k=10, device=torch.device("cpu"),
                                  prefer_cuml=False)
    pred = labelprop_communities(src, dst, w, len(latent), seed=0)
    assert _purity(pred, labels) > 0.95
    assert _completeness(pred, labels) > 0.9


def test_compact_labels_handles_singletons():
    out = _compact_labels(np.array([5, 5, -1, 3, -1]))
    assert set(out.tolist()) == {0, 1, 2, 3}  # 4 distinct -> two merged, two singletons
    assert out[0] == out[1]  # the two 5s stay together


def test_leiden_falls_back_to_medoid_on_failure(tmp_path, monkeypatch):
    # Simulate a RAPIDS/Blackwell failure in the GPU graph path; the dispatcher
    # must recover via the pure-torch medoid clusterer instead of crashing.
    from gbin.config import GBinConfig
    import gbin.cluster.clusterer as cl

    latent, labels = _angular_blobs()
    lengths = np.full(len(latent), 5000, dtype=np.int32)
    cfg = GBinConfig(outdir=tmp_path)
    cfg.cluster_method, cfg.knn_k, cfg.seed = "leiden", 10, 0

    monkeypatch.setattr(cl, "rapids_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("simulated cudf/numba sm_120 failure")

    monkeypatch.setattr(cl, "leiden_communities", boom)
    pred = cl.cluster_latent(latent, lengths, cfg, torch.device("cpu"))
    assert _purity(pred, labels) > 0.9  # medoid still recovered the blobs


def test_dispatcher_recovers_genomes_end_to_end(tmp_path):
    # Full path: synthetic -> composition -> abundance -> VAE -> cluster.
    from gbin.config import GBinConfig
    from gbin.features.composition import compute_composition
    from gbin.features.normalize import normalize_features
    from gbin.io.abundance import from_tsv
    from gbin.model.train import train_vae
    from gbin.cluster.clusterer import cluster_latent
    from make_synthetic import make_synthetic

    meta = make_synthetic(tmp_path, n_genomes=8, n_samples=5,
                          contigs_per_genome=8, seed=0)
    comp = compute_composition(meta["fasta"], "cpu", min_length=2000)
    ids = [str(x) for x in comp.identifiers]
    ab = from_tsv(meta["tsvs"], ids, comp.refhash)
    feats = normalize_features(ab.matrix, comp.tnf, comp.lengths, "cpu")
    n2g = {n: g for n, g in zip(meta["contig_names"], meta["contig_genome"])}
    labels = np.array([n2g[i] for i in ids])

    cfg = GBinConfig(outdir=tmp_path)
    cfg.resolve_paths()
    cfg.resolve_model_defaults(5)
    cfg.nhiddens, cfg.dropout, cfg.epochs = [256, 256], 0.0, 120
    cfg.latent, cfg.batch_size, cfg.batchsteps = 24, 64, []
    cfg.precision, cfg.seed, cfg.knn_k = "fp32", 0, 10
    _model, latent = train_vae(feats, cfg, torch.device("cpu"))

    for method in ("medoid", "leiden"):  # leiden falls back to labelprop on CPU
        cfg.cluster_method = method
        pred = cluster_latent(latent, comp.lengths, cfg, torch.device("cpu"))
        pur = _purity(pred, labels)
        comp_score = _completeness(pred, labels)
        assert pur > 0.9, f"{method}: purity {pur:.2f}"
        assert comp_score > 0.8, f"{method}: completeness {comp_score:.2f}"
