"""Train the VAE and encode contigs to a latent matrix.

Training runs entirely on-device: the (small) normalized feature tensors live on
the GPU and minibatches are formed by index shuffling, so there is no per-batch
host<->device copy. Mixed precision uses bf16 autocast on Blackwell (no loss
scaler needed); the loss itself is always computed in fp32 for numerical safety.
Batch size doubles at the configured epochs, which acts as an implicit
learning-rate schedule (as in VAMB).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import GBinConfig, supports_bf16
from ..features.normalize import NormalizedFeatures
from ..utils import logger, timed
from .loss import vae_loss
from .vae import VAE, VAEHyperParams


def _amp_settings(cfg: GBinConfig, device):
    """Return (enabled, dtype) for autocast given precision config and device."""
    import torch

    if device.type != "cuda" or cfg.precision == "fp32":
        return False, None
    if cfg.precision == "bf16":
        if not supports_bf16(device):
            logger.warning("bf16 unsupported on this GPU; using fp16.")
            return True, torch.float16
        return True, torch.bfloat16
    return True, torch.float16


def _cannot_link_loss(model, depths, tnf, ta, cl, n_sample, margin, amp_enabled, amp_dtype):
    """Hinge loss pushing cannot-link pairs >= margin apart in latent space."""
    import torch

    p = torch.randint(0, cl.shape[0], (n_sample,), device=cl.device)
    ia, ib = cl[p, 0], cl[p, 1]
    with torch.autocast(depths.device.type, dtype=amp_dtype, enabled=amp_enabled):
        mu_a = model._encode(torch.cat((depths[ia], tnf[ia], ta[ia]), dim=1))
        mu_b = model._encode(torch.cat((depths[ib], tnf[ib], ta[ib]), dim=1))
    dist = (mu_a.float() - mu_b.float()).norm(dim=1)
    return torch.relu(margin - dist).pow(2).mean()


def _iter_batches(perm, batch_size: int):
    """Yield index batches with drop-last semantics (>=2 items per batch)."""
    n = perm.numel()
    n_full = n // batch_size
    if n_full == 0:
        yield perm  # single batch of all n (n >= 2 guaranteed by caller)
        return
    for i in range(n_full):
        yield perm[i * batch_size : (i + 1) * batch_size]


def train_vae(
    features: NormalizedFeatures,
    cfg: GBinConfig,
    device,
    cannot_link: "np.ndarray | None" = None,
) -> tuple[VAE, np.ndarray]:
    """Train a VAE on the normalized features; return (model, latent[N, nlatent]).

    If ``cannot_link`` ((P, 2) contig index pairs) is given and
    ``cfg.marker_loss_weight > 0``, a hybrid margin loss pushes each constrained
    pair at least ``cfg.marker_margin`` apart in latent space.
    """
    import torch

    torch.manual_seed(cfg.seed)
    n = features.depths.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 contigs to train, got {n}")

    hp = VAEHyperParams(
        nsamples=features.nsamples,
        nhiddens=list(cfg.nhiddens),
        nlatent=cfg.latent,
        dropout=float(cfg.dropout),
        alpha=float(cfg.alpha),
        beta=cfg.beta,
    )
    model = VAE(hp).to(device)

    depths = torch.as_tensor(features.depths, device=device)
    tnf = torch.as_tensor(features.tnf, device=device)
    ta = torch.as_tensor(features.total_abundance, device=device)
    weights = torch.as_tensor(features.weights, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    amp_enabled, amp_dtype = _amp_settings(cfg, device)
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    # Optional hybrid cannot-link constraints (contigs sharing an SCG).
    cl = None
    if cannot_link is not None and len(cannot_link) and cfg.marker_loss_weight > 0:
        cl = torch.as_tensor(cannot_link, device=device, dtype=torch.long)
        logger.info(
            f"Hybrid loss: {len(cl)} cannot-link pairs, weight={cfg.marker_loss_weight}, "
            f"margin={cfg.marker_margin}"
        )

    logger.info(
        f"Training VAE: N={n}, samples={hp.nsamples}, hidden={hp.nhiddens}, "
        f"latent={hp.nlatent}, alpha={hp.alpha:.2f}, precision={cfg.precision}, "
        f"epochs={cfg.epochs}"
    )

    batch_size = cfg.batch_size
    batchsteps = set(cfg.batchsteps)
    with timed("VAE training"):
        for epoch in range(cfg.epochs):
            if epoch in batchsteps and batch_size * 2 < n:
                batch_size *= 2
            model.train()
            perm = torch.randperm(n, device=device)
            totals = np.zeros(5)  # total, ce, ab, sse, kld
            n_batches = 0
            for idx in _iter_batches(perm, batch_size):
                d, t, a, w = depths[idx], tnf[idx], ta[idx], weights[idx]
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
                    d_out, t_out, a_out, mu = model(d, t, a)
                comp = vae_loss(
                    d, d_out.float(), t, t_out.float(), a, a_out.float(),
                    mu.float(), w,
                    alpha=hp.alpha, beta=hp.beta,
                    nsamples=hp.nsamples, nlatent=hp.nlatent,
                )
                total_loss = comp.total
                if cl is not None:
                    total_loss = total_loss + cfg.marker_loss_weight * _cannot_link_loss(
                        model, depths, tnf, ta, cl, idx.numel(),
                        cfg.marker_margin, amp_enabled, amp_dtype,
                    )
                opt.zero_grad(set_to_none=True)
                if use_scaler:
                    scaler.scale(total_loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    total_loss.backward()
                    opt.step()
                totals += [comp.total.item(), comp.ce.item(), comp.ab.item(),
                           comp.sse.item(), comp.kld.item()]
                n_batches += 1
            if (epoch + 1) % max(1, cfg.epochs // 10) == 0 or epoch == 0:
                t_, ce_, ab_, sse_, kld_ = totals / max(1, n_batches)
                logger.info(
                    f"  epoch {epoch + 1:>4}/{cfg.epochs}  loss={t_:.4e}  "
                    f"CE={ce_:.4e} AB={ab_:.4e} SSE={sse_:.4e} KLD={kld_:.4e}  bs={batch_size}"
                )

    latent = encode_latent(model, features, device, batch_size=max(batch_size, 1024))
    return model, latent


def encode_latent(
    model: VAE,
    features: NormalizedFeatures,
    device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Encode all contigs to their latent ``mu`` (deterministic)."""
    import torch

    model.eval()
    depths = torch.as_tensor(features.depths, device=device)
    tnf = torch.as_tensor(features.tnf, device=device)
    ta = torch.as_tensor(features.total_abundance, device=device)
    n = depths.shape[0]
    out = np.empty((n, model.nlatent), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            mu = model.encode(depths[sl], tnf[sl], ta[sl])
            out[sl] = mu.float().cpu().numpy()
    return out
