"""VAE loss: weighted reconstruction (depths CE + TNF SSE + abundance SSE) + KLD.

Weighting follows VAMB's calc_loss: ``alpha`` trades composition against
abundance, ``beta`` scales the KLD, and the cross-entropy term is disabled for
single-sample data (where the depth distribution carries no information). The
optional SCG cannot-link term (M6) is added by the training loop, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .vae import NTNF


@dataclass
class LossComponents:
    total: torch.Tensor
    ce: torch.Tensor
    ab: torch.Tensor
    sse: torch.Tensor
    kld: torch.Tensor


def vae_loss(
    depths_in, depths_out,
    tnf_in, tnf_out,
    ab_in, ab_out,
    mu, weights,
    *, alpha: float, beta: float, nsamples: int, nlatent: int,
) -> LossComponents:
    # Per-contig reconstruction error of each modality.
    ab_sse = (ab_out - ab_in).pow(2).sum(dim=1)
    ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1)
    sse = (tnf_out - tnf_in).pow(2).sum(dim=1)
    kld = 0.5 * mu.pow(2).sum(dim=1)

    if nsamples == 1:
        ce_weight = 0.0
    else:
        ce_weight = ((1 - alpha) * (nsamples - 1)) / (nsamples * math.log(nsamples))
    ab_weight = (1 - alpha) * (1 / nsamples)
    sse_weight = alpha / NTNF
    kld_weight = 1 / (nlatent * beta)

    w_ce = ce * ce_weight
    w_ab = ab_sse * ab_weight
    w_sse = sse * sse_weight
    w_kld = kld * kld_weight

    reconstruction = w_ce + w_ab + w_sse
    loss = ((reconstruction + w_kld) * weights.squeeze(1)).mean()

    return LossComponents(
        total=loss,
        ce=w_ce.mean(),
        ab=w_ab.mean(),
        sse=w_sse.mean(),
        kld=w_kld.mean(),
    )
