"""Variational autoencoder over composition + abundance.

Architecture follows VAMB: a fixed-variance VAE (the posterior variance is fixed
to 1, so the latent is ``mu + N(0, 1)`` during training and ``mu`` at encode
time). Input is ``[depths(S) | tnf(103) | total_abundance(1)]``; the decoder
reconstructs all three, with depths passed through softmax to form a
distribution over samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

NTNF = 103


@dataclass
class VAEHyperParams:
    nsamples: int
    nhiddens: list[int]
    nlatent: int
    dropout: float
    alpha: float
    beta: float


class VAE(nn.Module):
    def __init__(self, hp: VAEHyperParams):
        super().__init__()
        self.hp = hp
        self.nsamples = hp.nsamples
        self.ntnf = NTNF
        self.nlatent = hp.nlatent
        ninput = hp.nsamples + self.ntnf + 1  # +1 for total abundance

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        dims = [ninput] + hp.nhiddens
        for nin, nout in zip(dims[:-1], dims[1:]):
            self.encoder.append(nn.Linear(nin, nout))
            self.encoder_bn.append(nn.BatchNorm1d(nout))
        self.mu = nn.Linear(hp.nhiddens[-1], hp.nlatent)

        # Decoder (mirror)
        self.decoder = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        dims = [hp.nlatent] + hp.nhiddens[::-1]
        for nin, nout in zip(dims[:-1], dims[1:]):
            self.decoder.append(nn.Linear(nin, nout))
            self.decoder_bn.append(nn.BatchNorm1d(nout))
        self.out = nn.Linear(hp.nhiddens[0], ninput)

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(hp.dropout)

    # ----------------------------------------------------------------- #
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer, bn in zip(self.encoder, self.encoder_bn):
            x = bn(self.dropout(self.relu(layer(x))))
        return self.mu(x)

    def _decode(self, z: torch.Tensor):
        for layer, bn in zip(self.decoder, self.decoder_bn):
            z = bn(self.dropout(self.relu(layer(z))))
        recon = self.out(z)
        depths = recon[:, : self.nsamples]
        tnf = recon[:, self.nsamples : self.nsamples + self.ntnf]
        ab = recon[:, self.nsamples + self.ntnf :]
        depths = torch.softmax(depths, dim=1)
        return depths, tnf, ab

    def reparameterize(self, mu: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu)
        return mu

    def forward(self, depths, tnf, abundance):
        x = torch.cat((depths, tnf, abundance), dim=1)
        mu = self._encode(x)
        z = self.reparameterize(mu)
        depths_out, tnf_out, ab_out = self._decode(z)
        return depths_out, tnf_out, ab_out, mu

    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def encode(self, depths, tnf, abundance) -> torch.Tensor:
        """Return the latent ``mu`` (deterministic, no noise)."""
        self.eval()
        x = torch.cat((depths, tnf, abundance), dim=1)
        return self._encode(x)

    def save(self, path) -> None:
        torch.save({"hp": self.hp, "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "VAE":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(blob["hp"])
        model.load_state_dict(blob["state"])
        return model
