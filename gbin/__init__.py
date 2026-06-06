"""gbin: GPU-accelerated metagenomic binning.

A hybrid binner that learns a latent embedding of contigs with a self-supervised
variational autoencoder (composition + abundance), refines it with single-copy
marker-gene constraints, and clusters the embedding on the GPU.

The public surface is intentionally small; most functionality is reached through
the ``gbin`` command-line interface (see :mod:`gbin.cli`).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
