"""Generate a tiny synthetic metagenome for integration tests.

Each "genome" gets its own 1st-order Markov chain (distinct tetranucleotide
composition) and a per-sample abundance profile. Contigs inherit both signals,
so a correct binner should recover the genomes from composition + abundance.

Usable as a fixture (``make_synthetic(tmp_path)``) or as a script:

    python tests/make_synthetic.py outdir --genomes 8 --samples 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BASES = "ACGT"


def _markov_genome(rng: np.random.Generator, trans: np.ndarray, length: int) -> str:
    """Generate a sequence from an order-3 Markov chain.

    ``trans`` is a (64, 4) row-stochastic matrix keyed by the previous 3 bases
    (state = b1*16 + b2*4 + b3). Order-3 makes the *tetranucleotide* frequencies
    genome-specific, which is exactly the signal the composition features encode.
    """
    out = list(rng.integers(0, 4, size=3))
    state = out[0] * 16 + out[1] * 4 + out[2]
    for _ in range(length - 3):
        nxt = rng.choice(4, p=trans[state])
        out.append(int(nxt))
        state = (state * 4 + nxt) % 64
    return "".join(BASES[i] for i in out)


def _genome_transition(rng: np.random.Generator) -> np.ndarray:
    # Peaked dirichlet (small alpha) -> sharply genome-specific 4-mer usage.
    return rng.dirichlet(alpha=np.full(4, 0.15), size=64)


def make_synthetic(
    outdir: Path,
    n_genomes: int = 8,
    n_samples: int = 5,
    contigs_per_genome: int = 6,
    contig_len: tuple[int, int] = (3000, 9000),
    seed: int = 0,
) -> dict:
    """Write contigs.fna and per-sample aemb TSVs; return ground-truth metadata."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Per-genome, per-sample latent abundance (genomes bloom in different samples).
    abund = rng.exponential(1.0, size=(n_genomes, n_samples)) * rng.uniform(
        1, 50, size=(n_genomes, 1)
    )

    contig_names: list[str] = []
    contig_genome: list[int] = []
    coverage_rows: list[np.ndarray] = []

    fasta_path = outdir / "contigs.fna"
    with open(fasta_path, "w") as fa:
        for g in range(n_genomes):
            trans = _genome_transition(rng)  # one composition profile per genome
            for c in range(contigs_per_genome):
                length = int(rng.integers(*contig_len))
                seq = _markov_genome(rng, trans, length)
                name = f"G{g}_C{c}"
                fa.write(f">{name}\n{seq}\n")
                contig_names.append(name)
                contig_genome.append(g)
                # Coverage ~ genome abundance * length-independent noise.
                noise = rng.lognormal(0.0, 0.15, size=n_samples)
                coverage_rows.append(abund[g] * noise)

    coverage = np.array(coverage_rows, dtype=np.float32)  # (n_contigs, n_samples)
    tsvs = []
    for s in range(n_samples):
        p = outdir / f"sample{s}.tsv"
        with open(p, "w") as f:
            for name, cov in zip(contig_names, coverage[:, s]):
                f.write(f"{name}\t{cov:.4f}\n")
        tsvs.append(p)

    return {
        "fasta": fasta_path,
        "tsvs": tsvs,
        "contig_names": contig_names,
        "contig_genome": np.array(contig_genome),
        "n_genomes": n_genomes,
        "n_samples": n_samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--genomes", type=int, default=8)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--contigs-per-genome", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = make_synthetic(
        args.outdir,
        n_genomes=args.genomes,
        n_samples=args.samples,
        contigs_per_genome=args.contigs_per_genome,
        seed=args.seed,
    )
    print(f"Wrote {len(meta['contig_names'])} contigs from {meta['n_genomes']} genomes "
          f"and {meta['n_samples']} sample TSVs to {args.outdir}")


if __name__ == "__main__":
    main()
