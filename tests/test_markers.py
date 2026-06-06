"""Integration tests for pyrodigal + pyhmmer marker detection.

Positive control uses the real bacterial genome fragment bundled with pyrodigal,
so we validate that real SCGs are actually detected (not just that the code
runs). Negative control uses random DNA (no real genes -> no SCGs).
"""

import gzip
import os
import shutil

import numpy as np
import pytest

from gbin.markers.find import compute_markers
from gbin.utils import hash_identifiers


def _bundled_genome(tmp_path, which="100kb"):
    pyrodigal = pytest.importorskip("pyrodigal")
    name = {
        "100kb": "GCF_001457455.1_NCTC11397_genomic_100kb.fna.gz",
        "full": "GCF_001457455.1_NCTC11397_genomic.fna.gz",
    }[which]
    src = os.path.join(os.path.dirname(pyrodigal.__file__), "tests", "data", name)
    if not os.path.exists(src):
        pytest.skip("pyrodigal bundled test genome not found")
    dst = tmp_path / "genome.fna"
    with gzip.open(src, "rt") as f, open(dst, "w") as o:
        shutil.copyfileobj(f, o)
    return dst


def test_detects_scgs_on_real_genome(tmp_path):
    pytest.importorskip("pyhmmer")
    fasta = _bundled_genome(tmp_path, "100kb")
    from gbin.io.fasta import iter_sequences

    ids = [n for n, _ in iter_sequences(fasta)]
    m = compute_markers(fasta, ids, hash_identifiers(ids), n_processes=1)
    assert m.n_seqs == len(ids)
    assert m.n_markers > 50  # ~104 in the bundled HMM set
    comp, cont = m.score_bin(range(len(ids)))
    # A real (clean) genome fragment: some SCGs present, no duplication.
    assert comp > 0.02, f"expected to detect SCGs, completeness={comp}"
    assert cont == 0.0, f"single genome should be uncontaminated, got {cont}"


def test_no_scgs_in_random_dna(tmp_path):
    pytest.importorskip("pyhmmer")
    rng = np.random.default_rng(0)
    fasta = tmp_path / "rand.fna"
    ids = []
    with open(fasta, "w") as f:
        for i in range(4):
            seq = "".join(rng.choice(list("ACGT"), size=5000))
            f.write(f">rand{i}\n{seq}\n")
            ids.append(f"rand{i}")
    m = compute_markers(fasta, ids, hash_identifiers(ids), n_processes=1)
    assert m.n_seqs == 4
    # Random DNA has no real conserved genes -> essentially no SCGs pass cutoff.
    assert all(x is None or x.dtype == np.uint8 for x in m.markers)
    comp, _ = m.score_bin(range(4))
    assert comp < 0.05


def test_contamination_detected_on_duplicated_genome(tmp_path):
    pytest.importorskip("pyhmmer")
    fasta = _bundled_genome(tmp_path, "100kb")
    from gbin.io.fasta import iter_sequences

    # Duplicate the genome under a second name: every present SCG now appears
    # twice -> contamination should jump above zero.
    dup = tmp_path / "dup.fna"
    with open(dup, "w") as out:
        for name, seq in iter_sequences(fasta):
            out.write(f">{name}_a\n{seq}\n>{name}_b\n{seq}\n")
    ids = [n for n, _ in iter_sequences(dup)]
    m = compute_markers(dup, ids, hash_identifiers(ids), n_processes=1)
    comp, cont = m.score_bin(range(len(ids)))
    assert cont > 0.0, "duplicated genome must show contamination"
