"""CheckM2-guided decontamination of bins (opt-in, GPU-accelerated splitting).

The internal SCG scorer (``markers/scg.py``) drives the *default* refinement loop
because it is fast enough to run per candidate split. It is, however, a coarse
universal-marker heuristic that can badly under-estimate contamination. This
module offers a stronger, opt-in pass: split the bins CheckM2 flags as
contaminated and **keep a split only if CheckM2 confirms it materially improves
quality**.

Per guided round (kept cheap -- one CheckM2 call scores all candidates at once):

1. Score the current bins with CheckM2 (round 0 reuses the bins already on disk).
2. Pick "contaminated" bins (contamination >= threshold, completeness high enough
   to be worth saving).
3. For each, **propose several candidate splits** on the GPU (SCG-seeded
   length-weighted *and* unweighted KMeans, plus farthest-point k=2..max_k),
   reusing ``cluster/refine.py``'s ``_kmeans_seeds`` / ``_weighted_kmeans``.
4. CheckM2-score every candidate sub-bin, then for each parent pick the best
   partition and accept it iff its children beat the parent's DAS-Tool score
   (``completeness - w * contamination``) by at least ``min_gain``. The margin
   rejects cosmetic "peel one contig" splits, which leave contamination untouched
   and so barely change the score.
5. Recurse into still-contaminated children on the next round (``--iters``).

GPU note: only the split *math* runs on the GPU (pure torch). CheckM2 itself is
CPU-bound (DIAMOND); nothing here changes that. CheckM2 is mocked in the tests so
the propose -> score -> accept logic is unit-tested without the external tool.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..cluster.refine import _kmeans_seeds, _weighted_kmeans
from ..io.write import write_bins
from ..markers.scg import Markers
from ..utils import logger
from .checkm2 import parse_quality_rows, run_checkm2


def quality_score(comp_pct: float, cont_pct: float, weight: float = 5.0) -> float:
    """DAS-Tool-style bin score: completeness minus a contamination penalty.

    Both inputs are CheckM2's native 0-100 percentages; the score is on the same
    scale. ``weight`` (default 5) is how many completeness points one point of
    contamination costs.
    """
    return comp_pct - weight * cont_pct


def _row_cc(row: dict) -> tuple[float, float]:
    """(completeness%, contamination%) from a CheckM2 report row."""
    return float(row["Completeness"]), float(row["Contamination"])


# --------------------------------------------------------------------------- #
# Split proposals (GPU)
# --------------------------------------------------------------------------- #
def _groups_from_assign(idxs: list[int], assign: np.ndarray) -> list[list[int]]:
    """Turn a per-contig cluster assignment into a list of (non-empty) index groups."""
    return [
        [idxs[j] for j in np.nonzero(assign == c)[0]] for c in np.unique(assign)
    ]


def _dedup_partitions(parts: list[list[list[int]]]) -> list[list[list[int]]]:
    """Drop partitions that are identical as sets of groups."""
    seen, out = set(), []
    for p in parts:
        key = frozenset(frozenset(g) for g in p)
        if len(p) >= 2 and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _farthest_seeds(idx_t, latent_t, lengths_t, k: int) -> Optional[list[int]]:
    """Pick k spread-out seed contigs (kmeans++-style) as global indices.

    Seeds the KMeans for bins where the SCGs don't pinpoint the duplication (or to
    try a different k). Returns ``None`` if fewer than 2 distinct points exist.
    """
    import torch

    X = latent_t[idx_t]
    w = lengths_t[idx_t]
    if X.shape[0] < 2:
        return None
    centroid = (X * w.unsqueeze(1)).sum(0) / w.sum()
    chosen = [int(torch.cdist(centroid.unsqueeze(0), X).squeeze(0).argmax())]
    while len(chosen) < k:
        d = torch.cdist(X, X[torch.as_tensor(chosen, device=X.device)]).min(dim=1).values
        d[torch.as_tensor(chosen, device=X.device)] = -1.0
        nxt = int(d.argmax())
        if float(d[nxt]) <= 0.0:
            break  # remaining points coincide with chosen seeds
        chosen.append(nxt)
    if len(chosen) < 2:
        return None
    return [int(idx_t[c]) for c in chosen]


def propose_splits(
    idxs: list[int],
    latent_t,
    lengths_t,
    lengths: np.ndarray,
    markers: Markers,
    device,
    *,
    max_k: int = 3,
) -> list[list[list[int]]]:
    """Propose several candidate splits of one bin (each a list of contig groups).

    Strategies (deduplicated): SCG-seeded length-weighted KMeans (the original
    behaviour), SCG-seeded *unweighted* KMeans (so a few huge contigs can't swamp
    the split), and farthest-point seeded k=2..max_k. CheckM2 later picks the best.
    Returns ``[]`` when no viable split exists.
    """
    import torch

    if len(idxs) < 2:
        return []
    counts = markers.counts(idxs)
    median = int(np.sort(counts)[len(counts) // 2]) if len(counts) else 0
    idx_t = torch.as_tensor(idxs, device=device)
    X = latent_t[idx_t]
    w = lengths_t[idx_t]
    ones = torch.ones_like(w)
    partitions: list[list[list[int]]] = []

    def attempt(seeds: Optional[list[int]], weights) -> None:
        if not seeds or len(seeds) < 2:
            return
        seed_t = torch.as_tensor(seeds, device=device)
        assign = _weighted_kmeans(X, latent_t[seed_t], weights).cpu().numpy()
        groups = [g for g in _groups_from_assign(idxs, assign) if g]
        if len(groups) >= 2:
            partitions.append(groups)

    if median >= 2:
        scg = _kmeans_seeds(idxs, markers, lengths, counts, median)
        attempt(scg, w)      # length-weighted (original)
        attempt(scg, ones)   # unweighted
    for k in sorted({2, min(max(median, 2), max_k)}):
        attempt(_farthest_seeds(idx_t, latent_t, lengths_t, k), ones)

    return _dedup_partitions(partitions)


# --------------------------------------------------------------------------- #
# CheckM2 scoring of candidate groups
# --------------------------------------------------------------------------- #
def _score_groups(cfg, comp, groups: dict[int, list[int]]) -> dict[int, dict]:
    """Write each group as a FASTA bin to a temp dir and CheckM2-score them.

    ``groups`` maps an arbitrary integer label -> contig indices. Returns
    ``{label: CheckM2 row dict}`` for every group CheckM2 scored. Reuses
    ``io.write.write_bins`` (streams sequences from the input FASTA) and
    ``qc.checkm2.run_checkm2``.
    """
    n = comp.n_contigs
    labels_arr = np.full(n, -1, dtype=np.int64)
    for lab, members in groups.items():
        labels_arr[members] = lab

    with tempfile.TemporaryDirectory(prefix="gbin_ckm2_") as td:
        td = Path(td)
        summary = write_bins(
            td, comp.identifiers, comp.lengths, labels_arr, cfg.fasta, min_bin_size=0
        )
        kept: dict[int, str] = summary["kept"]  # {label: bin_name}
        if not kept:
            return {}
        report = run_checkm2(
            td / "bins", td / "checkm2", cfg.num_threads, "fna",
            cfg.checkm2_db, cfg.checkm2_bin,
        )
        name_rows = parse_quality_rows(report)
    return {lab: name_rows[name] for lab, name in kept.items() if name in name_rows}


# --------------------------------------------------------------------------- #
# Guided refinement loop
# --------------------------------------------------------------------------- #
def checkm2_guided_refine(
    cfg,
    comp,
    latent: np.ndarray,
    labels: np.ndarray,
    markers: Markers,
    device,
    initial_bins_dir: Path,
    initial_kept: dict[int, str],
) -> tuple[np.ndarray, dict[int, dict]]:
    """Split CheckM2-contaminated bins; keep splits that improve the DAS score.

    ``initial_bins_dir`` / ``initial_kept`` are the bins already written by the
    caller (round 0 reuses them instead of re-writing). Returns
    ``(new_labels, label_rows)`` where ``label_rows`` maps each final label to its
    CheckM2 report row.
    """
    import torch

    latent_t = torch.as_tensor(latent, dtype=torch.float32, device=device)
    lengths_t = torch.as_tensor(comp.lengths, dtype=torch.float32, device=device)
    weight = cfg.checkm2_refine_score_weight
    min_gain = cfg.checkm2_refine_min_gain
    max_k = cfg.checkm2_refine_max_k

    # Work only on contigs that are in a written bin; everything else is unbinned.
    work = np.full(len(labels), -1, dtype=np.int64)
    for lab in initial_kept:
        work[labels == lab] = lab
    next_label = (int(work.max()) + 1) if work.max() >= 0 else 0

    # Round 0: CheckM2 on the already-written bins.
    logger.info("CheckM2-guided refine: scoring current bins (round 0)")
    report = run_checkm2(
        initial_bins_dir, cfg.outdir / "checkm2", cfg.num_threads, "fna",
        cfg.checkm2_db, cfg.checkm2_bin,
    )
    name_rows = parse_quality_rows(report)
    label_rows: dict[int, dict] = {
        lab: name_rows[name] for lab, name in initial_kept.items() if name in name_rows
    }

    total_split = 0
    for rnd in range(cfg.checkm2_refine_iters):
        members: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(work):
            if lab >= 0:
                members[int(lab)].append(i)

        targets = []
        for lab, row in label_rows.items():
            if lab not in members:
                continue
            comp_pct, cont_pct = _row_cc(row)
            if (cont_pct >= cfg.checkm2_refine_min_contamination
                    and comp_pct >= cfg.checkm2_refine_min_completeness):
                targets.append(lab)
        if not targets:
            logger.info(f"Guided refine round {rnd + 1}: no contaminated bins left")
            break

        # Propose splits (GPU). Identical sub-bins (across a parent's partitions)
        # share one candidate label, so CheckM2 scores each unique sub-bin once.
        candidates: dict[int, list[int]] = {}
        group_label: dict[frozenset, int] = {}
        parent_partitions: dict[int, list[list[int]]] = {}
        for parent in targets:
            parts = propose_splits(
                members[parent], latent_t, lengths_t, comp.lengths, markers,
                device, max_k=max_k,
            )
            plist = []
            for groups in parts:
                clabels = []
                for g in groups:
                    key = frozenset(g)
                    if key not in group_label:
                        group_label[key] = next_label
                        candidates[next_label] = list(g)
                        next_label += 1
                    clabels.append(group_label[key])
                plist.append(clabels)
            if plist:
                parent_partitions[parent] = plist
        if not candidates:
            logger.info(f"Guided refine round {rnd + 1}: no viable splits proposed")
            break

        logger.info(
            f"Guided refine round {rnd + 1}: scoring {len(candidates)} candidate "
            f"sub-bins from {len(parent_partitions)} contaminated bins"
        )
        cand_rows = _score_groups(cfg, comp, candidates)

        round_split = 0
        for parent, plist in parent_partitions.items():
            parent_score = quality_score(*_row_cc(label_rows[parent]), weight)
            best: Optional[tuple[float, list[int]]] = None
            for clabels in plist:
                rows = [cand_rows.get(c) for c in clabels]
                if any(r is None for r in rows):
                    continue
                total = sum(quality_score(*_row_cc(r), weight) for r in rows)
                if total >= parent_score + min_gain and (best is None or total > best[0]):
                    best = (total, clabels)
            if best is None:
                continue  # no partition meaningfully beats the parent; keep it
            for c in best[1]:
                work[candidates[c]] = c
                label_rows[c] = cand_rows[c]
            del label_rows[parent]
            round_split += 1

        total_split += round_split
        logger.info(
            f"Guided refine round {rnd + 1}: kept {round_split}/{len(parent_partitions)} splits"
        )
        if round_split == 0:
            break

    new_labels, label_rows = _relabel_contiguous(work, label_rows)
    logger.info(
        f"CheckM2-guided refine: split {total_split} bins; "
        f"{len(set(new_labels[new_labels >= 0].tolist()))} bins after refinement"
    )
    return new_labels, label_rows


def _relabel_contiguous(
    work: np.ndarray, label_rows: dict[int, dict]
) -> tuple[np.ndarray, dict[int, dict]]:
    """Map surviving labels to 0..K-1 (unbinned stays -1); remap the row map."""
    uniq = sorted(int(u) for u in np.unique(work) if u >= 0)
    remap = {old: new for new, old in enumerate(uniq)}
    new_labels = np.full(len(work), -1, dtype=np.int64)
    for old, new in remap.items():
        new_labels[work == old] = new
    new_rows = {remap[old]: r for old, r in label_rows.items() if old in remap}
    return new_labels, new_rows
