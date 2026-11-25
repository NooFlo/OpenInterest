"""Weighted reciprocal-rank fusion with a diversity (MMR-style) pass.

Pure Python, no ML dependencies. This is the layer Pinterest-grade quality
actually comes from at personal scale: per-signal reciprocal-rank fusion plus
demotion of candidates too similar to already-selected ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

DEFAULT_WEIGHTS = {"clip": 1.0, "dino": 1.0, "object": 1.0, "cosave": 0.7}

Scored = tuple[str, float]  # (pin_id, fused_score)


@dataclass
class Candidate:
    pin_id: str
    per_signal_rank: dict[str, int] = field(default_factory=dict)  # signal -> 1-based rank
    per_signal_score: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[str]],
    weights: dict[str, float] | None = None,
    k: float = 60.0,
) -> list[Candidate]:
    """Fuse multiple ranked lists into one via weighted RRF."""
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    candidates: dict[str, Candidate] = {}
    for signal, ranking in rankings.items():
        w = weights.get(signal, 0.0)
        if w <= 0:
            continue
        for rank, pin_id in enumerate(ranking, start=1):
            cand = candidates.setdefault(pin_id, Candidate(pin_id))
            cand.per_signal_rank[signal] = rank
            cand.per_signal_score[signal] = w / (k + rank)
    for cand in candidates.values():
        cand.per_signal_score["__fused__"] = sum(cand.per_signal_score.values())
    return sorted(candidates.values(), key=lambda c: -c.per_signal_score["__fused__"])


def diversity_pass(
    candidates: Sequence[Candidate],
    similarity: Callable[[str, str], float],
    lambda_penalty: float = 0.3,
    max_selected: int = 50,
) -> list[Candidate]:
    """MMR-style reselection: penalize candidates similar to already-selected ones."""
    selected: list[Candidate] = []
    pool = list(candidates)
    while pool and len(selected) < max_selected:
        best, best_val = None, -math.inf
        for i, cand in enumerate(pool):
            fused = cand.per_signal_score.get("__fused__", 0.0)
            if not selected:
                val = fused
            else:
                max_sim = max(similarity(cand.pin_id, s.pin_id) for s in selected)
                val = fused - lambda_penalty * max_sim
            if val > best_val:
                best, best_val = i, val
        selected.append(pool.pop(best))
    return selected


def fuse_and_diversify(
    rankings: dict[str, Sequence[str]],
    similarity: Callable[[str, str], float],
    weights: dict[str, float] | None = None,
    lambda_penalty: float = 0.3,
    limit: int = 50,
) -> list[Candidate]:
    fused = reciprocal_rank_fusion(rankings, weights)
    return diversity_pass(fused, similarity, lambda_penalty, max_selected=limit)
