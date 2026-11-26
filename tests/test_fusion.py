from openinterest.fusion import fuse_and_diversify, reciprocal_rank_fusion, Candidate
from openinterest.fusion import diversity_pass


def test_rrf_fuses_multiple_signals_and_prefers_consensus():
    rankings = {
        "clip": ["a", "b", "c"],
        "dino": ["b", "a", "d"],
        "cosave": ["c", "b"],
    }
    fused = reciprocal_rank_fusion(rankings, {"clip": 1.0, "dino": 1.0, "cosave": 0.7})
    ids = [c.pin_id for c in fused]
    assert ids[0] == "b"  # present in all three signals
    assert "a" in ids and "c" in ids and "d" in ids


def test_rrf_respects_zero_weights():
    rankings = {"clip": ["a"], "dino": ["b"]}
    fused = reciprocal_rank_fusion(rankings, {"clip": 1.0, "dino": 0.0})
    assert [c.pin_id for c in fused] == ["a"]


def test_diversity_penalizes_similar_to_selected():
    c1 = Candidate("x", per_signal_score={"__fused__": 1.0})
    c2 = Candidate("y", per_signal_score={"__fused__": 0.9})
    c3 = Candidate("z", per_signal_score={"__fused__": 0.7})

    # y is very similar to x; z is dissimilar from everything
    sim = lambda a, b: 0.9 if {a, b} == {"x", "y"} else 0.0
    out = diversity_pass([c1, c2, c3], sim, lambda_penalty=0.3)
    assert [c.pin_id for c in out] == ["x", "z", "y"]


def test_fuse_and_diversify_end_to_end():
    rankings = {"clip": ["a", "b"], "dino": ["b", "c"], "cosave": ["a"]}
    out = fuse_and_diversify(rankings, lambda a, b: 0.0, limit=5)
    assert len(out) == 3
    assert out[0].pin_id in ("a", "b")
    assert all("__fused__" in c.per_signal_score for c in out)
