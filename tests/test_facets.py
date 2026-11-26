from openinterest.caption import parse_vlm_json
from openinterest.config import Settings
from openinterest.facets import (
    accepted_facets,
    apply_user_edit,
    breadcrumb,
    compose_query,
    promote_facets,
)


def test_parse_vlm_json_extracts_facets_with_confidence():
    content = 'Sure! ```json\n{"object": {"value": "Armchair", "confidence": 0.9}, "style": "mid-century modern", "color": {"value": "walnut brown", "confidence": 0.7}, "era": null, "material": null, "caption": "A chair.", "tags": ["chair"]}\n```'
    out = parse_vlm_json(content)
    assert out["facets"]["object"] == {"value": "armchair", "confidence": 0.9}
    assert out["facets"]["style"] == {"value": "mid-century modern", "confidence": 0.6}
    assert "era" not in out["facets"] and "material" not in out["facets"]
    assert out["tags"] == ["chair"]


def test_promotion_only_above_threshold():
    state = {"facets": {}, "accepted": [], "seed_pins": [], "history": []}
    promote_facets(
        state,
        {"object": {"value": "chair", "confidence": 0.9},
         "material": {"value": "walnut", "confidence": 0.3}},
    )
    assert state["accepted"] == ["object"]
    assert state["facets"]["material"]["value"] == "walnut"  # visible but not accepted


def test_breadcrumb_and_query_compose_canonically():
    state = {"facets": {}, "accepted": [], "seed_pins": [], "history": []}
    promote_facets(
        state,
        {"object": {"value": "chair", "confidence": 0.9},
         "style": {"value": "mid-century modern", "confidence": 0.8},
         "material": {"value": "walnut", "confidence": 0.9}},
    )
    assert breadcrumb(state) == "chair · mid-century modern · walnut"
    assert compose_query(state) == "chair mid-century modern walnut"


def test_user_edit_wins_and_can_reject():
    state = {"facets": {}, "accepted": ["style"], "seed_pins": [], "history": []}
    state["facets"]["style"] = {"value": "gothic", "confidence": 0.8}
    apply_user_edit(state, "style", "art deco")
    assert state["facets"]["style"] == {"value": "art deco", "confidence": 1.0, "user": True}
    apply_user_edit(state, "style", None)
    assert "style" not in state["accepted"] and "style" not in state["facets"]


def test_query_caps_apply():
    s = Settings()
    state = {"facets": {}, "accepted": [], "seed_pins": [], "history": []}
    for i, key in enumerate(["object", "style", "material", "era", "color"]):
        state["facets"][key] = {"value": f"very long facet phrase {i}", "confidence": 1.0}
        state["accepted"].append(key)
    q = compose_query(state, s)
    accepted = accepted_facets(state, s)
    assert len(q.split()) <= s.max_query_words
    capped_values = [f["value"] for f in accepted[: s.facet_query_max_facets]]
    expected = (" ".join(capped_values)).split()[: s.max_query_words]
    assert q.split() == expected
