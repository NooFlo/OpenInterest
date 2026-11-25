"""Captioning + facet extraction.

Two backends:
- Mistral Pixtral VLM API (preferred: best captions/facets, no local ML cost)
- Free/tags heuristics as fallback when no API key is configured

Facets follow the fixed schema {object, style, material, era, color} with
confidence scores; promotion to accepted constraints is threshold + edit gated.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from .config import Settings

FACET_KEYS = ("object", "style", "material", "era", "color")

FACET_PROMPT = """You are a precise visual-metadata extractor for a search system.
Look at the image and extract facets. Reply with ONLY a JSON object with keys
"object", "style", "material", "era", "color" and "caption".

Rules:
- Each facet value is a short lower-case phrase (1-3 words) or null if not determinable.
- Each facet key maps to {"value": <phrase or null>, "confidence": <0.0-1.0>}.
- "caption": one sentence describing the image.
- "tags": list of 5-10 lowercase descriptive tags.
- Do not invent attributes you cannot see. Confidence must reflect certainty.
Example:
{"object": {"value": "armchair", "confidence": 0.93}, "style": {"value": "mid-century modern", "confidence": 0.71}, "material": {"value": "walnut", "confidence": 0.64}, "era": null, "color": {"value": "brown", "confidence": 0.9}, "caption": "A walnut mid-century armchair", "tags": ["armchair", "walnut", "furniture"]}
"""


class Captioner(Protocol):
    def describe(self, image_url: str) -> dict: ...


class MistralPixtralCaptioner:
    """Caption + facets via Mistral's vision API (pixtral)."""

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.mistral_api_key

    def describe(self, image_url: str) -> dict:
        resp = httpx.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": image_url},
                            {"type": "text", "text": FACET_PROMPT},
                        ],
                    }
                ],
                "temperature": 0.1,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return parse_vlm_json(content)


class HeuristicCaptioner:
    """Keyless fallback: derive light tags from pin title/description text."""

    def __init__(self, settings: Settings | None = None) -> None:
        pass

    def describe(self, image_url: str, title: str = "", description: str = "") -> dict:
        text = f"{title} {description}".lower()
        words = [
            w for w in re.findall(r"[a-z][a-z-]{2,}", text) if w not in ("the", "and", "with", "for")
        ]
        return {
            "caption": title or "",
            "tags": list(dict.fromkeys(words))[:8],
            "facets": {},
        }


def parse_vlm_json(content: str) -> dict:
    """Robustly extract the JSON object from a VLM reply."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("VLM reply contained no JSON object")
    data = json.loads(match.group(0))
    facets = {}
    for key in FACET_KEYS:
        f = data.get(key)
        if isinstance(f, dict) and f.get("value") is not None:
            facets[key] = {
                "value": str(f["value"]).strip().lower(),
                "confidence": float(min(max(f.get("confidence", 0.5), 0.0), 1.0)),
            }
        elif isinstance(f, str) and f:
            facets[key] = {"value": f.strip().lower(), "confidence": 0.6}
    return {
        "caption": str(data.get("caption", "")).strip(),
        "tags": [str(t).strip().lower() for t in (data.get("tags") or []) if str(t).strip()],
        "facets": facets,
    }


def make_captioner(settings: Settings) -> Captioner:
    if settings.has_mistral:
        return MistralPixtralCaptioner(settings)
    return HeuristicCaptioner()
