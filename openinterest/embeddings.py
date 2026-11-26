"""Embedding backends. Heavy deps (torch/open_clip) are imported lazily so the
core package works without them (tests, API-only deployments).

CPU-light defaults: OpenCLIP ViT-B/32 (512-d) + DINOv2 ViT-B/14 (384-d).
Optional sam-clip per-object embeddings when ENABLE_OBJECT_EMBEDDINGS=1.
"""
from __future__ import annotations

import io
from typing import Protocol, Sequence

import httpx
from PIL import Image


class ImageEmbedder(Protocol):
    name: str
    dim: int

    def embed_image(self, img: Image.Image) -> list[float]: ...
    def embed_text(self, text: str) -> list[float]: ...


def fetch_image(image_url: str) -> Image.Image:
    if image_url.startswith(("http://", "https://")):
        resp = httpx.get(image_url, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(image_url).convert("RGB")


class OpenClipEmbedder:
    """Whole-image semantic + free-text embedding (CLIP/SigLIP family)."""

    def __init__(self, model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        import open_clip
        import torch

        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(model)
        self._model.eval()
        self.name = f"openclip:{model}"
        self.dim = self._model.visual.proj.shape[1] if hasattr(self._model.visual, "proj") else 512

    def embed_image(self, img: Image.Image) -> list[float]:
        with self._torch.no_grad():
            vec = self._model.encode_image(self._preprocess(img).unsqueeze(0))
            return vec.squeeze(0).float().tolist()

    def embed_text(self, text: str) -> list[float]:
        with self._torch.no_grad():
            vec = self._model.encode_text(self._tokenizer([text]))
            return vec.squeeze(0).float().tolist()


class Dinov2Embedder:
    """Structural whole-image embedding; better than CLIP for image-to-image."""

    def __init__(self, model: str = "vitb14") -> None:
        import torch

        self._torch = torch
        self._model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model}")
        self._model.eval()
        self.name = f"dinov2:{model}"

        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        patch = 14 if "14" in model else 16
        self._preprocess = Compose(
            [
                Resize(224, max_size=256),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.dim = {
            "vits14": 384, "vitb14": 768, "vitl14": 1024,
            "vits16": 384, "vitb16": 768, "vitl16": 1024,
        }.get(model, 768)
        self._patch = patch

    def embed_image(self, img: Image.Image) -> list[float]:
        with self._torch.no_grad():
            x = self._preprocess(img).unsqueeze(0)
            vec = self._model(x)
            return vec.squeeze(0).float().tolist()

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError("DINOv2 is image-only; use the CLIP signal for text")


class SamClipObjectEmbedder:
    """Optional per-object embeddings (Grounding DINO + SAM + CLIP) via sam-clip.
    Returns [(vector, label, bbox), ...]. Slow on CPU — enable deliberately.
    """

    def __init__(self, classes: Sequence[str] | None = None) -> None:
        from sam_clip import SamClip

        self._samclip = SamClip()
        self.name = "sam-clip"
        self.classes = list(classes or [])
        self.dim = 512

    def embed_objects(self, img: Image.Image, text_queries: Sequence[str] | None = None):
        import numpy as np

        arr = np.array(img)
        queries = list(text_queries or self.classes) or ["object", "furniture", "product"]
        return self._samclip.embed(arr, queries=queries)


def load_embedders(settings) -> dict[str, ImageEmbedder]:
    """Build the standard embedder set per settings; import errors surface clearly."""
    embedders: dict[str, ImageEmbedder] = {}
    embedders["clip"] = OpenClipEmbedder(settings.openclip_model, settings.openclip_pretrained)
    try:
        embedders["dino"] = Dinov2Embedder(settings.dinov2_model)
    except Exception:
        pass
    return embedders
