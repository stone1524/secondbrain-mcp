"""Embedding provider abstraction and the offline fastembed implementation.

The production gateway embedded by spawning a long-lived Python child from Node
and exchanging newline-delimited JSON with **no request ids**. Any timeout left a
late response that could be consumed by the next request; the only mitigation
was killing the process. In a single-language implementation that entire class
of defect disappears — the provider is a plain callable.

The provider is an abstraction because a model swap is the expected next step:
the production pilot found this model weak across the Chinese/English boundary.
A swap must invalidate the index, so every vector carries a fingerprint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class EmbedderUnavailable(RuntimeError):
    """The embedding backend could not be loaded."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    dim: int
    revision: str
    prefix_scheme: str

    @property
    def fingerprint(self) -> str:
        material = f"{self.name}|{self.dim}|{self.revision}|{self.prefix_scheme}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class Embedder(Protocol):
    @property
    def info(self) -> ModelInfo: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class FastEmbedEmbedder:
    """Offline embeddings via fastembed/ONNX.

    Imported lazily: the core gateway must run with no embedding backend at all
    (lexical-only mode), and CI must not require the model cache.
    """

    def __init__(self, model_name: str, *, cache_dir: str | None = None, batch_size: int = 0):
        try:
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise EmbedderUnavailable(
                "fastembed is not installed; install the 'semantic' extra"
            ) from error
        options = {}
        if cache_dir:
            options["cache_dir"] = cache_dir
        if batch_size:
            options["batch_size"] = batch_size
        try:
            self._model = TextEmbedding(model_name=model_name, **options)
        except Exception as error:  # pragma: no cover - model download/cache
            raise EmbedderUnavailable(f"Could not load {model_name}: {error}") from error
        self._name = model_name
        # e5-family models require the query:/passage: prefixes; bge models do not.
        self._prefixes = model_name.lower().startswith(("intfloat/", "e5"))
        sample = np.asarray(next(iter(self._model.embed(["dimension probe"]))))
        self._info = ModelInfo(
            name=model_name,
            dim=int(sample.shape[0]),
            revision="fastembed",
            prefix_scheme="e5" if self._prefixes else "plain",
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        payload = [f"passage: {text}" for text in texts] if self._prefixes else list(texts)
        if not payload:
            return np.zeros((0, self.info.dim), dtype=np.float32)
        return _normalize(np.asarray(list(self._model.embed(payload)), dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        payload = [f"query: {text}" if self._prefixes else text]
        # Callers multiply this against an (n_chunks, dim) matrix, so the query
        # must be 1-D; fastembed returns a (1, dim) batch even for one string.
        return _normalize(np.asarray(list(self._model.embed(payload)), dtype=np.float32))[0]


@dataclass(slots=True)
class LexicalOnlyEmbedder:
    """Null object: keeps call sites free of ``if embedder is None``."""

    _info: ModelInfo = ModelInfo(
        name="lexical-only", dim=0, revision="none", prefix_scheme="none"
    )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((0, 0), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)


def build_embedder(model_name: str, *, enabled: bool, cache_dir: str | None = None) -> Embedder:
    if not enabled:
        return LexicalOnlyEmbedder()
    return FastEmbedEmbedder(model_name, cache_dir=cache_dir)
