"""
Embedding service for converting text into dense vector representations.

Key design decisions:
  - The local SentenceTransformer model is loaded once in ``__init__`` (the
    service is a singleton managed by ``ServiceContainer``).
  - ``SentenceTransformer.encode`` is a CPU-bound blocking call.  It is
    wrapped with ``asyncio.to_thread`` so it never stalls the event loop.
  - The OpenAI embedding path uses the current ``openai>=1.0`` async client
    (``AsyncOpenAI``), replacing the removed ``openai.Embedding.acreate``.
"""
import asyncio
import logging
from typing import Any, Dict, List, Union

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = structlog.get_logger(__name__)

# Magic number constants
_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_OPENAI_EMBEDDING_DIM = 1536


class EmbeddingService:
    """
    Generate text embeddings using a local SentenceTransformer model.

    An optional OpenAI fallback is available when ``OPENAI_API_KEY`` is set.
    The local model is always the default because it requires no network call
    and keeps all data on-premises.
    """

    def __init__(self) -> None:
        self.model_name: str = settings.embedding_model
        self.dimension: int = settings.embedding_dimension
        self._local_model: SentenceTransformer = self._load_local_model()

        # Build the async OpenAI client once if a key is available.
        self._openai_client = None
        if settings.openai_api_key:
            try:
                from openai import AsyncOpenAI

                self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI async embedding client ready")
            except ImportError:
                logger.warning("openai package not installed — OpenAI embeddings disabled")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_local_model(self) -> SentenceTransformer:
        """Load the SentenceTransformer model synchronously (done once at startup)."""
        logger.info("Loading local embedding model", model=self.model_name)
        model = SentenceTransformer(self.model_name)
        logger.info("Local embedding model ready", model=self.model_name)
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed_text(
        self,
        text: Union[str, List[str]],
        model_provider: str = "local",
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for one or more text inputs.

        Args:
            text: A single string or a list of strings.
            model_provider: ``"local"`` (default) or ``"openai"``.

        Returns:
            A single embedding vector (``List[float]``) when *text* is a
            string, or a list of vectors when *text* is a list.
        """
        single = isinstance(text, str)
        texts: List[str] = [text] if single else text  # type: ignore[list-item]

        if model_provider == "openai" and self._openai_client is not None:
            embeddings = await self._embed_with_openai(texts)
        else:
            embeddings = await self._embed_with_local_model(texts)

        return embeddings[0] if single else embeddings

    async def embed_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        model_provider: str = "local",
    ) -> List[Dict[str, Any]]:
        """
        Embed all text chunks from a document and attach the vectors in-place.

        Args:
            chunks: Dicts produced by :meth:`chunk_text_for_embedding`.
            model_provider: Embedding backend to use.

        Returns:
            The same list of dicts, each augmented with ``embedding``,
            ``embedding_model``, and ``embedding_dimension`` keys.
        """
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        embeddings: List[List[float]] = await self.embed_text(texts, model_provider)  # type: ignore[assignment]

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
            chunk["embedding_model"] = self.model_name
            chunk["embedding_dimension"] = len(embedding)

        return chunks

    def chunk_text_for_embedding(
        self,
        text: str,
        max_chunk_size: int = 512,
        overlap_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Split *text* into overlapping character-level chunks.

        Chunks are broken at word boundaries when possible to avoid splitting
        tokens mid-word.

        Args:
            text: Source text to split.
            max_chunk_size: Maximum characters per chunk (default 512).
            overlap_size: Characters of overlap between consecutive chunks (default 50).

        Returns:
            List of dicts with keys: ``text``, ``chunk_index``, ``start_char``,
            ``end_char``, ``chunk_size``.
        """
        if not text:
            return []

        if len(text) <= max_chunk_size:
            return [
                {
                    "text": text,
                    "chunk_index": 0,
                    "start_char": 0,
                    "end_char": len(text),
                    "chunk_size": len(text),
                }
            ]

        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + max_chunk_size, len(text))

            # Walk back to a word boundary so we don't split mid-word.
            if end < len(text):
                boundary = max(start + max_chunk_size - 100, start + 1)
                for i in range(end, boundary, -1):
                    if text[i] == " ":
                        end = i
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                        "chunk_size": len(chunk_text),
                    }
                )
                chunk_index += 1

            # Advance with overlap
            start = max(end - overlap_size, end)
            if start >= len(text):
                break

        return chunks

    def get_embedding_dimension(self, model_provider: str = "local") -> int:
        """Return the embedding dimensionality for the given provider."""
        if model_provider == "openai":
            return _OPENAI_EMBEDDING_DIM
        return self.dimension

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute the cosine similarity between two embedding vectors.

        Returns 0.0 for zero-norm vectors to avoid division by zero.
        """
        vec1 = np.asarray(embedding1, dtype=np.float32)
        vec2 = np.asarray(embedding2, dtype=np.float32)

        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _embed_with_local_model(self, texts: List[str]) -> List[List[float]]:
        """
        Run SentenceTransformer inference off the event loop.

        ``encode`` is a CPU-bound operation.  Wrapping it in
        ``asyncio.to_thread`` prevents it from blocking the async event loop
        under concurrent load.
        """
        embeddings = await asyncio.to_thread(
            self._local_model.encode,
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()  # type: ignore[union-attr]

    async def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """
        Call the OpenAI Embeddings API (openai>=1.0 async client).

        Falls back to the local model if the API call fails.
        """
        assert self._openai_client is not None
        try:
            response = await self._openai_client.embeddings.create(
                input=texts,
                model=_OPENAI_EMBEDDING_MODEL,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.warning(
                "OpenAI embedding failed, falling back to local model",
                error=str(exc),
            )
            return await self._embed_with_local_model(texts)