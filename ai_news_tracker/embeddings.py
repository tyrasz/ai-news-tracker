"""Content analysis using sentence embeddings."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache for embeddings."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        """Get item from cache, returning None if not found."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, key: str, value: np.ndarray) -> None:
        """Add item to cache, evicting oldest if full."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.maxsize:
                    self.cache.popitem(last=False)
                self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }


class EmbeddingEngine:
    """Generates embeddings for articles using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        """
        Initialize the embedding engine.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       "all-MiniLM-L6-v2" is fast and good for semantic search.
                       "all-mpnet-base-v2" is more accurate but slower.
            cache_size: Maximum number of embeddings to cache in memory.
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self._cache = LRUCache(maxsize=cache_size)

    def _hash_text(self, text: str) -> str:
        """Generate hash key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        if use_cache:
            key = self._hash_text(text)
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        embedding = self.model.encode(text, convert_to_numpy=True)

        if use_cache:
            self._cache.put(key, embedding)

        return embedding

    def embed_article(
        self, title: str, content: str | None = None, summary: str | None = None, use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for an article.

        Combines title with available content/summary for richer representation.
        """
        # Build combined text, prioritizing title + content
        parts = [title]

        if content:
            # Use first ~1000 chars of content to stay within model limits
            parts.append(content[:1000])
        elif summary:
            parts.append(summary)

        combined = " ".join(parts)
        return self.embed_text(combined, use_cache=use_cache)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def rank_by_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Rank candidates by similarity to query.

        Returns indices sorted by descending similarity.
        """
        # Normalize for efficient cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidates_norm = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

        # Compute all similarities at once
        similarities = candidates_norm @ query_norm

        # Return indices sorted by descending similarity
        return np.argsort(similarities)[::-1]

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.debug("Embedding cache cleared")


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize numpy embedding to bytes for database storage."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dim: int = 384) -> np.ndarray:
    """Deserialize bytes back to numpy embedding."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
