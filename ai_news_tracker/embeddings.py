"""Content analysis using sentence embeddings."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Generates embeddings for articles using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       "all-MiniLM-L6-v2" is fast and good for semantic search.
                       "all-mpnet-base-v2" is more accurate but slower.
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_article(self, title: str, content: str | None = None, summary: str | None = None) -> np.ndarray:
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
        return self.embed_text(combined)

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


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize numpy embedding to bytes for database storage."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dim: int = 384) -> np.ndarray:
    """Deserialize bytes back to numpy embedding."""
    return np.frombuffer(data, dtype=np.float32).reshape(dim)
