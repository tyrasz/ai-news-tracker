"""Tests for the embedding engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_news_tracker.embeddings import (
    EmbeddingEngine,
    embedding_to_bytes,
    bytes_to_embedding,
)


class TestEmbeddingToBytesConversion:
    """Tests for embedding serialization functions."""

    def test_embedding_to_bytes(self):
        """Test converting numpy array to bytes."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        data = embedding_to_bytes(embedding)

        assert isinstance(data, bytes)
        assert len(data) == 4 * 4  # 4 floats * 4 bytes each

    def test_bytes_to_embedding(self):
        """Test converting bytes back to numpy array."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        data = embedding.tobytes()

        restored = bytes_to_embedding(data, dim=4)

        np.testing.assert_array_almost_equal(embedding, restored)

    def test_roundtrip_conversion(self):
        """Test that conversion is lossless."""
        original = np.random.randn(384).astype(np.float32)
        data = embedding_to_bytes(original)
        restored = bytes_to_embedding(data, dim=384)

        np.testing.assert_array_equal(original, restored)

    def test_different_dimensions(self):
        """Test conversion with various embedding dimensions."""
        for dim in [128, 256, 384, 768, 1024]:
            original = np.random.randn(dim).astype(np.float32)
            data = embedding_to_bytes(original)
            restored = bytes_to_embedding(data, dim=dim)
            np.testing.assert_array_equal(original, restored)


class TestEmbeddingEngine:
    """Tests for the EmbeddingEngine class."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        with patch("ai_news_tracker.embeddings.SentenceTransformer") as mock_class:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384

            def encode(text, convert_to_numpy=True, show_progress_bar=False):
                if isinstance(text, str):
                    np.random.seed(hash(text) % (2**32))
                    return np.random.randn(384).astype(np.float32)
                else:
                    return np.array([
                        np.random.randn(384).astype(np.float32)
                        for _ in text
                    ])

            mock_model.encode = encode
            mock_class.return_value = mock_model
            yield mock_model

    def test_engine_initialization(self, mock_sentence_transformer):
        """Test embedding engine initialization."""
        engine = EmbeddingEngine()

        assert engine.embedding_dim == 384
        assert engine.model is not None

    def test_embed_text(self, mock_sentence_transformer):
        """Test embedding a single text."""
        engine = EmbeddingEngine()
        embedding = engine.embed_text("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_text_deterministic(self, mock_sentence_transformer):
        """Test that same text produces same embedding."""
        engine = EmbeddingEngine()

        emb1 = engine.embed_text("Test text")
        emb2 = engine.embed_text("Test text")

        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_article_title_only(self, mock_sentence_transformer):
        """Test embedding article with only title."""
        engine = EmbeddingEngine()
        embedding = engine.embed_article("Article Title")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_article_with_content(self, mock_sentence_transformer):
        """Test embedding article with title and content."""
        engine = EmbeddingEngine()

        emb_title_only = engine.embed_article("Title")
        emb_with_content = engine.embed_article("Title", content="Some content here")

        # Embeddings should be different when content is added
        assert not np.array_equal(emb_title_only, emb_with_content)

    def test_embed_article_with_summary(self, mock_sentence_transformer):
        """Test embedding article with title and summary."""
        engine = EmbeddingEngine()

        emb_title_only = engine.embed_article("Title")
        emb_with_summary = engine.embed_article("Title", summary="Article summary")

        assert not np.array_equal(emb_title_only, emb_with_summary)

    def test_embed_article_content_over_summary(self, mock_sentence_transformer):
        """Test that content is preferred over summary."""
        engine = EmbeddingEngine()

        emb_content = engine.embed_article("Title", content="Content", summary="Summary")
        emb_summary = engine.embed_article("Title", content=None, summary="Summary")

        # When content is provided, it should be used instead of summary
        assert not np.array_equal(emb_content, emb_summary)

    def test_embed_article_truncates_long_content(self, mock_sentence_transformer):
        """Test that long content is truncated."""
        engine = EmbeddingEngine()

        long_content = "x" * 5000
        embedding = engine.embed_article("Title", content=long_content)

        # Should not raise error and produce valid embedding
        assert embedding.shape == (384,)

    def test_embed_batch(self, mock_sentence_transformer):
        """Test batch embedding of multiple texts."""
        engine = EmbeddingEngine()
        texts = ["Text 1", "Text 2", "Text 3"]

        embeddings = engine.embed_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_cosine_similarity_identical(self, mock_sentence_transformer):
        """Test cosine similarity of identical vectors."""
        engine = EmbeddingEngine()

        vec = np.array([1.0, 2.0, 3.0])
        similarity = engine.cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self, mock_sentence_transformer):
        """Test cosine similarity of orthogonal vectors."""
        engine = EmbeddingEngine()

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = engine.cosine_similarity(vec1, vec2)

        assert abs(similarity) < 1e-6

    def test_cosine_similarity_opposite(self, mock_sentence_transformer):
        """Test cosine similarity of opposite vectors."""
        engine = EmbeddingEngine()

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = engine.cosine_similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 1e-6

    def test_cosine_similarity_range(self, mock_sentence_transformer):
        """Test that cosine similarity is in [-1, 1] range."""
        engine = EmbeddingEngine()

        for _ in range(10):
            vec1 = np.random.randn(384)
            vec2 = np.random.randn(384)
            similarity = engine.cosine_similarity(vec1, vec2)

            assert -1.0 <= similarity <= 1.0

    def test_rank_by_similarity(self, mock_sentence_transformer):
        """Test ranking candidates by similarity."""
        engine = EmbeddingEngine()

        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([
            [0.0, 1.0, 0.0],  # orthogonal
            [1.0, 0.0, 0.0],  # identical
            [0.5, 0.5, 0.0],  # partial match
        ])

        ranking = engine.rank_by_similarity(query, candidates)

        # Index 1 should be first (identical), then 2 (partial), then 0 (orthogonal)
        assert ranking[0] == 1  # identical
        assert ranking[1] == 2  # partial match
        assert ranking[2] == 0  # orthogonal

    def test_rank_by_similarity_returns_indices(self, mock_sentence_transformer):
        """Test that ranking returns valid indices."""
        engine = EmbeddingEngine()

        query = np.random.randn(384)
        candidates = np.random.randn(10, 384)

        ranking = engine.rank_by_similarity(query, candidates)

        assert len(ranking) == 10
        assert set(ranking) == set(range(10))  # All indices present
