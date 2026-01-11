"""Tests for the embedding engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_news_tracker.embeddings import (
    EmbeddingEngine,
    LRUCache,
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


class TestLRUCache:
    """Tests for the LRU cache implementation."""

    def test_cache_basic_operations(self):
        """Test basic get/put operations."""
        cache = LRUCache(maxsize=3)

        # Initially empty
        assert cache.get("key1") is None

        # Add and retrieve
        arr = np.array([1.0, 2.0, 3.0])
        cache.put("key1", arr)
        result = cache.get("key1")
        np.testing.assert_array_equal(result, arr)

    def test_cache_eviction(self):
        """Test that oldest items are evicted when full."""
        cache = LRUCache(maxsize=2)

        cache.put("key1", np.array([1.0]))
        cache.put("key2", np.array([2.0]))
        cache.put("key3", np.array([3.0]))  # This should evict key1

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_cache_lru_order(self):
        """Test that accessing items updates their position."""
        cache = LRUCache(maxsize=2)

        cache.put("key1", np.array([1.0]))
        cache.put("key2", np.array([2.0]))

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key3, should evict key2 (least recently used)
        cache.put("key3", np.array([3.0]))

        assert cache.get("key1") is not None  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = LRUCache(maxsize=5)

        cache.put("key1", np.array([1.0]))
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["maxsize"] == 5
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = LRUCache(maxsize=5)

        cache.put("key1", np.array([1.0]))
        cache.put("key2", np.array([2.0]))
        cache.get("key1")

        cache.clear()

        # Stats should be reset
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Keys should no longer exist (these calls add to misses)
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestEmbeddingCaching:
    """Tests for embedding caching functionality."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer that tracks calls."""
        with patch("ai_news_tracker.embeddings.SentenceTransformer") as mock_class:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.call_count = 0

            def encode(text, convert_to_numpy=True, show_progress_bar=False):
                mock_model.call_count += 1
                np.random.seed(hash(text) % (2**32))
                return np.random.randn(384).astype(np.float32)

            mock_model.encode = encode
            mock_class.return_value = mock_model
            yield mock_model

    def test_caching_reduces_model_calls(self, mock_sentence_transformer):
        """Test that caching reduces actual model calls."""
        engine = EmbeddingEngine()

        # First call - should hit the model
        engine.embed_text("Hello world")
        assert mock_sentence_transformer.call_count == 1

        # Second call with same text - should use cache
        engine.embed_text("Hello world")
        assert mock_sentence_transformer.call_count == 1  # No new call

        # Different text - should hit the model
        engine.embed_text("Different text")
        assert mock_sentence_transformer.call_count == 2

    def test_cache_can_be_disabled(self, mock_sentence_transformer):
        """Test that caching can be disabled."""
        engine = EmbeddingEngine()

        engine.embed_text("Hello world", use_cache=False)
        engine.embed_text("Hello world", use_cache=False)

        # Both calls should hit the model
        assert mock_sentence_transformer.call_count == 2

    def test_cache_stats_accessible(self, mock_sentence_transformer):
        """Test that cache stats are accessible."""
        engine = EmbeddingEngine()

        engine.embed_text("Text 1")
        engine.embed_text("Text 1")  # Cache hit
        engine.embed_text("Text 2")

        stats = engine.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2

    def test_cache_clear(self, mock_sentence_transformer):
        """Test clearing the cache."""
        engine = EmbeddingEngine()

        engine.embed_text("Text 1")
        engine.clear_cache()

        stats = engine.cache_stats()
        assert stats["size"] == 0

    def test_cache_size_configurable(self, mock_sentence_transformer):
        """Test that cache size is configurable."""
        engine = EmbeddingEngine(cache_size=2)

        engine.embed_text("Text 1")
        engine.embed_text("Text 2")
        engine.embed_text("Text 3")  # Should evict Text 1

        stats = engine.cache_stats()
        assert stats["size"] == 2
        assert stats["maxsize"] == 2
