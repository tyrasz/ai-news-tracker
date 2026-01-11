"""Tests for the neural re-ranker module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_news_tracker.reranker import (
    Reranker,
    RerankerConfig,
    FeedbackDataset,
    TORCH_AVAILABLE,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return [np.random.randn(384).astype(np.float32) for _ in range(20)]


class TestRerankerConfig:
    """Tests for RerankerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RerankerConfig()

        assert config.embedding_dim == 384
        assert config.hidden_dim == 128
        assert config.dropout == 0.2
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.min_samples == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = RerankerConfig(
            embedding_dim=512,
            hidden_dim=256,
            min_samples=20,
        )

        assert config.embedding_dim == 512
        assert config.hidden_dim == 256
        assert config.min_samples == 20


class TestFeedbackDataset:
    """Tests for FeedbackDataset."""

    def test_empty_dataset(self):
        """Test empty dataset."""
        dataset = FeedbackDataset(embedding_dim=384)
        assert len(dataset) == 0

    def test_add_sample(self, sample_embeddings):
        """Test adding samples."""
        dataset = FeedbackDataset(embedding_dim=384)

        dataset.add_sample(sample_embeddings[0], sample_embeddings[1], 1.0)
        assert len(dataset) == 1

        dataset.add_sample(sample_embeddings[2], sample_embeddings[3], 0.0)
        assert len(dataset) == 2

    def test_save_and_load(self, temp_dir, sample_embeddings):
        """Test saving and loading dataset."""
        dataset = FeedbackDataset(embedding_dim=384)
        dataset.add_sample(sample_embeddings[0], sample_embeddings[1], 1.0)
        dataset.add_sample(sample_embeddings[2], sample_embeddings[3], 0.0)

        path = temp_dir / "dataset.json"
        dataset.save(path)

        loaded = FeedbackDataset.load(path)
        assert len(loaded) == 2
        assert loaded.embedding_dim == 384

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_to_tensors(self, sample_embeddings):
        """Test converting to PyTorch tensors."""
        dataset = FeedbackDataset(embedding_dim=384)
        dataset.add_sample(sample_embeddings[0], sample_embeddings[1], 1.0)
        dataset.add_sample(sample_embeddings[2], sample_embeddings[3], 0.0)

        article_t, user_t, labels_t = dataset.to_tensors()

        assert article_t.shape == (2, 384)
        assert user_t.shape == (2, 384)
        assert labels_t.shape == (2,)

    def test_to_tensors_empty(self):
        """Test converting empty dataset to tensors."""
        dataset = FeedbackDataset(embedding_dim=384)
        result = dataset.to_tensors()
        assert result == (None, None, None)


class TestReranker:
    """Tests for the Reranker class."""

    def test_initialization(self, temp_dir):
        """Test reranker initialization."""
        reranker = Reranker(model_dir=temp_dir)

        assert reranker.model_dir == temp_dir
        assert reranker.is_trained is False
        assert len(reranker.dataset) == 0

    def test_add_feedback(self, temp_dir, sample_embeddings):
        """Test adding feedback samples."""
        reranker = Reranker(model_dir=temp_dir)

        reranker.add_feedback(sample_embeddings[0], sample_embeddings[1], True)
        assert len(reranker.dataset) == 1

        reranker.add_feedback(sample_embeddings[2], sample_embeddings[3], False)
        assert len(reranker.dataset) == 2

    def test_can_train_insufficient_samples(self, temp_dir, sample_embeddings):
        """Test can_train with insufficient samples."""
        config = RerankerConfig(min_samples=10)
        reranker = Reranker(model_dir=temp_dir, config=config)

        for i in range(5):
            reranker.add_feedback(sample_embeddings[i], sample_embeddings[i + 5], True)

        assert not reranker.can_train()

    def test_can_train_sufficient_samples(self, temp_dir, sample_embeddings):
        """Test can_train with sufficient samples."""
        config = RerankerConfig(min_samples=5)
        reranker = Reranker(model_dir=temp_dir, config=config)

        for i in range(6):
            reranker.add_feedback(sample_embeddings[i], sample_embeddings[i + 6], i % 2 == 0)

        assert reranker.can_train()

    def test_get_stats(self, temp_dir, sample_embeddings):
        """Test getting statistics."""
        config = RerankerConfig(min_samples=10)
        reranker = Reranker(model_dir=temp_dir, config=config)

        reranker.add_feedback(sample_embeddings[0], sample_embeddings[1], True)

        stats = reranker.get_stats()

        assert stats["is_trained"] is False
        assert stats["feedback_samples"] == 1
        assert stats["can_train"] is False
        assert stats["min_samples_required"] == 10
        assert stats["pytorch_available"] == TORCH_AVAILABLE

    def test_save_and_load_state(self, temp_dir, sample_embeddings):
        """Test saving and loading reranker state."""
        reranker = Reranker(model_dir=temp_dir)
        reranker.add_feedback(sample_embeddings[0], sample_embeddings[1], True)
        reranker.save_state()

        # Create new reranker and verify it loads the state
        reranker2 = Reranker(model_dir=temp_dir)
        assert len(reranker2.dataset) == 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_train(self, temp_dir, sample_embeddings):
        """Test training the reranker."""
        config = RerankerConfig(min_samples=10, epochs=5)
        reranker = Reranker(model_dir=temp_dir, config=config)

        # Add enough samples
        for i in range(15):
            liked = i % 3 != 0  # Mix of likes and dislikes
            reranker.add_feedback(sample_embeddings[i % len(sample_embeddings)],
                                  sample_embeddings[(i + 1) % len(sample_embeddings)],
                                  liked)

        results = reranker.train(verbose=False)

        assert "samples" in results
        assert "best_val_loss" in results
        assert "final_accuracy" in results
        assert reranker.is_trained

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_rerank_untrained(self, temp_dir, sample_embeddings):
        """Test reranking without trained model falls back to original scores."""
        reranker = Reranker(model_dir=temp_dir)

        # Create mock articles with scores
        articles_with_scores = [
            (MagicMock(), 0.9, 0.8, sample_embeddings[0]),
            (MagicMock(), 0.7, 0.6, sample_embeddings[1]),
            (MagicMock(), 0.5, 0.4, sample_embeddings[2]),
        ]

        result = reranker.rerank(articles_with_scores, sample_embeddings[10])

        # Should return original scores since not trained
        assert len(result) == 3
        assert result[0][1] == 0.9
        assert result[1][1] == 0.7
        assert result[2][1] == 0.5


class TestRerankerIntegration:
    """Integration tests for the reranker."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_full_workflow(self, temp_dir, sample_embeddings):
        """Test complete reranker workflow."""
        config = RerankerConfig(min_samples=10, epochs=10)
        reranker = Reranker(model_dir=temp_dir, config=config)

        # 1. Add feedback samples
        for i in range(15):
            liked = i % 2 == 0
            reranker.add_feedback(
                sample_embeddings[i % len(sample_embeddings)],
                sample_embeddings[(i + 5) % len(sample_embeddings)],
                liked,
            )

        # 2. Verify can train
        assert reranker.can_train()

        # 3. Train
        results = reranker.train(verbose=False)
        assert reranker.is_trained

        # 4. Create articles for reranking
        articles = [
            (MagicMock(id=i), 0.5 + i * 0.1, 0.8, sample_embeddings[i])
            for i in range(5)
        ]

        # 5. Rerank
        user_emb = sample_embeddings[10]
        reranked = reranker.rerank(articles, user_emb, blend_weight=0.5)

        # Should have reranked results
        assert len(reranked) == 5
        # Scores should be blended (not exactly original)
        for article, score, freshness in reranked:
            assert isinstance(score, float)
            assert 0 <= score <= 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_persistence(self, temp_dir, sample_embeddings):
        """Test that trained model persists across instances."""
        config = RerankerConfig(min_samples=10, epochs=5)

        # Train first instance
        reranker1 = Reranker(model_dir=temp_dir, config=config)
        for i in range(12):
            reranker1.add_feedback(
                sample_embeddings[i % len(sample_embeddings)],
                sample_embeddings[(i + 3) % len(sample_embeddings)],
                i % 2 == 0,
            )
        reranker1.train(verbose=False)
        assert reranker1.is_trained

        # Load second instance
        reranker2 = Reranker(model_dir=temp_dir)
        assert reranker2.is_trained
        assert len(reranker2.dataset) == 12
