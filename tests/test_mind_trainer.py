"""Tests for the MIND training module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_news_tracker.mind_trainer import (
    MINDConfig,
    MINDDataset,
    NewsArticle,
    UserBehavior,
    TORCH_AVAILABLE,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_news():
    """Create sample news articles."""
    return {
        "N1": NewsArticle(
            news_id="N1",
            category="tech",
            subcategory="ai",
            title="AI breakthrough in machine learning",
            abstract="New advances in neural networks",
        ),
        "N2": NewsArticle(
            news_id="N2",
            category="sports",
            subcategory="football",
            title="Team wins championship game",
            abstract="Exciting final match",
        ),
        "N3": NewsArticle(
            news_id="N3",
            category="tech",
            subcategory="security",
            title="New security vulnerability discovered",
            abstract="Critical patch released",
        ),
    }


@pytest.fixture
def sample_behaviors():
    """Create sample user behaviors."""
    return [
        UserBehavior(
            impression_id="I1",
            user_id="U1",
            history=["N1", "N2"],
            impressions=[("N3", 1), ("N1", 0)],
        ),
        UserBehavior(
            impression_id="I2",
            user_id="U2",
            history=["N2"],
            impressions=[("N1", 1), ("N3", 0)],
        ),
    ]


class TestMINDConfig:
    """Tests for MINDConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MINDConfig()

        assert config.dataset_size == "small"
        assert config.max_title_len == 30
        assert config.max_history_len == 50
        assert config.embedding_dim == 256  # Must be divisible by num_attention_heads
        assert config.num_attention_heads == 16
        assert config.epochs == 5

    def test_custom_config(self, temp_dir):
        """Test custom configuration."""
        config = MINDConfig(
            data_dir=temp_dir,
            dataset_size="large",
            epochs=10,
        )

        assert config.data_dir == temp_dir
        assert config.dataset_size == "large"
        assert config.epochs == 10


class TestNewsArticle:
    """Tests for NewsArticle dataclass."""

    def test_article_creation(self):
        """Test creating a news article."""
        article = NewsArticle(
            news_id="N123",
            category="tech",
            subcategory="ai",
            title="Test Title",
            abstract="Test abstract",
        )

        assert article.news_id == "N123"
        assert article.category == "tech"
        assert article.title == "Test Title"
        assert article.title_tokens == []  # Default empty

    def test_article_with_tokens(self):
        """Test article with tokenized title."""
        article = NewsArticle(
            news_id="N123",
            category="tech",
            subcategory="ai",
            title="Test Title",
            abstract="Test abstract",
            title_tokens=[1, 2, 3, 0, 0],
        )

        assert article.title_tokens == [1, 2, 3, 0, 0]


class TestUserBehavior:
    """Tests for UserBehavior dataclass."""

    def test_behavior_creation(self):
        """Test creating user behavior."""
        behavior = UserBehavior(
            impression_id="I1",
            user_id="U1",
            history=["N1", "N2", "N3"],
            impressions=[("N4", 1), ("N5", 0)],
        )

        assert behavior.impression_id == "I1"
        assert behavior.user_id == "U1"
        assert len(behavior.history) == 3
        assert len(behavior.impressions) == 2

    def test_behavior_empty_history(self):
        """Test behavior with empty history."""
        behavior = UserBehavior(
            impression_id="I1",
            user_id="U1",
            history=[],
            impressions=[("N1", 1)],
        )

        assert behavior.history == []


class TestMINDDataset:
    """Tests for MINDDataset."""

    def test_dataset_initialization(self, temp_dir):
        """Test dataset initialization."""
        config = MINDConfig(data_dir=temp_dir)
        dataset = MINDDataset(config)

        assert dataset.config == config
        assert dataset.data_dir == temp_dir
        assert len(dataset.news) == 0
        assert len(dataset.word2idx) == 2  # PAD and UNK

    def test_build_vocab(self, temp_dir, sample_news):
        """Test vocabulary building."""
        config = MINDConfig(data_dir=temp_dir)
        dataset = MINDDataset(config)

        dataset.build_vocab(sample_news, min_freq=1)

        # Should have PAD, UNK, and words from titles
        assert len(dataset.word2idx) > 2
        assert "<PAD>" in dataset.word2idx
        assert "<UNK>" in dataset.word2idx
        assert dataset.word2idx["<PAD>"] == 0
        assert dataset.word2idx["<UNK>"] == 1

    def test_tokenize_title(self, temp_dir, sample_news):
        """Test title tokenization."""
        config = MINDConfig(data_dir=temp_dir, max_title_len=10)
        dataset = MINDDataset(config)

        # Build vocab first
        dataset.build_vocab(sample_news, min_freq=1)

        # Tokenize
        tokens = dataset.tokenize_title("AI breakthrough in learning")

        assert len(tokens) == 10  # Padded to max length
        assert tokens[0] == dataset.word2idx.get("ai", dataset.word2idx["<UNK>"])

    def test_tokenize_title_unknown_words(self, temp_dir):
        """Test tokenization with unknown words."""
        config = MINDConfig(data_dir=temp_dir, max_title_len=5)
        dataset = MINDDataset(config)

        # Empty vocab (only PAD and UNK)
        tokens = dataset.tokenize_title("hello world")

        # All should be UNK
        assert tokens[0] == dataset.word2idx["<UNK>"]
        assert tokens[1] == dataset.word2idx["<UNK>"]

    def test_tokenize_title_truncation(self, temp_dir, sample_news):
        """Test that long titles are truncated."""
        config = MINDConfig(data_dir=temp_dir, max_title_len=3)
        dataset = MINDDataset(config)

        dataset.build_vocab(sample_news, min_freq=1)
        tokens = dataset.tokenize_title("one two three four five")

        assert len(tokens) == 3

    def test_save_and_load_vocab(self, temp_dir, sample_news):
        """Test vocabulary save and load."""
        config = MINDConfig(data_dir=temp_dir)
        dataset = MINDDataset(config)

        dataset.build_vocab(sample_news, min_freq=1)
        vocab_path = temp_dir / "vocab.json"
        dataset.save_vocab(vocab_path)

        # Load into new dataset
        dataset2 = MINDDataset(config)
        dataset2.load_vocab(vocab_path)

        assert dataset2.word2idx == dataset.word2idx

    def test_tokenize_all_news(self, temp_dir, sample_news):
        """Test tokenizing all news articles."""
        config = MINDConfig(data_dir=temp_dir, max_title_len=10)
        dataset = MINDDataset(config)
        dataset.news = sample_news

        dataset.build_vocab(sample_news, min_freq=1)
        dataset.tokenize_all_news()

        for article in dataset.news.values():
            assert len(article.title_tokens) == 10
            assert all(isinstance(t, int) for t in article.title_tokens)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMINDModels:
    """Tests for MIND neural network models."""

    def test_multi_head_attention(self):
        """Test MultiHeadSelfAttention module."""
        from ai_news_tracker.mind_trainer import MultiHeadSelfAttention
        import torch

        # attention_dim must be divisible by num_heads
        attention = MultiHeadSelfAttention(
            input_dim=256,
            num_heads=16,
            attention_dim=256,  # 256 / 16 = 16 per head
        )

        x = torch.randn(4, 30, 256)  # (batch, seq, dim)
        output = attention(x)

        assert output.shape == x.shape

    def test_additive_attention(self):
        """Test AdditiveAttention module."""
        from ai_news_tracker.mind_trainer import AdditiveAttention
        import torch

        attention = AdditiveAttention(input_dim=300, attention_dim=200)

        x = torch.randn(4, 30, 300)  # (batch, seq, dim)
        output = attention(x)

        assert output.shape == (4, 300)

    def test_news_encoder(self):
        """Test NewsEncoder module."""
        from ai_news_tracker.mind_trainer import NewsEncoder
        import torch

        config = MINDConfig()
        encoder = NewsEncoder(config, vocab_size=10000)

        # Input: token indices
        tokens = torch.randint(0, 10000, (4, 30))  # (batch, max_title_len)
        output = encoder(tokens)

        assert output.shape == (4, config.embedding_dim)

    def test_user_encoder(self):
        """Test UserEncoder module."""
        from ai_news_tracker.mind_trainer import UserEncoder, NewsEncoder
        import torch

        config = MINDConfig()
        news_encoder = NewsEncoder(config, vocab_size=10000)
        user_encoder = UserEncoder(config, news_encoder)

        # Input: history of articles
        history = torch.randint(0, 10000, (4, 50, 30))  # (batch, history, title_len)
        mask = torch.ones(4, 50)  # (batch, history)
        output = user_encoder(history, mask)

        assert output.shape == (4, config.embedding_dim)

    def test_nrms_model(self):
        """Test full NRMS model."""
        from ai_news_tracker.mind_trainer import NRMSModel
        import torch

        config = MINDConfig()
        model = NRMSModel(config, vocab_size=10000)

        # Inputs
        candidates = torch.randint(0, 10000, (4, 5, 30))  # (batch, candidates, title)
        history = torch.randint(0, 10000, (4, 50, 30))  # (batch, history, title)
        history_mask = torch.ones(4, 50)

        scores = model(candidates, history, history_mask)

        assert scores.shape == (4, 5)  # (batch, num_candidates)

    def test_nrms_get_embeddings(self):
        """Test NRMS embedding extraction."""
        from ai_news_tracker.mind_trainer import NRMSModel
        import torch

        config = MINDConfig()
        model = NRMSModel(config, vocab_size=10000)

        # Get news embedding
        tokens = torch.randint(0, 10000, (1, 30))
        news_emb = model.get_news_embedding(tokens)
        assert news_emb.shape == (1, config.embedding_dim)

        # Get user embedding
        history = torch.randint(0, 10000, (1, 10, 30))
        mask = torch.ones(1, 10)
        user_emb = model.get_user_embedding(history, mask)
        assert user_emb.shape == (1, config.embedding_dim)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMINDTrainingDataset:
    """Tests for MINDTrainingDataset."""

    def test_dataset_creation(self, sample_news, sample_behaviors):
        """Test creating training dataset."""
        from ai_news_tracker.mind_trainer import MINDTrainingDataset

        # Tokenize news first
        for article in sample_news.values():
            article.title_tokens = [1, 2, 3] + [0] * 27  # 30 tokens

        config = MINDConfig(negative_samples=1)
        dataset = MINDTrainingDataset(
            sample_behaviors,
            sample_news,
            config,
            negative_samples=1,
        )

        assert len(dataset) > 0

    def test_dataset_getitem(self, sample_news, sample_behaviors):
        """Test getting items from dataset."""
        from ai_news_tracker.mind_trainer import MINDTrainingDataset

        # Tokenize news
        for article in sample_news.values():
            article.title_tokens = list(range(30))

        config = MINDConfig(negative_samples=1, max_history_len=10)
        dataset = MINDTrainingDataset(
            sample_behaviors,
            sample_news,
            config,
            negative_samples=1,
        )

        if len(dataset) > 0:
            item = dataset[0]

            assert "history_tokens" in item
            assert "history_mask" in item
            assert "candidate_tokens" in item
            assert "labels" in item

            assert item["history_tokens"].shape[0] == config.max_history_len
            assert item["candidate_tokens"].shape[0] == 2  # 1 pos + 1 neg


class TestMINDTrainer:
    """Tests for MINDTrainer class."""

    def test_trainer_initialization(self, temp_dir):
        """Test trainer initialization."""
        from ai_news_tracker.mind_trainer import MINDTrainer

        config = MINDConfig(data_dir=temp_dir)
        trainer = MINDTrainer(config)

        assert trainer.config == config
        assert trainer.model is None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_trainer_save_load_model(self, temp_dir):
        """Test saving and loading model."""
        from ai_news_tracker.mind_trainer import MINDTrainer, NRMSModel

        config = MINDConfig(data_dir=temp_dir)
        trainer = MINDTrainer(config)

        # Initialize model manually for testing
        trainer.dataset.word2idx = {"<PAD>": 0, "<UNK>": 1, "test": 2}
        trainer.model = NRMSModel(config, vocab_size=3)

        # Save
        model_path = trainer.save_model()
        assert model_path.exists()

        # Load into new trainer
        trainer2 = MINDTrainer(config)
        trainer2.dataset.word2idx = trainer.dataset.word2idx
        trainer2.load_model(model_path)

        assert trainer2.model is not None
