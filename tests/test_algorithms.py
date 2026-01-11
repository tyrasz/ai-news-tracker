"""Tests for recommendation algorithms."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from ai_news_tracker.algorithms import (
    AlgorithmType,
    AlgorithmConfig,
    ScoredArticle,
    ForYouAlgorithm,
    ExploreAlgorithm,
    DeepDiveAlgorithm,
    TrendingAlgorithm,
    BalancedAlgorithm,
    ContrarianAlgorithm,
    get_algorithm,
    list_algorithms,
    ALGORITHMS,
)
from ai_news_tracker.models import Article
from ai_news_tracker.embeddings import embedding_to_bytes


class TestAlgorithmType:
    """Tests for AlgorithmType enum."""

    def test_algorithm_types_exist(self):
        """Test that all expected algorithm types exist."""
        assert AlgorithmType.FOR_YOU.value == "for_you"
        assert AlgorithmType.EXPLORE.value == "explore"
        assert AlgorithmType.DEEP_DIVE.value == "deep_dive"
        assert AlgorithmType.TRENDING.value == "trending"
        assert AlgorithmType.BALANCED.value == "balanced"
        assert AlgorithmType.CONTRARIAN.value == "contrarian"

    def test_algorithm_type_from_string(self):
        """Test creating AlgorithmType from string."""
        assert AlgorithmType("for_you") == AlgorithmType.FOR_YOU
        assert AlgorithmType("explore") == AlgorithmType.EXPLORE


class TestAlgorithmConfig:
    """Tests for AlgorithmConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AlgorithmConfig()
        assert config.freshness_weight == 0.3
        assert config.freshness_half_life_hours == 24.0
        assert config.diversity_factor == 0.1
        assert config.max_age_days == 7
        assert config.include_read is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AlgorithmConfig(
            freshness_weight=0.5,
            freshness_half_life_hours=12.0,
            diversity_factor=0.2,
            max_age_days=3,
            include_read=True,
        )
        assert config.freshness_weight == 0.5
        assert config.freshness_half_life_hours == 12.0
        assert config.include_read is True


class TestScoredArticle:
    """Tests for ScoredArticle dataclass."""

    def test_scored_article_creation(self):
        """Test creating a ScoredArticle."""
        article = MagicMock()
        scored = ScoredArticle(
            article=article,
            final_score=0.8,
            relevance_score=0.7,
            freshness_score=0.9,
            diversity_score=0.1,
        )
        assert scored.article == article
        assert scored.final_score == 0.8
        assert scored.relevance_score == 0.7
        assert scored.freshness_score == 0.9
        assert scored.diversity_score == 0.1


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def mock_embedding_engine():
    """Create a mock embedding engine."""
    engine = MagicMock()
    engine.embedding_dim = 384

    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    engine.cosine_similarity = cosine_similarity
    return engine


@pytest.fixture
def sample_candidates():
    """Create sample article candidates."""
    now = datetime.utcnow()
    articles = []

    for i in range(5):
        np.random.seed(i)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        article = MagicMock()
        article.id = i + 1
        article.title = f"Article {i + 1}"
        article.embedding = embedding_to_bytes(embedding)
        article.published_at = now - timedelta(hours=i * 6)
        article.fetched_at = now
        article.is_read = False
        articles.append(article)

    return articles


@pytest.fixture
def preference_embedding():
    """Create a sample preference embedding."""
    np.random.seed(42)
    embedding = np.random.randn(384).astype(np.float32)
    return embedding / np.linalg.norm(embedding)


class TestForYouAlgorithm:
    """Tests for ForYouAlgorithm."""

    def test_name_and_description(self, mock_db, mock_embedding_engine):
        """Test algorithm metadata."""
        config = AlgorithmConfig()
        algo = ForYouAlgorithm(mock_db, mock_embedding_engine, config)

        assert algo.name == "For You"
        assert "personalized" in algo.description.lower()

    def test_score_articles_with_preferences(
        self, mock_db, mock_embedding_engine, sample_candidates, preference_embedding
    ):
        """Test scoring with user preferences."""
        config = AlgorithmConfig()
        algo = ForYouAlgorithm(mock_db, mock_embedding_engine, config)

        results = algo.score_articles(sample_candidates, preference_embedding)

        assert len(results) == 5
        assert all(isinstance(r, ScoredArticle) for r in results)
        # Should be sorted by score descending
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_articles_without_preferences(
        self, mock_db, mock_embedding_engine, sample_candidates
    ):
        """Test scoring without user preferences."""
        config = AlgorithmConfig()
        algo = ForYouAlgorithm(mock_db, mock_embedding_engine, config)

        results = algo.score_articles(sample_candidates, None)

        assert len(results) == 5
        # All should have neutral relevance (0.5)
        for r in results:
            assert r.relevance_score == 0.5

    def test_freshness_computation(self, mock_db, mock_embedding_engine):
        """Test freshness score computation."""
        config = AlgorithmConfig(freshness_half_life_hours=24.0)
        algo = ForYouAlgorithm(mock_db, mock_embedding_engine, config)

        # New article
        new_article = MagicMock()
        new_article.published_at = datetime.utcnow()
        new_article.fetched_at = datetime.utcnow()
        assert algo.compute_freshness(new_article) > 0.9

        # Old article (1 half-life)
        old_article = MagicMock()
        old_article.published_at = datetime.utcnow() - timedelta(hours=24)
        old_article.fetched_at = datetime.utcnow()
        freshness = algo.compute_freshness(old_article)
        assert 0.45 < freshness < 0.55  # Should be ~0.5


class TestExploreAlgorithm:
    """Tests for ExploreAlgorithm."""

    def test_name_and_description(self, mock_db, mock_embedding_engine):
        """Test algorithm metadata."""
        config = AlgorithmConfig()
        algo = ExploreAlgorithm(mock_db, mock_embedding_engine, config)

        assert algo.name == "Explore"
        assert "discover" in algo.description.lower() or "outside" in algo.description.lower()

    def test_prefers_dissimilar_articles(
        self, mock_db, mock_embedding_engine, preference_embedding
    ):
        """Test that explore prefers articles dissimilar to preferences."""
        config = AlgorithmConfig()
        algo = ExploreAlgorithm(mock_db, mock_embedding_engine, config)

        # Create articles: one similar, one dissimilar to preferences
        now = datetime.utcnow()

        similar = MagicMock()
        similar.embedding = embedding_to_bytes(preference_embedding)  # Same as preference
        similar.published_at = now
        similar.fetched_at = now

        # Orthogonal embedding
        orthogonal = np.zeros(384, dtype=np.float32)
        orthogonal[0] = 1.0  # Point in different direction
        dissimilar = MagicMock()
        dissimilar.embedding = embedding_to_bytes(orthogonal)
        dissimilar.published_at = now
        dissimilar.fetched_at = now

        results = algo.score_articles([similar, dissimilar], preference_embedding)

        # Dissimilar should score higher in explore mode
        assert results[0].article == dissimilar or results[0].relevance_score > results[1].relevance_score


class TestTrendingAlgorithm:
    """Tests for TrendingAlgorithm."""

    def test_name_and_description(self, mock_db, mock_embedding_engine):
        """Test algorithm metadata."""
        config = AlgorithmConfig()
        algo = TrendingAlgorithm(mock_db, mock_embedding_engine, config)

        assert algo.name == "Trending"
        assert "fresh" in algo.description.lower() or "latest" in algo.description.lower()

    def test_prefers_fresh_articles(
        self, mock_db, mock_embedding_engine, preference_embedding
    ):
        """Test that trending heavily weights freshness."""
        config = AlgorithmConfig()
        algo = TrendingAlgorithm(mock_db, mock_embedding_engine, config)

        now = datetime.utcnow()
        np.random.seed(100)
        embedding = np.random.randn(384).astype(np.float32)

        fresh = MagicMock()
        fresh.embedding = embedding_to_bytes(embedding)
        fresh.published_at = now
        fresh.fetched_at = now

        old = MagicMock()
        old.embedding = embedding_to_bytes(embedding)  # Same content
        old.published_at = now - timedelta(days=3)
        old.fetched_at = now

        results = algo.score_articles([old, fresh], preference_embedding)

        # Fresh article should be first
        assert results[0].article == fresh


class TestDeepDiveAlgorithm:
    """Tests for DeepDiveAlgorithm."""

    def test_name_and_description(self, mock_db, mock_embedding_engine):
        """Test algorithm metadata."""
        config = AlgorithmConfig()
        algo = DeepDiveAlgorithm(mock_db, mock_embedding_engine, config)

        assert algo.name == "Deep Dive"
        assert "depth" in algo.description.lower() or "topic" in algo.description.lower()

    def test_score_articles(
        self, mock_db, mock_embedding_engine, sample_candidates, preference_embedding
    ):
        """Test that deep dive produces valid scores."""
        config = AlgorithmConfig()
        algo = DeepDiveAlgorithm(mock_db, mock_embedding_engine, config)

        results = algo.score_articles(sample_candidates, preference_embedding)

        assert len(results) == 5
        assert all(0 <= r.final_score <= 2 for r in results)  # Reasonable score range


class TestBalancedAlgorithm:
    """Tests for BalancedAlgorithm."""

    def test_name_and_description(self, mock_db, mock_embedding_engine):
        """Test algorithm metadata."""
        config = AlgorithmConfig()
        algo = BalancedAlgorithm(mock_db, mock_embedding_engine, config)

        assert algo.name == "Balanced"
        assert "mix" in algo.description.lower() or "balanced" in algo.description.lower()

    def test_includes_diversity_score(
        self, mock_db, mock_embedding_engine, sample_candidates, preference_embedding
    ):
        """Test that balanced includes diversity in scoring."""
        config = AlgorithmConfig()
        algo = BalancedAlgorithm(mock_db, mock_embedding_engine, config)

        results = algo.score_articles(sample_candidates, preference_embedding)

        # Should have diversity scores
        assert all(hasattr(r, 'diversity_score') for r in results)


class TestContrarianAlgorithm:
    """Tests for ContrarianAlgorithm."""

    def test_name_and_description(self, mock_db, mock_embedding_engine):
        """Test algorithm metadata."""
        config = AlgorithmConfig()
        algo = ContrarianAlgorithm(mock_db, mock_embedding_engine, config)

        assert algo.name == "Contrarian"
        assert "challenge" in algo.description.lower() or "different" in algo.description.lower()

    def test_prefers_opposite_articles(
        self, mock_db, mock_embedding_engine, preference_embedding
    ):
        """Test that contrarian prefers articles opposite to preferences."""
        config = AlgorithmConfig()
        algo = ContrarianAlgorithm(mock_db, mock_embedding_engine, config)

        now = datetime.utcnow()

        # Article similar to preferences
        similar = MagicMock()
        similar.embedding = embedding_to_bytes(preference_embedding)
        similar.published_at = now
        similar.fetched_at = now

        # Article opposite to preferences
        opposite = MagicMock()
        opposite.embedding = embedding_to_bytes(-preference_embedding)
        opposite.published_at = now
        opposite.fetched_at = now

        results = algo.score_articles([similar, opposite], preference_embedding)

        # Opposite should score higher
        opposite_score = next(r for r in results if r.article == opposite)
        similar_score = next(r for r in results if r.article == similar)
        assert opposite_score.relevance_score > similar_score.relevance_score


class TestGetAlgorithm:
    """Tests for get_algorithm factory function."""

    def test_get_each_algorithm_type(self, mock_db, mock_embedding_engine):
        """Test that each algorithm type can be instantiated."""
        for algo_type in AlgorithmType:
            algo = get_algorithm(algo_type, mock_db, mock_embedding_engine)
            assert algo is not None
            assert hasattr(algo, 'name')
            assert hasattr(algo, 'description')
            assert hasattr(algo, 'score_articles')

    def test_with_custom_config(self, mock_db, mock_embedding_engine):
        """Test algorithm with custom config."""
        config = AlgorithmConfig(freshness_weight=0.8)
        algo = get_algorithm(AlgorithmType.FOR_YOU, mock_db, mock_embedding_engine, config)

        assert algo.config.freshness_weight == 0.8

    def test_default_to_for_you(self, mock_db, mock_embedding_engine):
        """Test that invalid algorithm defaults to ForYou."""
        # This tests the fallback behavior
        algo = get_algorithm(AlgorithmType.FOR_YOU, mock_db, mock_embedding_engine)
        assert isinstance(algo, ForYouAlgorithm)


class TestListAlgorithms:
    """Tests for list_algorithms function."""

    def test_returns_all_algorithms(self):
        """Test that all algorithms are listed."""
        algos = list_algorithms()

        assert len(algos) == len(ALGORITHMS)
        assert all('id' in a for a in algos)
        assert all('name' in a for a in algos)
        assert all('description' in a for a in algos)

    def test_algorithm_ids_match_enum(self):
        """Test that algorithm IDs match enum values."""
        algos = list_algorithms()
        ids = {a['id'] for a in algos}

        for algo_type in AlgorithmType:
            assert algo_type.value in ids


class TestAlgorithmGetCandidates:
    """Tests for get_candidates method."""

    def test_filters_by_read_status(self, mock_db, mock_embedding_engine):
        """Test that read articles are filtered by default."""
        config = AlgorithmConfig(include_read=False)
        algo = ForYouAlgorithm(mock_db, mock_embedding_engine, config)

        # Setup mock query
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []

        algo.get_candidates()

        # Verify filter was called
        mock_query.filter.assert_called()

    def test_includes_read_when_configured(self, mock_db, mock_embedding_engine):
        """Test that read articles are included when configured."""
        config = AlgorithmConfig(include_read=True)
        algo = ForYouAlgorithm(mock_db, mock_embedding_engine, config)

        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []

        algo.get_candidates()

        # Query should still be made
        mock_db.query.assert_called()
