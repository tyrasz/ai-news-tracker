"""Tests for the article clustering module."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_news_tracker.clustering import ArticleGroup, cluster_similar_articles, flatten_groups
from ai_news_tracker.models import Article
from ai_news_tracker.embeddings import embedding_to_bytes


@pytest.fixture
def mock_embedding_engine():
    """Create a mock embedding engine."""
    engine = MagicMock()
    engine.embedding_dim = 384
    return engine


@pytest.fixture
def sample_articles():
    """Create sample articles with embeddings."""
    now = datetime.utcnow()
    articles = []

    for i in range(5):
        embedding = np.random.randn(384).astype(np.float32)
        article = Article(
            id=i + 1,
            url=f"https://example.com/article{i}",
            title=f"Test Article {i}",
            summary=f"Summary for article {i}",
            source=f"Source {i}",
            published_at=now - timedelta(hours=i * 2),
            fetched_at=now,
            embedding=embedding_to_bytes(embedding),
        )
        articles.append(article)

    return articles


class TestArticleGroup:
    """Tests for the ArticleGroup dataclass."""

    def test_article_group_creation(self, sample_articles):
        """Test creating an ArticleGroup."""
        primary = sample_articles[0]
        group = ArticleGroup(
            primary=primary,
            primary_score=0.9,
            primary_freshness=0.8,
        )

        assert group.primary == primary
        assert group.primary_score == 0.9
        assert group.primary_freshness == 0.8
        assert group.related == []

    def test_article_group_with_related(self, sample_articles):
        """Test ArticleGroup with related articles."""
        primary = sample_articles[0]
        related = [
            (sample_articles[1], 0.85, 0.7),
            (sample_articles[2], 0.8, 0.6),
        ]

        group = ArticleGroup(
            primary=primary,
            primary_score=0.9,
            primary_freshness=0.8,
            related=related,
        )

        assert len(group.related) == 2
        assert group.related[0][0] == sample_articles[1]

    def test_count_property_single(self, sample_articles):
        """Test count property with only primary article."""
        group = ArticleGroup(
            primary=sample_articles[0],
            primary_score=0.9,
            primary_freshness=0.8,
        )

        assert group.count == 1

    def test_count_property_with_related(self, sample_articles):
        """Test count property with related articles."""
        group = ArticleGroup(
            primary=sample_articles[0],
            primary_score=0.9,
            primary_freshness=0.8,
            related=[
                (sample_articles[1], 0.85, 0.7),
                (sample_articles[2], 0.8, 0.6),
            ],
        )

        assert group.count == 3

    def test_sources_property_single(self, sample_articles):
        """Test sources property with only primary article."""
        group = ArticleGroup(
            primary=sample_articles[0],
            primary_score=0.9,
            primary_freshness=0.8,
        )

        assert group.sources == ["Source 0"]

    def test_sources_property_with_related(self, sample_articles):
        """Test sources property with related articles."""
        group = ArticleGroup(
            primary=sample_articles[0],
            primary_score=0.9,
            primary_freshness=0.8,
            related=[
                (sample_articles[1], 0.85, 0.7),
                (sample_articles[2], 0.8, 0.6),
            ],
        )

        sources = group.sources
        assert "Source 0" in sources
        assert "Source 1" in sources
        assert "Source 2" in sources
        assert len(sources) == 3

    def test_sources_property_no_duplicates(self, sample_articles):
        """Test that sources doesn't have duplicates."""
        # Make articles have same source
        sample_articles[1].source = "Source 0"
        sample_articles[2].source = "Source 0"

        group = ArticleGroup(
            primary=sample_articles[0],
            primary_score=0.9,
            primary_freshness=0.8,
            related=[
                (sample_articles[1], 0.85, 0.7),
                (sample_articles[2], 0.8, 0.6),
            ],
        )

        assert group.sources == ["Source 0"]

    def test_sources_property_handles_none(self, sample_articles):
        """Test that sources handles None source values."""
        sample_articles[0].source = None

        group = ArticleGroup(
            primary=sample_articles[0],
            primary_score=0.9,
            primary_freshness=0.8,
            related=[
                (sample_articles[1], 0.85, 0.7),
            ],
        )

        sources = group.sources
        assert "Source 1" in sources
        assert None not in sources


class TestClusterSimilarArticles:
    """Tests for the cluster_similar_articles function."""

    def test_empty_input(self, mock_embedding_engine):
        """Test clustering with empty input."""
        result = cluster_similar_articles([], mock_embedding_engine)
        assert result == []

    def test_single_article(self, mock_embedding_engine, sample_articles):
        """Test clustering with single article."""
        scored = [(sample_articles[0], 0.9, 0.8)]

        result = cluster_similar_articles(scored, mock_embedding_engine)

        assert len(result) == 1
        assert result[0].primary == sample_articles[0]
        assert result[0].related == []

    def test_no_similar_articles(self, mock_embedding_engine, sample_articles):
        """Test clustering when no articles are similar."""
        # Make all similarities low
        mock_embedding_engine.cosine_similarity.return_value = 0.3

        scored = [
            (sample_articles[0], 0.9, 0.8),
            (sample_articles[1], 0.85, 0.7),
            (sample_articles[2], 0.8, 0.6),
        ]

        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.75)

        # Each article should be its own group
        assert len(result) == 3
        for group in result:
            assert group.related == []

    def test_all_similar_articles(self, mock_embedding_engine, sample_articles):
        """Test clustering when all articles are similar."""
        # Make all similarities high
        mock_embedding_engine.cosine_similarity.return_value = 0.9

        scored = [
            (sample_articles[0], 0.9, 0.8),
            (sample_articles[1], 0.85, 0.7),
            (sample_articles[2], 0.8, 0.6),
        ]

        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.75)

        # All articles should be in one group
        assert len(result) == 1
        assert result[0].primary == sample_articles[0]
        assert len(result[0].related) == 2

    def test_some_similar_articles(self, mock_embedding_engine, sample_articles):
        """Test clustering when some articles are similar."""
        # The algorithm checks primary (0) against all others first: 0v1, 0v2, 0v3
        # Then 1 is already assigned, so skipped
        # Then 2 is checked against remaining: 2v3
        mock_embedding_engine.cosine_similarity.side_effect = [
            0.9,  # 0 vs 1 - similar (1 gets grouped with 0)
            0.3,  # 0 vs 2 - different
            0.3,  # 0 vs 3 - different
            # 1 is skipped (already assigned)
            0.9,  # 2 vs 3 - similar (3 gets grouped with 2)
        ]

        scored = [
            (sample_articles[0], 0.9, 0.8),
            (sample_articles[1], 0.85, 0.7),
            (sample_articles[2], 0.8, 0.6),
            (sample_articles[3], 0.75, 0.5),
        ]

        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.75)

        # Should have 2 groups
        assert len(result) == 2
        # First group has articles 0 and 1
        assert result[0].primary == sample_articles[0]
        assert len(result[0].related) == 1
        # Second group has articles 2 and 3
        assert result[1].primary == sample_articles[2]
        assert len(result[1].related) == 1

    def test_preserves_score_order(self, mock_embedding_engine, sample_articles):
        """Test that highest scored article becomes primary."""
        mock_embedding_engine.cosine_similarity.return_value = 0.9

        # Articles in score order (highest first)
        scored = [
            (sample_articles[2], 0.95, 0.9),  # Highest score
            (sample_articles[0], 0.85, 0.8),
            (sample_articles[1], 0.75, 0.7),
        ]

        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.75)

        # Primary should be the highest scored
        assert result[0].primary == sample_articles[2]
        assert result[0].primary_score == 0.95

    def test_threshold_boundary(self, mock_embedding_engine, sample_articles):
        """Test behavior at threshold boundary."""
        # Exactly at threshold
        mock_embedding_engine.cosine_similarity.return_value = 0.75

        scored = [
            (sample_articles[0], 0.9, 0.8),
            (sample_articles[1], 0.85, 0.7),
        ]

        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.75)

        # Should be grouped (>= threshold)
        assert len(result) == 1
        assert len(result[0].related) == 1

    def test_below_threshold(self, mock_embedding_engine, sample_articles):
        """Test behavior just below threshold."""
        mock_embedding_engine.cosine_similarity.return_value = 0.749

        scored = [
            (sample_articles[0], 0.9, 0.8),
            (sample_articles[1], 0.85, 0.7),
        ]

        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.75)

        # Should not be grouped
        assert len(result) == 2

    def test_articles_without_embeddings(self, mock_embedding_engine, sample_articles):
        """Test that articles without embeddings are skipped."""
        sample_articles[1].embedding = None

        scored = [
            (sample_articles[0], 0.9, 0.8),
            (sample_articles[1], 0.85, 0.7),  # No embedding
            (sample_articles[2], 0.8, 0.6),
        ]

        mock_embedding_engine.cosine_similarity.return_value = 0.9

        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.75)

        # Article 1 should be skipped, so only 2 articles form one group
        assert len(result) == 1
        assert result[0].count == 2

    def test_custom_threshold(self, mock_embedding_engine, sample_articles):
        """Test with custom similarity threshold."""
        mock_embedding_engine.cosine_similarity.return_value = 0.85

        scored = [
            (sample_articles[0], 0.9, 0.8),
            (sample_articles[1], 0.85, 0.7),
        ]

        # High threshold - should not group
        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.9)
        assert len(result) == 2

        # Lower threshold - should group
        result = cluster_similar_articles(scored, mock_embedding_engine, similarity_threshold=0.8)
        assert len(result) == 1


class TestFlattenGroups:
    """Tests for the flatten_groups function."""

    def test_flatten_empty(self):
        """Test flattening empty list."""
        result = flatten_groups([])
        assert result == []

    def test_flatten_single_group(self, sample_articles):
        """Test flattening single group."""
        groups = [
            ArticleGroup(
                primary=sample_articles[0],
                primary_score=0.9,
                primary_freshness=0.8,
            )
        ]

        result = flatten_groups(groups)

        assert len(result) == 1
        assert result[0] == (sample_articles[0], 0.9, 0.8)

    def test_flatten_multiple_groups(self, sample_articles):
        """Test flattening multiple groups."""
        groups = [
            ArticleGroup(
                primary=sample_articles[0],
                primary_score=0.9,
                primary_freshness=0.8,
                related=[
                    (sample_articles[1], 0.85, 0.7),
                ],
            ),
            ArticleGroup(
                primary=sample_articles[2],
                primary_score=0.8,
                primary_freshness=0.6,
            ),
        ]

        result = flatten_groups(groups)

        # Should only return primary articles
        assert len(result) == 2
        assert result[0] == (sample_articles[0], 0.9, 0.8)
        assert result[1] == (sample_articles[2], 0.8, 0.6)

    def test_flatten_ignores_related(self, sample_articles):
        """Test that flatten only returns primary articles."""
        groups = [
            ArticleGroup(
                primary=sample_articles[0],
                primary_score=0.9,
                primary_freshness=0.8,
                related=[
                    (sample_articles[1], 0.85, 0.7),
                    (sample_articles[2], 0.8, 0.6),
                    (sample_articles[3], 0.75, 0.5),
                ],
            ),
        ]

        result = flatten_groups(groups)

        # Should only have 1 article (the primary)
        assert len(result) == 1
        assert result[0][0] == sample_articles[0]


class TestClusteringIntegration:
    """Integration tests for clustering with real embedding engine."""

    def test_clustering_with_identical_embeddings(self, sample_articles):
        """Test clustering articles with identical embeddings."""
        # Create mock engine that returns high similarity for identical embeddings
        engine = MagicMock()
        engine.embedding_dim = 384
        engine.cosine_similarity.return_value = 1.0

        # Give all articles the same embedding
        same_embedding = np.random.randn(384).astype(np.float32)
        for article in sample_articles:
            article.embedding = embedding_to_bytes(same_embedding)

        scored = [(a, 0.9 - i * 0.1, 0.8 - i * 0.1) for i, a in enumerate(sample_articles)]

        result = cluster_similar_articles(scored, engine, similarity_threshold=0.75)

        # All should be in one group
        assert len(result) == 1
        assert result[0].count == 5

    def test_clustering_preserves_data(self, mock_embedding_engine, sample_articles):
        """Test that clustering preserves article data correctly."""
        mock_embedding_engine.cosine_similarity.return_value = 0.9

        scored = [
            (sample_articles[0], 0.95, 0.85),
            (sample_articles[1], 0.88, 0.75),
        ]

        result = cluster_similar_articles(scored, mock_embedding_engine)

        # Check primary data
        assert result[0].primary.id == sample_articles[0].id
        assert result[0].primary.title == sample_articles[0].title
        assert result[0].primary_score == 0.95
        assert result[0].primary_freshness == 0.85

        # Check related data
        related_article, score, freshness = result[0].related[0]
        assert related_article.id == sample_articles[1].id
        assert score == 0.88
        assert freshness == 0.75
