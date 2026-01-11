"""Tests for the recommendation engine."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_news_tracker.models import Article, FeedSource, UserProfile
from ai_news_tracker.recommender import NewsRecommender
from ai_news_tracker.embeddings import embedding_to_bytes, bytes_to_embedding


class TestNewsRecommender:
    """Tests for the NewsRecommender class."""

    @pytest.fixture
    def recommender(self, temp_db, mock_embedding_engine):
        """Create a recommender with mock dependencies."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(
                db_session=session,
                embedding_engine=mock_embedding_engine,
            )
            yield recommender

    def test_init_with_defaults(self, temp_db):
        """Test recommender initialization with default engines."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.EmbeddingEngine") as mock_engine:
            with patch("ai_news_tracker.recommender.FeedFetcher"):
                mock_engine.return_value.embedding_dim = 384
                recommender = NewsRecommender(session)

                assert recommender.db == session
                assert recommender.embedding_engine is not None
                assert recommender.preference_learner is not None

    def test_compute_freshness_new_article(self, recommender):
        """Test freshness score for new article."""
        article = Article(
            url="https://example.com/test",
            title="Test",
            published_at=datetime.utcnow(),
        )

        freshness = recommender.compute_freshness(article)

        # New article should have freshness close to 1
        assert freshness > 0.9

    def test_compute_freshness_old_article(self, recommender):
        """Test freshness score for old article."""
        article = Article(
            url="https://example.com/test",
            title="Test",
            published_at=datetime.utcnow() - timedelta(days=7),
        )

        freshness = recommender.compute_freshness(article)

        # 7-day old article with 24h half-life should be very low
        assert freshness < 0.01

    def test_compute_freshness_one_half_life(self, recommender):
        """Test freshness score after one half-life."""
        article = Article(
            url="https://example.com/test",
            title="Test",
            published_at=datetime.utcnow() - timedelta(hours=24),
        )

        freshness = recommender.compute_freshness(article, half_life_hours=24.0)

        # Should be approximately 0.5
        assert abs(freshness - 0.5) < 0.01

    def test_compute_freshness_no_date(self, recommender):
        """Test freshness score when no date available."""
        article = Article(
            url="https://example.com/test",
            title="Test",
            published_at=None,
            fetched_at=None,
        )

        freshness = recommender.compute_freshness(article)

        # Unknown age should return neutral score
        assert freshness == 0.5

    def test_compute_freshness_custom_half_life(self, recommender):
        """Test freshness with custom half-life."""
        article = Article(
            url="https://example.com/test",
            title="Test",
            published_at=datetime.utcnow() - timedelta(hours=12),
        )

        # With 12h half-life, 12-hour-old article should be ~0.5
        freshness = recommender.compute_freshness(article, half_life_hours=12.0)
        assert abs(freshness - 0.5) < 0.01

    def test_get_recommendations_empty(self, recommender):
        """Test recommendations when no articles exist."""
        results = recommender.get_recommendations(limit=10)
        assert results == []

    def test_get_recommendations_no_preferences(self, temp_db, mock_embedding_engine, sample_articles):
        """Test recommendations without user preferences."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.get_recommendations(limit=5)

            # Should return articles sorted by recency
            assert len(results) <= 5
            # Each result should be (article, score, freshness)
            for article, score, freshness in results:
                assert isinstance(article, Article)
                assert 0 <= freshness <= 1

    def test_get_recommendations_with_preferences(
        self, temp_db, mock_embedding_engine, sample_articles, sample_user_profile
    ):
        """Test recommendations with user preferences."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.get_recommendations(limit=5)

            assert len(results) > 0
            for article, score, freshness in results:
                assert isinstance(score, float)
                assert isinstance(freshness, float)

    def test_get_recommendations_excludes_read(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test that read articles are excluded by default."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.get_recommendations(include_read=False)

            for article, _, _ in results:
                assert article.is_read is False

    def test_get_recommendations_includes_read(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test including read articles."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.get_recommendations(include_read=True)

            urls = [a.url for a, _, _ in results]
            assert "https://example.com/article5" in urls  # The read article

    def test_get_recommendations_respects_limit(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test that limit is respected."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.get_recommendations(limit=2)

            assert len(results) <= 2

    def test_get_recommendations_max_age_filter(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test max age filtering."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            # Very short max age should filter most articles
            results = recommender.get_recommendations(max_age_days=1)

            # Should only get recent articles
            for article, _, _ in results:
                age = datetime.utcnow() - article.fetched_at
                assert age.days <= 1

    def test_record_feedback_like(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test recording like feedback."""
        session, _ = temp_db
        article = sample_articles[0]

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            recommender.record_feedback(article.id, liked=True)

            session.refresh(article)
            assert article.is_liked is True
            assert article.is_read is True

    def test_record_feedback_dislike(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test recording dislike feedback."""
        session, _ = temp_db
        article = sample_articles[0]

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            recommender.record_feedback(article.id, liked=False)

            session.refresh(article)
            assert article.is_liked is False

    def test_record_feedback_nonexistent(self, recommender):
        """Test feedback for nonexistent article."""
        # Should not raise error
        recommender.record_feedback(99999, liked=True)

    def test_mark_read(self, temp_db, mock_embedding_engine, sample_articles):
        """Test marking article as read."""
        session, _ = temp_db
        article = sample_articles[0]
        assert article.is_read is False

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            recommender.mark_read(article.id)

            session.refresh(article)
            assert article.is_read is True
            assert article.read_at is not None

    def test_mark_read_nonexistent(self, recommender):
        """Test marking nonexistent article as read."""
        # Should not raise error
        recommender.mark_read(99999)

    def test_add_feed_source(self, temp_db, mock_embedding_engine):
        """Test adding a feed source."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            source = recommender.add_feed_source("Test Feed", "https://test.com/feed")

            assert source.id is not None
            assert source.name == "Test Feed"
            assert source.url == "https://test.com/feed"

    def test_search_by_topic_empty(self, recommender):
        """Test topic search with no articles."""
        results = recommender.search_by_topic("machine learning")
        assert results == []

    def test_search_by_topic(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test topic search returns results."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.search_by_topic("machine learning")

            assert len(results) > 0
            for article, score, freshness in results:
                assert isinstance(article, Article)
                assert 0 <= score <= 1
                assert 0 <= freshness <= 1

    def test_search_by_topic_respects_limit(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test topic search respects limit."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.search_by_topic("technology", limit=2)

            assert len(results) <= 2

    def test_search_by_topic_excludes_read(
        self, temp_db, mock_embedding_engine, sample_articles
    ):
        """Test topic search excludes read articles by default."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            results = recommender.search_by_topic("space", include_read=False)

            for article, _, _ in results:
                assert article.is_read is False

    def test_get_stats_empty(self, temp_db, mock_embedding_engine):
        """Test stats with empty database."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            stats = recommender.get_stats()

            assert stats["total_articles"] == 0
            assert stats["unread_articles"] == 0
            assert stats["liked_articles"] == 0
            assert stats["disliked_articles"] == 0
            assert stats["active_sources"] == 0
            assert stats["has_preferences"] is False

    def test_get_stats_with_data(
        self, temp_db, mock_embedding_engine, sample_articles, sample_feed_sources, sample_user_profile
    ):
        """Test stats with data."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher"):
            recommender = NewsRecommender(session, mock_embedding_engine)
            stats = recommender.get_stats()

            assert stats["total_articles"] == 5
            assert stats["active_sources"] == 2  # 2 active sources
            assert stats["has_preferences"] is True

    def test_ingest_from_feed(self, temp_db, mock_embedding_engine, mock_feedparser):
        """Test ingesting articles from a feed."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            from ai_news_tracker.sources import RawArticle
            mock_fetcher.fetch_feed.return_value = iter([
                RawArticle(
                    url="https://example.com/new1",
                    title="New Article 1",
                    summary="Summary 1",
                    source="Test Source",
                ),
                RawArticle(
                    url="https://example.com/new2",
                    title="New Article 2",
                    source="Test Source",
                ),
            ])

            recommender = NewsRecommender(session, mock_embedding_engine)
            new_articles = recommender.ingest_from_feed("https://test.com/feed")

            assert len(new_articles) == 2
            assert new_articles[0].title == "New Article 1"
            assert new_articles[0].embedding is not None

    def test_ingest_from_feed_skips_existing(self, temp_db, mock_embedding_engine, sample_articles):
        """Test that existing articles are skipped."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            from ai_news_tracker.sources import RawArticle
            mock_fetcher.fetch_feed.return_value = iter([
                RawArticle(
                    url="https://example.com/article1",  # Existing
                    title="Duplicate Article",
                ),
                RawArticle(
                    url="https://example.com/brand_new",
                    title="Brand New Article",
                ),
            ])

            recommender = NewsRecommender(session, mock_embedding_engine)
            new_articles = recommender.ingest_from_feed("https://test.com/feed")

            assert len(new_articles) == 1
            assert new_articles[0].url == "https://example.com/brand_new"

    def test_refresh_all_feeds(self, temp_db, mock_embedding_engine, sample_feed_sources):
        """Test refreshing all feeds."""
        session, _ = temp_db

        with patch("ai_news_tracker.recommender.FeedFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            from ai_news_tracker.sources import RawArticle
            call_count = [0]

            def mock_fetch_feed(url, source_name=None):
                call_count[0] += 1
                return iter([
                    RawArticle(
                        url=f"https://example.com/feed{call_count[0]}_article",
                        title=f"Article from feed {call_count[0]}",
                        source=source_name,
                    )
                ])

            mock_fetcher.fetch_feed.side_effect = mock_fetch_feed

            recommender = NewsRecommender(session, mock_embedding_engine)
            total_new = recommender.refresh_all_feeds()

            # Should have fetched from 2 active sources
            assert total_new == 2
            assert mock_fetcher.fetch_feed.call_count == 2
