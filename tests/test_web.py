"""Tests for the web API endpoints."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_news_tracker.models import Base, Article, FeedSource, UserProfile
from ai_news_tracker.embeddings import embedding_to_bytes


@pytest.fixture
def test_app():
    """Create a test app with mocked dependencies."""
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    os.environ["NEWS_DB_PATH"] = db_path

    # Import app after setting env var
    from ai_news_tracker.web import app

    # Initialize database
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Add sample data
    now = datetime.utcnow()
    for i in range(5):
        embedding = np.random.randn(384).astype(np.float32)
        article = Article(
            url=f"https://example.com/article{i}",
            title=f"Test Article {i}",
            summary=f"Summary for article {i}",
            source="Test Source",
            published_at=now - timedelta(hours=i * 2),
            fetched_at=now,
            embedding=embedding_to_bytes(embedding),
            is_read=i == 0,  # First article is read
            is_liked=True if i == 1 else (False if i == 2 else None),
        )
        session.add(article)

    # Add feed sources
    source = FeedSource(name="Test Feed", url="https://test.com/feed", is_active=True)
    session.add(source)

    session.commit()
    session.close()
    engine.dispose()

    # Create mock recommender
    with patch("ai_news_tracker.web.recommender") as mock_recommender:
        with patch("ai_news_tracker.web.db_session") as mock_session:
            # Setup mock methods
            mock_recommender.get_recommendations.return_value = []
            mock_recommender.search_by_topic.return_value = []
            mock_recommender.get_stats.return_value = {
                "total_articles": 5,
                "unread_articles": 4,
                "liked_articles": 1,
                "disliked_articles": 1,
                "active_sources": 1,
                "has_preferences": False,
            }
            mock_recommender.record_feedback.return_value = None
            mock_recommender.mark_read.return_value = None
            mock_recommender.refresh_all_feeds.return_value = 3

            # Create test client
            client = TestClient(app)

            yield client, mock_recommender, db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def test_client_with_data():
    """Create test client with actual data (not mocked)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    os.environ["NEWS_DB_PATH"] = db_path

    from ai_news_tracker.web import app, startup

    # Run startup to initialize
    with patch("ai_news_tracker.web.EmbeddingEngine") as mock_engine:
        mock_engine_instance = MagicMock()
        mock_engine_instance.embedding_dim = 384
        mock_engine_instance.embed_article.return_value = np.random.randn(384).astype(np.float32)
        mock_engine_instance.embed_text.return_value = np.random.randn(384).astype(np.float32)
        mock_engine_instance.cosine_similarity.return_value = 0.5
        mock_engine.return_value = mock_engine_instance

        startup()

    client = TestClient(app)

    yield client, db_path

    try:
        os.unlink(db_path)
    except:
        pass


class TestIndexEndpoint:
    """Tests for the index endpoint."""

    def test_index_returns_html(self, test_app):
        """Test that index returns HTML."""
        client, _, _ = test_app
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_index_has_cache_control(self, test_app):
        """Test that index has cache control headers."""
        client, _, _ = test_app
        response = client.get("/")

        assert "no-cache" in response.headers.get("cache-control", "")


class TestRecommendationsEndpoint:
    """Tests for the recommendations endpoint."""

    def test_get_recommendations_default(self, test_app):
        """Test getting recommendations with defaults."""
        client, mock_recommender, _ = test_app

        # Setup mock to return actual data
        mock_article = MagicMock()
        mock_article.id = 1
        mock_article.title = "Test Article"
        mock_article.url = "https://example.com/test"
        mock_article.source = "Test"
        mock_article.author = "Author"
        mock_article.summary = "Summary"
        mock_article.published_at = datetime.utcnow()
        mock_article.is_read = False
        mock_article.is_liked = None

        mock_recommender.get_recommendations.return_value = [
            (mock_article, 0.8, 0.9)
        ]

        response = client.get("/api/recommendations")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_recommendations_with_params(self, test_app):
        """Test recommendations with query parameters."""
        client, mock_recommender, _ = test_app
        mock_recommender.get_recommendations_v2.return_value = []

        response = client.get(
            "/api/recommendations",
            params={
                "limit": 10,
                "freshness_weight": 0.5,
                "include_read": True,
                "algorithm": "trending",
            }
        )

        assert response.status_code == 200
        mock_recommender.get_recommendations_v2.assert_called_once_with(
            algorithm="trending",
            limit=10,
            include_read=True,
            freshness_weight=0.5,
        )

    def test_get_recommendations_limit_validation(self, test_app):
        """Test that limit is validated."""
        client, _, _ = test_app

        # Too low
        response = client.get("/api/recommendations", params={"limit": 0})
        assert response.status_code == 422

        # Too high
        response = client.get("/api/recommendations", params={"limit": 201})
        assert response.status_code == 422


class TestAlgorithmsEndpoint:
    """Tests for the algorithms endpoint."""

    def test_list_algorithms(self, test_app):
        """Test listing available algorithms."""
        client, mock_recommender, _ = test_app
        mock_recommender.list_algorithms.return_value = [
            {"id": "for_you", "name": "For You", "description": "Personalized"},
            {"id": "explore", "name": "Explore", "description": "Discover new topics"},
        ]

        response = client.get("/api/algorithms")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "for_you"


class TestGroupedRecommendationsEndpoint:
    """Tests for the grouped recommendations endpoint."""

    def test_get_grouped_recommendations(self, test_app):
        """Test getting grouped recommendations."""
        client, mock_recommender, _ = test_app

        # Create mock article group
        from ai_news_tracker.clustering import ArticleGroup

        mock_primary = MagicMock()
        mock_primary.id = 1
        mock_primary.title = "Breaking News"
        mock_primary.url = "https://example.com/news1"
        mock_primary.source = "Source A"
        mock_primary.author = None
        mock_primary.summary = "Summary"
        mock_primary.published_at = datetime.utcnow()
        mock_primary.is_read = False
        mock_primary.is_liked = None

        mock_related = MagicMock()
        mock_related.id = 2
        mock_related.title = "Same Story Different Source"
        mock_related.url = "https://example.com/news2"
        mock_related.source = "Source B"
        mock_related.author = None
        mock_related.summary = "Related summary"
        mock_related.published_at = datetime.utcnow()
        mock_related.is_read = False
        mock_related.is_liked = None

        mock_group = ArticleGroup(
            primary=mock_primary,
            primary_score=0.9,
            primary_freshness=0.8,
            related=[(mock_related, 0.85, 0.75)],
        )

        mock_recommender.get_recommendations_grouped.return_value = [mock_group]

        response = client.get("/api/recommendations/grouped")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

        group = data[0]
        assert group["count"] == 2
        assert "Source A" in group["sources"]
        assert "Source B" in group["sources"]
        assert group["primary"]["title"] == "Breaking News"
        assert len(group["related"]) == 1
        assert group["related"][0]["title"] == "Same Story Different Source"

    def test_get_grouped_recommendations_with_params(self, test_app):
        """Test grouped recommendations with query parameters."""
        client, mock_recommender, _ = test_app
        mock_recommender.get_recommendations_grouped.return_value = []

        response = client.get(
            "/api/recommendations/grouped",
            params={
                "limit": 10,
                "freshness_weight": 0.5,
                "include_read": True,
                "algorithm": "trending",
                "similarity_threshold": 0.8,
            }
        )

        assert response.status_code == 200
        mock_recommender.get_recommendations_grouped.assert_called_once_with(
            algorithm="trending",
            limit=10,
            include_read=True,
            freshness_weight=0.5,
            similarity_threshold=0.8,
        )

    def test_grouped_similarity_threshold_validation(self, test_app):
        """Test that similarity threshold is validated."""
        client, _, _ = test_app

        # Too low
        response = client.get("/api/recommendations/grouped", params={"similarity_threshold": 0.4})
        assert response.status_code == 422

        # Too high
        response = client.get("/api/recommendations/grouped", params={"similarity_threshold": 0.99})
        assert response.status_code == 422


class TestTopicSearchEndpoint:
    """Tests for the topic search endpoint."""

    def test_search_topic(self, test_app):
        """Test topic search."""
        client, mock_recommender, _ = test_app

        mock_article = MagicMock()
        mock_article.id = 1
        mock_article.title = "ML Article"
        mock_article.url = "https://example.com/ml"
        mock_article.source = "Tech"
        mock_article.author = None
        mock_article.summary = "About ML"
        mock_article.published_at = datetime.utcnow()
        mock_article.is_read = False
        mock_article.is_liked = None

        mock_recommender.search_by_topic.return_value = [
            (mock_article, 0.9, 0.8)
        ]

        response = client.get("/api/topic/machine%20learning")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_search_topic_with_params(self, test_app):
        """Test topic search with parameters."""
        client, mock_recommender, _ = test_app
        mock_recommender.search_by_topic.return_value = []

        response = client.get(
            "/api/topic/python",
            params={"limit": 5, "freshness_weight": 0.3}
        )

        assert response.status_code == 200
        mock_recommender.search_by_topic.assert_called_once()


class TestFeedbackEndpoint:
    """Tests for the feedback endpoint."""

    def test_record_like(self, test_app):
        """Test recording a like."""
        client, mock_recommender, _ = test_app

        response = client.post(
            "/api/feedback",
            json={"article_id": 1, "liked": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["article_id"] == 1
        assert data["liked"] is True
        mock_recommender.record_feedback.assert_called_once_with(1, True)

    def test_record_dislike(self, test_app):
        """Test recording a dislike."""
        client, mock_recommender, _ = test_app

        response = client.post(
            "/api/feedback",
            json={"article_id": 2, "liked": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["liked"] is False
        mock_recommender.record_feedback.assert_called_once_with(2, False)

    def test_feedback_validation(self, test_app):
        """Test feedback validation."""
        client, _, _ = test_app

        # Missing fields
        response = client.post("/api/feedback", json={})
        assert response.status_code == 422

        # Invalid article_id type
        response = client.post(
            "/api/feedback",
            json={"article_id": "abc", "liked": True}
        )
        assert response.status_code == 422


class TestMarkReadEndpoint:
    """Tests for the mark read endpoint."""

    def test_mark_read(self, test_app):
        """Test marking article as read."""
        client, mock_recommender, _ = test_app

        response = client.post("/api/read/1")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["article_id"] == 1
        mock_recommender.mark_read.assert_called_once_with(1)


class TestArticleEndpoint:
    """Tests for the article detail endpoint."""

    def test_get_article(self, test_app):
        """Test getting article details."""
        client, _, db_path = test_app

        # We need to use actual database for this test
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        session = Session()

        # Add an article
        article = Article(
            url="https://example.com/detail",
            title="Detail Article",
            summary="Detailed summary",
            content="Full content here",
            source="Test",
            published_at=datetime.utcnow(),
        )
        session.add(article)
        session.commit()
        article_id = article.id
        session.close()

        # Patch db_session query
        with patch("ai_news_tracker.web.db_session") as mock_session:
            mock_article = MagicMock()
            mock_article.id = article_id
            mock_article.title = "Detail Article"
            mock_article.url = "https://example.com/detail"
            mock_article.source = "Test"
            mock_article.author = None
            mock_article.summary = "Detailed summary"
            mock_article.content = "Full content here"
            mock_article.published_at = datetime.utcnow()
            mock_article.is_read = False
            mock_article.is_liked = None

            mock_query = MagicMock()
            mock_query.get.return_value = mock_article
            mock_session.query.return_value = mock_query

            response = client.get(f"/api/article/{article_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Detail Article"
            assert data["content"] == "Full content here"

    def test_get_article_not_found(self, test_app):
        """Test getting nonexistent article returns 404."""
        client, _, _ = test_app

        with patch("ai_news_tracker.web.db_session") as mock_session:
            mock_query = MagicMock()
            mock_query.get.return_value = None
            mock_session.query.return_value = mock_query

            response = client.get("/api/article/99999")

            assert response.status_code == 404
            data = response.json()
            assert data["status"] == "error"
            assert "not found" in data["message"].lower()


class TestStatsEndpoint:
    """Tests for the stats endpoint."""

    def test_get_stats(self, test_app):
        """Test getting statistics."""
        client, mock_recommender, _ = test_app

        response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_articles" in data
        assert "unread_articles" in data
        assert "liked_articles" in data
        assert "disliked_articles" in data
        assert "active_sources" in data
        assert "has_preferences" in data


class TestRefreshEndpoint:
    """Tests for the refresh endpoint."""

    def test_refresh_success(self, test_app):
        """Test successful feed refresh."""
        client, mock_recommender, _ = test_app

        response = client.post("/api/refresh")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["new_articles"] == 3
        mock_recommender.refresh_all_feeds.assert_called_once_with(fetch_content=False)

    def test_refresh_error(self, test_app):
        """Test refresh with error."""
        client, mock_recommender, _ = test_app
        mock_recommender.refresh_all_feeds.side_effect = Exception("Network error")

        response = client.post("/api/refresh")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Network error" in data["message"]


class TestCacheEndpoints:
    """Tests for cache management endpoints."""

    def test_get_cache_stats(self, test_app):
        """Test getting cache statistics."""
        client, mock_recommender, _ = test_app

        mock_engine = MagicMock()
        mock_engine.cache_stats.return_value = {
            "size": 100,
            "maxsize": 1000,
            "hits": 50,
            "misses": 25,
            "hit_rate": 0.67,
        }
        mock_recommender.embedding_engine = mock_engine

        response = client.get("/api/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert "size" in data
        mock_engine.cache_stats.assert_called_once()

    def test_clear_cache(self, test_app):
        """Test clearing the cache."""
        client, mock_recommender, _ = test_app

        mock_engine = MagicMock()
        mock_recommender.embedding_engine = mock_engine

        response = client.post("/api/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        mock_engine.clear_cache.assert_called_once()


class TestTopicSearchValidation:
    """Tests for topic search input validation."""

    def test_search_empty_query_after_strip(self, test_app):
        """Test that whitespace-only query returns 400."""
        client, _, _ = test_app

        response = client.get("/api/topic/%20%20%20")  # Just whitespace

        assert response.status_code == 400
        data = response.json()
        assert "empty" in data["message"].lower()

    def test_search_query_too_long(self, test_app):
        """Test that very long query is rejected."""
        client, _, _ = test_app

        long_query = "a" * 250
        response = client.get(f"/api/topic/{long_query}")

        assert response.status_code == 422


class TestExceptionHandlers:
    """Tests for exception handlers."""

    def test_validation_error_response_format(self, test_app):
        """Test validation error response format."""
        client, _, _ = test_app

        response = client.get("/api/recommendations", params={"limit": -5})

        assert response.status_code == 422
        data = response.json()
        assert data["status"] == "error"
        assert "Validation error" in data["message"]
        assert "details" in data

    def test_invalid_algorithm_type(self, test_app):
        """Test invalid algorithm type returns validation error."""
        client, _, _ = test_app

        response = client.get("/api/recommendations", params={"algorithm": "invalid_algo"})

        assert response.status_code == 422


class TestCleanSummary:
    """Tests for the _clean_summary helper function."""

    def test_clean_summary_import(self):
        """Test that _clean_summary can be imported."""
        from ai_news_tracker.web import _clean_summary
        assert callable(_clean_summary)

    def test_clean_summary_none(self):
        """Test cleaning None summary."""
        from ai_news_tracker.web import _clean_summary
        assert _clean_summary(None) is None

    def test_clean_summary_html_entities(self):
        """Test cleaning HTML entities."""
        from ai_news_tracker.web import _clean_summary

        result = _clean_summary("It&#8217;s a &#8220;test&#8221;")
        assert result == "It's a \"test\""

    def test_clean_summary_html_tags(self):
        """Test removing HTML tags."""
        from ai_news_tracker.web import _clean_summary

        result = _clean_summary("<p>Paragraph</p><br>Line")
        assert "<p>" not in result
        assert "</p>" not in result
        assert "<br>" not in result

    def test_clean_summary_ampersand(self):
        """Test cleaning ampersand entity."""
        from ai_news_tracker.web import _clean_summary

        result = _clean_summary("Tom &amp; Jerry")
        assert result == "Tom & Jerry"

    def test_clean_summary_strips_whitespace(self):
        """Test that summary is stripped."""
        from ai_news_tracker.web import _clean_summary

        result = _clean_summary("  Some text  ")
        assert result == "Some text"
