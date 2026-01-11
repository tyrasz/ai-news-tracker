"""Shared test fixtures for AI News Tracker."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ai_news_tracker.models import Base, Article, FeedSource, UserProfile


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session, db_path

    session.close()
    engine.dispose()
    os.unlink(db_path)


@pytest.fixture
def mock_embedding_engine():
    """Create a mock embedding engine that returns predictable vectors."""
    mock = MagicMock()
    mock.embedding_dim = 384

    def embed_text(text):
        # Generate a deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(384).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def embed_article(title, content=None, summary=None):
        combined = title + (content or "") + (summary or "")
        return embed_text(combined)

    def embed_batch(texts):
        return np.array([embed_text(t) for t in texts])

    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    mock.embed_text = embed_text
    mock.embed_article = embed_article
    mock.embed_batch = embed_batch
    mock.cosine_similarity = cosine_similarity

    return mock


@pytest.fixture
def sample_articles(temp_db):
    """Create sample articles in the database."""
    session, _ = temp_db

    articles = []
    now = datetime.utcnow()

    # Create diverse articles for testing
    article_data = [
        {
            "url": "https://example.com/article1",
            "title": "Machine Learning Advances in 2024",
            "summary": "New breakthrough in neural networks",
            "source": "TechNews",
            "published_at": now - timedelta(hours=2),
        },
        {
            "url": "https://example.com/article2",
            "title": "Python 4.0 Release Announced",
            "summary": "Major language update coming soon",
            "source": "DevBlog",
            "published_at": now - timedelta(hours=12),
        },
        {
            "url": "https://example.com/article3",
            "title": "Climate Change Summit Results",
            "summary": "World leaders agree on new targets",
            "source": "WorldNews",
            "published_at": now - timedelta(days=2),
        },
        {
            "url": "https://example.com/article4",
            "title": "Stock Market Hits Record High",
            "summary": "Tech stocks lead the rally",
            "source": "Finance",
            "published_at": now - timedelta(days=5),
        },
        {
            "url": "https://example.com/article5",
            "title": "New Space Mission Launched",
            "summary": "NASA sends probe to Mars",
            "source": "Science",
            "published_at": now - timedelta(hours=6),
            "is_read": True,
        },
    ]

    for data in article_data:
        # Generate a fake embedding
        np.random.seed(hash(data["title"]) % (2**32))
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        article = Article(
            url=data["url"],
            title=data["title"],
            summary=data.get("summary"),
            source=data.get("source"),
            published_at=data.get("published_at"),
            fetched_at=now,
            embedding=embedding.tobytes(),
            is_read=data.get("is_read", False),
        )
        session.add(article)
        articles.append(article)

    session.commit()
    return articles


@pytest.fixture
def sample_feed_sources(temp_db):
    """Create sample feed sources in the database."""
    session, _ = temp_db

    sources = [
        FeedSource(name="Hacker News", url="https://hnrss.org/frontpage", is_active=True),
        FeedSource(name="TechCrunch", url="https://techcrunch.com/feed/", is_active=True),
        FeedSource(name="Inactive Source", url="https://example.com/feed", is_active=False),
    ]

    for source in sources:
        session.add(source)

    session.commit()
    return sources


@pytest.fixture
def sample_user_profile(temp_db, mock_embedding_engine):
    """Create a sample user profile with preferences."""
    session, _ = temp_db

    # Generate a preference embedding
    np.random.seed(42)
    preference = np.random.randn(384).astype(np.float32)
    preference = preference / np.linalg.norm(preference)

    profile = UserProfile(
        name="default",
        preference_embedding=preference.tobytes(),
        articles_liked=5,
        articles_disliked=2,
        decay_factor=0.95,
    )
    session.add(profile)
    session.commit()

    return profile


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for testing HTTP requests."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        yield mock_client


@pytest.fixture
def mock_feedparser():
    """Create a mock feedparser response."""
    with patch("feedparser.parse") as mock_parse:
        mock_feed = MagicMock()
        mock_feed.feed = {"title": "Test Feed"}
        mock_feed.entries = [
            {
                "link": "https://example.com/post1",
                "title": "Test Article 1",
                "summary": "Summary of article 1",
                "author": "Author 1",
                "published_parsed": (2024, 1, 15, 10, 30, 0, 0, 0, 0),
            },
            {
                "link": "https://example.com/post2",
                "title": "Test Article 2",
                "summary": "Summary of article 2",
            },
        ]
        mock_parse.return_value = mock_feed
        yield mock_parse
