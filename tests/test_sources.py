"""Tests for news source fetching."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ai_news_tracker.sources import (
    RawArticle,
    FeedFetcher,
    FEED_CATEGORIES,
    DEFAULT_FEEDS,
    ALL_FEEDS,
)


class TestRawArticle:
    """Tests for the RawArticle dataclass."""

    def test_raw_article_creation(self):
        """Test creating a RawArticle with required fields."""
        article = RawArticle(
            url="https://example.com/article",
            title="Test Article",
        )

        assert article.url == "https://example.com/article"
        assert article.title == "Test Article"
        assert article.content is None
        assert article.summary is None

    def test_raw_article_all_fields(self):
        """Test creating a RawArticle with all fields."""
        now = datetime.now()
        article = RawArticle(
            url="https://example.com/article",
            title="Test Article",
            content="Full content here",
            summary="Summary text",
            source="Test Source",
            author="Test Author",
            published_at=now,
        )

        assert article.url == "https://example.com/article"
        assert article.title == "Test Article"
        assert article.content == "Full content here"
        assert article.summary == "Summary text"
        assert article.source == "Test Source"
        assert article.author == "Test Author"
        assert article.published_at == now


class TestFeedFetcher:
    """Tests for the FeedFetcher class."""

    def test_fetcher_initialization(self):
        """Test FeedFetcher initialization."""
        with patch("httpx.Client"):
            fetcher = FeedFetcher(timeout=60.0)
            assert fetcher.client is not None

    def test_fetcher_context_manager(self):
        """Test FeedFetcher as context manager."""
        with patch("httpx.Client") as mock_client:
            with FeedFetcher() as fetcher:
                assert fetcher is not None
            mock_client.return_value.close.assert_called_once()

    def test_fetch_feed_basic(self, mock_feedparser):
        """Test basic feed fetching."""
        with patch("httpx.Client"):
            fetcher = FeedFetcher()
            articles = list(fetcher.fetch_feed("https://example.com/feed"))

            assert len(articles) == 2
            assert articles[0].title == "Test Article 1"
            assert articles[0].url == "https://example.com/post1"
            assert articles[1].title == "Test Article 2"

    def test_fetch_feed_with_source_name(self, mock_feedparser):
        """Test feed fetching with custom source name."""
        with patch("httpx.Client"):
            fetcher = FeedFetcher()
            articles = list(fetcher.fetch_feed(
                "https://example.com/feed",
                source_name="Custom Source"
            ))

            assert all(a.source == "Custom Source" for a in articles)

    def test_fetch_feed_parses_date(self):
        """Test that published dates are parsed correctly."""
        with patch("feedparser.parse") as mock_parse:
            mock_feed = MagicMock()
            mock_feed.feed = {"title": "Test"}

            # Create entry with published_parsed attribute
            entry1 = MagicMock(spec=["get", "published_parsed"])
            entry1.get.side_effect = lambda k, d=None: {
                "link": "https://example.com/post1",
                "title": "Article 1",
                "summary": "Summary 1",
            }.get(k, d)
            entry1.published_parsed = (2024, 1, 15, 10, 30, 0, 0, 0, 0)

            # Second entry has no date attributes
            entry2 = MagicMock(spec=["get"])
            entry2.get.side_effect = lambda k, d=None: {
                "link": "https://example.com/post2",
                "title": "Article 2",
            }.get(k, d)

            mock_feed.entries = [entry1, entry2]
            mock_parse.return_value = mock_feed

            with patch("httpx.Client"):
                fetcher = FeedFetcher()
                articles = list(fetcher.fetch_feed("https://example.com/feed"))

                # First article has published_parsed
                assert articles[0].published_at is not None
                assert articles[0].published_at.year == 2024
                assert articles[0].published_at.month == 1
                assert articles[0].published_at.day == 15

                # Second article has no date
                assert articles[1].published_at is None

    def test_fetch_feed_extracts_author(self, mock_feedparser):
        """Test that author is extracted."""
        with patch("httpx.Client"):
            fetcher = FeedFetcher()
            articles = list(fetcher.fetch_feed("https://example.com/feed"))

            assert articles[0].author == "Author 1"
            assert articles[1].author is None

    def test_fetch_feed_extracts_summary(self, mock_feedparser):
        """Test that summary is extracted."""
        with patch("httpx.Client"):
            fetcher = FeedFetcher()
            articles = list(fetcher.fetch_feed("https://example.com/feed"))

            assert articles[0].summary == "Summary of article 1"
            assert articles[1].summary == "Summary of article 2"

    def test_fetch_feed_uses_feed_title_as_default_source(self, mock_feedparser):
        """Test that feed title is used as default source name."""
        with patch("httpx.Client"):
            fetcher = FeedFetcher()
            articles = list(fetcher.fetch_feed("https://example.com/feed"))

            assert all(a.source == "Test Feed" for a in articles)

    def test_fetch_feed_handles_updated_parsed(self):
        """Test fallback to updated_parsed for date."""
        with patch("feedparser.parse") as mock_parse:
            mock_feed = MagicMock()
            mock_feed.feed = {"title": "Test"}

            entry = MagicMock()
            entry.get.side_effect = lambda k, d=None: {
                "link": "https://example.com/post",
                "title": "Article",
            }.get(k, d)
            entry.published_parsed = None
            entry.updated_parsed = (2024, 6, 15, 12, 0, 0, 0, 0, 0)

            mock_feed.entries = [entry]
            mock_parse.return_value = mock_feed

            with patch("httpx.Client"):
                fetcher = FeedFetcher()
                articles = list(fetcher.fetch_feed("https://example.com/feed"))

                assert articles[0].published_at.month == 6

    def test_fetch_full_content_success(self):
        """Test successful full content extraction."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "<html><body><p>Article content</p></body></html>"
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            with patch("trafilatura.extract") as mock_extract:
                mock_extract.return_value = "Article content"

                fetcher = FeedFetcher()
                content = fetcher.fetch_full_content("https://example.com/article")

                assert content == "Article content"
                mock_extract.assert_called_once()

    def test_fetch_full_content_error(self):
        """Test content extraction on HTTP error."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client

            fetcher = FeedFetcher()
            content = fetcher.fetch_full_content("https://example.com/article")

            assert content is None

    def test_fetch_full_content_http_error(self):
        """Test content extraction on HTTP status error."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            fetcher = FeedFetcher()
            content = fetcher.fetch_full_content("https://example.com/article")

            assert content is None

    def test_fetcher_close(self):
        """Test closing the fetcher."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            fetcher = FeedFetcher()
            fetcher.close()

            mock_client.close.assert_called_once()


class TestFeedCategories:
    """Tests for feed category constants."""

    def test_feed_categories_structure(self):
        """Test that FEED_CATEGORIES has expected structure."""
        expected_categories = [
            "tech", "world", "us", "politics",
            "business", "science", "middle_east", "europe", "asia"
        ]

        for cat in expected_categories:
            assert cat in FEED_CATEGORIES
            assert isinstance(FEED_CATEGORIES[cat], list)
            assert len(FEED_CATEGORIES[cat]) > 0

    def test_feed_entries_are_tuples(self):
        """Test that feed entries are (name, url) tuples."""
        for category, feeds in FEED_CATEGORIES.items():
            for feed in feeds:
                assert isinstance(feed, tuple)
                assert len(feed) == 2
                name, url = feed
                assert isinstance(name, str)
                assert isinstance(url, str)
                assert url.startswith("http")

    def test_default_feeds_is_tech(self):
        """Test that DEFAULT_FEEDS is the tech category."""
        assert DEFAULT_FEEDS == FEED_CATEGORIES["tech"]

    def test_all_feeds_contains_all_categories(self):
        """Test that ALL_FEEDS contains feeds from all categories."""
        total_expected = sum(len(feeds) for feeds in FEED_CATEGORIES.values())
        assert len(ALL_FEEDS) == total_expected

    def test_all_feeds_contains_tech_feeds(self):
        """Test that ALL_FEEDS contains all tech feeds."""
        for feed in FEED_CATEGORIES["tech"]:
            assert feed in ALL_FEEDS

    def test_category_has_valid_urls(self):
        """Test that feed URLs look valid."""
        for category, feeds in FEED_CATEGORIES.items():
            for name, url in feeds:
                assert "://" in url
                assert len(url) > 10
