"""Tests for the CLI commands."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from ai_news_tracker.cli import main, get_recommender
from ai_news_tracker.models import Base, Article, FeedSource, UserProfile
from ai_news_tracker.embeddings import embedding_to_bytes


@pytest.fixture
def cli_runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        os.unlink(db_path)
        # Also remove WAL files if they exist
        for ext in ["-wal", "-shm"]:
            try:
                os.unlink(db_path + ext)
            except:
                pass
    except:
        pass


@pytest.fixture
def mock_recommender():
    """Create a mock recommender."""
    mock = MagicMock()
    mock.get_recommendations.return_value = []
    mock.search_by_topic.return_value = []
    mock.get_stats.return_value = {
        "total_articles": 10,
        "unread_articles": 8,
        "liked_articles": 1,
        "disliked_articles": 1,
        "active_sources": 3,
        "has_preferences": True,
    }
    return mock


class TestGetRecommender:
    """Tests for the get_recommender helper."""

    def test_get_recommender_creates_db(self, temp_db_path):
        """Test that get_recommender creates database."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384
            recommender, session = get_recommender(temp_db_path)

            assert recommender is not None
            assert session is not None
            assert os.path.exists(temp_db_path)


class TestInitCommand:
    """Tests for the init command."""

    def test_init_default_tech_feeds(self, cli_runner, temp_db_path):
        """Test init with default tech feeds."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            result = cli_runner.invoke(main, ["--db", temp_db_path, "init"])

            assert result.exit_code == 0
            assert "Initialized" in result.output
            assert "news fetch" in result.output

    def test_init_all_feeds(self, cli_runner, temp_db_path):
        """Test init with all feeds."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            result = cli_runner.invoke(main, ["--db", temp_db_path, "init", "--all"])

            assert result.exit_code == 0
            assert "all news sources" in result.output.lower()

    def test_init_specific_category(self, cli_runner, temp_db_path):
        """Test init with specific category."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            result = cli_runner.invoke(main, ["--db", temp_db_path, "init", "-c", "world"])

            assert result.exit_code == 0
            assert "world" in result.output.lower()

    def test_init_multiple_categories(self, cli_runner, temp_db_path):
        """Test init with multiple categories."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            result = cli_runner.invoke(
                main, ["--db", temp_db_path, "init", "-c", "world", "-c", "politics"]
            )

            assert result.exit_code == 0

    def test_init_unknown_category(self, cli_runner, temp_db_path):
        """Test init with unknown category."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            result = cli_runner.invoke(main, ["--db", temp_db_path, "init", "-c", "unknown"])

            assert "Unknown category" in result.output

    def test_init_skips_existing(self, cli_runner, temp_db_path):
        """Test that init skips existing sources."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            # First init
            cli_runner.invoke(main, ["--db", temp_db_path, "init"])

            # Second init should skip existing
            result = cli_runner.invoke(main, ["--db", temp_db_path, "init"])

            assert result.exit_code == 0
            assert "0 new sources" in result.output


class TestCategoriesCommand:
    """Tests for the categories command."""

    def test_categories_list(self, cli_runner, temp_db_path):
        """Test listing categories."""
        result = cli_runner.invoke(main, ["categories"])

        assert result.exit_code == 0
        assert "tech" in result.output.lower()
        assert "world" in result.output.lower()
        assert "politics" in result.output.lower()


class TestFetchCommand:
    """Tests for the fetch command."""

    def test_fetch_no_sources(self, cli_runner, temp_db_path):
        """Test fetch with no configured sources."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            result = cli_runner.invoke(main, ["--db", temp_db_path, "fetch"])

            assert "No feed sources" in result.output

    def test_fetch_with_sources(self, cli_runner, temp_db_path):
        """Test fetch with sources configured."""
        with patch("ai_news_tracker.cli.EmbeddingEngine") as mock_engine:
            mock_engine.return_value.embedding_dim = 384

            # First init
            cli_runner.invoke(main, ["--db", temp_db_path, "init"])

            # Mock the refresh to avoid actual network calls
            with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
                mock_rec = MagicMock()
                mock_rec.refresh_all_feeds.return_value = 5
                mock_session = MagicMock()
                mock_session.query.return_value.filter_by.return_value.count.return_value = 3
                mock_get_rec.return_value = (mock_rec, mock_session)

                result = cli_runner.invoke(main, ["--db", temp_db_path, "fetch"])

                assert result.exit_code == 0
                assert "Fetched" in result.output


class TestReadCommand:
    """Tests for the read command."""

    def test_read_no_articles(self, cli_runner, temp_db_path):
        """Test read with no articles."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_rec = MagicMock()
            mock_rec.get_recommendations.return_value = []
            mock_get_rec.return_value = (mock_rec, MagicMock())

            result = cli_runner.invoke(main, ["--db", temp_db_path, "read"])

            assert "No articles found" in result.output

    def test_read_with_articles(self, cli_runner, temp_db_path):
        """Test read with articles."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_article = MagicMock()
            mock_article.id = 1
            mock_article.title = "Test Article"
            mock_article.source = "TestSource"

            mock_rec = MagicMock()
            mock_rec.get_recommendations.return_value = [
                (mock_article, 0.8, 0.9)
            ]
            mock_get_rec.return_value = (mock_rec, MagicMock())

            result = cli_runner.invoke(main, ["--db", temp_db_path, "read"])

            assert result.exit_code == 0
            assert "Test Article" in result.output

    def test_read_with_limit(self, cli_runner, temp_db_path):
        """Test read with custom limit."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_rec = MagicMock()
            mock_rec.get_recommendations.return_value = []
            mock_get_rec.return_value = (mock_rec, MagicMock())

            cli_runner.invoke(main, ["--db", temp_db_path, "read", "-n", "5"])

            mock_rec.get_recommendations.assert_called_once()
            call_kwargs = mock_rec.get_recommendations.call_args[1]
            assert call_kwargs["limit"] == 5


class TestTopicCommand:
    """Tests for the topic command."""

    def test_topic_search(self, cli_runner, temp_db_path):
        """Test topic search."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_article = MagicMock()
            mock_article.id = 1
            mock_article.title = "ML Article"
            mock_article.source = "Tech"

            mock_rec = MagicMock()
            mock_rec.search_by_topic.return_value = [
                (mock_article, 0.9, 0.8)
            ]
            mock_get_rec.return_value = (mock_rec, MagicMock())

            result = cli_runner.invoke(
                main, ["--db", temp_db_path, "topic", "machine", "learning"]
            )

            assert result.exit_code == 0
            assert "machine learning" in result.output.lower()

    def test_topic_no_results(self, cli_runner, temp_db_path):
        """Test topic search with no results."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_rec = MagicMock()
            mock_rec.search_by_topic.return_value = []
            mock_get_rec.return_value = (mock_rec, MagicMock())

            result = cli_runner.invoke(main, ["--db", temp_db_path, "topic", "xyz123"])

            assert "No matching articles" in result.output


class TestLikeDislikeCommands:
    """Tests for the like and dislike commands."""

    def test_like_article(self, cli_runner, temp_db_path):
        """Test liking an article."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_rec = MagicMock()
            mock_get_rec.return_value = (mock_rec, MagicMock())

            result = cli_runner.invoke(main, ["--db", temp_db_path, "like", "1"])

            assert result.exit_code == 0
            assert "Liked article 1" in result.output
            mock_rec.record_feedback.assert_called_once_with(1, liked=True)

    def test_dislike_article(self, cli_runner, temp_db_path):
        """Test disliking an article."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_rec = MagicMock()
            mock_get_rec.return_value = (mock_rec, MagicMock())

            result = cli_runner.invoke(main, ["--db", temp_db_path, "dislike", "2"])

            assert result.exit_code == 0
            assert "Disliked article 2" in result.output
            mock_rec.record_feedback.assert_called_once_with(2, liked=False)


class TestOpenCommand:
    """Tests for the open command."""

    def test_open_article_not_found(self, cli_runner, temp_db_path):
        """Test opening nonexistent article."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_session = MagicMock()
            mock_session.query.return_value.get.return_value = None
            mock_get_rec.return_value = (MagicMock(), mock_session)

            result = cli_runner.invoke(main, ["--db", temp_db_path, "open", "999"])

            assert "not found" in result.output.lower()

    def test_open_article_found(self, cli_runner, temp_db_path):
        """Test opening existing article."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_article = MagicMock()
            mock_article.id = 1
            mock_article.title = "Test Article"
            mock_article.url = "https://example.com/test"
            mock_article.source = "Test"
            mock_article.author = "Author"
            mock_article.published_at = datetime.utcnow()
            mock_article.summary = "Test summary"
            mock_article.content = None

            mock_session = MagicMock()
            mock_session.query.return_value.get.return_value = mock_article
            mock_rec = MagicMock()
            mock_get_rec.return_value = (mock_rec, mock_session)

            # Answer "no" to browser prompt
            result = cli_runner.invoke(
                main, ["--db", temp_db_path, "open", "1"],
                input="n\n"
            )

            assert result.exit_code == 0
            assert "Test Article" in result.output
            mock_rec.mark_read.assert_called_once_with(1)


class TestAddSourceCommand:
    """Tests for the add_source command."""

    def test_add_source(self, cli_runner, temp_db_path):
        """Test adding a feed source."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_rec = MagicMock()
            mock_get_rec.return_value = (mock_rec, MagicMock())

            # Note: Click command is add_source but CLI uses add-source
            result = cli_runner.invoke(
                main,
                ["--db", temp_db_path, "add-source", "My Feed", "https://example.com/feed"]
            )

            assert result.exit_code == 0
            assert "Added source" in result.output
            mock_rec.add_feed_source.assert_called_once_with("My Feed", "https://example.com/feed")


class TestSourcesCommand:
    """Tests for the sources command."""

    def test_sources_empty(self, cli_runner, temp_db_path):
        """Test sources with no sources configured."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_session = MagicMock()
            mock_session.query.return_value.all.return_value = []
            mock_get_rec.return_value = (MagicMock(), mock_session)

            result = cli_runner.invoke(main, ["--db", temp_db_path, "sources"])

            assert "No sources configured" in result.output

    def test_sources_list(self, cli_runner, temp_db_path):
        """Test listing sources."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_source = MagicMock()
            mock_source.id = 1
            mock_source.name = "Test Feed"
            mock_source.url = "https://test.com/feed"
            mock_source.is_active = True
            mock_source.last_fetched = None

            mock_session = MagicMock()
            mock_session.query.return_value.all.return_value = [mock_source]
            mock_get_rec.return_value = (MagicMock(), mock_session)

            result = cli_runner.invoke(main, ["--db", temp_db_path, "sources"])

            assert result.exit_code == 0
            assert "Test Feed" in result.output


class TestStatsCommand:
    """Tests for the stats command."""

    def test_stats(self, cli_runner, temp_db_path):
        """Test stats display."""
        with patch("ai_news_tracker.cli.get_recommender") as mock_get_rec:
            mock_rec = MagicMock()
            mock_rec.get_stats.return_value = {
                "total_articles": 100,
                "unread_articles": 90,
                "liked_articles": 5,
                "disliked_articles": 2,
                "active_sources": 10,
                "has_preferences": True,
            }
            mock_get_rec.return_value = (mock_rec, MagicMock())

            result = cli_runner.invoke(main, ["--db", temp_db_path, "stats"])

            assert result.exit_code == 0
            assert "100" in result.output  # Total articles
            assert "90" in result.output  # Unread
            assert "5" in result.output  # Liked


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_starts_server(self, cli_runner, temp_db_path):
        """Test that serve command starts server."""
        # run_server is imported inside the serve command, so patch at web module level
        with patch("ai_news_tracker.web.run_server") as mock_run:
            result = cli_runner.invoke(
                main,
                ["--db", temp_db_path, "serve", "--host", "0.0.0.0", "-p", "9000"]
            )

            mock_run.assert_called_once_with(
                host="0.0.0.0",
                port=9000,
                db_path=temp_db_path,
            )
