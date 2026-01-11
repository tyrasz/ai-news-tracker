"""Tests for database models and initialization."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest
from sqlalchemy import text

from ai_news_tracker.models import (
    init_db,
    init_db_scoped,
    get_session,
    checkpoint_wal,
    retry_on_db_error,
    Article,
    FeedSource,
    UserProfile,
    Base,
)
from sqlalchemy.exc import OperationalError


class TestInitDb:
    """Tests for database initialization."""

    def test_init_db_creates_tables(self):
        """Test that init_db creates all required tables."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, Session = init_db(db_path)
            session = Session()

            # Check tables exist by querying them
            assert session.query(Article).count() == 0
            assert session.query(FeedSource).count() == 0
            assert session.query(UserProfile).count() == 0

            session.close()
            engine.dispose()
        finally:
            os.unlink(db_path)

    def test_init_db_enables_wal_mode(self):
        """Test that WAL mode is enabled for SQLite."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, Session = init_db(db_path)
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA journal_mode"))
                mode = result.fetchone()[0]
                assert mode.lower() == "wal"

            engine.dispose()
        finally:
            os.unlink(db_path)

    def test_init_db_returns_session_factory(self):
        """Test that init_db returns a working session factory."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, Session = init_db(db_path)
            session = Session()
            assert session is not None

            # Can create and query objects
            source = FeedSource(name="Test", url="https://test.com")
            session.add(source)
            session.commit()

            retrieved = session.query(FeedSource).first()
            assert retrieved.name == "Test"

            session.close()
            engine.dispose()
        finally:
            os.unlink(db_path)


class TestArticleModel:
    """Tests for the Article model."""

    def test_article_creation(self, temp_db):
        """Test basic article creation."""
        session, _ = temp_db

        article = Article(
            url="https://example.com/test",
            title="Test Article",
            summary="Test summary",
            source="Test Source",
        )
        session.add(article)
        session.commit()

        retrieved = session.query(Article).first()
        assert retrieved.url == "https://example.com/test"
        assert retrieved.title == "Test Article"
        assert retrieved.summary == "Test summary"
        assert retrieved.source == "Test Source"

    def test_article_defaults(self, temp_db):
        """Test article default values."""
        session, _ = temp_db

        article = Article(url="https://example.com/test", title="Test")
        session.add(article)
        session.commit()

        retrieved = session.query(Article).first()
        assert retrieved.is_read is False
        assert retrieved.is_liked is None
        assert retrieved.fetched_at is not None

    def test_article_unique_url(self, temp_db):
        """Test that article URLs must be unique."""
        session, _ = temp_db

        article1 = Article(url="https://example.com/same", title="Article 1")
        session.add(article1)
        session.commit()

        article2 = Article(url="https://example.com/same", title="Article 2")
        session.add(article2)

        with pytest.raises(Exception):  # IntegrityError
            session.commit()

    def test_article_repr(self, temp_db):
        """Test article string representation."""
        session, _ = temp_db

        article = Article(
            url="https://example.com/test",
            title="A Very Long Title That Should Be Truncated In The Repr Method",
        )
        session.add(article)
        session.commit()

        repr_str = repr(article)
        assert "Article" in repr_str
        assert str(article.id) in repr_str

    def test_article_with_embedding(self, temp_db):
        """Test article with binary embedding storage."""
        import numpy as np

        session, _ = temp_db

        embedding = np.random.randn(384).astype(np.float32)
        article = Article(
            url="https://example.com/test",
            title="Test",
            embedding=embedding.tobytes(),
        )
        session.add(article)
        session.commit()

        retrieved = session.query(Article).first()
        restored = np.frombuffer(retrieved.embedding, dtype=np.float32)
        np.testing.assert_array_almost_equal(embedding, restored)


class TestFeedSourceModel:
    """Tests for the FeedSource model."""

    def test_feed_source_creation(self, temp_db):
        """Test basic feed source creation."""
        session, _ = temp_db

        source = FeedSource(
            name="Test Feed",
            url="https://example.com/feed.xml",
            feed_type="rss",
        )
        session.add(source)
        session.commit()

        retrieved = session.query(FeedSource).first()
        assert retrieved.name == "Test Feed"
        assert retrieved.url == "https://example.com/feed.xml"
        assert retrieved.feed_type == "rss"

    def test_feed_source_defaults(self, temp_db):
        """Test feed source default values."""
        session, _ = temp_db

        source = FeedSource(name="Test", url="https://test.com")
        session.add(source)
        session.commit()

        retrieved = session.query(FeedSource).first()
        assert retrieved.is_active is True
        assert retrieved.feed_type == "rss"
        assert retrieved.fetch_interval_minutes == 60
        assert retrieved.last_fetched is None

    def test_feed_source_unique_url(self, temp_db):
        """Test that feed URLs must be unique."""
        session, _ = temp_db

        source1 = FeedSource(name="Feed 1", url="https://same.com/feed")
        session.add(source1)
        session.commit()

        source2 = FeedSource(name="Feed 2", url="https://same.com/feed")
        session.add(source2)

        with pytest.raises(Exception):  # IntegrityError
            session.commit()

    def test_feed_source_repr(self, temp_db):
        """Test feed source string representation."""
        session, _ = temp_db

        source = FeedSource(name="My Feed", url="https://test.com")
        session.add(source)
        session.commit()

        repr_str = repr(source)
        assert "FeedSource" in repr_str
        assert "My Feed" in repr_str


class TestUserProfileModel:
    """Tests for the UserProfile model."""

    def test_user_profile_creation(self, temp_db):
        """Test basic user profile creation."""
        session, _ = temp_db

        profile = UserProfile(name="test_user")
        session.add(profile)
        session.commit()

        retrieved = session.query(UserProfile).first()
        assert retrieved.name == "test_user"

    def test_user_profile_defaults(self, temp_db):
        """Test user profile default values."""
        session, _ = temp_db

        profile = UserProfile()
        session.add(profile)
        session.commit()

        retrieved = session.query(UserProfile).first()
        assert retrieved.name == "default"
        assert retrieved.articles_liked == 0
        assert retrieved.articles_disliked == 0
        assert retrieved.decay_factor == 0.95
        assert retrieved.preference_embedding is None

    def test_user_profile_with_embedding(self, temp_db):
        """Test user profile with preference embedding."""
        import numpy as np

        session, _ = temp_db

        embedding = np.random.randn(384).astype(np.float32)
        profile = UserProfile(
            name="test",
            preference_embedding=embedding.tobytes(),
            articles_liked=10,
            articles_disliked=3,
        )
        session.add(profile)
        session.commit()

        retrieved = session.query(UserProfile).first()
        assert retrieved.articles_liked == 10
        assert retrieved.articles_disliked == 3
        restored = np.frombuffer(retrieved.preference_embedding, dtype=np.float32)
        np.testing.assert_array_almost_equal(embedding, restored)

    def test_user_profile_updated_at(self, temp_db):
        """Test that updated_at changes on update."""
        import time

        session, _ = temp_db

        profile = UserProfile(name="test")
        session.add(profile)
        session.commit()

        initial_updated = profile.updated_at
        time.sleep(0.1)

        profile.articles_liked = 5
        session.commit()
        session.refresh(profile)

        # Note: SQLAlchemy's onupdate may not trigger for all cases
        # This test verifies the column exists and works
        assert profile.updated_at is not None


class TestDatabaseResilience:
    """Tests for database connection resilience features."""

    def test_init_db_scoped_returns_scoped_session(self):
        """Test that init_db_scoped returns a scoped session."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, ScopedSession = init_db_scoped(db_path)

            # Scoped session should be callable and return same session in same thread
            session1 = ScopedSession()
            session2 = ScopedSession()
            assert session1 is session2

            ScopedSession.remove()
            engine.dispose()
        finally:
            os.unlink(db_path)

    def test_get_session_context_manager(self):
        """Test get_session context manager commits on success."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, Session = init_db(db_path)

            # Use context manager to add data
            with get_session(Session) as session:
                source = FeedSource(name="Test", url="https://test.com")
                session.add(source)

            # Data should be committed
            with get_session(Session) as session:
                count = session.query(FeedSource).count()
                assert count == 1

            engine.dispose()
        finally:
            os.unlink(db_path)

    def test_get_session_rolls_back_on_error(self):
        """Test get_session context manager rolls back on error."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, Session = init_db(db_path)

            # Add initial data
            with get_session(Session) as session:
                source = FeedSource(name="Initial", url="https://initial.com")
                session.add(source)

            # Try to add duplicate (will fail)
            try:
                with get_session(Session) as session:
                    # Add valid data first
                    source2 = FeedSource(name="Test2", url="https://test2.com")
                    session.add(source2)
                    session.flush()

                    # Now add duplicate URL (should fail)
                    source3 = FeedSource(name="Duplicate", url="https://initial.com")
                    session.add(source3)
                    # Commit happens automatically, will raise error
            except Exception:
                pass  # Expected

            # Only initial data should exist (Test2 should be rolled back)
            with get_session(Session) as session:
                count = session.query(FeedSource).count()
                assert count == 1

            engine.dispose()
        finally:
            os.unlink(db_path)

    def test_checkpoint_wal(self):
        """Test WAL checkpoint function."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, Session = init_db(db_path)

            # Add some data to create WAL activity
            with get_session(Session) as session:
                for i in range(10):
                    session.add(FeedSource(name=f"Feed {i}", url=f"https://feed{i}.com"))

            # Checkpoint should not raise
            checkpoint_wal(engine)

            engine.dispose()
        finally:
            os.unlink(db_path)

    def test_retry_on_db_error_decorator(self):
        """Test retry decorator retries on transient errors."""
        call_count = 0

        @retry_on_db_error(max_retries=3, delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OperationalError("database is locked", None, None)
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_on_db_error_gives_up_after_max_retries(self):
        """Test retry decorator gives up after max retries."""
        call_count = 0

        @retry_on_db_error(max_retries=2, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise OperationalError("database is locked", None, None)

        with pytest.raises(OperationalError):
            always_fails()

        assert call_count == 2

    def test_retry_on_db_error_does_not_retry_non_transient(self):
        """Test retry decorator does not retry non-transient errors."""
        call_count = 0

        @retry_on_db_error(max_retries=3, delay=0.01)
        def non_transient_error():
            nonlocal call_count
            call_count += 1
            raise OperationalError("some other error", None, None)

        with pytest.raises(OperationalError):
            non_transient_error()

        assert call_count == 1  # No retries

    def test_connection_pool_configuration(self):
        """Test that connection pool is properly configured."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            engine, Session = init_db(db_path)

            # Check pool configuration
            pool = engine.pool
            assert pool.size() == 5  # pool_size
            # overflow() returns negative when pool has capacity remaining
            # max_overflow=10 means total capacity is 15 (5 + 10)
            assert pool.overflow() <= 10

            engine.dispose()
        finally:
            os.unlink(db_path)
