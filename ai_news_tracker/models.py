"""Database models for storing articles and user preferences."""

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import logging
import time
from typing import Optional

from sqlalchemy import create_engine, event, text, String, Text, Float, DateTime, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, SQLAlchemyError


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


logger = logging.getLogger(__name__)


def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode and other optimizations for SQLite."""
    cursor = dbapi_connection.cursor()
    # WAL mode allows concurrent reads while writing
    cursor.execute("PRAGMA journal_mode=WAL")
    # Synchronous NORMAL is safer than OFF but faster than FULL
    cursor.execute("PRAGMA synchronous=NORMAL")
    # Increase cache size for better performance (negative = KB)
    cursor.execute("PRAGMA cache_size=-64000")
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Article(Base):
    """Stored news article with metadata and embedding."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, default=None)
    summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    source: Mapped[Optional[str]] = mapped_column(String(256), default=None)  # e.g., "Hacker News", "TechCrunch"
    author: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)

    # Embedding stored as binary (numpy array bytes)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, default=None)

    # User interaction
    is_read: Mapped[bool] = mapped_column(default=False)
    is_liked: Mapped[Optional[bool]] = mapped_column(default=None)  # None = no feedback, True = liked, False = disliked
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    def __repr__(self):
        return f"<Article {self.id}: {self.title[:50]}>"


class UserProfile(Base):
    """User preference profile stored as embedding centroid."""

    __tablename__ = "user_profile"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), default="default")

    # Preference embedding (weighted average of liked articles)
    preference_embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, default=None)

    # Stats
    articles_liked: Mapped[int] = mapped_column(default=0)
    articles_disliked: Mapped[int] = mapped_column(default=0)

    # Settings
    decay_factor: Mapped[float] = mapped_column(Float, default=0.95)

    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeedSource(Base):
    """RSS/Atom feed source configuration."""

    __tablename__ = "feed_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    feed_type: Mapped[str] = mapped_column(String(50), default="rss")  # rss, atom, json
    is_active: Mapped[bool] = mapped_column(default=True)
    last_fetched: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    fetch_interval_minutes: Mapped[int] = mapped_column(default=60)

    def __repr__(self):
        return f"<FeedSource {self.name}>"


def retry_on_db_error(max_retries: int = 3, delay: float = 0.1):
    """
    Decorator that retries database operations on transient errors.

    Handles SQLite locking issues and connection problems with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    last_error = e
                    error_msg = str(e).lower()
                    # Retry on transient errors
                    if any(err in error_msg for err in ['locked', 'busy', 'timeout']):
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Database locked, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    raise
                except SQLAlchemyError:
                    raise
            raise last_error
        return wrapper
    return decorator


def init_db(db_path: str = "news_tracker.db"):
    """
    Initialize the database and return engine + session factory.

    Configures connection pooling, WAL mode, and other optimizations.
    """
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,  # Verify connections before use
        connect_args={"check_same_thread": False},  # Allow multi-threaded access
    )

    # Register SQLite optimizations (WAL mode, etc.)
    event.listen(engine, "connect", _set_sqlite_pragma)

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


def init_db_scoped(db_path: str = "news_tracker.db"):
    """
    Initialize database with thread-safe scoped sessions.

    Use this for multi-threaded applications like web servers.
    Returns engine and a scoped session factory.
    """
    engine, Session = init_db(db_path)
    ScopedSession = scoped_session(Session)
    return engine, ScopedSession


@contextmanager
def get_session(Session):
    """
    Context manager for database sessions with automatic cleanup.

    Usage:
        engine, Session = init_db()
        with get_session(Session) as session:
            articles = session.query(Article).all()

    Automatically commits on success, rolls back on error, and closes session.
    """
    session = Session()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def checkpoint_wal(engine):
    """
    Checkpoint the WAL file to prevent unbounded growth.

    Call this periodically in long-running applications.
    """
    with engine.connect() as conn:
        conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        logger.debug("WAL checkpoint completed")
