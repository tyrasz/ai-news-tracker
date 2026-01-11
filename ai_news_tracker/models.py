"""Database models for storing articles and user preferences."""

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import logging
import time

from sqlalchemy import create_engine, event, text, Column, Integer, String, Text, Float, DateTime, Boolean, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, SQLAlchemyError

Base = declarative_base()
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

    id = Column(Integer, primary_key=True)
    url = Column(String(2048), unique=True, nullable=False)
    title = Column(String(512), nullable=False)
    content = Column(Text)
    summary = Column(Text)
    source = Column(String(256))  # e.g., "Hacker News", "TechCrunch"
    author = Column(String(256))
    published_at = Column(DateTime)
    fetched_at = Column(DateTime, default=datetime.utcnow)

    # Embedding stored as binary (numpy array bytes)
    embedding = Column(LargeBinary)

    # User interaction
    is_read = Column(Boolean, default=False)
    is_liked = Column(Boolean, default=None)  # None = no feedback, True = liked, False = disliked
    read_at = Column(DateTime)

    def __repr__(self):
        return f"<Article {self.id}: {self.title[:50]}>"


class UserProfile(Base):
    """User preference profile stored as embedding centroid."""

    __tablename__ = "user_profile"

    id = Column(Integer, primary_key=True)
    name = Column(String(256), default="default")

    # Preference embedding (weighted average of liked articles)
    preference_embedding = Column(LargeBinary)

    # Stats
    articles_liked = Column(Integer, default=0)
    articles_disliked = Column(Integer, default=0)

    # Settings
    decay_factor = Column(Float, default=0.95)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeedSource(Base):
    """RSS/Atom feed source configuration."""

    __tablename__ = "feed_sources"

    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    url = Column(String(2048), unique=True, nullable=False)
    feed_type = Column(String(50), default="rss")  # rss, atom, json
    is_active = Column(Boolean, default=True)
    last_fetched = Column(DateTime)
    fetch_interval_minutes = Column(Integer, default=60)

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
