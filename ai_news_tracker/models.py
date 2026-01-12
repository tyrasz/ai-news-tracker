"""Database models for storing articles and user preferences."""

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import hashlib
import logging
import secrets
import time
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy import create_engine, event, text, String, Text, Float, DateTime, LargeBinary, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, scoped_session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, SQLAlchemyError


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


logger = logging.getLogger(__name__)


def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode and other optimizations for SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((salt + password).encode())
    return f"{salt}:{hash_obj.hexdigest()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    try:
        salt, hash_value = hashed.split(":")
        hash_obj = hashlib.sha256((salt + password).encode())
        return hash_obj.hexdigest() == hash_value
    except ValueError:
        return False


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


class User(Base):
    """User account for multi-user support."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    api_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, default=generate_api_key)
    display_name: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    # Relationships
    interactions: Mapped[List["UserArticleInteraction"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    profile: Mapped[Optional["UserProfile"]] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    notes: Mapped[List["ArticleNote"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        """Set the user's password."""
        self.password_hash = hash_password(password)

    def check_password(self, password: str) -> bool:
        """Check if password matches."""
        return verify_password(password, self.password_hash)

    def regenerate_api_key(self) -> str:
        """Generate a new API key."""
        self.api_key = generate_api_key()
        return self.api_key

    def __repr__(self):
        return f"<User {self.id}: {self.email}>"


class Article(Base):
    """Stored news article with metadata and embedding."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, default=None)
    summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    source: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    author: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)

    # Embedding stored as binary (numpy array bytes)
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, default=None)

    # Legacy single-user fields (kept for backwards compatibility, use UserArticleInteraction for multi-user)
    is_read: Mapped[bool] = mapped_column(default=False)
    is_liked: Mapped[Optional[bool]] = mapped_column(default=None)
    is_bookmarked: Mapped[bool] = mapped_column(default=False)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    bookmarked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    # Relationships
    interactions: Mapped[List["UserArticleInteraction"]] = relationship(back_populates="article", cascade="all, delete-orphan")
    notes: Mapped[List["ArticleNote"]] = relationship(back_populates="article", cascade="all, delete-orphan")

    # Average reading speed in words per minute
    WORDS_PER_MINUTE = 238

    @property
    def reading_time_minutes(self) -> int:
        """Estimate reading time in minutes based on content length."""
        text = self.content or self.summary or ""
        word_count = len(text.split())
        minutes = max(1, round(word_count / self.WORDS_PER_MINUTE))
        return minutes

    @property
    def word_count(self) -> int:
        """Count words in article content."""
        text = self.content or self.summary or ""
        return len(text.split())

    def __repr__(self):
        return f"<Article {self.id}: {self.title[:50]}>"


class UserArticleInteraction(Base):
    """User-specific interaction data for articles (multi-user support)."""

    __tablename__ = "user_article_interactions"
    __table_args__ = (
        Index("ix_user_article", "user_id", "article_id", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), nullable=False)

    is_read: Mapped[bool] = mapped_column(default=False)
    is_liked: Mapped[Optional[bool]] = mapped_column(default=None)  # None = no feedback
    is_bookmarked: Mapped[bool] = mapped_column(default=False)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    bookmarked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    liked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="interactions")
    article: Mapped["Article"] = relationship(back_populates="interactions")

    def __repr__(self):
        return f"<UserArticleInteraction user={self.user_id} article={self.article_id}>"


class UserProfile(Base):
    """User preference profile stored as embedding centroid."""

    __tablename__ = "user_profile"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), default=None)
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

    # Relationships
    user: Mapped[Optional["User"]] = relationship(back_populates="profile")


class FeedSource(Base):
    """RSS/Atom feed source configuration."""

    __tablename__ = "feed_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    feed_type: Mapped[str] = mapped_column(String(50), default="rss")
    is_active: Mapped[bool] = mapped_column(default=True)
    last_fetched: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    fetch_interval_minutes: Mapped[int] = mapped_column(default=60)

    def __repr__(self):
        return f"<FeedSource {self.name}>"


class ArticleNote(Base):
    """User notes and highlights for articles."""

    __tablename__ = "article_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), default=None)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    highlight_text: Mapped[Optional[str]] = mapped_column(Text, default=None)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user: Mapped[Optional["User"]] = relationship(back_populates="notes")
    article: Mapped["Article"] = relationship(back_populates="notes")

    def __repr__(self):
        return f"<ArticleNote {self.id} for Article {self.article_id}>"


def retry_on_db_error(max_retries: int = 3, delay: float = 0.1):
    """Decorator that retries database operations on transient errors."""
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
    """Initialize the database and return engine + session factory."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )

    event.listen(engine, "connect", _set_sqlite_pragma)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


def init_db_scoped(db_path: str = "news_tracker.db"):
    """Initialize database with thread-safe scoped sessions."""
    engine, Session = init_db(db_path)
    ScopedSession = scoped_session(Session)
    return engine, ScopedSession


@contextmanager
def get_session(Session):
    """Context manager for database sessions with automatic cleanup."""
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
    """Checkpoint the WAL file to prevent unbounded growth."""
    with engine.connect() as conn:
        conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        logger.debug("WAL checkpoint completed")


def get_or_create_user(session, email: str, password: str) -> User:
    """Get existing user or create a new one."""
    user = session.query(User).filter(User.email == email).first()
    if user:
        return user

    user = User(email=email)
    user.set_password(password)
    session.add(user)
    session.flush()
    return user


def get_user_by_api_key(session, api_key: str) -> Optional[User]:
    """Get user by API key."""
    return session.query(User).filter(User.api_key == api_key, User.is_active == True).first()  # noqa: E712


def get_or_create_interaction(session, user_id: int, article_id: int) -> UserArticleInteraction:
    """Get existing interaction or create a new one."""
    interaction = (
        session.query(UserArticleInteraction)
        .filter(UserArticleInteraction.user_id == user_id, UserArticleInteraction.article_id == article_id)
        .first()
    )
    if interaction:
        return interaction

    interaction = UserArticleInteraction(user_id=user_id, article_id=article_id)
    session.add(interaction)
    session.flush()
    return interaction
