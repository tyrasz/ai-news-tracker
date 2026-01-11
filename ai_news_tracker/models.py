"""Database models for storing articles and user preferences."""

from datetime import datetime
from sqlalchemy import create_engine, event, Column, Integer, String, Text, Float, DateTime, Boolean, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


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


def init_db(db_path: str = "news_tracker.db"):
    """Initialize the database and return engine + session factory."""
    engine = create_engine(f"sqlite:///{db_path}")

    # Register SQLite optimizations (WAL mode, etc.)
    event.listen(engine, "connect", _set_sqlite_pragma)

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session
