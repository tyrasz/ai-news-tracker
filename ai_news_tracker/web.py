"""Web frontend for AI News Tracker."""

import asyncio
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, Body, Path as PathParam, Depends, Header
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import (
    init_db_scoped, checkpoint_wal, Article, ArticleNote, FeedSource,
    User, UserArticleInteraction, UserProfile,
    get_user_by_api_key, get_or_create_interaction, get_or_create_user,
)
from .embeddings import EmbeddingEngine
from .preferences import PreferenceLearner
from .recommender import NewsRecommender
from .logging_config import setup_logging, get_logger
from .scheduler import FeedRefreshScheduler
from .opml import export_opml, parse_opml, export_feeds_json
from .auth import AuthContext, UserCreate, UserLogin, UserResponse, create_auth_dependency

logger = get_logger(__name__)

# Global instances (initialized on startup)
recommender = None
db_session = None
engine = None
_scoped_session = None
wal_checkpoint_task = None
feed_scheduler: Optional[FeedRefreshScheduler] = None

STATIC_DIR = Path(__file__).parent / "static"


class AlgorithmType(str, Enum):
    """Valid recommendation algorithm types."""
    FOR_YOU = "for_you"
    EXPLORE = "explore"
    DEEP_DIVE = "deep_dive"
    TRENDING = "trending"
    BALANCED = "balanced"
    CONTRARIAN = "contrarian"


class ArticleResponse(BaseModel):
    id: int
    title: str
    url: str
    source: Optional[str]
    author: Optional[str]
    summary: Optional[str]
    published_at: Optional[str]
    score: float
    freshness: float
    is_read: bool
    is_liked: Optional[bool]
    is_bookmarked: bool = False
    reading_time_minutes: int = 1


class FeedbackRequest(BaseModel):
    article_id: int = Field(..., ge=1, description="Article ID to provide feedback for")
    liked: bool = Field(..., description="True for like, False for dislike")


class StatsResponse(BaseModel):
    total_articles: int
    unread_articles: int
    liked_articles: int
    disliked_articles: int
    active_sources: int
    has_preferences: bool


class ArticleGroupResponse(BaseModel):
    """A group of similar articles about the same story."""
    primary: ArticleResponse
    related: List[ArticleResponse]
    count: int
    sources: List[str]


class NoteRequest(BaseModel):
    """Request to create or update a note."""
    content: str = Field(..., min_length=1, max_length=10000, description="Note content")
    highlight_text: Optional[str] = Field(None, max_length=2000, description="Highlighted text from article")


class NoteResponse(BaseModel):
    """Response containing a note."""
    id: int
    article_id: int
    content: str
    highlight_text: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


async def periodic_wal_checkpoint():
    """Periodically checkpoint WAL to prevent unbounded growth."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        try:
            checkpoint_wal(engine)
            logger.debug("Periodic WAL checkpoint completed")
        except Exception as e:
            logger.warning(f"WAL checkpoint failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global recommender, db_session, engine, _scoped_session, wal_checkpoint_task, feed_scheduler

    # Startup
    db_path = os.environ.get("NEWS_DB_PATH", "news_tracker.db")
    engine, _scoped_session = init_db_scoped(db_path)
    db_session = _scoped_session()
    embedding_engine = EmbeddingEngine()
    preference_learner = PreferenceLearner(embedding_engine)
    recommender = NewsRecommender(db_session, embedding_engine, preference_learner)

    # Start background WAL checkpoint task
    wal_checkpoint_task = asyncio.create_task(periodic_wal_checkpoint())

    # Start feed refresh scheduler
    refresh_interval = int(os.environ.get("FEED_REFRESH_INTERVAL", "30"))
    scheduler_enabled = os.environ.get("FEED_REFRESH_ENABLED", "true").lower() == "true"
    feed_scheduler = FeedRefreshScheduler(
        refresh_callback=lambda: recommender.refresh_all_feeds(fetch_content=False) if recommender else 0,
        interval_minutes=refresh_interval,
        enabled=scheduler_enabled,
    )
    await feed_scheduler.start()

    logger.info(f"Started web server with database: {db_path}")
    logger.info(f"Feed scheduler: interval={refresh_interval}min, enabled={scheduler_enabled}")

    yield

    # Shutdown
    if feed_scheduler:
        await feed_scheduler.stop()

    if wal_checkpoint_task:
        wal_checkpoint_task.cancel()
        try:
            await wal_checkpoint_task
        except asyncio.CancelledError:
            pass

    # Final WAL checkpoint on shutdown
    try:
        checkpoint_wal(engine)
    except Exception as e:
        logger.warning(f"Final WAL checkpoint failed: {e}")

    # Clean up session
    if db_session:
        db_session.close()
    if _scoped_session:
        _scoped_session.remove()

    logger.info("Web server shutdown complete")


app = FastAPI(title="AI News Tracker", lifespan=lifespan)


# Auth dependency - uses lambda to defer session access until request time
def _get_db_session():
    """Get the current database session."""
    return db_session


get_auth = create_auth_dependency(_get_db_session)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Return readable validation errors."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        errors.append(f"{field}: {error['msg']}")
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": "Validation error", "details": errors}
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Return consistent error format for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all handler for unexpected errors."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


# ============== Authentication Endpoints ==============


@app.post("/api/auth/register", response_model=UserResponse)
def register_user(user_data: UserCreate):
    """Register a new user account."""
    assert db_session is not None, "Database session not initialized"

    # Check if email already exists
    existing = db_session.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    user = User(
        email=user_data.email,
        display_name=user_data.display_name,
    )
    user.set_password(user_data.password)
    db_session.add(user)

    # Create default user profile
    profile = UserProfile(user=user, name="default")
    db_session.add(profile)

    db_session.commit()

    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        api_key=user.api_key,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


@app.post("/api/auth/login", response_model=UserResponse)
def login_user(credentials: UserLogin):
    """Login with email and password to get API key."""
    assert db_session is not None, "Database session not initialized"

    user = db_session.query(User).filter(User.email == credentials.email).first()
    if not user or not user.check_password(credentials.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    # Update last login time
    user.last_login = datetime.utcnow()
    db_session.commit()

    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        api_key=user.api_key,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


@app.get("/api/auth/me", response_model=UserResponse)
def get_current_user(auth: AuthContext = Depends(get_auth)):
    """Get the current authenticated user's profile."""
    auth.require_auth()
    user = auth.user
    assert user is not None  # guaranteed by require_auth

    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        api_key=user.api_key,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


@app.post("/api/auth/regenerate-key", response_model=UserResponse)
def regenerate_api_key(auth: AuthContext = Depends(get_auth)):
    """Generate a new API key for the current user."""
    assert db_session is not None, "Database session not initialized"
    auth.require_auth()
    user = auth.user
    assert user is not None

    user.regenerate_api_key()
    db_session.commit()

    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        api_key=user.api_key,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else None,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the main HTML page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(
        content=html_path.read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/recommendations", response_model=List[ArticleResponse])
def get_recommendations(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of articles to return"),
    freshness_weight: float = Query(0.3, ge=0, le=1, description="Weight for freshness vs relevance (0=relevance only, 1=freshness only)"),
    include_read: bool = Query(False, description="Include articles already marked as read"),
    algorithm: AlgorithmType = Query(AlgorithmType.FOR_YOU, description="Recommendation algorithm to use"),
):
    """Get personalized article recommendations using the specified algorithm."""
    assert recommender is not None, "Recommender not initialized"
    results = recommender.get_recommendations_v2(
        algorithm=algorithm.value,
        limit=limit,
        include_read=include_read,
        freshness_weight=freshness_weight,
    )

    return [
        ArticleResponse(
            id=article.id,
            title=article.title,
            url=article.url,
            source=article.source,
            author=article.author,
            summary=_clean_summary(article.summary),
            published_at=article.published_at.isoformat() if article.published_at else None,
            score=score,
            freshness=freshness,
            is_read=article.is_read or False,
            is_liked=article.is_liked,
            is_bookmarked=article.is_bookmarked or False,
            reading_time_minutes=article.reading_time_minutes,
        )
        for article, score, freshness in results
    ]


@app.get("/api/algorithms")
def get_algorithms():
    """List available recommendation algorithms."""
    assert recommender is not None, "Recommender not initialized"
    return recommender.list_algorithms()


@app.get("/api/recommendations/grouped", response_model=List[ArticleGroupResponse])
def get_recommendations_grouped(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of article groups to return"),
    freshness_weight: float = Query(0.3, ge=0, le=1, description="Weight for freshness vs relevance"),
    include_read: bool = Query(False, description="Include articles already marked as read"),
    algorithm: AlgorithmType = Query(AlgorithmType.FOR_YOU, description="Recommendation algorithm to use"),
    similarity_threshold: float = Query(0.75, ge=0.5, le=0.95, description="Similarity threshold for grouping (0.75 = quite similar)"),
):
    """Get recommendations with similar articles grouped together."""
    assert recommender is not None, "Recommender not initialized"
    groups = recommender.get_recommendations_grouped(
        algorithm=algorithm.value,
        limit=limit,
        include_read=include_read,
        freshness_weight=freshness_weight,
        similarity_threshold=similarity_threshold,
    )

    def article_to_response(article, score, freshness):
        return ArticleResponse(
            id=article.id,
            title=article.title,
            url=article.url,
            source=article.source,
            author=article.author,
            summary=_clean_summary(article.summary),
            published_at=article.published_at.isoformat() if article.published_at else None,
            score=score,
            freshness=freshness,
            is_read=article.is_read or False,
            is_liked=article.is_liked,
            is_bookmarked=article.is_bookmarked or False,
            reading_time_minutes=article.reading_time_minutes,
        )

    return [
        ArticleGroupResponse(
            primary=article_to_response(group.primary, group.primary_score, group.primary_freshness),
            related=[
                article_to_response(article, score, freshness)
                for article, score, freshness in group.related
            ],
            count=group.count,
            sources=group.sources,
        )
        for group in groups
    ]


@app.get("/api/topic/{query}", response_model=List[ArticleResponse])
def search_topic(
    query: str = PathParam(..., min_length=1, max_length=200, description="Search query (1-200 characters)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    freshness_weight: float = Query(0.2, ge=0, le=1, description="Weight for freshness vs relevance"),
    include_read: bool = Query(False, description="Include articles already marked as read"),
    min_relevance: float = Query(0.25, ge=0, le=1, description="Minimum similarity threshold to filter irrelevant results"),
):
    """Search for articles about a specific topic."""
    assert recommender is not None, "Recommender not initialized"
    # Strip and validate query
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    results = recommender.search_by_topic(
        query=query,
        limit=limit,
        include_read=include_read,
        freshness_weight=freshness_weight,
        min_relevance=min_relevance,
    )

    return [
        ArticleResponse(
            id=article.id,
            title=article.title,
            url=article.url,
            source=article.source,
            author=article.author,
            summary=_clean_summary(article.summary),
            published_at=article.published_at.isoformat() if article.published_at else None,
            score=score,
            freshness=freshness,
            is_read=article.is_read or False,
            is_liked=article.is_liked,
            is_bookmarked=article.is_bookmarked or False,
            reading_time_minutes=article.reading_time_minutes,
        )
        for article, score, freshness in results
    ]


@app.post("/api/feedback")
def record_feedback(feedback: FeedbackRequest):
    """Record like/dislike feedback for an article."""
    assert recommender is not None, "Recommender not initialized"
    recommender.record_feedback(feedback.article_id, feedback.liked)
    return {"status": "ok", "article_id": feedback.article_id, "liked": feedback.liked}


@app.post("/api/read/{article_id}")
def mark_read(article_id: int = PathParam(..., ge=1, description="Article ID")):
    """Mark an article as read."""
    assert recommender is not None, "Recommender not initialized"
    recommender.mark_read(article_id)
    return {"status": "ok", "article_id": article_id}


@app.get("/api/article/{article_id}")
def get_article(article_id: int = PathParam(..., ge=1, description="Article ID")):
    """Get full article details."""
    assert db_session is not None, "Database session not initialized"
    article = db_session.query(Article).get(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return {
        "id": article.id,
        "title": article.title,
        "url": article.url,
        "source": article.source,
        "author": article.author,
        "summary": _clean_summary(article.summary),
        "content": article.content,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "is_read": article.is_read or False,
        "is_liked": article.is_liked,
        "is_bookmarked": article.is_bookmarked or False,
        "reading_time_minutes": article.reading_time_minutes,
        "word_count": article.word_count,
    }


@app.post("/api/bookmark/{article_id}")
def toggle_bookmark(
    article_id: int = PathParam(..., ge=1, description="Article ID"),
    auth: AuthContext = Depends(get_auth),
):
    """Toggle bookmark status for an article."""
    assert db_session is not None, "Database session not initialized"
    article = db_session.query(Article).get(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    if auth.is_authenticated:
        # Use user-specific interaction
        assert auth.user_id is not None  # guaranteed by is_authenticated
        interaction = get_or_create_interaction(db_session, auth.user_id, article_id)
        interaction.is_bookmarked = not interaction.is_bookmarked
        interaction.bookmarked_at = datetime.utcnow() if interaction.is_bookmarked else None
        db_session.commit()
        is_bookmarked = interaction.is_bookmarked
    else:
        # Legacy single-user mode
        article.is_bookmarked = not article.is_bookmarked
        article.bookmarked_at = datetime.utcnow() if article.is_bookmarked else None
        db_session.commit()
        is_bookmarked = article.is_bookmarked

    return {
        "status": "ok",
        "article_id": article_id,
        "is_bookmarked": is_bookmarked,
    }


@app.get("/api/bookmarks", response_model=List[ArticleResponse])
def get_bookmarks(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of bookmarks to return"),
    auth: AuthContext = Depends(get_auth),
):
    """Get all bookmarked articles, sorted by bookmark date."""
    assert db_session is not None, "Database session not initialized"

    if auth.is_authenticated:
        # Get user-specific bookmarks
        results = (
            db_session.query(Article, UserArticleInteraction)
            .join(UserArticleInteraction, UserArticleInteraction.article_id == Article.id)
            .filter(
                UserArticleInteraction.user_id == auth.user_id,
                UserArticleInteraction.is_bookmarked == True,  # noqa: E712
            )
            .order_by(UserArticleInteraction.bookmarked_at.desc())
            .limit(limit)
            .all()
        )
        return [
            ArticleResponse(
                id=article.id,
                title=article.title,
                url=article.url,
                source=article.source,
                author=article.author,
                summary=_clean_summary(article.summary),
                published_at=article.published_at.isoformat() if article.published_at else None,
                score=0.0,
                freshness=0.0,
                is_read=interaction.is_read or False,
                is_liked=interaction.is_liked,
                is_bookmarked=True,
                reading_time_minutes=article.reading_time_minutes,
            )
            for article, interaction in results
        ]
    else:
        # Legacy single-user mode
        articles = (
            db_session.query(Article)
            .filter(Article.is_bookmarked == True)  # noqa: E712
            .order_by(Article.bookmarked_at.desc())
            .limit(limit)
            .all()
        )
        return [
            ArticleResponse(
                id=article.id,
                title=article.title,
                url=article.url,
                source=article.source,
                author=article.author,
                summary=_clean_summary(article.summary),
                published_at=article.published_at.isoformat() if article.published_at else None,
                score=0.0,
                freshness=0.0,
                is_read=article.is_read or False,
                is_liked=article.is_liked,
                is_bookmarked=True,
                reading_time_minutes=article.reading_time_minutes,
            )
            for article in articles
        ]


@app.post("/api/article/{article_id}/notes", response_model=NoteResponse)
def create_note(
    article_id: int = PathParam(..., ge=1, description="Article ID"),
    note: NoteRequest = Body(...),
    auth: AuthContext = Depends(get_auth),
):
    """Create a note for an article."""
    assert db_session is not None, "Database session not initialized"

    # Verify article exists
    article = db_session.query(Article).get(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    new_note = ArticleNote(
        article_id=article_id,
        user_id=auth.user_id,  # Will be None for anonymous users
        content=note.content,
        highlight_text=note.highlight_text,
    )
    db_session.add(new_note)
    db_session.commit()

    return NoteResponse(
        id=new_note.id,
        article_id=new_note.article_id,
        content=new_note.content,
        highlight_text=new_note.highlight_text,
        created_at=new_note.created_at.isoformat() if new_note.created_at else None,
        updated_at=new_note.updated_at.isoformat() if new_note.updated_at else None,
    )


@app.get("/api/article/{article_id}/notes", response_model=List[NoteResponse])
def get_article_notes(
    article_id: int = PathParam(..., ge=1, description="Article ID"),
    auth: AuthContext = Depends(get_auth),
):
    """Get all notes for an article (filtered by user if authenticated)."""
    assert db_session is not None, "Database session not initialized"

    query = db_session.query(ArticleNote).filter(ArticleNote.article_id == article_id)

    if auth.is_authenticated:
        query = query.filter(ArticleNote.user_id == auth.user_id)
    else:
        query = query.filter(ArticleNote.user_id == None)  # noqa: E711

    notes = query.order_by(ArticleNote.created_at.desc()).all()

    return [
        NoteResponse(
            id=note.id,
            article_id=note.article_id,
            content=note.content,
            highlight_text=note.highlight_text,
            created_at=note.created_at.isoformat() if note.created_at else None,
            updated_at=note.updated_at.isoformat() if note.updated_at else None,
        )
        for note in notes
    ]


@app.put("/api/notes/{note_id}", response_model=NoteResponse)
def update_note(
    note_id: int = PathParam(..., ge=1, description="Note ID"),
    note: NoteRequest = Body(...),
    auth: AuthContext = Depends(get_auth),
):
    """Update an existing note (must own the note)."""
    assert db_session is not None, "Database session not initialized"

    existing_note = db_session.query(ArticleNote).get(note_id)
    if not existing_note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Verify ownership
    if auth.is_authenticated:
        if existing_note.user_id != auth.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to edit this note")
    else:
        if existing_note.user_id is not None:
            raise HTTPException(status_code=403, detail="Not authorized to edit this note")

    existing_note.content = note.content
    existing_note.highlight_text = note.highlight_text
    db_session.commit()

    return NoteResponse(
        id=existing_note.id,
        article_id=existing_note.article_id,
        content=existing_note.content,
        highlight_text=existing_note.highlight_text,
        created_at=existing_note.created_at.isoformat() if existing_note.created_at else None,
        updated_at=existing_note.updated_at.isoformat() if existing_note.updated_at else None,
    )


@app.delete("/api/notes/{note_id}")
def delete_note(
    note_id: int = PathParam(..., ge=1, description="Note ID"),
    auth: AuthContext = Depends(get_auth),
):
    """Delete a note (must own the note)."""
    assert db_session is not None, "Database session not initialized"

    note = db_session.query(ArticleNote).get(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    # Verify ownership
    if auth.is_authenticated:
        if note.user_id != auth.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this note")
    else:
        if note.user_id is not None:
            raise HTTPException(status_code=403, detail="Not authorized to delete this note")

    db_session.delete(note)
    db_session.commit()

    return {"status": "ok", "note_id": note_id}


@app.get("/api/notes", response_model=List[NoteResponse])
def get_all_notes(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of notes to return"),
    auth: AuthContext = Depends(get_auth),
):
    """Get all notes across all articles (filtered by user if authenticated)."""
    assert db_session is not None, "Database session not initialized"

    query = db_session.query(ArticleNote)

    if auth.is_authenticated:
        query = query.filter(ArticleNote.user_id == auth.user_id)
    else:
        query = query.filter(ArticleNote.user_id == None)  # noqa: E711

    notes = query.order_by(ArticleNote.created_at.desc()).limit(limit).all()

    return [
        NoteResponse(
            id=note.id,
            article_id=note.article_id,
            content=note.content,
            highlight_text=note.highlight_text,
            created_at=note.created_at.isoformat() if note.created_at else None,
            updated_at=note.updated_at.isoformat() if note.updated_at else None,
        )
        for note in notes
    ]


@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    """Get statistics about articles and preferences."""
    assert recommender is not None, "Recommender not initialized"
    stats = recommender.get_stats()
    return StatsResponse(**stats)


@app.post("/api/refresh")
def refresh_feeds():
    """Fetch new articles from all feed sources."""
    assert recommender is not None, "Recommender not initialized"
    try:
        new_count = recommender.refresh_all_feeds(fetch_content=False)
        return {"status": "ok", "new_articles": new_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/cache/stats")
def get_cache_stats():
    """Get embedding cache statistics."""
    assert recommender is not None, "Recommender not initialized"
    return recommender.embedding_engine.cache_stats()


@app.post("/api/cache/clear")
def clear_cache():
    """Clear the embedding cache."""
    assert recommender is not None, "Recommender not initialized"
    recommender.embedding_engine.clear_cache()
    return {"status": "ok", "message": "Cache cleared"}


@app.get("/api/scheduler/status")
def get_scheduler_status():
    """Get the feed refresh scheduler status."""
    assert feed_scheduler is not None, "Scheduler not initialized"
    return feed_scheduler.get_status()


@app.post("/api/scheduler/enable")
def enable_scheduler():
    """Enable the feed refresh scheduler."""
    assert feed_scheduler is not None, "Scheduler not initialized"
    feed_scheduler.set_enabled(True)
    return {"status": "ok", "enabled": True}


@app.post("/api/scheduler/disable")
def disable_scheduler():
    """Disable the feed refresh scheduler."""
    assert feed_scheduler is not None, "Scheduler not initialized"
    feed_scheduler.set_enabled(False)
    return {"status": "ok", "enabled": False}


@app.post("/api/scheduler/interval")
def set_scheduler_interval(
    minutes: int = Query(..., ge=1, le=1440, description="Refresh interval in minutes (1-1440)")
):
    """Set the feed refresh interval."""
    assert feed_scheduler is not None, "Scheduler not initialized"
    feed_scheduler.set_interval(minutes)
    return {"status": "ok", "interval_minutes": minutes}


@app.post("/api/scheduler/refresh-now")
async def trigger_refresh_now():
    """Trigger an immediate feed refresh."""
    assert feed_scheduler is not None, "Scheduler not initialized"
    result = await feed_scheduler.refresh_now()
    return result


@app.get("/api/feeds/export/opml")
def export_feeds_opml():
    """Export all feed sources as OPML."""
    assert db_session is not None, "Database session not initialized"
    sources = db_session.query(FeedSource).filter(FeedSource.is_active == True).all()  # noqa: E712
    opml_content = export_opml(sources, title="AI News Tracker Feeds")
    return HTMLResponse(
        content=opml_content,
        media_type="application/xml",
        headers={"Content-Disposition": "attachment; filename=feeds.opml"},
    )


@app.get("/api/feeds/export/json")
def export_feeds_as_json():
    """Export all feed sources as JSON."""
    assert db_session is not None, "Database session not initialized"
    sources = db_session.query(FeedSource).all()
    return export_feeds_json(sources)


@app.post("/api/feeds/import/opml")
async def import_feeds_opml(file: bytes = Body(..., media_type="application/xml")):
    """
    Import feed sources from OPML.

    Upload OPML file content as request body.
    Returns count of imported feeds and any skipped duplicates.
    """
    assert db_session is not None, "Database session not initialized"

    try:
        opml_content = file.decode("utf-8")
        feeds = parse_opml(opml_content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid file encoding, expected UTF-8")

    imported = 0
    skipped = 0

    for name, url, feed_type in feeds:
        # Check if URL already exists
        existing = db_session.query(FeedSource).filter(FeedSource.url == url).first()
        if existing:
            skipped += 1
            continue

        source = FeedSource(
            name=name,
            url=url,
            feed_type=feed_type,
            is_active=True,
        )
        db_session.add(source)
        imported += 1

    db_session.commit()

    return {
        "status": "ok",
        "imported": imported,
        "skipped": skipped,
        "total_in_file": len(feeds),
    }


@app.get("/api/feeds")
def list_feeds():
    """List all configured feed sources."""
    assert db_session is not None, "Database session not initialized"
    sources = db_session.query(FeedSource).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "url": s.url,
            "feed_type": s.feed_type,
            "is_active": s.is_active,
            "last_fetched": s.last_fetched.isoformat() if s.last_fetched else None,
            "fetch_interval_minutes": s.fetch_interval_minutes,
        }
        for s in sources
    ]


@app.delete("/api/feeds/{feed_id}")
def delete_feed(feed_id: int = PathParam(..., ge=1, description="Feed ID")):
    """Delete a feed source."""
    assert db_session is not None, "Database session not initialized"
    source = db_session.query(FeedSource).get(feed_id)
    if not source:
        raise HTTPException(status_code=404, detail="Feed not found")

    db_session.delete(source)
    db_session.commit()
    return {"status": "ok", "feed_id": feed_id}


def _clean_summary(summary: Optional[str]) -> Optional[str]:
    """Clean up HTML entities in summary."""
    if not summary:
        return None
    # Basic HTML entity cleanup
    return (
        summary
        .replace("&#8217;", "'")
        .replace("&#8216;", "'")
        .replace("&#8220;", '"')
        .replace("&#8221;", '"')
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("<p>", "")
        .replace("</p>", " ")
        .replace("<br>", " ")
        .replace("<br/>", " ")
        .strip()
    )


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    db_path: str = "news_tracker.db",
    log_level: str = "INFO",
):
    """Run the web server."""
    import uvicorn

    # Setup application logging
    setup_logging(level=log_level)

    os.environ["NEWS_DB_PATH"] = db_path
    print(f"Starting web server at http://{host}:{port}")
    print(f"Using database: {db_path}")
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
