"""Web frontend for AI News Tracker."""

import asyncio
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, Path as PathParam
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import init_db_scoped, checkpoint_wal, Article
from .embeddings import EmbeddingEngine
from .preferences import PreferenceLearner
from .recommender import NewsRecommender

logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
recommender = None
db_session = None
engine = None
_scoped_session = None
wal_checkpoint_task = None

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
    global recommender, db_session, engine, _scoped_session, wal_checkpoint_task

    # Startup
    db_path = os.environ.get("NEWS_DB_PATH", "news_tracker.db")
    engine, _scoped_session = init_db_scoped(db_path)
    db_session = _scoped_session()
    embedding_engine = EmbeddingEngine()
    preference_learner = PreferenceLearner(embedding_engine)
    recommender = NewsRecommender(db_session, embedding_engine, preference_learner)

    # Start background WAL checkpoint task
    wal_checkpoint_task = asyncio.create_task(periodic_wal_checkpoint())
    logger.info(f"Started web server with database: {db_path}")

    yield

    # Shutdown
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
        )
        for article, score, freshness in results
    ]


@app.get("/api/algorithms")
def get_algorithms():
    """List available recommendation algorithms."""
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
        )
        for article, score, freshness in results
    ]


@app.post("/api/feedback")
def record_feedback(feedback: FeedbackRequest):
    """Record like/dislike feedback for an article."""
    recommender.record_feedback(feedback.article_id, feedback.liked)
    return {"status": "ok", "article_id": feedback.article_id, "liked": feedback.liked}


@app.post("/api/read/{article_id}")
def mark_read(article_id: int = PathParam(..., ge=1, description="Article ID")):
    """Mark an article as read."""
    recommender.mark_read(article_id)
    return {"status": "ok", "article_id": article_id}


@app.get("/api/article/{article_id}")
def get_article(article_id: int = PathParam(..., ge=1, description="Article ID")):
    """Get full article details."""
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
    }


@app.get("/api/stats", response_model=StatsResponse)
def get_stats():
    """Get statistics about articles and preferences."""
    stats = recommender.get_stats()
    return StatsResponse(**stats)


@app.post("/api/refresh")
def refresh_feeds():
    """Fetch new articles from all feed sources."""
    try:
        new_count = recommender.refresh_all_feeds(fetch_content=False)
        return {"status": "ok", "new_articles": new_count}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/cache/stats")
def get_cache_stats():
    """Get embedding cache statistics."""
    return recommender.embedding_engine.cache_stats()


@app.post("/api/cache/clear")
def clear_cache():
    """Clear the embedding cache."""
    recommender.embedding_engine.clear_cache()
    return {"status": "ok", "message": "Cache cleared"}


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


def run_server(host: str = "127.0.0.1", port: int = 8000, db_path: str = "news_tracker.db"):
    """Run the web server."""
    import uvicorn
    os.environ["NEWS_DB_PATH"] = db_path
    uvicorn.run(app, host=host, port=port)
