"""Web frontend for AI News Tracker."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import init_db, Article
from .embeddings import EmbeddingEngine
from .preferences import PreferenceLearner
from .recommender import NewsRecommender

app = FastAPI(title="AI News Tracker")

# Global instances (initialized on startup)
recommender = None
db_session = None

STATIC_DIR = Path(__file__).parent / "static"


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
    article_id: int
    liked: bool


class StatsResponse(BaseModel):
    total_articles: int
    unread_articles: int
    liked_articles: int
    disliked_articles: int
    active_sources: int
    has_preferences: bool


@app.on_event("startup")
def startup():
    global recommender, db_session
    db_path = os.environ.get("NEWS_DB_PATH", "news_tracker.db")
    engine, Session = init_db(db_path)
    db_session = Session()
    embedding_engine = EmbeddingEngine()
    preference_learner = PreferenceLearner(embedding_engine)
    recommender = NewsRecommender(db_session, embedding_engine, preference_learner)


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
    limit: int = Query(20, ge=1, le=100),
    freshness_weight: float = Query(0.3, ge=0, le=1),
    include_read: bool = Query(False),
):
    """Get personalized article recommendations."""
    results = recommender.get_recommendations(
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


@app.get("/api/topic/{query}", response_model=List[ArticleResponse])
def search_topic(
    query: str,
    limit: int = Query(20, ge=1, le=100),
    freshness_weight: float = Query(0.2, ge=0, le=1),
    include_read: bool = Query(False),
):
    """Search for articles about a specific topic."""
    results = recommender.search_by_topic(
        query=query,
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


@app.post("/api/feedback")
def record_feedback(feedback: FeedbackRequest):
    """Record like/dislike feedback for an article."""
    recommender.record_feedback(feedback.article_id, feedback.liked)
    return {"status": "ok", "article_id": feedback.article_id, "liked": feedback.liked}


@app.post("/api/read/{article_id}")
def mark_read(article_id: int):
    """Mark an article as read."""
    recommender.mark_read(article_id)
    return {"status": "ok", "article_id": article_id}


@app.get("/api/article/{article_id}")
def get_article(article_id: int):
    """Get full article details."""
    article = db_session.query(Article).get(article_id)
    if not article:
        return {"error": "Article not found"}

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
