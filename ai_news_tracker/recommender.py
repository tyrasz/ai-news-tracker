"""Recommendation engine that ranks articles by user preferences."""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_

from .models import Article, FeedSource
from .embeddings import EmbeddingEngine, embedding_to_bytes, bytes_to_embedding
from .preferences import PreferenceLearner
from .sources import FeedFetcher, RawArticle


class NewsRecommender:
    """
    Main recommendation engine that combines:
    - Article fetching from feeds
    - Embedding generation
    - Preference-based ranking
    - Diversity injection
    """

    def __init__(
        self,
        db_session: Session,
        embedding_engine: EmbeddingEngine | None = None,
        preference_learner: PreferenceLearner | None = None,
    ):
        self.db = db_session
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.preference_learner = preference_learner or PreferenceLearner(self.embedding_engine)
        self.fetcher = FeedFetcher()

    def ingest_from_feed(
        self,
        feed_url: str,
        source_name: str | None = None,
        fetch_content: bool = False,
    ) -> list[Article]:
        """
        Fetch articles from a feed and store them in the database.

        Args:
            feed_url: URL of the RSS/Atom feed
            source_name: Human-readable name for the source
            fetch_content: If True, also fetch full article content (slower)

        Returns:
            List of newly added articles
        """
        new_articles = []

        for raw in self.fetcher.fetch_feed(feed_url, source_name):
            # Skip if we already have this URL
            existing = self.db.query(Article).filter_by(url=raw.url).first()
            if existing:
                continue

            # Optionally fetch full content
            content = None
            if fetch_content and raw.url:
                content = self.fetcher.fetch_full_content(raw.url)

            # Create article
            article = Article(
                url=raw.url,
                title=raw.title,
                content=content,
                summary=raw.summary,
                source=raw.source,
                author=raw.author,
                published_at=raw.published_at,
            )

            # Generate embedding
            embedding = self.embedding_engine.embed_article(
                article.title,
                article.content,
                article.summary,
            )
            article.embedding = embedding_to_bytes(embedding)

            self.db.add(article)
            new_articles.append(article)

        self.db.commit()
        return new_articles

    def refresh_all_feeds(self, fetch_content: bool = False) -> int:
        """Refresh all active feed sources. Returns count of new articles."""
        sources = self.db.query(FeedSource).filter_by(is_active=True).all()
        total_new = 0

        for source in sources:
            new_articles = self.ingest_from_feed(
                source.url,
                source.name,
                fetch_content=fetch_content,
            )
            total_new += len(new_articles)
            source.last_fetched = datetime.utcnow()

        self.db.commit()
        return total_new

    def compute_freshness(
        self,
        article: Article,
        half_life_hours: float = 24.0,
    ) -> float:
        """
        Compute freshness score using exponential decay.

        Args:
            article: The article to score
            half_life_hours: Hours until freshness drops to 50%

        Returns:
            Freshness score from 0 to 1 (1 = brand new, 0 = very old)
        """
        pub_time = article.published_at or article.fetched_at
        if not pub_time:
            return 0.5  # Unknown age, neutral score

        age_hours = (datetime.utcnow() - pub_time).total_seconds() / 3600
        # Exponential decay: score = 0.5^(age/half_life)
        return 0.5 ** (age_hours / half_life_hours)

    def get_recommendations(
        self,
        limit: int = 20,
        profile_name: str = "default",
        include_read: bool = False,
        diversity_factor: float = 0.1,
        max_age_days: int | None = 7,
        freshness_weight: float = 0.3,
        freshness_half_life_hours: float = 24.0,
    ) -> list[tuple[Article, float, float]]:
        """
        Get personalized article recommendations.

        Args:
            limit: Maximum number of articles to return
            profile_name: User profile to use for personalization
            include_read: If True, include articles already marked as read
            diversity_factor: 0-1, how much to inject random diversity
            max_age_days: Only consider articles from the last N days
            freshness_weight: 0-1, how much freshness matters vs relevance
                             0 = only relevance, 1 = only freshness
            freshness_half_life_hours: Hours until freshness drops to 50%

        Returns:
            List of (article, combined_score, freshness) tuples, sorted by score
        """
        # Build query filters
        filters = []
        if not include_read:
            filters.append(Article.is_read == False)
        if max_age_days:
            cutoff = datetime.utcnow() - timedelta(days=max_age_days)
            filters.append(Article.fetched_at >= cutoff)

        # Get candidate articles
        query = self.db.query(Article).filter(Article.embedding.isnot(None))
        if filters:
            query = query.filter(and_(*filters))

        candidates = query.all()

        if not candidates:
            return []

        # Get preference embedding
        preference = self.preference_learner.get_preference_embedding(self.db, profile_name)

        if preference is None:
            # No preferences yet - return by recency with freshness scores
            candidates.sort(key=lambda a: a.published_at or a.fetched_at, reverse=True)
            return [
                (a, 0.5, self.compute_freshness(a, freshness_half_life_hours))
                for a in candidates[:limit]
            ]

        # Score all candidates
        scored = []
        for article in candidates:
            embedding = bytes_to_embedding(article.embedding, self.embedding_engine.embedding_dim)

            # Relevance score from preference similarity (normalize to 0-1 range)
            relevance = (self.embedding_engine.cosine_similarity(preference, embedding) + 1) / 2

            # Freshness score using exponential decay
            freshness = self.compute_freshness(article, freshness_half_life_hours)

            # Combined score: weighted blend of relevance and freshness
            combined = (1 - freshness_weight) * relevance + freshness_weight * freshness

            # Add diversity noise
            diversity_noise = np.random.uniform(-diversity_factor, diversity_factor)
            final_score = combined + diversity_noise

            scored.append((article, final_score, combined, freshness))

        # Sort by final score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top articles with combined score and freshness
        return [(article, combined, freshness) for article, _, combined, freshness in scored[:limit]]

    def record_feedback(
        self,
        article_id: int,
        liked: bool,
        profile_name: str = "default",
    ) -> None:
        """Record user feedback (like/dislike) on an article."""
        article = self.db.query(Article).get(article_id)
        if article:
            self.preference_learner.update_profile(self.db, article, liked, profile_name)

    def mark_read(self, article_id: int) -> None:
        """Mark an article as read without explicit like/dislike."""
        article = self.db.query(Article).get(article_id)
        if article:
            article.is_read = True
            article.read_at = datetime.utcnow()
            self.db.commit()

    def add_feed_source(self, name: str, url: str) -> FeedSource:
        """Add a new feed source."""
        source = FeedSource(name=name, url=url)
        self.db.add(source)
        self.db.commit()
        return source

    def search_by_topic(
        self,
        query: str,
        limit: int = 20,
        include_read: bool = False,
        max_age_days: int | None = 7,
        freshness_weight: float = 0.2,
        freshness_half_life_hours: float = 24.0,
    ) -> list[tuple[Article, float, float]]:
        """
        Search for articles matching a topic/query using semantic search.

        Args:
            query: Natural language topic or query (e.g., "machine learning", "space exploration")
            limit: Maximum number of articles to return
            include_read: If True, include articles already marked as read
            max_age_days: Only consider articles from the last N days
            freshness_weight: 0-1, how much freshness matters vs relevance
            freshness_half_life_hours: Hours until freshness drops to 50%

        Returns:
            List of (article, combined_score, freshness) tuples, sorted by relevance
        """
        # Embed the query
        query_embedding = self.embedding_engine.embed_text(query)

        # Build query filters
        filters = []
        if not include_read:
            filters.append(Article.is_read == False)
        if max_age_days:
            cutoff = datetime.utcnow() - timedelta(days=max_age_days)
            filters.append(Article.fetched_at >= cutoff)

        # Get candidate articles
        db_query = self.db.query(Article).filter(Article.embedding.isnot(None))
        if filters:
            db_query = db_query.filter(and_(*filters))

        candidates = db_query.all()

        if not candidates:
            return []

        # Score all candidates by similarity to query
        scored = []
        for article in candidates:
            embedding = bytes_to_embedding(article.embedding, self.embedding_engine.embedding_dim)

            # Relevance = similarity to query (normalize to 0-1)
            relevance = (self.embedding_engine.cosine_similarity(query_embedding, embedding) + 1) / 2

            # Freshness score
            freshness = self.compute_freshness(article, freshness_half_life_hours)

            # Combined score
            combined = (1 - freshness_weight) * relevance + freshness_weight * freshness

            scored.append((article, combined, freshness, relevance))

        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(article, combined, freshness) for article, combined, freshness, _ in scored[:limit]]

    def get_stats(self, profile_name: str = "default") -> dict:
        """Get statistics about articles and preferences."""
        total = self.db.query(Article).count()
        unread = self.db.query(Article).filter_by(is_read=False).count()
        liked = self.db.query(Article).filter_by(is_liked=True).count()
        disliked = self.db.query(Article).filter_by(is_liked=False).count()
        sources = self.db.query(FeedSource).filter_by(is_active=True).count()

        preference = self.preference_learner.get_preference_embedding(self.db, profile_name)

        return {
            "total_articles": total,
            "unread_articles": unread,
            "liked_articles": liked,
            "disliked_articles": disliked,
            "active_sources": sources,
            "has_preferences": preference is not None,
        }
