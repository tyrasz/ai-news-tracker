"""Multiple recommendation algorithms for personalized news delivery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Tuple, Optional, Any

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, ColumnElement

from .models import Article, UserProfile
from .embeddings import EmbeddingEngine, bytes_to_embedding


class AlgorithmType(str, Enum):
    """Available recommendation algorithms."""
    FOR_YOU = "for_you"           # Preference-based (current default)
    EXPLORE = "explore"           # Discover new topics outside your bubble
    DEEP_DIVE = "deep_dive"       # Cluster-focused, more depth per topic
    TRENDING = "trending"         # Recency and freshness focused
    BALANCED = "balanced"         # Mix of all signals
    CONTRARIAN = "contrarian"     # Opposite of your preferences


@dataclass
class AlgorithmConfig:
    """Configuration for recommendation algorithms."""
    freshness_weight: float = 0.3
    freshness_half_life_hours: float = 24.0
    diversity_factor: float = 0.1
    max_age_days: int = 7
    include_read: bool = False


@dataclass
class ScoredArticle:
    """Article with computed scores."""
    article: Article
    final_score: float
    relevance_score: float
    freshness_score: float
    diversity_score: float = 0.0


class RecommendationAlgorithm(ABC):
    """Base class for recommendation algorithms."""

    def __init__(
        self,
        db: Session,
        embedding_engine: EmbeddingEngine,
        config: AlgorithmConfig,
    ):
        self.db = db
        self.embedding_engine = embedding_engine
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this algorithm does."""
        pass

    @abstractmethod
    def score_articles(
        self,
        candidates: List[Article],
        preference_embedding: Optional[np.ndarray],
    ) -> List[ScoredArticle]:
        """Score and rank articles according to this algorithm."""
        pass

    def compute_freshness(self, article: Article) -> float:
        """Compute freshness score using exponential decay."""
        pub_time = article.published_at or article.fetched_at
        if not pub_time:
            return 0.5

        age_hours = (datetime.utcnow() - pub_time).total_seconds() / 3600
        return 0.5 ** (age_hours / self.config.freshness_half_life_hours)

    def get_candidates(self) -> List[Article]:
        """Get candidate articles from database."""
        filters: List[ColumnElement[bool]] = [Article.embedding.isnot(None)]

        if not self.config.include_read:
            filters.append(Article.is_read == False)  # noqa: E712

        if self.config.max_age_days:
            cutoff = datetime.utcnow() - timedelta(days=self.config.max_age_days)
            filters.append(Article.fetched_at >= cutoff)

        return self.db.query(Article).filter(and_(*filters)).all()


class ForYouAlgorithm(RecommendationAlgorithm):
    """
    Preference-based recommendations using embedding similarity.

    This is the default algorithm that matches articles to user preferences
    learned from their likes/dislikes.
    """

    @property
    def name(self) -> str:
        return "For You"

    @property
    def description(self) -> str:
        return "Personalized recommendations based on your reading history"

    def score_articles(
        self,
        candidates: List[Article],
        preference_embedding: Optional[np.ndarray],
    ) -> List[ScoredArticle]:
        scored = []

        for article in candidates:
            assert article.embedding is not None  # Filtered by get_candidates
            embedding = bytes_to_embedding(article.embedding, self.embedding_engine.embedding_dim)
            freshness = self.compute_freshness(article)

            if preference_embedding is not None:
                # Relevance from preference similarity (normalize to 0-1)
                similarity = self.embedding_engine.cosine_similarity(preference_embedding, embedding)
                relevance = (similarity + 1) / 2
            else:
                relevance = 0.5

            # Combined score
            combined = (1 - self.config.freshness_weight) * relevance + self.config.freshness_weight * freshness

            # Add diversity noise
            diversity = np.random.uniform(-self.config.diversity_factor, self.config.diversity_factor)

            scored.append(ScoredArticle(
                article=article,
                final_score=combined + diversity,
                relevance_score=relevance,
                freshness_score=freshness,
                diversity_score=diversity,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored


class ExploreAlgorithm(RecommendationAlgorithm):
    """
    Discovery algorithm that surfaces articles OUTSIDE your usual interests.

    Helps break filter bubbles by recommending content that is dissimilar
    to your preference profile but still high quality (fresh, from good sources).
    """

    @property
    def name(self) -> str:
        return "Explore"

    @property
    def description(self) -> str:
        return "Discover new topics outside your usual interests"

    def score_articles(
        self,
        candidates: List[Article],
        preference_embedding: Optional[np.ndarray],
    ) -> List[ScoredArticle]:
        scored = []

        for article in candidates:
            assert article.embedding is not None  # Filtered by get_candidates
            embedding = bytes_to_embedding(article.embedding, self.embedding_engine.embedding_dim)
            freshness = self.compute_freshness(article)

            if preference_embedding is not None:
                # INVERSE relevance - prefer dissimilar articles
                similarity = self.embedding_engine.cosine_similarity(preference_embedding, embedding)
                # Transform: high similarity -> low score, low similarity -> high score
                # But filter out extremely negative (opposite) content
                dissimilarity = 1 - abs(similarity)  # Prefer orthogonal, not opposite
                relevance = dissimilarity
            else:
                relevance = 0.5

            # Freshness matters more in explore mode
            combined = 0.4 * relevance + 0.6 * freshness

            scored.append(ScoredArticle(
                article=article,
                final_score=combined,
                relevance_score=relevance,
                freshness_score=freshness,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored


class DeepDiveAlgorithm(RecommendationAlgorithm):
    """
    Cluster-focused algorithm for deep exploration of topics.

    Groups similar articles together and presents multiple perspectives
    on fewer topics, rather than breadth across many topics.
    """

    @property
    def name(self) -> str:
        return "Deep Dive"

    @property
    def description(self) -> str:
        return "Multiple articles on fewer topics for deeper understanding"

    def score_articles(
        self,
        candidates: List[Article],
        preference_embedding: Optional[np.ndarray],
    ) -> List[ScoredArticle]:
        if not candidates:
            return []

        # Get all embeddings (candidates are filtered to have embeddings)
        embeddings = np.array([
            bytes_to_embedding(a.embedding, self.embedding_engine.embedding_dim)  # type: ignore[arg-type]
            for a in candidates
        ])

        # Simple clustering: find topic centroids from top articles
        # First, score by preference to find seed articles
        top_indices: np.ndarray
        if preference_embedding is not None:
            similarities = embeddings @ preference_embedding
            top_indices = np.argsort(similarities)[-5:]  # Top 5 as cluster seeds
        else:
            # Use most recent as seeds
            sorted_by_date = sorted(
                enumerate(candidates),
                key=lambda x: x[1].published_at or x[1].fetched_at or datetime.min,
                reverse=True
            )
            top_indices = np.array([i for i, _ in sorted_by_date[:5]])

        # Compute cluster centroids
        cluster_centroids = embeddings[top_indices]

        scored = []
        for i, article in enumerate(candidates):
            embedding = embeddings[i]
            freshness = self.compute_freshness(article)

            # Score by max similarity to any cluster centroid
            cluster_similarities = [
                self.embedding_engine.cosine_similarity(embedding, centroid)
                for centroid in cluster_centroids
            ]
            cluster_score = max(cluster_similarities)
            relevance = (cluster_score + 1) / 2

            # Less freshness weight - depth over recency
            combined = 0.8 * relevance + 0.2 * freshness

            scored.append(ScoredArticle(
                article=article,
                final_score=combined,
                relevance_score=relevance,
                freshness_score=freshness,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored


class TrendingAlgorithm(RecommendationAlgorithm):
    """
    Recency-focused algorithm for breaking news.

    Heavily weights freshness over personalization, surfacing
    the newest content regardless of topic preferences.
    """

    @property
    def name(self) -> str:
        return "Trending"

    @property
    def description(self) -> str:
        return "Latest and freshest news, regardless of your preferences"

    def score_articles(
        self,
        candidates: List[Article],
        preference_embedding: Optional[np.ndarray],
    ) -> List[ScoredArticle]:
        scored = []

        for article in candidates:
            freshness = self.compute_freshness(article)

            # Minimal preference influence
            if preference_embedding is not None:
                assert article.embedding is not None  # Filtered by get_candidates
                embedding = bytes_to_embedding(article.embedding, self.embedding_engine.embedding_dim)
                similarity = self.embedding_engine.cosine_similarity(preference_embedding, embedding)
                relevance = (similarity + 1) / 2
            else:
                relevance = 0.5

            # Heavily weight freshness (90%)
            combined = 0.1 * relevance + 0.9 * freshness

            scored.append(ScoredArticle(
                article=article,
                final_score=combined,
                relevance_score=relevance,
                freshness_score=freshness,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored


class BalancedAlgorithm(RecommendationAlgorithm):
    """
    Balanced algorithm mixing relevance, freshness, and diversity.

    Provides a mix of personalized content, fresh news, and diverse
    perspectives in a single feed.
    """

    @property
    def name(self) -> str:
        return "Balanced"

    @property
    def description(self) -> str:
        return "A balanced mix of personalized, fresh, and diverse content"

    def score_articles(
        self,
        candidates: List[Article],
        preference_embedding: Optional[np.ndarray],
    ) -> List[ScoredArticle]:
        if not candidates:
            return []

        # Get all embeddings for diversity calculation (candidates filtered to have embeddings)
        embeddings = np.array([
            bytes_to_embedding(a.embedding, self.embedding_engine.embedding_dim)  # type: ignore[arg-type]
            for a in candidates
        ])

        # Compute centroid for diversity measurement
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        scored = []
        for i, article in enumerate(candidates):
            embedding = embeddings[i]
            freshness = self.compute_freshness(article)

            # Relevance from preferences
            if preference_embedding is not None:
                similarity = self.embedding_engine.cosine_similarity(preference_embedding, embedding)
                relevance = (similarity + 1) / 2
            else:
                relevance = 0.5

            # Diversity: distance from centroid (normalized)
            centroid_sim = self.embedding_engine.cosine_similarity(embedding, centroid)
            diversity = 1 - abs(centroid_sim)

            # Equal weighting of all factors
            combined = 0.4 * relevance + 0.4 * freshness + 0.2 * diversity

            scored.append(ScoredArticle(
                article=article,
                final_score=combined,
                relevance_score=relevance,
                freshness_score=freshness,
                diversity_score=diversity,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored


class ContrarianAlgorithm(RecommendationAlgorithm):
    """
    Contrarian algorithm showing opposite perspectives.

    Surfaces content that contradicts or challenges your usual
    preferences, helping expose you to different viewpoints.
    """

    @property
    def name(self) -> str:
        return "Contrarian"

    @property
    def description(self) -> str:
        return "Challenge your views with different perspectives"

    def score_articles(
        self,
        candidates: List[Article],
        preference_embedding: Optional[np.ndarray],
    ) -> List[ScoredArticle]:
        scored = []

        for article in candidates:
            assert article.embedding is not None  # Filtered by get_candidates
            embedding = bytes_to_embedding(article.embedding, self.embedding_engine.embedding_dim)
            freshness = self.compute_freshness(article)

            if preference_embedding is not None:
                # Prefer articles with NEGATIVE similarity (opposite)
                similarity = self.embedding_engine.cosine_similarity(preference_embedding, embedding)
                # Transform: -1 similarity -> 1.0 score, +1 similarity -> 0.0 score
                relevance = (1 - similarity) / 2
            else:
                relevance = 0.5

            # Moderate freshness weight
            combined = 0.6 * relevance + 0.4 * freshness

            scored.append(ScoredArticle(
                article=article,
                final_score=combined,
                relevance_score=relevance,
                freshness_score=freshness,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored


# Algorithm registry
ALGORITHMS = {
    AlgorithmType.FOR_YOU: ForYouAlgorithm,
    AlgorithmType.EXPLORE: ExploreAlgorithm,
    AlgorithmType.DEEP_DIVE: DeepDiveAlgorithm,
    AlgorithmType.TRENDING: TrendingAlgorithm,
    AlgorithmType.BALANCED: BalancedAlgorithm,
    AlgorithmType.CONTRARIAN: ContrarianAlgorithm,
}


def get_algorithm(
    algorithm_type: AlgorithmType,
    db: Session,
    embedding_engine: EmbeddingEngine,
    config: Optional[AlgorithmConfig] = None,
) -> RecommendationAlgorithm:
    """Factory function to get an algorithm instance."""
    config = config or AlgorithmConfig()
    algorithm_class = ALGORITHMS.get(algorithm_type, ForYouAlgorithm)
    return algorithm_class(db, embedding_engine, config)  # type: ignore[abstract]


def list_algorithms() -> List[dict]:
    """List all available algorithms with metadata."""
    result: List[dict[str, Any]] = []
    for algo_type, algo_class in ALGORITHMS.items():
        # Create a dummy instance to get name/description
        # We use None for db/engine since we only need metadata
        instance = algo_class.__new__(algo_class)  # type: ignore[type-abstract]
        result.append({
            "id": algo_type.value,
            "name": instance.name,
            "description": instance.description,
        })
    return result
