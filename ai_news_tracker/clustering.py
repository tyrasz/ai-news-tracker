"""Article clustering for deduplication and grouping similar stories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .models import Article
from .embeddings import bytes_to_embedding, EmbeddingEngine


@dataclass
class ArticleGroup:
    """A group of similar articles about the same story."""
    primary: Article
    primary_score: float
    primary_freshness: float
    related: List[tuple[Article, float, float]] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total number of articles in this group."""
        return 1 + len(self.related)

    @property
    def sources(self) -> List[str]:
        """List of all sources covering this story."""
        sources = [self.primary.source] if self.primary.source else []
        for article, _, _ in self.related:
            if article.source and article.source not in sources:
                sources.append(article.source)
        return sources


def cluster_similar_articles(
    scored_articles: List[tuple[Article, float, float]],
    embedding_engine: EmbeddingEngine,
    similarity_threshold: float = 0.75,
) -> List[ArticleGroup]:
    """
    Cluster similar articles together based on embedding similarity.

    Args:
        scored_articles: List of (article, score, freshness) tuples
        embedding_engine: Engine for computing similarity
        similarity_threshold: Minimum cosine similarity to group articles (0.75 = quite similar)

    Returns:
        List of ArticleGroups, with similar articles grouped together
    """
    if not scored_articles:
        return []

    # Extract embeddings
    articles_with_embeddings = []
    for article, score, freshness in scored_articles:
        if article.embedding:
            embedding = bytes_to_embedding(article.embedding, embedding_engine.embedding_dim)
            articles_with_embeddings.append((article, score, freshness, embedding))

    if not articles_with_embeddings:
        return []

    # Track which articles have been assigned to a group
    assigned = set()
    groups = []

    # Process articles in score order (best first becomes primary)
    for i, (article, score, freshness, embedding) in enumerate(articles_with_embeddings):
        if i in assigned:
            continue

        # Start a new group with this article as primary
        group = ArticleGroup(
            primary=article,
            primary_score=score,
            primary_freshness=freshness,
        )
        assigned.add(i)

        # Find similar articles to add to this group
        for j, (other_article, other_score, other_freshness, other_embedding) in enumerate(articles_with_embeddings):
            if j in assigned:
                continue

            # Compute similarity
            similarity = embedding_engine.cosine_similarity(embedding, other_embedding)

            if similarity >= similarity_threshold:
                group.related.append((other_article, other_score, other_freshness))
                assigned.add(j)

        groups.append(group)

    return groups


def flatten_groups(groups: List[ArticleGroup]) -> List[tuple[Article, float, float]]:
    """Convert groups back to flat list (primary articles only)."""
    return [(g.primary, g.primary_score, g.primary_freshness) for g in groups]
