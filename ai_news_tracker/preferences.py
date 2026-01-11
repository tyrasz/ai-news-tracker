"""User preference learning using embedding-based profiles."""

from __future__ import annotations

import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session

from .models import UserProfile as UserProfileModel, Article
from .embeddings import embedding_to_bytes, bytes_to_embedding, EmbeddingEngine


class PreferenceLearner:
    """
    Learns user preferences from article interactions.

    Uses an exponential moving average of liked article embeddings
    to build a preference profile. Dislikes contribute negatively.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        decay: float = 0.95,
        dislike_weight: float = -0.3,
    ):
        """
        Initialize the preference learner.

        Args:
            embedding_engine: Engine for generating embeddings
            decay: Exponential decay factor (0.9-0.99). Higher = slower adaptation.
            dislike_weight: How much dislikes affect the profile (negative value).
        """
        self.embedding_engine = embedding_engine
        self.decay = decay
        self.dislike_weight = dislike_weight
        self.embedding_dim = embedding_engine.embedding_dim

    def get_or_create_profile(self, db: Session, profile_name: str = "default") -> UserProfileModel:
        """Get existing profile or create a new one."""
        profile = db.query(UserProfileModel).filter_by(name=profile_name).first()
        if not profile:
            profile = UserProfileModel(
                name=profile_name,
                decay_factor=self.decay,
            )
            db.add(profile)
            db.commit()
        return profile

    def update_profile(
        self,
        db: Session,
        article: Article,
        liked: bool,
        profile_name: str = "default",
    ) -> UserProfileModel:
        """
        Update user profile based on article interaction.

        Args:
            db: Database session
            article: The article that was interacted with
            liked: True if liked, False if disliked
            profile_name: Name of the profile to update
        """
        profile = self.get_or_create_profile(db, profile_name)

        # Get article embedding
        if article.embedding is None:
            article_embedding = self.embedding_engine.embed_article(
                article.title,
                article.content,
                article.summary,
            )
            article.embedding = embedding_to_bytes(article_embedding)
        else:
            article_embedding = bytes_to_embedding(article.embedding, self.embedding_dim)

        # Compute weight for this update
        weight = 1.0 if liked else self.dislike_weight

        # Update preference embedding using exponential moving average
        if profile.preference_embedding is None:
            # First interaction - initialize profile
            new_embedding = article_embedding * weight
        else:
            current_embedding = bytes_to_embedding(profile.preference_embedding, self.embedding_dim)
            new_embedding = (
                self.decay * current_embedding +
                (1 - self.decay) * article_embedding * weight
            )

        # Normalize to unit vector
        norm = np.linalg.norm(new_embedding)
        if norm > 0:
            new_embedding = new_embedding / norm

        profile.preference_embedding = embedding_to_bytes(new_embedding)

        # Update stats
        if liked:
            profile.articles_liked += 1
        else:
            profile.articles_disliked += 1

        profile.updated_at = datetime.utcnow()

        # Update article feedback
        article.is_liked = liked
        article.is_read = True
        article.read_at = datetime.utcnow()

        db.commit()
        return profile

    def get_preference_embedding(self, db: Session, profile_name: str = "default") -> np.ndarray | None:
        """Get the current preference embedding for a profile."""
        profile = db.query(UserProfileModel).filter_by(name=profile_name).first()
        if profile and profile.preference_embedding:
            return bytes_to_embedding(profile.preference_embedding, self.embedding_dim)
        return None

    def compute_affinity_score(
        self,
        db: Session,
        article_embedding: np.ndarray,
        profile_name: str = "default",
    ) -> float:
        """
        Compute how well an article matches user preferences.

        Returns a score from -1 to 1, where:
        - 1 = perfect match to preferences
        - 0 = neutral / no preference data
        - -1 = opposite of preferences
        """
        preference = self.get_preference_embedding(db, profile_name)
        if preference is None:
            return 0.0

        return self.embedding_engine.cosine_similarity(preference, article_embedding)

    def explain_recommendation(
        self,
        db: Session,
        article: Article,
        profile_name: str = "default",
    ) -> str:
        """Generate a simple explanation for why an article was recommended."""
        if article.embedding is None:
            return "No embedding available for this article."

        score = self.compute_affinity_score(
            db,
            bytes_to_embedding(article.embedding, self.embedding_dim),
            profile_name,
        )

        profile = db.query(UserProfileModel).filter_by(name=profile_name).first()
        if not profile or profile.articles_liked == 0:
            return "Not enough preference data yet. Like/dislike articles to improve recommendations."

        if score > 0.7:
            return f"Highly relevant to your interests (score: {score:.2f})"
        elif score > 0.4:
            return f"Moderately relevant to your interests (score: {score:.2f})"
        elif score > 0.1:
            return f"Somewhat relevant to your interests (score: {score:.2f})"
        else:
            return f"Exploring new topics (score: {score:.2f})"
