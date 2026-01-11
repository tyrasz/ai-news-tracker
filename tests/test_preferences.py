"""Tests for preference learning."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from ai_news_tracker.models import Article, UserProfile
from ai_news_tracker.preferences import PreferenceLearner
from ai_news_tracker.embeddings import embedding_to_bytes, bytes_to_embedding


class TestPreferenceLearner:
    """Tests for the PreferenceLearner class."""

    def test_init_with_defaults(self, mock_embedding_engine):
        """Test initialization with default parameters."""
        learner = PreferenceLearner(mock_embedding_engine)

        assert learner.decay == 0.95
        assert learner.dislike_weight == -0.3
        assert learner.embedding_dim == 384

    def test_init_with_custom_params(self, mock_embedding_engine):
        """Test initialization with custom parameters."""
        learner = PreferenceLearner(
            mock_embedding_engine,
            decay=0.9,
            dislike_weight=-0.5,
        )

        assert learner.decay == 0.9
        assert learner.dislike_weight == -0.5

    def test_get_or_create_profile_creates_new(self, temp_db, mock_embedding_engine):
        """Test creating a new profile."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        profile = learner.get_or_create_profile(session, "test_user")

        assert profile is not None
        assert profile.name == "test_user"
        assert profile.decay_factor == 0.95

    def test_get_or_create_profile_returns_existing(self, temp_db, mock_embedding_engine):
        """Test returning existing profile."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        # Create profile
        profile1 = learner.get_or_create_profile(session, "test_user")
        profile1_id = profile1.id

        # Get same profile
        profile2 = learner.get_or_create_profile(session, "test_user")

        assert profile2.id == profile1_id

    def test_update_profile_first_like(self, temp_db, mock_embedding_engine):
        """Test updating profile with first liked article."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        # Create article
        article = Article(
            url="https://example.com/test",
            title="Machine Learning Article",
            summary="About ML",
        )
        session.add(article)
        session.commit()

        # Like the article
        profile = learner.update_profile(session, article, liked=True)

        assert profile.articles_liked == 1
        assert profile.articles_disliked == 0
        assert profile.preference_embedding is not None
        assert article.is_liked is True
        assert article.is_read is True

    def test_update_profile_first_dislike(self, temp_db, mock_embedding_engine):
        """Test updating profile with first disliked article."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        article = Article(
            url="https://example.com/test",
            title="Boring Article",
        )
        session.add(article)
        session.commit()

        profile = learner.update_profile(session, article, liked=False)

        assert profile.articles_liked == 0
        assert profile.articles_disliked == 1
        assert article.is_liked is False

    def test_update_profile_multiple_likes(self, temp_db, mock_embedding_engine):
        """Test updating profile with multiple liked articles."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        for i in range(3):
            article = Article(
                url=f"https://example.com/article{i}",
                title=f"Article {i}",
            )
            session.add(article)
            session.commit()
            learner.update_profile(session, article, liked=True)

        profile = session.query(UserProfile).filter_by(name="default").first()
        assert profile.articles_liked == 3

    def test_update_profile_uses_existing_embedding(self, temp_db, mock_embedding_engine):
        """Test that existing article embedding is reused."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        # Create article with embedding
        existing_embedding = np.random.randn(384).astype(np.float32)
        existing_embedding = existing_embedding / np.linalg.norm(existing_embedding)
        article = Article(
            url="https://example.com/test",
            title="Test",
            embedding=embedding_to_bytes(existing_embedding),
        )
        session.add(article)
        session.commit()

        learner.update_profile(session, article, liked=True)

        # Article embedding should not have changed
        restored = bytes_to_embedding(article.embedding, 384)
        np.testing.assert_array_almost_equal(existing_embedding, restored)

    def test_update_profile_normalizes_embedding(self, temp_db, mock_embedding_engine):
        """Test that preference embedding is normalized."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        article = Article(url="https://example.com/test", title="Test")
        session.add(article)
        session.commit()

        profile = learner.update_profile(session, article, liked=True)
        embedding = bytes_to_embedding(profile.preference_embedding, 384)

        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_update_profile_sets_read_at(self, temp_db, mock_embedding_engine):
        """Test that read_at timestamp is set."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        article = Article(url="https://example.com/test", title="Test")
        session.add(article)
        session.commit()

        before = datetime.utcnow()
        learner.update_profile(session, article, liked=True)
        after = datetime.utcnow()

        assert article.read_at is not None
        assert before <= article.read_at <= after

    def test_get_preference_embedding_none(self, temp_db, mock_embedding_engine):
        """Test getting embedding when no profile exists."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        embedding = learner.get_preference_embedding(session, "nonexistent")
        assert embedding is None

    def test_get_preference_embedding_no_embedding(self, temp_db, mock_embedding_engine):
        """Test getting embedding when profile has no embedding."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        profile = UserProfile(name="empty")
        session.add(profile)
        session.commit()

        embedding = learner.get_preference_embedding(session, "empty")
        assert embedding is None

    def test_get_preference_embedding_exists(self, temp_db, mock_embedding_engine):
        """Test getting existing preference embedding."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        # Create profile with embedding
        original = np.random.randn(384).astype(np.float32)
        profile = UserProfile(
            name="test",
            preference_embedding=embedding_to_bytes(original),
        )
        session.add(profile)
        session.commit()

        embedding = learner.get_preference_embedding(session, "test")

        assert embedding is not None
        np.testing.assert_array_almost_equal(original, embedding)

    def test_compute_affinity_score_no_profile(self, temp_db, mock_embedding_engine):
        """Test affinity score when no profile exists."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        article_embedding = np.random.randn(384).astype(np.float32)
        score = learner.compute_affinity_score(session, article_embedding, "nonexistent")

        assert score == 0.0

    def test_compute_affinity_score_with_profile(self, temp_db, mock_embedding_engine):
        """Test affinity score with existing profile."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        # Create profile with known embedding
        preference = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        profile = UserProfile(
            name="test",
            preference_embedding=embedding_to_bytes(preference),
        )
        session.add(profile)
        session.commit()

        # Test with identical embedding
        identical = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        score = learner.compute_affinity_score(session, identical, "test")
        assert abs(score - 1.0) < 1e-5

        # Test with orthogonal embedding
        orthogonal = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)
        score = learner.compute_affinity_score(session, orthogonal, "test")
        assert abs(score) < 1e-5

    def test_explain_recommendation_no_embedding(self, temp_db, mock_embedding_engine):
        """Test explanation when article has no embedding."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        article = Article(url="https://example.com/test", title="Test")
        session.add(article)
        session.commit()

        explanation = learner.explain_recommendation(session, article)
        assert "No embedding" in explanation

    def test_explain_recommendation_no_profile(self, temp_db, mock_embedding_engine):
        """Test explanation when no preference data."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        embedding = np.random.randn(384).astype(np.float32)
        article = Article(
            url="https://example.com/test",
            title="Test",
            embedding=embedding_to_bytes(embedding),
        )
        session.add(article)
        session.commit()

        explanation = learner.explain_recommendation(session, article)
        assert "Not enough preference data" in explanation

    def test_explain_recommendation_high_score(self, temp_db, mock_embedding_engine):
        """Test explanation for highly relevant article."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        # Create profile and article with similar embeddings
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        profile = UserProfile(
            name="default",
            preference_embedding=embedding_to_bytes(embedding),
            articles_liked=5,
        )
        session.add(profile)

        article = Article(
            url="https://example.com/test",
            title="Test",
            embedding=embedding_to_bytes(embedding),
        )
        session.add(article)
        session.commit()

        explanation = learner.explain_recommendation(session, article)
        assert "Highly relevant" in explanation

    def test_explain_recommendation_moderate_score(self, temp_db, mock_embedding_engine):
        """Test explanation for moderately relevant article."""
        session, _ = temp_db
        learner = PreferenceLearner(mock_embedding_engine)

        # Create embeddings with moderate similarity
        pref = np.array([1.0, 0.5] + [0.0] * 382, dtype=np.float32)
        pref = pref / np.linalg.norm(pref)

        art = np.array([1.0, -0.2] + [0.0] * 382, dtype=np.float32)
        art = art / np.linalg.norm(art)

        profile = UserProfile(
            name="default",
            preference_embedding=embedding_to_bytes(pref),
            articles_liked=5,
        )
        session.add(profile)

        article = Article(
            url="https://example.com/test",
            title="Test",
            embedding=embedding_to_bytes(art),
        )
        session.add(article)
        session.commit()

        explanation = learner.explain_recommendation(session, article)
        assert "score" in explanation.lower()
