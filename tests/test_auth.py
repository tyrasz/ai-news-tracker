"""Tests for authentication module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from ai_news_tracker.auth import (
    UserCreate,
    UserLogin,
    UserResponse,
    AuthContext,
    create_auth_dependency,
    require_auth,
)


class TestUserCreateModel:
    """Tests for UserCreate Pydantic model."""

    def test_valid_user_create(self):
        """Test creating a valid UserCreate model."""
        user = UserCreate(
            email="test@example.com",
            password="securepassword123",
            display_name="Test User",
        )
        assert user.email == "test@example.com"
        assert user.password == "securepassword123"
        assert user.display_name == "Test User"

    def test_user_create_without_display_name(self):
        """Test creating UserCreate without display name."""
        user = UserCreate(
            email="test@example.com",
            password="securepassword123",
        )
        assert user.display_name is None

    def test_user_create_email_too_short(self):
        """Test that email must be at least 3 characters."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(email="ab", password="securepassword123")

    def test_user_create_password_too_short(self):
        """Test that password must be at least 8 characters."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com", password="short")


class TestUserLoginModel:
    """Tests for UserLogin Pydantic model."""

    def test_valid_user_login(self):
        """Test creating a valid UserLogin model."""
        login = UserLogin(
            email="test@example.com",
            password="anypassword",
        )
        assert login.email == "test@example.com"
        assert login.password == "anypassword"

    def test_user_login_allows_short_password(self):
        """Test that login allows any password length >= 1."""
        login = UserLogin(
            email="test@example.com",
            password="x",
        )
        assert login.password == "x"


class TestUserResponseModel:
    """Tests for UserResponse Pydantic model."""

    def test_valid_user_response(self):
        """Test creating a valid UserResponse model."""
        response = UserResponse(
            id=1,
            email="test@example.com",
            display_name="Test User",
            api_key="abc123xyz",
            is_active=True,
            created_at="2024-01-15T10:30:00",
        )
        assert response.id == 1
        assert response.email == "test@example.com"
        assert response.display_name == "Test User"
        assert response.api_key == "abc123xyz"
        assert response.is_active is True

    def test_user_response_without_display_name(self):
        """Test UserResponse with None display name."""
        response = UserResponse(
            id=1,
            email="test@example.com",
            display_name=None,
            api_key="abc123xyz",
            is_active=True,
            created_at=None,
        )
        assert response.display_name is None
        assert response.created_at is None


class TestAuthContext:
    """Tests for AuthContext class."""

    def test_auth_context_with_user(self):
        """Test AuthContext with an authenticated user."""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 42

        ctx = AuthContext(mock_db, user=mock_user)

        assert ctx.db == mock_db
        assert ctx.user == mock_user
        assert ctx.user_id == 42
        assert ctx.is_authenticated is True

    def test_auth_context_anonymous(self):
        """Test AuthContext without a user (anonymous)."""
        mock_db = MagicMock()

        ctx = AuthContext(mock_db, user=None)

        assert ctx.db == mock_db
        assert ctx.user is None
        assert ctx.user_id is None
        assert ctx.is_authenticated is False

    def test_require_auth_when_authenticated(self):
        """Test require_auth does not raise when authenticated."""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 1

        ctx = AuthContext(mock_db, user=mock_user)

        # Should not raise
        ctx.require_auth()

    def test_require_auth_when_anonymous(self):
        """Test require_auth raises HTTPException when not authenticated."""
        mock_db = MagicMock()

        ctx = AuthContext(mock_db, user=None)

        with pytest.raises(HTTPException) as exc_info:
            ctx.require_auth()

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Authentication required"
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


class TestCreateAuthDependency:
    """Tests for create_auth_dependency function."""

    def test_anonymous_access_no_headers(self):
        """Test anonymous access when no auth headers provided."""
        mock_db = MagicMock()

        def db_getter():
            return mock_db

        get_auth_context = create_auth_dependency(db_getter)

        ctx = get_auth_context(x_api_key=None, authorization=None)

        assert ctx.db == mock_db
        assert ctx.user is None
        assert ctx.is_authenticated is False

    def test_valid_x_api_key_header(self):
        """Test authentication via X-API-Key header."""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 1

        def db_getter():
            return mock_db

        with patch("ai_news_tracker.auth.get_user_by_api_key") as mock_get_user:
            mock_get_user.return_value = mock_user

            get_auth_context = create_auth_dependency(db_getter)
            ctx = get_auth_context(x_api_key="valid-api-key", authorization=None)

            mock_get_user.assert_called_once_with(mock_db, "valid-api-key")
            assert ctx.user == mock_user
            assert ctx.is_authenticated is True

    def test_valid_bearer_authorization_header(self):
        """Test authentication via Authorization: Bearer header."""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 1

        def db_getter():
            return mock_db

        with patch("ai_news_tracker.auth.get_user_by_api_key") as mock_get_user:
            mock_get_user.return_value = mock_user

            get_auth_context = create_auth_dependency(db_getter)
            ctx = get_auth_context(x_api_key=None, authorization="Bearer my-api-key")

            mock_get_user.assert_called_once_with(mock_db, "my-api-key")
            assert ctx.user == mock_user
            assert ctx.is_authenticated is True

    def test_x_api_key_takes_precedence_over_bearer(self):
        """Test that X-API-Key header takes precedence over Authorization."""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 1

        def db_getter():
            return mock_db

        with patch("ai_news_tracker.auth.get_user_by_api_key") as mock_get_user:
            mock_get_user.return_value = mock_user

            get_auth_context = create_auth_dependency(db_getter)
            ctx = get_auth_context(
                x_api_key="x-api-key-value",
                authorization="Bearer bearer-value",
            )

            # Should use X-API-Key, not Bearer
            mock_get_user.assert_called_once_with(mock_db, "x-api-key-value")

    def test_invalid_api_key_raises_401(self):
        """Test that invalid API key raises HTTPException."""
        mock_db = MagicMock()

        def db_getter():
            return mock_db

        with patch("ai_news_tracker.auth.get_user_by_api_key") as mock_get_user:
            mock_get_user.return_value = None  # No user found

            get_auth_context = create_auth_dependency(db_getter)

            with pytest.raises(HTTPException) as exc_info:
                get_auth_context(x_api_key="invalid-key", authorization=None)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Invalid API key"

    def test_non_bearer_authorization_ignored(self):
        """Test that non-Bearer Authorization headers are ignored."""
        mock_db = MagicMock()

        def db_getter():
            return mock_db

        get_auth_context = create_auth_dependency(db_getter)
        ctx = get_auth_context(x_api_key=None, authorization="Basic dXNlcjpwYXNz")

        # Should treat as anonymous since it's not Bearer
        assert ctx.is_authenticated is False


class TestRequireAuthDependency:
    """Tests for require_auth dependency function."""

    def test_require_auth_with_authenticated_context(self):
        """Test require_auth returns context when authenticated."""
        mock_db = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 1

        auth_ctx = AuthContext(mock_db, user=mock_user)

        result = require_auth(auth=auth_ctx)

        assert result == auth_ctx

    def test_require_auth_with_anonymous_context(self):
        """Test require_auth raises when not authenticated."""
        mock_db = MagicMock()
        auth_ctx = AuthContext(mock_db, user=None)

        with pytest.raises(HTTPException) as exc_info:
            require_auth(auth=auth_ctx)

        assert exc_info.value.status_code == 401
