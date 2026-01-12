"""Authentication module for multi-user support."""

from typing import Optional
from fastapi import Header, HTTPException, Depends
from pydantic import BaseModel, Field, EmailStr

from .models import User, get_user_by_api_key


class UserCreate(BaseModel):
    """Request model for user registration."""
    email: str = Field(..., min_length=3, max_length=256)
    password: str = Field(..., min_length=8, max_length=128)
    display_name: Optional[str] = Field(None, max_length=256)


class UserLogin(BaseModel):
    """Request model for user login."""
    email: str = Field(..., min_length=3, max_length=256)
    password: str = Field(..., min_length=1, max_length=128)


class UserResponse(BaseModel):
    """Response model for user data."""
    id: int
    email: str
    display_name: Optional[str]
    api_key: str
    is_active: bool
    created_at: Optional[str]


class AuthContext:
    """
    Authentication context that can be injected into route handlers.

    Supports both authenticated and anonymous access.
    """

    def __init__(self, db_session, user: Optional[User] = None):
        self.db = db_session
        self.user = user
        self.user_id: Optional[int] = user.id if user else None
        self.is_authenticated = user is not None

    def require_auth(self):
        """Raise an exception if not authenticated."""
        if not self.is_authenticated:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )


def create_auth_dependency(db_session_getter):
    """
    Create an authentication dependency for FastAPI.

    Args:
        db_session_getter: A callable that returns the database session

    Returns:
        A FastAPI dependency function
    """

    def get_auth_context(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
        authorization: Optional[str] = Header(None),
    ) -> AuthContext:
        """
        Extract and validate authentication from request headers.

        Supports:
        - X-API-Key header
        - Authorization: Bearer <api_key> header

        Returns AuthContext with user if authenticated, or anonymous context.
        """
        db = db_session_getter()

        # Try X-API-Key header first
        api_key = x_api_key

        # Fall back to Authorization header
        if not api_key and authorization:
            if authorization.startswith("Bearer "):
                api_key = authorization[7:]

        # No auth provided - return anonymous context
        if not api_key:
            return AuthContext(db, user=None)

        # Validate API key
        user = get_user_by_api_key(db, api_key)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return AuthContext(db, user=user)

    return get_auth_context


def require_auth(auth: AuthContext = Depends()) -> AuthContext:
    """Dependency that requires authentication."""
    auth.require_auth()
    return auth
