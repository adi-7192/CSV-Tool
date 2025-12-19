"""
User model for authentication
"""
from pydantic import BaseModel, EmailStr
from typing import Literal, Optional
from datetime import datetime


class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr
    role: Literal["user", "admin"] = "user"
    plan: Literal["free", "pro", "enterprise"] = "free"
    onboarded: bool = False
    tenant_id: Optional[str] = None
    is_active: bool = True
    last_login_at: Optional[datetime] = None


class UserCreate(UserBase):
    """User creation model (includes password)"""
    password: str


class UserResponse(UserBase):
    """User response model (excludes password)"""
    id: int
    created_at: datetime
    is_active: Optional[bool] = True
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserInDB(UserBase):
    """User model as stored in database"""
    id: int
    password_hash: str
    created_at: datetime
    token_version: int = 0  # Incremented on password change to invalidate existing sessions
    is_active: bool = True
    last_login_at: Optional[datetime] = None

