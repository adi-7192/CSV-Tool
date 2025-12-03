"""
Authentication API routes - Registration, login, and user management
"""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from services.auth_service import verify_password, create_access_token
from services.user_service import create_user, get_user_by_email
from models.user import UserCreate, UserResponse
from api.deps.auth_deps import get_current_user
from models.user import UserInDB
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class RegisterRequest(BaseModel):
    """User registration request
    
    Note: Role is NOT accepted in registration - all new users are created with role="user".
    Only admins can promote users via admin endpoints (to be implemented).
    """
    email: EmailStr
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters")
    name: Optional[str] = None
    # Explicitly exclude role field - users cannot set their own role


class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user.
    
    Creates a new user account with role "user" and returns an access token.
    
    Args:
        request: Registration data (email, password, optional name)
        
    Returns:
        TokenResponse with access token and user info
        
    Raises:
        HTTPException: If email already exists or validation fails
    """
    try:
        # Security: Explicitly ignore any role field if present in request
        # Registration always creates users with role="user"
        # Only admins can promote users (via admin endpoints, not implemented yet)
        
        # Check if user already exists
        existing_user = get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="EMAIL_ALREADY_EXISTS"
            )
        
        # Create user - ALWAYS with role="user" (role cannot be set via registration)
        # Any role value in request.data or request.dict() is explicitly ignored for security
        user_data = UserCreate(
            email=request.email,
            password=request.password,
            role="user",  # Enforced: all registrations create regular users, never admin
            plan="free",  # All new users start on free plan
            onboarded=False,  # New users need onboarding
            tenant_id=None
        )
        
        user = create_user(user_data)
        
        # Generate token
        token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            tenant_id=user.tenant_id
        )
        
        logger.info(f"User registered: {user.email} (ID: {user.id})")
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse(
                id=user.id,
                email=user.email,
                role=user.role,
                plan=user.plan,
                onboarded=user.onboarded,
                tenant_id=user.tenant_id,
                created_at=user.created_at
            )
        )
        
    except ValueError as e:
        logger.warning(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return access token.
    
    Args:
        request: Login credentials (email, password)
        
    Returns:
        TokenResponse with access token and user info
        
    Raises:
        HTTPException: If credentials are invalid
    """
    # Get user by email
    user = get_user_by_email(request.email)
    
    if not user:
        logger.warning(f"Login attempt with non-existent email: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        logger.warning(f"Invalid password attempt for user: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate token
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
        tenant_id=user.tenant_id
    )
    
    logger.info(f"User logged in: {user.email} (ID: {user.id}, Role: {user.role})")
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            email=user.email,
            role=user.role,
            plan=user.plan,
            onboarded=user.onboarded,
            tenant_id=user.tenant_id,
            created_at=user.created_at
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Get current authenticated user information.
    
    Args:
        current_user: Current authenticated user (from dependency)
        
    Returns:
        UserResponse with user info
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role,
        plan=current_user.plan,
        onboarded=current_user.onboarded,
        tenant_id=current_user.tenant_id,
        created_at=current_user.created_at
    )

