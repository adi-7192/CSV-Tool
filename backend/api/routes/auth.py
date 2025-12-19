"""
Authentication API routes - Registration, login, user management, and password reset
"""
from fastapi import APIRouter, HTTPException, status, Depends, Request, BackgroundTasks
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from services.auth_service import verify_password, create_access_token, hash_password
from services.user_service import create_user, get_user_by_email, get_user_by_id
from services.password_reset_service import (
    create_password_reset_token,
    validate_password_reset_token,
    mark_token_as_used,
    update_user_password,
    validate_password_strength,
)
from services.email_service import send_password_reset_email, send_password_changed_email
from models.user import UserCreate, UserResponse
from api.deps.auth_deps import get_current_user
from models.user import UserInDB
from utils.rate_limiter import check_forgot_password_rate_limit
from core.config import settings
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


class ForgotPasswordRequest(BaseModel):
    """Forgot password request"""
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    """Forgot password response - always returns success for security"""
    message: str = "If an account with that email exists, a password reset link has been sent."


class ResetPasswordRequest(BaseModel):
    """Reset password request"""
    token: str = Field(..., min_length=1, description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")


class ChangePasswordRequest(BaseModel):
    """Change password request (for authenticated users)"""
    current_password: str = Field(..., min_length=1, description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")


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
        
        # Generate token with token_version (new users start at 0)
        token = create_access_token(
            user_id=user.id,
            email=user.email,
            role=user.role,
            tenant_id=user.tenant_id,
            token_version=user.token_version
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
                created_at=user.created_at,
                is_active=user.is_active,
                last_login_at=None  # New users haven't logged in yet
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
    
    # Check if user is active
    if not user.is_active:
        logger.warning(f"Login attempt by deactivated user: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact support.",
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        logger.warning(f"Invalid password attempt for user: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last_login_at
    from core.database import get_connection
    from datetime import datetime, timezone
    conn = get_connection()
    now = datetime.now(timezone.utc)
    try:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            [now, user.id]
        )
    except Exception as e:
        logger.warning(f"Failed to update last_login_at for user {user.id}: {e}")
        # Don't fail login if this update fails
    
    # Generate token with current token_version
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        role=user.role,
        tenant_id=user.tenant_id,
        token_version=user.token_version
    )
    
    logger.info(f"User logged in: {user.email} (ID: {user.id}, Role: {user.role})")
    
    # Refresh user to get updated last_login_at
    user = get_user_by_id(user.id)
    
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
            created_at=user.created_at,
            is_active=user.is_active,
            last_login_at=user.last_login_at
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Get current authenticated user information.
    
    Returns user details for the authenticated user.
    Inactive users will receive 403 before reaching this endpoint.
    
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
        created_at=current_user.created_at,
        is_active=current_user.is_active,
        last_login_at=current_user.last_login_at
    )


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    """
    Request a password reset link.
    
    Security notes:
    - Always returns 200 with generic message (prevents email enumeration)
    - Rate limited per IP and per email
    - Sends email only if user exists
    - Token expires after 15 minutes
    
    Args:
        request: ForgotPasswordRequest with email
        background_tasks: FastAPI BackgroundTasks for async email sending
        http_request: HTTP request for IP extraction
        
    Returns:
        ForgotPasswordResponse with generic message
    """
    email = request.email.lower()
    
    # Get client IP (check for proxy headers)
    forwarded_for = http_request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = http_request.client.host if http_request.client else "unknown"
    
    # Get user agent
    user_agent = http_request.headers.get("User-Agent", "")[:500]  # Limit length
    
    # Check rate limits
    try:
        is_allowed, error_message = check_forgot_password_rate_limit(client_ip, email)
    except RuntimeError as e:
        # Redis is required but unavailable - return 503
        logger.error(f"Rate limiting service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiting service is temporarily unavailable. Please try again later."
        )
    
    if not is_allowed:
        # Still return 200 with generic message to prevent enumeration
        # But log the rate limit hit
        if error_message and "unavailable" in error_message.lower():
            # Rate limiting service is down - return 503
            logger.error(f"Rate limiting service unavailable: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiting service is temporarily unavailable. Please try again later."
            )
        logger.warning(f"Rate limit hit for forgot-password: IP={client_ip}, email={email[:20]}...")
        return ForgotPasswordResponse()
    
    # Check if user exists
    user = get_user_by_email(email)
    
    if user:
        try:
            # Create reset token
            token = create_password_reset_token(
                user_id=user.id,
                request_ip=client_ip,
                user_agent=user_agent
            )
            
            # Send email in background (non-blocking)
            # Wrap in error handler to catch and log any exceptions
            def send_email_with_error_handling():
                try:
                    result = send_password_reset_email(
                        to_email=user.email,
                        reset_token=token,
                        expires_minutes=settings.PASSWORD_RESET_TOKEN_EXPIRES_MINUTES
                    )
                    if not result:
                        logger.error(f"Failed to send password reset email to {user.email} - send_email returned False")
                    return result
                except Exception as e:
                    logger.error(f"Exception in background task sending password reset email to {user.email}: {e}", exc_info=True)
                    return False
            
            background_tasks.add_task(send_email_with_error_handling)
            
            logger.info(f"Password reset requested for user: {email} - email queued for sending")
            
        except Exception as e:
            # Log error but don't expose to user
            logger.error(f"Error creating password reset token: {e}", exc_info=True)
    else:
        # User doesn't exist - log but don't reveal
        logger.info(f"Password reset requested for non-existent email: {email[:20]}...")
    
    # Always return success (prevents email enumeration)
    return ForgotPasswordResponse()


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest,
    background_tasks: BackgroundTasks,
):
    """
    Reset password using a valid token.
    
    Args:
        request: ResetPasswordRequest with token and new password
        background_tasks: FastAPI BackgroundTasks for async email notification
        
    Returns:
        Success message
        
    Raises:
        HTTPException: 400 if token is invalid/expired or password doesn't meet requirements
    """
    # Validate password strength
    is_valid, error_message = validate_password_strength(request.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )
    
    # Validate token
    token_valid, user_id, error_code = validate_password_reset_token(request.token)
    
    if not token_valid:
        error_messages = {
            "INVALID_TOKEN": "Invalid or expired password reset link. Please request a new one.",
            "TOKEN_ALREADY_USED": "This password reset link has already been used. Please request a new one.",
            "TOKEN_EXPIRED": "This password reset link has expired. Please request a new one.",
        }
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_messages.get(error_code, "Invalid password reset link.")
        )
    
    # Get user to verify they still exist
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account not found."
        )
    
    try:
        # Hash new password
        new_password_hash = hash_password(request.new_password)
        
        # Update password and password_changed_at (invalidates old sessions)
        update_user_password(user_id, new_password_hash)
        
        # Mark token as used
        mark_token_as_used(request.token)
        
        # Send notification email in background
        background_tasks.add_task(
            send_password_changed_email,
            to_email=user.email
        )
        
        logger.info(f"Password reset successful for user: {user.email}")
        
        return {
            "message": "Password has been reset successfully. You can now log in with your new password."
        }
        
    except Exception as e:
        logger.error(f"Error resetting password: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password. Please try again."
        )


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Change password for authenticated user.
    
    Requires current password verification and invalidates all existing sessions.
    
    Args:
        request: ChangePasswordRequest with current_password and new_password
        current_user: Current authenticated user (from dependency)
        
    Returns:
        Success message
        
    Raises:
        HTTPException: 400 if current password is incorrect or new password doesn't meet requirements
    """
    from services.auth_service import verify_password, hash_password
    from services.password_reset_service import update_user_password
    
    # Validate current password
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect."
        )
    
    # Validate new password strength
    is_valid, error_message = validate_password_strength(request.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )
    
    # Check that new password is different from current
    if verify_password(request.new_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password."
        )
    
    try:
        # Hash new password
        new_password_hash = hash_password(request.new_password)
        
        # Update password (this also increments token_version to invalidate sessions)
        update_user_password(current_user.id, new_password_hash)
        
        logger.info(f"Password changed successfully for user: {current_user.email}")
        
        return {
            "success": True,
            "message": "Password changed successfully. All active sessions have been invalidated. Please log in again."
        }
        
    except Exception as e:
        logger.error(f"Error changing password: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password. Please try again."
        )

