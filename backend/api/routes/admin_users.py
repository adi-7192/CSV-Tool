"""
Admin API routes - User management
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from models.user import UserResponse
from api.deps.auth_deps import get_current_admin
from services.user_service import get_all_users
from models.user import UserInDB
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/users", response_model=List[UserResponse])
async def get_all_users_endpoint(
    current_admin: UserInDB = Depends(get_current_admin)
):
    """
    Get all users (admin only).
    
    Returns list of all registered users with their details.
    """
    try:
        users = get_all_users()
        return [
            UserResponse(
                id=user.id,
                email=user.email,
                role=user.role,
                plan=user.plan,
                onboarded=user.onboarded,
                tenant_id=user.tenant_id,
                created_at=user.created_at
            )
            for user in users
        ]
    except Exception as e:
        logger.error(f"Error fetching users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch users")

