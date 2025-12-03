"""
User management API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from api.deps.auth_deps import get_current_user
from models.user import UserInDB, UserResponse
from core.database import get_connection
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/onboarded", response_model=UserResponse)
async def mark_onboarded(
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Mark the current user as onboarded.
    
    This endpoint is called when the user completes onboarding (e.g., uploads first file).
    
    Args:
        current_user: Current authenticated user (from dependency)
        
    Returns:
        Updated user info with onboarded=True
    """
    try:
        conn = get_connection()
        
        # Update onboarded status
        conn.execute("""
            UPDATE users 
            SET onboarded = ? 
            WHERE id = ?
        """, [True, current_user.id])
        
        logger.info(f"User {current_user.email} marked as onboarded")
        
        # Fetch updated user
        from services.user_service import get_user_by_id
        updated_user = get_user_by_id(current_user.id)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=updated_user.id,
            email=updated_user.email,
            role=updated_user.role,
            plan=updated_user.plan,
            onboarded=updated_user.onboarded,
            tenant_id=updated_user.tenant_id,
            created_at=updated_user.created_at
        )
        
    except Exception as e:
        logger.error(f"Error marking user as onboarded: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update onboarded status"
        )

