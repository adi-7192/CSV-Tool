"""
Admin Statistics API routes - Overall system statistics

Provides admin-only access to aggregated system statistics.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime, timedelta
from api.deps.auth_deps import get_current_admin
from models.user import UserInDB
from core.database import get_connection, table_exists
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats/summary")
async def get_admin_stats_summary(
    current_admin: UserInDB = Depends(get_current_admin),
) -> Dict[str, Any]:
    """
    Get overall system statistics (admin only).
    
    Returns:
    - total_users: Total number of users
    - active_users: Number of active users (is_active = TRUE)
    - total_records: Total number of records across all tenants
    - api_calls_30d: Number of API calls in last 30 days (from system_events)
    """
    try:
        conn = get_connection()
        stats = {
            "total_users": 0,
            "active_users": 0,
            "total_records": 0,
            "api_calls_30d": 0,
        }
        
        # Get user statistics
        if table_exists('users'):
            users_df = conn.execute("SELECT COUNT(*) as total, COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active FROM users").fetchdf()
            if not users_df.empty:
                stats["total_users"] = int(users_df.iloc[0]['total'])
                stats["active_users"] = int(users_df.iloc[0]['active'])
        
        # Get total records across all tenants
        if table_exists('sales'):
            records_df = conn.execute("SELECT COUNT(*) as total FROM sales").fetchdf()
            if not records_df.empty:
                stats["total_records"] = int(records_df.iloc[0]['total'])
        
        # Get API calls from last 30 days (from system_events)
        if table_exists('system_events'):
            thirty_days_ago = datetime.now() - timedelta(days=30)
            api_calls_df = conn.execute(
                "SELECT COUNT(*) as total FROM system_events WHERE created_at >= ?",
                [thirty_days_ago]
            ).fetchdf()
            if not api_calls_df.empty:
                stats["api_calls_30d"] = int(api_calls_df.iloc[0]['total'])
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching admin stats summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

