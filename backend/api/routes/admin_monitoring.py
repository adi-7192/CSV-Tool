"""
Admin Monitoring API routes - System observability endpoints

Provides admin-only access to system events, health status, and monitoring data.
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from api.deps.auth_deps import get_current_admin
from models.user import UserInDB
from core.database import get_connection, table_exists
from core.config import settings
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/monitoring/events")
async def get_monitoring_events(
    level: Optional[str] = Query(None, description="Filter by level (INFO, WARN, ERROR)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint (partial match)"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant_id"),
    from_date: Optional[str] = Query(None, alias="from", description="Start date (YYYY-MM-DD or ISO format)"),
    to_date: Optional[str] = Query(None, alias="to", description="End date (YYYY-MM-DD or ISO format)"),
    limit: int = Query(200, ge=1, le=1000, description="Maximum number of events to return"),
    current_admin: UserInDB = Depends(get_current_admin),
) -> Dict[str, Any]:
    """
    Get system events with filtering (admin only).
    
    Returns paginated list of system events matching the filters.
    """
    try:
        if not table_exists('system_events'):
            return {
                "events": [],
                "total": 0,
                "limit": limit,
            }
        
        # Build WHERE clause
        where_conditions = []
        params = []
        
        if level:
            where_conditions.append("level = ?")
            params.append(level.upper())
        
        if category:
            where_conditions.append("category = ?")
            params.append(category)
        
        if endpoint:
            where_conditions.append("endpoint LIKE ?")
            params.append(f"%{endpoint}%")
        
        if tenant_id:
            where_conditions.append("tenant_id = ?")
            params.append(tenant_id)
        
        if from_date:
            where_conditions.append("created_at >= ?")
            params.append(from_date)
        
        if to_date:
            where_conditions.append("created_at <= ?")
            params.append(to_date)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # Get total count
        conn = get_connection()
        count_sql = f"SELECT COUNT(*) as count FROM system_events WHERE {where_clause}"
        if params:
            count_df = conn.execute(count_sql, params).fetchdf()
        else:
            count_df = conn.execute(count_sql).fetchdf()
        total = int(count_df.iloc[0]['count']) if not count_df.empty else 0
        
        # Get events (ordered by created_at DESC, most recent first)
        events_sql = f"""
        SELECT 
            id, created_at, level, category, message, endpoint, method,
            status_code, duration_ms, tenant_id, user_id, request_id, meta
        FROM system_events
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
        """
        events_params = params + [limit] if params else [limit]
        events_df = conn.execute(events_sql, events_params).fetchdf()
        
        # Convert to list of dicts
        events = []
        for _, row in events_df.iterrows():
            meta = None
            if row.get('meta'):
                try:
                    meta = json.loads(row['meta'])
                except (json.JSONDecodeError, TypeError):
                    meta = {"raw": str(row['meta'])}
            
            events.append({
                "id": int(row['id']),
                "created_at": str(row['created_at']),
                "level": row['level'],
                "category": row['category'],
                "message": row['message'],
                "endpoint": row['endpoint'],
                "method": row['method'],
                "status_code": int(row['status_code']) if row['status_code'] is not None else None,
                "duration_ms": int(row['duration_ms']) if row['duration_ms'] is not None else None,
                "tenant_id": row['tenant_id'],
                "user_id": row['user_id'],
                "request_id": row['request_id'],
                "meta": meta,
            })
        
        return {
            "events": events,
            "total": total,
            "limit": limit,
        }
        
    except Exception as e:
        logger.error(f"Error fetching monitoring events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching events: {str(e)}")


@router.get("/monitoring/summary")
async def get_monitoring_summary(
    current_admin: UserInDB = Depends(get_current_admin),
) -> Dict[str, Any]:
    """
    Get monitoring summary statistics (admin only).
    
    Returns:
    - Counts by level (last 24h)
    - Top endpoints by error count
    - Top endpoints by average duration
    """
    try:
        if not table_exists('system_events'):
            return {
                "counts_by_level": {},
                "top_error_endpoints": [],
                "top_slow_endpoints": [],
            }
        
        # Get counts by level (last 24h)
        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
        counts_sql = """
        SELECT level, COUNT(*) as count
        FROM system_events
        WHERE created_at >= ?
        GROUP BY level
        """
        conn = get_connection()
        counts_df = conn.execute(counts_sql, [twenty_four_hours_ago]).fetchdf()
        counts_by_level = {
            row['level']: int(row['count'])
            for _, row in counts_df.iterrows()
        }
        
        # Top endpoints by error count (last 24h)
        top_errors_sql = """
        SELECT endpoint, method, COUNT(*) as error_count
        FROM system_events
        WHERE created_at >= ? AND level = 'ERROR'
        GROUP BY endpoint, method
        ORDER BY error_count DESC
        LIMIT 10
        """
        top_errors_df = conn.execute(top_errors_sql, [twenty_four_hours_ago]).fetchdf()
        top_error_endpoints = [
            {
                "endpoint": row['endpoint'],
                "method": row['method'],
                "error_count": int(row['error_count']),
            }
            for _, row in top_errors_df.iterrows()
        ]
        
        # Top endpoints by average duration (last 24h, only requests with duration)
        top_slow_sql = """
        SELECT endpoint, method, AVG(duration_ms) as avg_duration_ms, COUNT(*) as request_count
        FROM system_events
        WHERE created_at >= ? AND duration_ms IS NOT NULL
        GROUP BY endpoint, method
        ORDER BY avg_duration_ms DESC
        LIMIT 10
        """
        top_slow_df = conn.execute(top_slow_sql, [twenty_four_hours_ago]).fetchdf()
        top_slow_endpoints = [
            {
                "endpoint": row['endpoint'],
                "method": row['method'],
                "avg_duration_ms": float(row['avg_duration_ms']) if row['avg_duration_ms'] is not None else None,
                "request_count": int(row['request_count']),
            }
            for _, row in top_slow_df.iterrows()
        ]
        
        return {
            "counts_by_level": counts_by_level,
            "top_error_endpoints": top_error_endpoints,
            "top_slow_endpoints": top_slow_endpoints,
        }
        
    except Exception as e:
        logger.error(f"Error fetching monitoring summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching summary: {str(e)}")


@router.get("/monitoring/health")
async def get_monitoring_health(
    current_admin: UserInDB = Depends(get_current_admin),
) -> Dict[str, Any]:
    """
    Get system health status (admin only).
    
    Returns:
    - redis_ok: Redis connection status (if configured)
    - db_ok: Database connection status
    - app_version: Application version
    - uptime_seconds: Server uptime (if available)
    """
    health_status = {
        "db_ok": False,
        "redis_ok": None,  # None means not configured
        "app_version": "2.0.0",
        "uptime_seconds": None,
    }
    
    # Check database
    try:
        conn = get_connection()
        conn.execute("SELECT 1")
        health_status["db_ok"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["db_ok"] = False
    
    # Check Redis (if configured)
    if settings.REDIS_URL:
        try:
            from utils.rate_limiter import get_rate_limiter
            limiter = get_rate_limiter()
            if hasattr(limiter, 'redis_available'):
                health_status["redis_ok"] = limiter.redis_available
            else:
                health_status["redis_ok"] = False
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            health_status["redis_ok"] = False
    else:
        health_status["redis_ok"] = None  # Not configured
    
    # Uptime would require storing startup time, skip for now
    # health_status["uptime_seconds"] = (datetime.now() - startup_time).total_seconds()
    
    return health_status

