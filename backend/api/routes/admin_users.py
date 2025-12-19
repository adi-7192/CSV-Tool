"""
Admin API routes - User management
"""
from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List, Dict, Any
from models.user import UserResponse
from api.deps.auth_deps import get_current_admin
from services.user_service import get_all_users, get_user_by_id
from models.user import UserInDB
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class DeleteTenantRequest(BaseModel):
    """Request model for delete tenant endpoint"""
    confirm: str  # Must be "DELETE" to confirm
    delete_user: bool = False  # Whether to also delete the user record(s)


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
                created_at=user.created_at,
                is_active=user.is_active,
                last_login_at=user.last_login_at
            )
            for user in users
        ]
    except Exception as e:
        logger.error(f"Error fetching users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch users")


@router.delete("/tenants/{tenant_id}")
async def delete_tenant_endpoint(
    tenant_id: str,
    request: DeleteTenantRequest = Body(...),
    current_admin: UserInDB = Depends(get_current_admin),
):
    """
    Delete all data for a specific tenant (admin only).
    
    This endpoint allows admins to delete all data for a tenant, including:
    - All sales rows with tenant_id = {tenant_id}
    - Tenant's ChromaDB collection (rag_{tenant_id})
    - Ingestion logs for this tenant's files
    - Optionally: user record(s) with this tenant_id
    
    Requires confirmation: request body must include {"confirm": "DELETE"}
    
    Args:
        tenant_id: Tenant ID to delete
        request: Confirmation request body
        current_admin: Authenticated admin user (from JWT token)
    
    Returns:
        {
            "success": bool,
            "deleted_rows": int,
            "users_deleted": int,
            "message": str
        }
    """
    from core.database import get_connection, table_exists
    from utils.tenant_filter import get_tenant_filter_sql
    from services.embedding_service import get_chroma_client
    
    # Require confirmation
    if request.confirm != "DELETE":
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Send {'confirm': 'DELETE'} to delete tenant data."
        )
    
    try:
        conn = get_connection()
        tenant_filter = get_tenant_filter_sql(tenant_id)
        
        deleted_rows = 0
        users_deleted = 0
        
        # 1. Delete all sales rows for this tenant
        if table_exists('sales'):
            count_sql = f"SELECT COUNT(*) as count FROM sales WHERE {tenant_filter}"
            count_df = conn.execute(count_sql).fetchdf()
            if not count_df.empty:
                deleted_rows = int(count_df.iloc[0]['count'])
            
            if deleted_rows > 0:
                delete_sql = f"DELETE FROM sales WHERE {tenant_filter}"
                conn.execute(delete_sql)
        
        # 2. Delete ingestion logs for this tenant's files
        # Directly delete by tenant_id from ingestion_log (more reliable than checking sales table)
        if table_exists('ingestion_log'):
            # Delete all ingestion_log entries for this tenant
            delete_log_sql = f"DELETE FROM ingestion_log WHERE {tenant_filter}"
            conn.execute(delete_log_sql)
            
            # Also clean up any orphaned ingestion_log entries that might exist
            # (entries where sales data was already deleted but ingestion_log wasn't cleaned up)
            if table_exists('sales'):
                cleanup_sql = """
                DELETE FROM ingestion_log
                WHERE ingestion_id NOT IN (
                    SELECT DISTINCT ingestion_id FROM sales WHERE ingestion_id IS NOT NULL
                )
                """
                try:
                    conn.execute(cleanup_sql)
                except Exception as e:
                    # If cleanup fails, log but don't fail the operation
                    logger.warning(f"Failed to cleanup orphaned ingestion_log entries: {e}")
        
        # 3. Delete tenant's ChromaDB collection
        try:
            chroma_client = get_chroma_client()
            collection_name = f"rag_{tenant_id}"
            try:
                collection = chroma_client.get_collection(name=collection_name)
                collection.delete()  # Delete all documents
            except Exception:
                # Collection doesn't exist, which is fine
                pass
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB collection for tenant {tenant_id}: {e}")
        
        # 4. Optionally delete user record(s) with this tenant_id
        if request.delete_user:
            if table_exists('users'):
                # Find users with this tenant_id
                users_sql = "SELECT id FROM users WHERE tenant_id = ?"
                users_df = conn.execute(users_sql, [tenant_id]).fetchdf()
                user_ids = users_df['id'].tolist() if not users_df.empty else []
                
                for user_id in user_ids:
                    delete_user_sql = "DELETE FROM users WHERE id = ?"
                    conn.execute(delete_user_sql, [user_id])
                    users_deleted += 1
        
        return {
            "success": True,
            "deleted_rows": deleted_rows,
            "users_deleted": users_deleted,
            "message": f"Successfully deleted tenant {tenant_id}. Deleted {deleted_rows} rows and {users_deleted} user(s)."
        }
    
    except Exception as e:
        logger.error(f"Error deleting tenant {tenant_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting tenant: {str(e)}")


@router.patch("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    current_admin: UserInDB = Depends(get_current_admin),
):
    """
    Deactivate a user account (admin only).
    
    Deactivated users cannot login and will receive 403 on /auth/me.
    
    Args:
        user_id: User ID to deactivate
        current_admin: Authenticated admin user
    
    Returns:
        {
            "success": bool,
            "message": str
        }
    """
    from core.database import get_connection
    
    # Prevent deactivating yourself
    if user_id == current_admin.id:
        raise HTTPException(
            status_code=400,
            detail="Cannot deactivate your own account"
        )
    
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user.is_active:
        return {
            "success": True,
            "message": f"User {user.email} is already deactivated"
        }
    
    try:
        from core.database import get_connection
        conn = get_connection()
        
        # Ensure is_active column exists (migration safety check)
        try:
            # Try to check if column exists by querying it
            conn.execute("SELECT is_active FROM users WHERE id = ? LIMIT 1", [user_id]).fetchdf()
        except Exception as e:
            # Column doesn't exist, add it
            logger.warning(f"is_active column not found, attempting to add it: {e}")
            try:
                conn.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE")
                logger.info("✅ Added 'is_active' column to users table (migration)")
                # Set all existing users to active by default
                try:
                    conn.execute("UPDATE users SET is_active = TRUE WHERE is_active IS NULL")
                except:
                    pass  # Ignore if update fails
            except Exception as e2:
                logger.error(f"❌ Could not add is_active column: {e2}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Database schema error: is_active column missing. Please run: python backend/scripts/fix_is_active_column.py"
                )
        
        # Now update the user
        conn.execute(
            "UPDATE users SET is_active = FALSE WHERE id = ?",
            [user_id]
        )
        logger.info(f"Admin {current_admin.email} deactivated user {user.email} (ID: {user_id})")
        return {
            "success": True,
            "message": f"User {user.email} has been deactivated"
        }
    except Exception as e:
        logger.error(f"Error deactivating user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deactivating user: {str(e)}")


@router.patch("/users/{user_id}/activate")
async def activate_user(
    user_id: int,
    current_admin: UserInDB = Depends(get_current_admin),
):
    """
    Activate a user account (admin only).
    
    Args:
        user_id: User ID to activate
        current_admin: Authenticated admin user
    
    Returns:
        {
            "success": bool,
            "message": str
        }
    """
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_active:
        return {
            "success": True,
            "message": f"User {user.email} is already active"
        }
    
    try:
        from core.database import get_connection
        conn = get_connection()
        
        # Ensure is_active column exists (migration safety check)
        try:
            # Try to check if column exists by querying it
            conn.execute("SELECT is_active FROM users WHERE id = ? LIMIT 1", [user_id]).fetchdf()
        except Exception as e:
            # Column doesn't exist, add it
            logger.warning(f"is_active column not found, attempting to add it: {e}")
            try:
                conn.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE")
                logger.info("✅ Added 'is_active' column to users table (migration)")
                # Set all existing users to active by default
                try:
                    conn.execute("UPDATE users SET is_active = TRUE WHERE is_active IS NULL")
                except:
                    pass  # Ignore if update fails
            except Exception as e2:
                logger.error(f"❌ Could not add is_active column: {e2}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Database schema error: is_active column missing. Please run: python backend/scripts/fix_is_active_column.py"
                )
        
        # Now update the user
        conn.execute(
            "UPDATE users SET is_active = TRUE WHERE id = ?",
            [user_id]
        )
        logger.info(f"Admin {current_admin.email} activated user {user.email} (ID: {user_id})")
        return {
            "success": True,
            "message": f"User {user.email} has been activated"
        }
    except Exception as e:
        logger.error(f"Error activating user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error activating user: {str(e)}")


@router.get("/tenants/usage")
async def get_tenants_usage(
    current_admin: UserInDB = Depends(get_current_admin),
):
    """
    Get usage statistics for all tenants (admin only).
    
    Returns per-tenant usage stats:
    - tenant_id
    - user_email
    - file_count (from ingestion_log)
    - row_count (from sales)
    - last_upload_at (from ingestion_log)
    
    Returns:
        List of tenant usage objects
    """
    from core.database import get_connection, table_exists
    from utils.tenant_filter import get_tenant_filter_sql
    import pandas as pd
    
    try:
        conn = get_connection()
        usage_stats = []
        
        # Get all users with their tenant_ids
        users_df = conn.execute("SELECT id, email, tenant_id FROM users").fetchdf()
        
        if users_df.empty:
            return []
        
        for _, user_row in users_df.iterrows():
            tenant_id = user_row.get('tenant_id') or str(user_row['id'])
            user_email = user_row['email']
            
            # Get file count and last upload from ingestion_log
            file_count = 0
            last_upload_at = None
            if table_exists('ingestion_log'):
                ingestion_sql = f"""
                SELECT COUNT(*) as count, MAX(uploaded_at) as last_upload
                FROM ingestion_log
                WHERE {get_tenant_filter_sql(tenant_id)}
                """
                ingestion_df = conn.execute(ingestion_sql).fetchdf()
                if not ingestion_df.empty:
                    file_count = int(ingestion_df.iloc[0]['count'])
                    last_upload = ingestion_df.iloc[0].get('last_upload')
                    if pd.notna(last_upload):
                        last_upload_at = str(last_upload)
            
            # Get row count from sales
            row_count = 0
            if table_exists('sales'):
                row_sql = f"""
                SELECT COUNT(*) as count
                FROM sales
                WHERE {get_tenant_filter_sql(tenant_id)}
                """
                row_df = conn.execute(row_sql).fetchdf()
                if not row_df.empty:
                    row_count = int(row_df.iloc[0]['count'])
            
            usage_stats.append({
                "tenant_id": tenant_id,
                "user_email": user_email,
                "file_count": file_count,
                "row_count": row_count,
                "last_upload_at": last_upload_at
            })
        
        return usage_stats
    
    except Exception as e:
        logger.error(f"Error fetching tenant usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching tenant usage: {str(e)}")


@router.post("/backups/create")
async def create_backup_endpoint(
    current_admin: UserInDB = Depends(get_current_admin),
):
    """
    Create a backup of DuckDB and ChromaDB (admin only).
    
    This endpoint triggers a backup job and returns the backup path.
    The backup is created in backend/backups/ directory with a timestamp.
    
    Returns:
        {
            "success": bool,
            "backup_path": str,
            "message": str
        }
    """
    import subprocess
    from pathlib import Path
    
    try:
        # Run backup script
        # Path calculation: admin_users.py is in backend/api/routes/
        # So we go: parent (routes) -> parent (api) -> parent (backend) -> scripts/backup.py
        script_path = Path(__file__).parent.parent.parent / "scripts" / "backup.py"
        
        if not script_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Backup script not found at {script_path}"
            )
        
        # Run backup script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=script_path.parent.parent  # Run from backend/ directory
        )
        
        if result.returncode != 0:
            logger.error(f"Backup script failed: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Backup failed: {result.stderr}"
            )
        
        # Find latest backup directory
        backups_dir = Path(__file__).parent.parent.parent / "backups"
        backup_dirs = sorted(backups_dir.glob("backup_*"), reverse=True) if backups_dir.exists() else []
        
        if not backup_dirs:
            raise HTTPException(
                status_code=500,
                detail="Backup created but backup directory not found"
            )
        
        backup_path = str(backup_dirs[0])
        logger.info(f"Admin {current_admin.email} created backup: {backup_path}")
        
        return {
            "success": True,
            "backup_path": backup_path,
            "message": f"Backup created successfully at {backup_path}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import sys
        logger.error(f"Error creating backup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating backup: {str(e)}")

