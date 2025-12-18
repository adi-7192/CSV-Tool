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
                created_at=user.created_at
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
        ingestion_ids = []
        if table_exists('ingestion_log') and table_exists('sales'):
            ingestion_sql = f"""
            SELECT DISTINCT ingestion_id
            FROM sales
            WHERE {tenant_filter}
            """
            ingestion_df = conn.execute(ingestion_sql).fetchdf()
            ingestion_ids = ingestion_df['ingestion_id'].tolist() if not ingestion_df.empty else []
            
            for ingestion_id in ingestion_ids:
                other_tenants_sql = f"""
                SELECT COUNT(*) as count
                FROM sales
                WHERE ingestion_id = ? AND NOT ({tenant_filter})
                """
                other_tenants_df = conn.execute(other_tenants_sql, [ingestion_id]).fetchdf()
                other_tenants_count = int(other_tenants_df.iloc[0]['count']) if not other_tenants_df.empty else 0
                
                if other_tenants_count == 0:
                    delete_log_sql = "DELETE FROM ingestion_log WHERE ingestion_id = ?"
                    conn.execute(delete_log_sql, [ingestion_id])
        
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

