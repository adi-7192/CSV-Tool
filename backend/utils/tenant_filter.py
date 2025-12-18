"""
Tenant Filter Utilities - Enforce data isolation across all SQL queries

This module provides utilities to inject tenant filters into SQL queries,
ensuring strict data isolation between users.

SECURITY: Never accept tenant_id from client requests. Always derive from
the authenticated user's session.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def inject_tenant_filter(sql: str, tenant_id: str, table_name: str = 'sales') -> str:
    """
    Inject tenant_id filter into SQL query for data isolation.
    
    This function modifies SQL queries to add tenant_id filtering,
    ensuring users can only see their own data.
    
    Args:
        sql: Original SQL query
        tenant_id: User's tenant ID (from authenticated session)
        table_name: Table to filter (default: 'sales')
    
    Returns:
        Modified SQL with tenant filter injected
    
    Examples:
        Input:  SELECT * FROM sales WHERE order_date >= '2025-01-01'
        Output: SELECT * FROM sales WHERE tenant_id = 'user123' AND order_date >= '2025-01-01'
        
        Input:  SELECT * FROM sales
        Output: SELECT * FROM sales WHERE tenant_id = 'user123'
    """
    if not tenant_id:
        logger.warning("No tenant_id provided - query will return all data (SECURITY RISK)")
        return sql
    
    # Escape tenant_id to prevent SQL injection (though it should come from trusted source)
    safe_tenant_id = tenant_id.replace("'", "''")
    tenant_condition = f'tenant_id = \'{safe_tenant_id}\''
    
    # Check if query already has WHERE clause
    # Use case-insensitive matching
    sql_upper = sql.upper()
    
    # Find position of WHERE keyword (if exists)
    where_match = re.search(r'\bWHERE\b', sql_upper)
    
    if where_match:
        # Insert tenant filter after WHERE
        where_pos = where_match.end()
        # Check if there's already a condition
        rest_of_query = sql[where_pos:].strip()
        if rest_of_query and not rest_of_query.upper().startswith('ORDER') and not rest_of_query.upper().startswith('GROUP'):
            # There's already a condition, add AND
            modified_sql = sql[:where_pos] + f' {tenant_condition} AND' + sql[where_pos:]
        else:
            # WHERE is followed by ORDER BY or GROUP BY, or nothing - add condition
            modified_sql = sql[:where_pos] + f' {tenant_condition}' + sql[where_pos:]
    else:
        # No WHERE clause, need to add one
        # Find appropriate position (before ORDER BY, GROUP BY, LIMIT, etc.)
        
        # Check for FROM clause
        from_match = re.search(r'\bFROM\s+\w+', sql, re.IGNORECASE)
        if not from_match:
            logger.warning(f"Could not find FROM clause in SQL: {sql[:100]}...")
            return sql
        
        # Find position to insert WHERE (after FROM table_name and before ORDER BY/GROUP BY/LIMIT)
        insert_positions = []
        for keyword in ['ORDER BY', 'GROUP BY', 'HAVING', 'LIMIT', 'OFFSET']:
            match = re.search(rf'\b{keyword}\b', sql_upper)
            if match:
                insert_positions.append(match.start())
        
        if insert_positions:
            # Insert before the first keyword
            insert_pos = min(insert_positions)
            modified_sql = sql[:insert_pos] + f' WHERE {tenant_condition} ' + sql[insert_pos:]
        else:
            # No keywords found, append WHERE at the end
            modified_sql = sql + f' WHERE {tenant_condition}'
    
    logger.debug(f"Injected tenant filter: {tenant_id}")
    return modified_sql


def build_tenant_where_clause(
    existing_conditions: str,
    tenant_id: str,
    column_name: str = 'tenant_id'
) -> str:
    """
    Build a WHERE clause with tenant filter and existing conditions.
    
    Args:
        existing_conditions: Existing WHERE conditions (without 'WHERE' keyword)
        tenant_id: User's tenant ID
        column_name: Name of tenant column (default: 'tenant_id')
    
    Returns:
        Combined WHERE clause
    
    Examples:
        build_tenant_where_clause("order_date >= '2025-01-01'", "user123")
        -> "tenant_id = 'user123' AND order_date >= '2025-01-01'"
        
        build_tenant_where_clause("1=1", "user123")
        -> "tenant_id = 'user123'"
    """
    if not tenant_id:
        return existing_conditions
    
    safe_tenant_id = tenant_id.replace("'", "''")
    tenant_condition = f'{column_name} = \'{safe_tenant_id}\''
    
    # Clean up existing conditions
    existing = existing_conditions.strip()
    
    if not existing or existing == '1=1':
        return tenant_condition
    
    return f'{tenant_condition} AND {existing}'


def get_tenant_filter_sql(tenant_id: str, column_name: str = 'tenant_id') -> str:
    """
    Get a simple tenant filter condition for use in SQL queries.
    
    Args:
        tenant_id: User's tenant ID
        column_name: Name of tenant column
    
    Returns:
        SQL condition string (e.g., "tenant_id = 'user123'")
    """
    if not tenant_id:
        return '1=1'  # No filter if no tenant_id
    
    safe_tenant_id = tenant_id.replace("'", "''")
    return f'{column_name} = \'{safe_tenant_id}\''


def validate_tenant_access(requested_tenant_id: str, user_tenant_id: str) -> bool:
    """
    Validate that user has access to requested tenant data.
    
    SECURITY: This prevents users from accessing other tenants' data
    by manipulating request parameters.
    
    Args:
        requested_tenant_id: Tenant ID from request
        user_tenant_id: User's actual tenant ID from session
    
    Returns:
        True if access is allowed, False otherwise
    """
    if not user_tenant_id:
        logger.warning("User has no tenant_id - denying access")
        return False
    
    if requested_tenant_id != user_tenant_id:
        logger.warning(f"Tenant access violation: user {user_tenant_id} tried to access {requested_tenant_id}")
        return False
    
    return True

