"""
RAG Service - Retrieval-Augmented Generation with ChromaDB

Uses ChromaDB for semantic search to retrieve relevant context:
- Business rules (formulas, calculations)
- Schema information (column names, types)
- Query patterns (SQL templates)

This context augments LLM prompts for better SQL generation and responses.
"""

import logging
from typing import Dict, Any, Optional, List

from services.embedding_service import (
    get_relevant_context,
    semantic_search,
    index_all_documents,
    get_collection,
    is_chromadb_available,
)
from services.sql_validator import (
    get_date_column,
    get_revenue_column,
    get_transaction_type_column,
    get_actual_columns,
)

logger = logging.getLogger(__name__)


def retrieve_context(question: str, schema: Optional[Dict[str, Any]] = None, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve relevant context from ChromaDB based on question with tenant isolation.
    
    Args:
        question: User's natural language question
        schema: Optional database schema (if None, will fetch)
        tenant_id: User's tenant ID for data isolation (REQUIRED for multi-tenant)
        
    Returns:
        Dictionary with retrieved context including:
        - semantic_matches: Results from ChromaDB (tenant-isolated)
        - schema: Actual database schema with correct column names
        - column_mappings: Correct column names to use
        - business_rules: Relevant business rules for the query type
    """
    if not tenant_id:
        raise ValueError("tenant_id is required for RAG context retrieval. Cannot query without tenant isolation.")
    # Get schema if not provided
    if schema is None:
        from core.ai_service import get_database_schema
        schema = get_database_schema()
    
    context = {
        'question': question,
        'schema': schema,
    }
    
    # Get correct column mappings (critical for SQL generation)
    context['column_mappings'] = {
        'date_column': get_date_column(),
        'revenue_column': get_revenue_column(),
        'transaction_type_column': get_transaction_type_column(),
        'all_columns': list(get_actual_columns()),
    }
    
    # Perform semantic search in ChromaDB (TENANT ISOLATED)
    try:
        # Ensure documents are indexed for this tenant
        collection = get_collection(tenant_id=tenant_id)
        if collection.count() == 0:
            logger.info(f"Indexing documents for tenant {tenant_id} for first-time use...")
            index_all_documents(tenant_id=tenant_id)
        
        # Search for relevant context (tenant-isolated)
        semantic_results = semantic_search(question, top_k=5, tenant_id=tenant_id)
        context['semantic_matches'] = semantic_results
        
        # Extract business rules from results
        business_rules = []
        sql_templates = []
        
        for result in semantic_results:
            metadata = result.get('metadata', {})
            content = result.get('content', '')
            
            if metadata.get('type') == 'business_rule':
                business_rules.append({
                    'category': metadata.get('category', 'general'),
                    'content': content,
                })
            
            # Extract SQL templates from content
            if 'SQL Template:' in content:
                template_start = content.find('SQL Template:')
                template = content[template_start:]
                sql_templates.append(template)
        
        context['business_rules'] = business_rules
        context['sql_templates'] = sql_templates
        
        logger.info(f"RAG: Retrieved {len(semantic_results)} semantic matches, "
                   f"{len(business_rules)} business rules")
        
    except Exception as e:
        logger.error(f"RAG: Semantic search failed: {e}")
        context['semantic_matches'] = []
        context['business_rules'] = []
        context['sql_templates'] = []
    
    return context


def enhance_prompt_with_rag(original_prompt: str, context: Dict[str, Any]) -> str:
    """
    Enhance SQL generation prompt with RAG context
    
    Args:
        original_prompt: Original prompt for SQL generation
        context: Retrieved context dictionary from retrieve_context()
        
    Returns:
        Enhanced prompt with RAG context included
    """
    rag_sections = []
    
    # Add column mappings (CRITICAL)
    column_mappings = context.get('column_mappings', {})
    if column_mappings:
        rag_sections.append("=== CRITICAL: COLUMN NAME MAPPINGS ===")
        rag_sections.append("Use ONLY these column names (they are the actual columns in the database):")
        rag_sections.append(f"  - Date column: \"{column_mappings.get('date_column', 'order_date')}\"")
        rag_sections.append(f"  - Revenue column: \"{column_mappings.get('revenue_column', 'revenue_amount')}\"")
        rag_sections.append(f"  - Transaction type column: \"{column_mappings.get('transaction_type_column', 'transaction_type')}\"")
        
        all_cols = column_mappings.get('all_columns', [])
        if all_cols:
            rag_sections.append(f"  - All available columns: {', '.join(all_cols[:15])}")
        
        rag_sections.append("")
        rag_sections.append("DO NOT use 'Invoice Date', 'Invoice Amount', etc. - these are WRONG!")
        rag_sections.append("")
    
    # Add business rules from semantic search
    business_rules = context.get('business_rules', [])
    if business_rules:
        rag_sections.append("=== RELEVANT BUSINESS RULES ===")
        for rule in business_rules[:3]:  # Limit to top 3
            category = rule.get('category', 'general')
            content = rule.get('content', '')
            # Truncate long content
            if len(content) > 400:
                content = content[:400] + "..."
            rag_sections.append(f"[{category.upper()}]")
            rag_sections.append(content)
            rag_sections.append("")
    
    # Add SQL templates if relevant
    sql_templates = context.get('sql_templates', [])
    if sql_templates:
        rag_sections.append("=== EXAMPLE SQL TEMPLATES ===")
        for template in sql_templates[:2]:  # Limit to top 2
            # Truncate
            if len(template) > 300:
                template = template[:300] + "..."
            rag_sections.append(template)
            rag_sections.append("")
    
    # Combine with original prompt
    if rag_sections:
        rag_context = "\n".join(rag_sections)
        enhanced_prompt = f"""CONTEXT FROM KNOWLEDGE BASE:

{rag_context}

---

{original_prompt}

REMINDER: Use the EXACT column names from the COLUMN NAME MAPPINGS section above!
"""
    else:
        enhanced_prompt = original_prompt
    
    logger.debug(f"RAG: Enhanced prompt length: {len(enhanced_prompt)} chars (original: {len(original_prompt)} chars)")
    
    return enhanced_prompt


def get_data_statistics(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current data statistics for context with tenant isolation.
    
    Args:
        tenant_id: User's tenant ID for data isolation (REQUIRED for multi-tenant)
    """
    from core.database import execute_query
    from utils.tenant_filter import get_tenant_filter_sql
    
    if not tenant_id:
        raise ValueError("tenant_id is required for data statistics. Cannot query without tenant isolation.")
    
    stats = {
        'has_data': False,
        'total_records': 0,
        'date_range': None,
        'transaction_types': [],
    }
    
    try:
        date_col = get_date_column()
        tenant_filter = get_tenant_filter_sql(tenant_id)
        
        # Get basic stats (TENANT ISOLATED)
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            MIN("{date_col}") as min_date,
            MAX("{date_col}") as max_date
        FROM sales
        WHERE {tenant_filter}
        """
        df = execute_query(query)
        if not df.empty:
            stats['has_data'] = True
            stats['total_records'] = int(df.iloc[0]['total_records'])
            stats['date_range'] = {
                'min': str(df.iloc[0]['min_date']),
                'max': str(df.iloc[0]['max_date']),
            }
        
        # Get transaction types (TENANT ISOLATED)
        txn_col = get_transaction_type_column()
        query = f"""
        SELECT DISTINCT "{txn_col}" as txn_type
        FROM sales
        WHERE {tenant_filter} AND "{txn_col}" IS NOT NULL
        """
        df = execute_query(query)
        if not df.empty:
            stats['transaction_types'] = df['txn_type'].tolist()
        
    except Exception as e:
        logger.warning(f"Could not get data statistics: {e}")
    
    return stats


def format_rag_response(question: str, context: Dict[str, Any]) -> str:
    """
    Format a simple response using only RAG context (no LLM)
    Used as fallback when LLM is not available
    """
    response_parts = []
    
    # Check for business rule matches
    business_rules = context.get('business_rules', [])
    if business_rules:
        response_parts.append("Based on our knowledge base:\n")
        for rule in business_rules[:2]:
            content = rule.get('content', '')
            # Extract just the explanation, not SQL
            if 'SQL Template:' in content:
                content = content[:content.find('SQL Template:')]
            response_parts.append(f"• {content.strip()}\n")
    
    if not response_parts:
        response_parts.append("I found relevant information but need to query the database to give you accurate numbers.")
    
    return "\n".join(response_parts)
