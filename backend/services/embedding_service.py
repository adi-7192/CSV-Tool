"""
Embedding Service - ChromaDB for semantic search

Embeds:
- Database schema (column names, types, meanings)
- Business rules (net revenue calculation, transaction types)
- Query patterns (common SQL templates)
- Data statistics (date ranges, value distributions)

Uses sentence-transformers for embeddings, ChromaDB for vector storage.
"""

import logging
import os
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Singleton instance
_chroma_client = None
# Thread-local storage for collections (one per tenant per thread)
_collections_cache: Dict[str, Any] = {}
_collections_lock = threading.Lock()


def is_chromadb_available() -> bool:
    """Check if ChromaDB is available - always True with Python 3.12"""
    return True


def get_chroma_client():
    """Get or create ChromaDB client with persistent storage"""
    global _chroma_client
    
    if _chroma_client is None:
        # Store in backend/data/chromadb
        persist_dir = Path(__file__).parent.parent / "data" / "chromadb"
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info(f"ChromaDB initialized at {persist_dir}")
    
    return _chroma_client


def get_collection(tenant_id: Optional[str] = None, name: Optional[str] = None):
    """
    Get or create a collection for a specific tenant.
    
    TENANT ISOLATION: Each tenant gets their own collection to ensure
    strict data isolation in RAG queries.
    
    Args:
        tenant_id: User's tenant ID (REQUIRED for multi-tenant)
        name: Optional collection name (defaults to f"rag_{tenant_id}")
        
    Returns:
        ChromaDB collection for the tenant
        
    Raises:
        ValueError: If tenant_id is None (safety check)
    """
    if not tenant_id:
        raise ValueError("tenant_id is required for ChromaDB collection. Cannot query without tenant isolation.")
    
    # Use tenant-specific collection name
    collection_name = name or f"rag_{tenant_id}"
    
    # Check cache first (thread-safe)
    with _collections_lock:
        if collection_name in _collections_cache:
            return _collections_cache[collection_name]
    
    # Create or get collection
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": f"Sales data context for RAG - Tenant {tenant_id}",
            "tenant_id": tenant_id
        }
    )
    
    # Cache it
    with _collections_lock:
        _collections_cache[collection_name] = collection
    
    logger.info(f"Collection '{collection_name}' ready for tenant {tenant_id} with {collection.count()} documents")
    return collection


def build_schema_documents(tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build documents from database schema for embedding with tenant isolation.
    
    Args:
        tenant_id: User's tenant ID for data isolation (REQUIRED for multi-tenant)
    
    Returns list of documents with 'id', 'content', 'metadata'
    """
    from core.database import get_connection, execute_query
    from utils.tenant_filter import get_tenant_filter_sql
    
    if not tenant_id:
        raise ValueError("tenant_id is required for schema documents. Cannot index without tenant isolation.")
    
    documents = []
    
    try:
        conn = get_connection()
        
        # Check if sales table exists
        tables = conn.execute("SHOW TABLES").fetchdf()
        if 'sales' not in tables['name'].values:
            logger.warning("Sales table does not exist yet")
            return documents
        
        # Get column information
        columns_df = conn.execute("DESCRIBE sales").fetchdf()
        
        # Build column documents
        for _, row in columns_df.iterrows():
            col_name = row['column_name']
            col_type = row['column_type']
            
            # Add column description document
            col_doc = {
                'id': f'column_{col_name}',
                'content': f"Column '{col_name}' of type {col_type} in sales table. Used for {get_column_purpose(col_name)}.",
                'metadata': {
                    'type': 'schema',
                    'column_name': col_name,
                    'column_type': col_type,
                }
            }
            documents.append(col_doc)
        
        # Get data statistics (TENANT ISOLATED)
        tenant_filter = get_tenant_filter_sql(tenant_id)
        stats_query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT sku) as distinct_products,
            COUNT(DISTINCT region) as distinct_regions,
            MIN(order_date) as min_date,
            MAX(order_date) as max_date
        FROM sales
        WHERE {tenant_filter}
        """
        try:
            stats_df = execute_query(stats_query)
            if not stats_df.empty:
                stats = stats_df.iloc[0]
                stats_doc = {
                    'id': 'data_statistics',
                    'content': f"Sales database contains {stats['total_records']:,} records, {stats['distinct_products']} distinct products, {stats['distinct_regions']} regions. Date range: {stats['min_date']} to {stats['max_date']}.",
                    'metadata': {
                        'type': 'statistics',
                        'total_records': int(stats['total_records']),
                        'distinct_products': int(stats['distinct_products']),
                    }
                }
                documents.append(stats_doc)
        except Exception as e:
            logger.warning(f"Could not get data statistics: {e}")
        
    except Exception as e:
        logger.error(f"Error building schema documents: {e}")
    
    return documents


def get_column_purpose(col_name: str) -> str:
    """Get human-readable purpose for a column"""
    purposes = {
        'order_id': 'unique order identification',
        'order_date': 'filtering and grouping by date, time-series analysis',
        'revenue_amount': 'calculating revenue, sales totals, financial metrics',
        'transaction_type': 'filtering by Shipment, Refund, Cancel, FreeReplacement',
        'sku': 'product identification, product-level analysis',
        'quantity': 'order quantities, volume analysis',
        'region': 'geographic analysis, city/region breakdown',
        'shipping_amount': 'shipping costs, logistics analysis',
        'source_file': 'data lineage, tracking data sources',
        'ingestion_id': 'batch tracking, data management',
        'loaded_at': 'data freshness, ETL tracking',
        'updated_at': 'change tracking',
    }
    return purposes.get(col_name.lower(), 'data analysis')


def build_business_rule_documents() -> List[Dict[str, Any]]:
    """Build documents for business rules and formulas"""
    
    rules = [
        {
            'id': 'rule_net_revenue',
            'content': """Net Revenue Calculation: 
            net_revenue = gross_revenue - refunds - cancellations - free_replacements
            Where:
            - gross_revenue = SUM(revenue_amount) WHERE transaction_type = 'Shipment'
            - refunds = ABS(SUM(revenue_amount)) WHERE transaction_type = 'Refund'
            - cancellations = ABS(SUM(revenue_amount)) WHERE transaction_type = 'Cancel'
            - free_replacements = ABS(SUM(revenue_amount)) WHERE transaction_type = 'FreeReplacement'
            
            SQL Template:
            SELECT 
                SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as gross_revenue,
                ABS(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END)) as refunds,
                ABS(SUM(CASE WHEN transaction_type = 'Cancel' THEN revenue_amount ELSE 0 END)) as cancellations,
                ABS(SUM(CASE WHEN transaction_type = 'FreeReplacement' THEN revenue_amount ELSE 0 END)) as free_replacements,
                (SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) 
                 - ABS(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END))
                 - ABS(SUM(CASE WHEN transaction_type = 'Cancel' THEN revenue_amount ELSE 0 END))
                 - ABS(SUM(CASE WHEN transaction_type = 'FreeReplacement' THEN revenue_amount ELSE 0 END))) as net_revenue
            FROM sales
            WHERE order_date >= 'start_date' AND order_date <= 'end_date'
            """,
            'metadata': {'type': 'business_rule', 'category': 'revenue'}
        },
        {
            'id': 'rule_gross_revenue',
            'content': """Gross Revenue Calculation:
            gross_revenue = SUM(revenue_amount) WHERE transaction_type = 'Shipment'
            Only count Shipment transactions for gross revenue.
            Revenue is already in Indian Rupees (₹) - no conversion needed.
            
            SQL Template:
            SELECT SUM(revenue_amount) as gross_revenue
            FROM sales
            WHERE transaction_type = 'Shipment'
            AND order_date >= 'start_date' AND order_date <= 'end_date'
            """,
            'metadata': {'type': 'business_rule', 'category': 'revenue'}
        },
        {
            'id': 'rule_refund_rate',
            'content': """Refund Rate Calculation:
            refund_rate = (refund_count / shipment_count) * 100
            Or by value: (refund_amount / gross_revenue) * 100
            
            SQL Template:
            SELECT 
                COUNT(CASE WHEN transaction_type = 'Refund' THEN 1 END) * 100.0 / 
                NULLIF(COUNT(CASE WHEN transaction_type = 'Shipment' THEN 1 END), 0) as refund_rate_by_count,
                ABS(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END)) * 100.0 /
                NULLIF(SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END), 0) as refund_rate_by_value
            FROM sales
            WHERE order_date >= 'start_date' AND order_date <= 'end_date'
            """,
            'metadata': {'type': 'business_rule', 'category': 'metrics'}
        },
        {
            'id': 'rule_transaction_types',
            'content': """Transaction Types in the system:
            1. 'Shipment' - Revenue from successful orders
            2. 'Refund' - Returns and refunds (negative amounts, use ABS)
            3. 'Cancel' - Cancelled orders (negative amounts, use ABS)
            4. 'FreeReplacement' - Free replacements (cost to business, use ABS)
            
            Always use transaction_type column for filtering.
            """,
            'metadata': {'type': 'business_rule', 'category': 'transactions'}
        },
        {
            'id': 'rule_date_handling',
            'content': """Date Handling Rules:
            The date column is 'order_date' (not 'Invoice Date').
            Date filtering: WHERE order_date >= 'YYYY-MM-DD' AND order_date <= 'YYYY-MM-DD'
            Month extraction: DATE_PART('month', CAST(order_date AS DATE))
            Year extraction: DATE_PART('year', CAST(order_date AS DATE))
            
            IMPORTANT: Always cast to DATE if the column type is VARCHAR:
            CAST(order_date AS DATE)
            """,
            'metadata': {'type': 'business_rule', 'category': 'dates'}
        },
        {
            'id': 'rule_top_products',
            'content': """Top Products by Revenue:
            Group by SKU, sum revenue from Shipments, order descending.
            
            SQL Template:
            SELECT sku, SUM(revenue_amount) as total_revenue, COUNT(*) as order_count
            FROM sales
            WHERE transaction_type = 'Shipment'
            AND order_date >= 'start_date' AND order_date <= 'end_date'
            GROUP BY sku
            ORDER BY total_revenue DESC
            LIMIT N
            """,
            'metadata': {'type': 'business_rule', 'category': 'products'}
        },
        {
            'id': 'rule_region_analysis',
            'content': """Regional Analysis:
            Use 'region' column for geographic breakdown.
            
            SQL Template:
            SELECT region, SUM(revenue_amount) as total_revenue, COUNT(*) as order_count
            FROM sales
            WHERE transaction_type = 'Shipment'
            AND order_date >= 'start_date' AND order_date <= 'end_date'
            GROUP BY region
            ORDER BY total_revenue DESC
            """,
            'metadata': {'type': 'business_rule', 'category': 'geography'}
        },
        {
            'id': 'rule_monthly_analysis',
            'content': """Monthly Sales Analysis:
            For questions about "last month", "monthly trends", "month by month":
            
            SQL Template:
            SELECT 
                DATE_TRUNC('month', CAST(order_date AS DATE)) as month,
                SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as revenue,
                COUNT(CASE WHEN transaction_type = 'Shipment' THEN 1 END) as orders
            FROM sales
            WHERE order_date >= 'start_date' AND order_date <= 'end_date'
            GROUP BY DATE_TRUNC('month', CAST(order_date AS DATE))
            ORDER BY month
            """,
            'metadata': {'type': 'business_rule', 'category': 'time_series'}
        },
    ]
    
    return rules


def build_data_insight_documents(tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build documents from actual data insights for embedding with tenant isolation.
    This indexes real business patterns, not just schema.
    
    Args:
        tenant_id: User's tenant ID for data isolation (REQUIRED for multi-tenant)
    """
    from core.database import execute_query, table_exists
    from services.metrics_service import (
        get_top_products, get_revenue_by_city, get_movers_decliners,
        calculate_metrics
    )
    from utils.tenant_filter import get_tenant_filter_sql
    
    if not tenant_id:
        raise ValueError("tenant_id is required for data insight documents. Cannot index without tenant isolation.")
    
    documents = []
    
    if not table_exists('sales'):
        return documents
    
    try:
        # Get date range from database (TENANT ISOLATED)
        tenant_filter = get_tenant_filter_sql(tenant_id)
        date_range = execute_query(f"SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM sales WHERE {tenant_filter}")
        if date_range.empty:
            return documents
        
        min_date = str(date_range.iloc[0]['min_date'])[:10]
        max_date = str(date_range.iloc[0]['max_date'])[:10]
        
        # 1. Top Products Summary (TENANT ISOLATED)
        try:
            top_products_df = get_top_products(limit=10, start_date=min_date, end_date=max_date, metric='revenue', tenant_id=tenant_id)
            if not top_products_df.empty:
                top_skus = top_products_df.head(5)['sku'].tolist()
                doc = {
                    'id': 'insight_top_products',
                    'content': f"Top 5 products by revenue ({min_date} to {max_date}): {', '.join(top_skus)}. These are the best-performing SKUs in the dataset.",
                    'metadata': {
                        'type': 'data_insight',
                        'category': 'products',
                        'date_range': f"{min_date} to {max_date}"
                    }
                }
                documents.append(doc)
        except Exception as e:
            logger.warning(f"Could not build top products insight: {e}")
        
        # 2. Regional Performance Summary (TENANT ISOLATED)
        try:
            city_revenue = get_revenue_by_city(start_date=min_date, end_date=max_date, limit=5, tenant_id=tenant_id)
            if city_revenue and city_revenue.get('data'):
                top_cities = [item.get('city', '') for item in city_revenue['data'][:3]]
                doc = {
                    'id': 'insight_top_cities',
                    'content': f"Top 3 cities by revenue ({min_date} to {max_date}): {', '.join(top_cities)}. These regions generate the most sales.",
                    'metadata': {
                        'type': 'data_insight',
                        'category': 'geography',
                        'date_range': f"{min_date} to {max_date}"
                    }
                }
                documents.append(doc)
        except Exception as e:
            logger.warning(f"Could not build city revenue insight: {e}")
        
        # 3. Overall Metrics Summary (TENANT ISOLATED)
        try:
            metrics = calculate_metrics(start_date=min_date, end_date=max_date, tenant_id=tenant_id)
            if metrics:
                gross_rev = metrics.get('gross_revenue', 0)
                net_rev = metrics.get('net_revenue', 0)
                refund_rate = metrics.get('refund_rate', 0)
                orders = metrics.get('orders', 0)
                
                doc = {
                    'id': 'insight_overall_metrics',
                    'content': f"Overall business metrics ({min_date} to {max_date}): Gross revenue ₹{gross_rev/100000:.1f}L, Net revenue ₹{net_rev/100000:.1f}L, {orders:,} orders, {refund_rate:.1f}% refund rate.",
                    'metadata': {
                        'type': 'data_insight',
                        'category': 'summary',
                        'date_range': f"{min_date} to {max_date}"
                    }
                }
                documents.append(doc)
        except Exception as e:
            logger.warning(f"Could not build overall metrics insight: {e}")
        
        # 4. Movers and Decliners (if date range is sufficient)
        try:
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(min_date, '%Y-%m-%d')
            end_dt = datetime.strptime(max_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days + 1
            
            if days >= 7:  # Minimum for movers/decliners
                movers_decliners = get_movers_decliners(start_date=min_date, end_date=max_date, limit=5, tenant_id=tenant_id)
                if movers_decliners:
                    decliners = movers_decliners.get('decliners', [])
                    movers = movers_decliners.get('movers', [])
                    
                    if decliners:
                        declining_skus = [item.get('sku', '') for item in decliners[:3]]
                        doc = {
                            'id': 'insight_declining_products',
                            'content': f"Declining products ({min_date} to {max_date}): {', '.join(declining_skus)}. These SKUs show negative growth trends and may need attention.",
                            'metadata': {
                                'type': 'data_insight',
                                'category': 'trends',
                                'trend': 'declining',
                                'date_range': f"{min_date} to {max_date}"
                            }
                        }
                        documents.append(doc)
                    
                    if movers:
                        growing_skus = [item.get('sku', '') for item in movers[:3]]
                        doc = {
                            'id': 'insight_growing_products',
                            'content': f"Growing products ({min_date} to {max_date}): {', '.join(growing_skus)}. These SKUs show positive growth trends and are performing well.",
                            'metadata': {
                                'type': 'data_insight',
                                'category': 'trends',
                                'trend': 'growing',
                                'date_range': f"{min_date} to {max_date}"
                            }
                        }
                        documents.append(doc)
        except Exception as e:
            logger.warning(f"Could not build movers/decliners insight: {e}")
        
    except Exception as e:
        logger.error(f"Error building data insight documents: {e}")
    
    return documents


def index_all_documents(tenant_id: Optional[str] = None):
    """
    Index all documents into ChromaDB with tenant isolation.
    
    Args:
        tenant_id: User's tenant ID for data isolation (REQUIRED for multi-tenant)
    """
    if not tenant_id:
        raise ValueError("tenant_id is required for indexing documents. Cannot index without tenant isolation.")
    
    collection = get_collection(tenant_id=tenant_id)
    
    # Clear existing documents
    try:
        existing = collection.get()
        if existing['ids']:
            collection.delete(ids=existing['ids'])
            logger.info(f"Cleared {len(existing['ids'])} existing documents")
    except Exception as e:
        logger.warning(f"Could not clear existing documents: {e}")
    
    # Build all documents
    all_docs = []
    
    # Schema documents (TENANT ISOLATED)
    schema_docs = build_schema_documents(tenant_id=tenant_id)
    all_docs.extend(schema_docs)
    logger.info(f"Built {len(schema_docs)} schema documents for tenant {tenant_id}")
    
    # Business rule documents (shared across all tenants - no tenant_id needed)
    rule_docs = build_business_rule_documents()
    all_docs.extend(rule_docs)
    logger.info(f"Built {len(rule_docs)} business rule documents")
    
    # Data insight documents (TENANT ISOLATED - actual data patterns)
    insight_docs = build_data_insight_documents(tenant_id=tenant_id)
    all_docs.extend(insight_docs)
    logger.info(f"Built {len(insight_docs)} data insight documents for tenant {tenant_id}")
    
    if not all_docs:
        logger.warning("No documents to index")
        return 0
    
    # Add to collection
    ids = [doc['id'] for doc in all_docs]
    contents = [doc['content'] for doc in all_docs]
    metadatas = [doc['metadata'] for doc in all_docs]
    
    collection.add(
        ids=ids,
        documents=contents,
        metadatas=metadatas
    )
    
    logger.info(f"Indexed {len(all_docs)} documents into ChromaDB")
    return len(all_docs)


def semantic_search(query: str, top_k: int = 5, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Perform semantic search on indexed documents with tenant isolation.
    
    Args:
        query: Natural language query
        top_k: Number of results to return
        tenant_id: User's tenant ID for data isolation (REQUIRED for multi-tenant)
        
    Returns:
        List of relevant documents with scores
    """
    if not tenant_id:
        raise ValueError("tenant_id is required for semantic search. Cannot query without tenant isolation.")
    
    collection = get_collection(tenant_id=tenant_id)
    
    # Check if collection has documents
    if collection.count() == 0:
        logger.warning(f"Collection is empty for tenant {tenant_id}, indexing documents first...")
        index_all_documents(tenant_id=tenant_id)
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted.append({
                    'id': doc_id,
                    'content': results['documents'][0][i] if results['documents'] else None,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                })
        
        logger.debug(f"Semantic search for '{query[:50]}...' returned {len(formatted)} results")
        return formatted
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def get_relevant_context(question: str, tenant_id: Optional[str] = None) -> str:
    """
    Get relevant context for a question as a formatted string with tenant isolation.
    
    Args:
        question: User's question
        tenant_id: User's tenant ID for data isolation (REQUIRED for multi-tenant)
        
    Returns:
        Formatted context string for LLM prompt
    """
    if not tenant_id:
        raise ValueError("tenant_id is required for context retrieval. Cannot query without tenant isolation.")
    
    results = semantic_search(question, top_k=5, tenant_id=tenant_id)
    
    if not results:
        return ""
    
    context_parts = ["RELEVANT CONTEXT FROM KNOWLEDGE BASE:"]
    
    for i, result in enumerate(results, 1):
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        doc_type = metadata.get('type', 'unknown')
        
        # Truncate long content
        if len(content) > 500:
            content = content[:500] + "..."
        
        context_parts.append(f"\n[{i}] ({doc_type})")
        context_parts.append(content)
    
    return "\n".join(context_parts)


# Initialize on import
def init_embeddings():
    """Initialize the embedding service (no-op, collections are created on-demand per tenant)"""
    try:
        client = get_chroma_client()
        logger.info("Embedding service initialized (collections created per tenant on-demand)")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")

