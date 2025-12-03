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
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Singleton instance
_chroma_client = None
_collection = None


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


def get_collection(name: str = "sales_context"):
    """Get or create the main collection"""
    global _collection
    
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=name,
            metadata={"description": "Sales data context for RAG"}
        )
        logger.info(f"Collection '{name}' ready with {_collection.count()} documents")
    
    return _collection


def build_schema_documents() -> List[Dict[str, Any]]:
    """
    Build documents from database schema for embedding
    
    Returns list of documents with 'id', 'content', 'metadata'
    """
    from core.database import get_connection, execute_query
    
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
        
        # Get data statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT sku) as distinct_products,
            COUNT(DISTINCT region) as distinct_regions,
            MIN(order_date) as min_date,
            MAX(order_date) as max_date
        FROM sales
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


def index_all_documents():
    """Index all documents into ChromaDB"""
    collection = get_collection()
    
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
    
    # Schema documents
    schema_docs = build_schema_documents()
    all_docs.extend(schema_docs)
    logger.info(f"Built {len(schema_docs)} schema documents")
    
    # Business rule documents
    rule_docs = build_business_rule_documents()
    all_docs.extend(rule_docs)
    logger.info(f"Built {len(rule_docs)} business rule documents")
    
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


def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search on indexed documents
    
    Args:
        query: Natural language query
        top_k: Number of results to return
        
    Returns:
        List of relevant documents with scores
    """
    collection = get_collection()
    
    # Check if collection has documents
    if collection.count() == 0:
        logger.warning("Collection is empty, indexing documents first...")
        index_all_documents()
    
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


def get_relevant_context(question: str) -> str:
    """
    Get relevant context for a question as a formatted string
    
    Args:
        question: User's question
        
    Returns:
        Formatted context string for LLM prompt
    """
    results = semantic_search(question, top_k=5)
    
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
    """Initialize the embedding service"""
    try:
        collection = get_collection()
        if collection.count() == 0:
            logger.info("Empty collection, will index on first query")
        else:
            logger.info(f"Embedding service ready with {collection.count()} documents")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")

