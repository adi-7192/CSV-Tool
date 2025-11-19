"""
Health check endpoints - verify system status
"""
from fastapi import APIRouter, HTTPException
from core.database import get_connection, execute_query, table_exists
from core.ai_service import check_ollama_connection
from core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Basic health check - returns OK if API is running
    """
    return {"status": "healthy", "version": "2.0.0"}


@router.get("/detailed")
async def detailed_health():
    """
    Detailed health check - verifies database and Ollama connections
    """
    health_status = {
        "api": "healthy",
        "database": "unknown",
        "ollama": "unknown",
    }

    # Check database
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"

    # Check Ollama
    try:
        ollama_status = check_ollama_connection()
        health_status["ollama"] = "connected" if ollama_status else "disconnected"
    except Exception as e:
        health_status["ollama"] = f"error: {str(e)}"

    # Overall status
    all_healthy = (
        health_status["database"] == "connected" and
        health_status["ollama"] == "connected"
    )

    if not all_healthy:
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


@router.get("/data-quality")
async def data_quality_check():
    """
    Check for data quality issues:
    - Duplicate records based on business keys
    - Data from multiple sources for same date ranges
    - Missing required columns
    """
    if not table_exists('sales'):
        return {
            "status": "no_data",
            "message": "Sales table does not exist"
        }
    
    diagnostics = {
        "duplicates": {},
        "multi_source_data": {},
        "source_files": [],
        "date_ranges_by_source": {},
        "warnings": [],
        "errors": []
    }
    
    try:
        conn = get_connection()
        
        # 1. Check for duplicate business keys
        try:
            column_info = execute_query("DESCRIBE sales")
            existing_columns = column_info['column_name'].tolist()
            
            # Find business key columns
            business_keys = []
            order_id_cols = [col for col in existing_columns if any(kw in col.lower() for kw in ['invoice number', 'order id', 'orderid', 'invoice id', 'order_id'])]
            if order_id_cols:
                business_keys.append(order_id_cols[0])
            
            if 'transaction_type' in existing_columns:
                business_keys.append('transaction_type')
            elif 'Transaction Type' in existing_columns:
                business_keys.append('Transaction Type')
            
            sku_cols = [col for col in existing_columns if col.lower() == 'sku']
            if sku_cols:
                business_keys.append(sku_cols[0])
            
            if len(business_keys) >= 2:
                group_by_cols = ', '.join([f'"{col}"' for col in business_keys])
                dup_check_sql = f"""
                SELECT {group_by_cols}, COUNT(*) as duplicate_count
                FROM sales
                GROUP BY {group_by_cols}
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
                LIMIT 50
                """
                duplicates_df = execute_query(dup_check_sql)
                
                if len(duplicates_df) > 0:
                    diagnostics["duplicates"] = {
                        "found": True,
                        "count": len(duplicates_df),
                        "total_duplicate_rows": int(duplicates_df['duplicate_count'].sum()) - len(duplicates_df),
                        "examples": duplicates_df.head(10).to_dict('records')
                    }
                    diagnostics["warnings"].append(f"Found {len(duplicates_df)} duplicate business key combinations")
                else:
                    diagnostics["duplicates"] = {
                        "found": False,
                        "message": "No duplicates found based on business keys"
                    }
            else:
                diagnostics["warnings"].append("Insufficient business key columns to check duplicates")
        
        except Exception as e:
            diagnostics["errors"].append(f"Error checking duplicates: {str(e)}")
        
        # 2. Check for data from multiple sources
        try:
            # Get all source files
            if 'source_file' in existing_columns:
                source_files_sql = """
                SELECT DISTINCT source_file, COUNT(*) as row_count
                FROM sales
                WHERE source_file IS NOT NULL
                GROUP BY source_file
                ORDER BY row_count DESC
                """
                source_files_df = execute_query(source_files_sql)
                diagnostics["source_files"] = source_files_df.to_dict('records') if not source_files_df.empty else []
                
                # Check for overlapping date ranges by source
                date_col = None
                for col in ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']:
                    if col in existing_columns:
                        date_col = col
                        break
                
                if date_col and len(diagnostics["source_files"]) > 1:
                    # For each source file, get date range
                    for source_file_info in diagnostics["source_files"]:
                        source = source_file_info['source_file']
                        date_range_sql = f"""
                        SELECT 
                            MIN(CAST("{date_col}" AS DATE)) as min_date,
                            MAX(CAST("{date_col}" AS DATE)) as max_date,
                            COUNT(*) as row_count
                        FROM sales
                        WHERE source_file = '{source}'
                        AND "{date_col}" IS NOT NULL
                        """
                        range_df = execute_query(date_range_sql)
                        if not range_df.empty:
                            diagnostics["date_ranges_by_source"][source] = {
                                "min_date": str(range_df.iloc[0]['min_date']) if range_df.iloc[0]['min_date'] else None,
                                "max_date": str(range_df.iloc[0]['max_date']) if range_df.iloc[0]['max_date'] else None,
                                "row_count": int(range_df.iloc[0]['row_count'])
                            }
                    
                    # Check for overlapping date ranges
                    sources = list(diagnostics["date_ranges_by_source"].keys())
                    overlaps = []
                    for i, source1 in enumerate(sources):
                        for source2 in sources[i+1:]:
                            range1 = diagnostics["date_ranges_by_source"][source1]
                            range2 = diagnostics["date_ranges_by_source"][source2]
                            
                            if range1['min_date'] and range1['max_date'] and range2['min_date'] and range2['max_date']:
                                # Check if ranges overlap
                                if (range1['min_date'] <= range2['max_date'] and range1['max_date'] >= range2['min_date']):
                                    overlaps.append({
                                        "source1": source1,
                                        "source2": source2,
                                        "overlap_detected": True
                                    })
                    
                    if overlaps:
                        diagnostics["multi_source_data"] = {
                            "overlap_detected": True,
                            "overlapping_sources": overlaps,
                            "warning": "Multiple CSV files contain data for overlapping date ranges. This may cause duplicate counting."
                        }
                        diagnostics["warnings"].append(f"Found {len(overlaps)} pairs of source files with overlapping date ranges")
                    else:
                        diagnostics["multi_source_data"] = {
                            "overlap_detected": False,
                            "message": "No overlapping date ranges detected between source files"
                        }
            else:
                diagnostics["warnings"].append("source_file column not found - cannot check for multi-source data")
        
        except Exception as e:
            diagnostics["errors"].append(f"Error checking multi-source data: {str(e)}")
        
        # 3. Overall status
        has_issues = (
            diagnostics["duplicates"].get("found", False) or
            diagnostics["multi_source_data"].get("overlap_detected", False) or
            len(diagnostics["errors"]) > 0
        )
        
        diagnostics["status"] = "issues_found" if has_issues else "ok"
        diagnostics["summary"] = {
            "total_warnings": len(diagnostics["warnings"]),
            "total_errors": len(diagnostics["errors"]),
            "has_duplicates": diagnostics["duplicates"].get("found", False),
            "has_overlaps": diagnostics["multi_source_data"].get("overlap_detected", False)
        }
        
        return diagnostics
    
    except Exception as e:
        logger.error(f"Error in data quality check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data quality check failed: {str(e)}")

