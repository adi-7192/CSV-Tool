"""
Data Status endpoint - summarizes what data is currently loaded
"""
from fastapi import APIRouter
from typing import Dict, Any
from pathlib import Path

from core.config import settings
from core.database import execute_query, table_exists, get_row_count


router = APIRouter()


def _resolve_db_path() -> Path:
    db_path = Path(settings.DATABASE_PATH)
    if not db_path.is_absolute():
        if str(db_path).startswith("../"):
            relative_path = str(db_path)[3:]
            db_path = Path(__file__).parent.parent.parent / relative_path
        else:
            db_path = Path(__file__).parent.parent.parent / db_path
    return db_path


@router.get("/status")
async def data_status() -> Dict[str, Any]:
    db_file = _resolve_db_path()
    database_exists = db_file.exists()

    if not database_exists or not table_exists('sales'):
        return {
            "database_exists": database_exists,
            "total_rows": 0,
            "date_range": {"earliest": None, "latest": None},
            "source_files": [],
            "transaction_breakdown": {},
            "data_quality": {
                "duplicates": 0,
                "rows_without_order_id": 0,
                "synthetic_ids_generated": 0,
            },
        }

    # Total rows
    total_rows = get_row_count('sales')

    # Determine date column and range
    earliest = latest = None
    try:
        info = execute_query("DESCRIBE sales")
        date_columns = ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']
        date_col = None
        for col in date_columns:
            if col in info['column_name'].values:
                date_col = col
                break
        if date_col:
            col_type = info[info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                rng = execute_query(f"SELECT MIN(CAST(\"{date_col}\" AS DATE)) as earliest, MAX(CAST(\"{date_col}\" AS DATE)) as latest FROM sales")
            else:
                rng = execute_query(f"SELECT MIN(\"{date_col}\") as earliest, MAX(\"{date_col}\") as latest FROM sales")
            if not rng.empty:
                earliest = str(rng['earliest'].iloc[0]) if rng['earliest'].iloc[0] is not None else None
                latest = str(rng['latest'].iloc[0]) if rng['latest'].iloc[0] is not None else None
    except Exception:
        pass

    # Source files breakdown
    source_files = []
    try:
        src = execute_query(
            """
            SELECT 
                source_file as filename,
                COUNT(*) as rows,
                MIN(loaded_at) as loaded_at,
                MAX(updated_at) as last_updated
            FROM sales
            GROUP BY source_file
            ORDER BY last_updated DESC
            """
        )
        for _, r in src.iterrows():
            source_files.append({
                "filename": r.get('filename'),
                "rows": int(r.get('rows') or 0),
                "loaded_at": str(r.get('loaded_at')) if r.get('loaded_at') is not None else None,
                "last_updated": str(r.get('last_updated')) if r.get('last_updated') is not None else None,
            })
    except Exception:
        pass

    # Transaction breakdown
    transaction_breakdown: Dict[str, int] = {}
    try:
        txn = execute_query("SELECT transaction_type, COUNT(*) as cnt FROM sales GROUP BY transaction_type")
        for _, r in txn.iterrows():
            key = r.get('transaction_type') if r.get('transaction_type') is not None else 'Unknown'
            transaction_breakdown[str(key)] = int(r.get('cnt') or 0)
    except Exception:
        pass

    # Data quality checks
    duplicates = 0
    rows_without_order_id = 0
    synthetic_ids_generated = 0
    try:
        dup_df = execute_query(
            """
            SELECT COUNT(*) as groups FROM (
                SELECT order_id, transaction_type, COUNT(*) as cnt
                FROM sales
                GROUP BY order_id, transaction_type
                HAVING COUNT(*) > 1
            )
            """
        )
        if not dup_df.empty:
            duplicates = int(dup_df['groups'].iloc[0] or 0)

        nulls_df = execute_query("SELECT COUNT(*) as c FROM sales WHERE order_id IS NULL OR TRIM(order_id) = ''")
        if not nulls_df.empty:
            rows_without_order_id = int(nulls_df['c'].iloc[0] or 0)

        syn_df = execute_query("SELECT COUNT(*) as c FROM sales WHERE order_id LIKE 'UNKNOWN_%'")
        if not syn_df.empty:
            synthetic_ids_generated = int(syn_df['c'].iloc[0] or 0)
    except Exception:
        pass

    return {
        "database_exists": database_exists,
        "total_rows": total_rows,
        "date_range": {"earliest": earliest, "latest": latest},
        "source_files": source_files,
        "transaction_breakdown": transaction_breakdown,
        "data_quality": {
            "duplicates": duplicates,
            "rows_without_order_id": rows_without_order_id,
            "synthetic_ids_generated": synthetic_ids_generated,
        },
    }




