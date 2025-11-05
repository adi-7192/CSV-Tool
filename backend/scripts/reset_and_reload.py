"""
Reset and Reload Script

Backs up the existing DB, resets the sales table schema, reloads JulyMonthly.csv
via the production upload service (with dedup + lineage), and verifies results.

Run: python backend/scripts/reset_and_reload.py
"""
import os
import sys
import shutil
from datetime import datetime

# Ensure project root on sys.path for imports
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.services.upload_service import process_csv_upload  # type: ignore
from backend.core.database import get_connection, table_exists  # type: ignore


# === Config ===
PRIMARY_DB = os.path.join(PROJECT_ROOT, 'sales_data.db')
ALT_DB = os.path.join(PROJECT_ROOT, 'data', 'analytics.duckdb')
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'JulyMonthly_raw.csv')
TABLE = 'sales'


# === Colors ===
class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def log(msg: str, color: str = Color.CYAN) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{color}[{ts}] {msg}{Color.RESET}")


def backup_database() -> str | None:
    """Create a timestamped backup of the current database if it exists."""
    db_path = PRIMARY_DB if os.path.exists(PRIMARY_DB) else ALT_DB
    if not os.path.exists(db_path):
        log("No database file found to backup.", Color.YELLOW)
        return None
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"sales_data_backup_{ts}.db"
    backup_path = os.path.join(PROJECT_ROOT, backup_name)
    shutil.copy2(db_path, backup_path)
    wal_src = db_path + '.wal'
    if os.path.exists(wal_src):
        try:
            shutil.copy2(wal_src, backup_path + '.wal')
        except Exception:
            pass
    log(f"Backup created: {backup_path}", Color.GREEN)
    return backup_path


def reset_sales_table() -> None:
    """Drop and recreate the sales table with the standardized schema."""
    conn = get_connection()
    log("Resetting sales table...", Color.CYAN)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        conn.execute(
            f"""
            CREATE TABLE {TABLE} (
                order_id VARCHAR,
                order_date DATE,
                revenue_amount DOUBLE,
                transaction_type VARCHAR,
                sku VARCHAR,
                quantity INTEGER,
                region VARCHAR,
                shipping_amount DOUBLE,
                source_file VARCHAR,
                ingestion_id VARCHAR,
                loaded_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        conn.commit()
        log("Sales table recreated with standardized schema.", Color.GREEN)
    finally:
        # Don't close the connection - it's a singleton that should be reused
        pass


def reload_csv() -> dict:
    """Process JulyMonthly.csv through the upload service."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    log(f"Loading CSV: {CSV_PATH}", Color.CYAN)
    with open(CSV_PATH, 'rb') as f:
        content = f.read()
    result = process_csv_upload(content, os.path.basename(CSV_PATH))
    if not result.get('success'):
        raise RuntimeError(result.get('error', 'Upload failed'))
    log("CSV uploaded via upload service.", Color.GREEN)
    return result


def verify() -> None:
    """Run basic verification queries and print a summary report."""
    # Use the existing connection instead of creating a new one
    conn = get_connection()
    log("Running verification checks...", Color.CYAN)

    # Duplicates
    dup = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT order_id, transaction_type, COUNT(*) AS cnt
            FROM sales
            GROUP BY order_id, transaction_type
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    # Counts by type
    by_type = conn.execute(
        """
        SELECT transaction_type, COUNT(*) AS rows
        FROM sales
        GROUP BY transaction_type
        ORDER BY transaction_type
        """
    ).fetchall()

    # Basic metrics
    gross, orders, refunds = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN transaction_type='Shipment' THEN revenue_amount ELSE 0 END),0) AS gross,
          COUNT(DISTINCT CASE WHEN transaction_type='Shipment' THEN order_id END) AS orders,
          ABS(COALESCE(SUM(CASE WHEN transaction_type='Refund' THEN revenue_amount ELSE 0 END),0)) AS refunds
        FROM sales
        """
    ).fetchone()
    # Don't close the connection - it's a singleton that should be reused

    # Report
    log("Verification Summary:", Color.CYAN)
    log(f"Duplicates (should be 0): {int(dup)}", Color.GREEN if int(dup) == 0 else Color.RED)
    for t, c in by_type:
        log(f"{t or 'Unknown'}: {c}")
    log(f"Gross Revenue: ₹{float(gross or 0):,.2f}", Color.CYAN)
    log(f"Orders: {int(orders or 0)}", Color.CYAN)
    log(f"Refunds: ₹{float(refunds or 0):,.2f}", Color.CYAN)


def main() -> None:
    log("Starting reset and reload...", Color.CYAN)
    backup_database()
    reset_sales_table()
    try:
        reload_csv()
    except Exception as e:
        log(f"Upload failed: {e}", Color.RED)
        sys.exit(1)
    verify()
    log("All done.", Color.GREEN)


if __name__ == '__main__':
    main()




