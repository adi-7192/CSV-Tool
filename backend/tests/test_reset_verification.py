"""End-to-end reset + reload verification tests.

Run with:
    pytest backend/tests/test_reset_verification.py -v
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / 'backend' / 'scripts' / 'reset_and_reload.py'
DB_PRIMARY = PROJECT_ROOT / 'sales_data.db'
DB_ALT = PROJECT_ROOT / 'data' / 'analytics.duckdb'


def _db_path() -> Path:
    return DB_PRIMARY if DB_PRIMARY.exists() else DB_ALT


def _connect_ro():
    return duckdb.connect(str(_db_path()), read_only=True)


def test_reset_and_reload_runs_successfully():
    assert SCRIPT_PATH.exists(), f"reset script missing at {SCRIPT_PATH}"
    # Run the script as a subprocess to simulate real usage
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, (
        "reset_and_reload.py failed to run successfully\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def test_no_duplicate_business_keys():
    conn = _connect_ro()
    dup_count = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT order_id, transaction_type, COUNT(*) AS cnt
            FROM sales
            GROUP BY order_id, transaction_type
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    conn.close()
    assert dup_count == 0, f"Expected 0 duplicate business keys, found {dup_count}"


def test_all_transaction_types_present():
    conn = _connect_ro()
    types = {t for (t,) in conn.execute(
        "SELECT DISTINCT transaction_type FROM sales"
    ).fetchall()}
    conn.close()
    expected = {"Shipment", "Refund", "Cancel", "FreeReplacement"}
    missing = expected - types
    assert not missing, f"Missing transaction types: {missing} (found: {types})"


def test_row_count_and_basic_metrics():
    conn = _connect_ro()
    total_rows = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    # Allow some tolerance; expected around 2170 rows for July
    assert 1800 <= total_rows <= 2500, (
        f"Unexpected row count: {total_rows} (expected ~2170 ± tolerance)"
    )

    gross, refunds = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN transaction_type='Shipment' THEN revenue_amount ELSE 0 END),0) AS gross,
          ABS(COALESCE(SUM(CASE WHEN transaction_type='Refund' THEN revenue_amount ELSE 0 END),0)) AS refunds
        FROM sales
        """
    ).fetchone()
    conn.close()

    gross = float(gross or 0)
    refunds = float(refunds or 0)
    # Gross approx check (~ 2,859,300 for July), allow ±10%
    assert 2_500_000 <= gross <= 3_200_000, (
        f"Gross revenue out of expected range: {gross:,.2f}"
    )

    expected_net = gross - refunds
    assert expected_net >= 0 or abs(expected_net) < 1e-6, (
        f"Net revenue calculation unexpected: gross={gross}, refunds={refunds}"
    )


def test_lineage_tracking_and_data_quality():
    conn = _connect_ro()
    lineage_missing = conn.execute(
        "SELECT COUNT(*) FROM sales WHERE source_file IS NULL OR ingestion_id IS NULL"
    ).fetchone()[0]
    null_order_ids = conn.execute(
        "SELECT COUNT(*) FROM sales WHERE order_id IS NULL OR TRIM(order_id) = ''"
    ).fetchone()[0]
    invalid_dates = conn.execute(
        "SELECT COUNT(*) FROM sales WHERE order_date IS NULL"
    ).fetchone()[0]
    missing_sku = conn.execute(
        "SELECT COUNT(*) FROM sales WHERE sku IS NULL OR TRIM(sku) = ''"
    ).fetchone()[0]
    conn.close()

    assert lineage_missing == 0, (
        f"Lineage columns missing in {lineage_missing} rows (source_file/ingestion_id)"
    )
    assert null_order_ids == 0, (
        f"Found {null_order_ids} rows with null/empty order_id (should be 0 after synthetic IDs)"
    )
    # Some datasets may have a few null dates filtered out on load; by now table should have valid dates
    assert invalid_dates == 0, f"Found {invalid_dates} rows with invalid/null order_date"
    # SKU should be fully populated after mapping fix; allow at most tiny noise
    assert missing_sku <= 5, f"Too many rows missing SKU: {missing_sku}"




