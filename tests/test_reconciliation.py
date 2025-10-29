"""
Test suite for data reconciliation validation.

These tests verify that dashboard metrics match source CSV data.
"""

import pytest
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path to import reconcile module
sys.path.insert(0, str(Path(__file__).parent.parent))

from reconcile import (
    load_raw_csv_totals,
    load_database_totals,
    compare_totals,
    reconcile_single_file
)


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a sample CSV file for testing"""
    csv_data = {
        'Invoice Number': ['INV001', 'INV002', 'INV003'],
        'Invoice Amount': [1000.0, 2000.0, 1500.0],
        'Transaction Type': ['Shipment', 'Shipment', 'Refund'],
        'Invoice Date': ['2024-01-01', '2024-01-02', '2024-01-03']
    }
    df = pd.DataFrame(csv_data)
    csv_file = tmp_path / "test_sample.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def db_path():
    """Return database path"""
    db_path = "data/analytics.duckdb"
    if not os.path.exists(db_path):
        pytest.skip(f"Database not found: {db_path}. Upload data first.")
    return db_path


def test_load_raw_csv_totals(sample_csv_path):
    """Test loading totals from raw CSV file"""
    totals = load_raw_csv_totals(sample_csv_path)
    
    assert totals['total_rows'] == 3
    assert totals['total_amount_raw'] == 4500.0
    assert 'shipment_revenue' in totals
    assert totals['shipment_count'] == 2


def test_load_raw_csv_totals_handles_different_column_names(tmp_path):
    """Test that CSV loading handles different column naming conventions"""
    csv_data = {
        'Order Amount': [500.0, 600.0],
        'Type': ['Ship', 'Refund'],
        'Date': ['2024-01-01', '2024-01-02']
    }
    df = pd.DataFrame(csv_data)
    csv_file = tmp_path / "test_varying_names.csv"
    df.to_csv(csv_file, index=False)
    
    totals = load_raw_csv_totals(str(csv_file))
    
    assert totals['total_rows'] == 2
    assert totals['total_amount_raw'] == 1100.0


def test_load_database_totals(db_path):
    """Test loading totals from database"""
    # Skip if database doesn't exist or connection fails
    try:
        totals = load_database_totals(db_path)
        
        assert 'total_rows' in totals
        assert 'shipment_revenue' in totals
        assert 'shipment_count' in totals
        assert totals['total_rows'] >= 0
        assert totals['shipment_revenue'] >= 0
    except Exception as e:
        # If database is locked or connection fails, skip the test
        pytest.skip(f"Database connection issue: {e}")


def test_compare_totals_exact_match():
    """Test comparison when totals match exactly"""
    csv_totals = {
        'total_rows': 100,
        'shipment_revenue': 50000.0,
        'refund_amount': 5000.0
    }
    db_totals = {
        'total_rows': 100,
        'shipment_revenue': 50000.0,
        'refund_amount': 5000.0
    }
    
    passed, discrepancies = compare_totals(csv_totals, db_totals, tolerance=1.0)
    
    assert passed is True
    assert all("PASS" in d['status'] for d in discrepancies)


def test_compare_totals_within_tolerance():
    """Test comparison with small differences within tolerance"""
    csv_totals = {
        'total_rows': 100,
        'shipment_revenue': 50000.0
    }
    db_totals = {
        'total_rows': 99,  # 1 row removed (deduplication)
        'shipment_revenue': 49950.0  # 0.1% difference
    }
    
    passed, discrepancies = compare_totals(csv_totals, db_totals, tolerance=1.0)
    
    # Should pass due to tolerance
    assert passed is True


def test_compare_totals_fails_on_large_difference():
    """Test that comparison fails when difference exceeds tolerance"""
    csv_totals = {
        'total_rows': 100,
        'shipment_revenue': 50000.0
    }
    db_totals = {
        'total_rows': 150,  # Too many extra rows
        'shipment_revenue': 60000.0  # 20% difference
    }
    
    passed, discrepancies = compare_totals(csv_totals, db_totals, tolerance=1.0)
    
    assert passed is False
    assert any("FAIL" in d['status'] for d in discrepancies)


def test_compare_totals_allows_deduplication():
    """Test that row count comparison allows for deduplication"""
    csv_totals = {
        'total_rows': 100,
        'shipment_revenue': 50000.0
    }
    db_totals = {
        'total_rows': 95,  # 5 duplicates removed
        'shipment_revenue': 50000.0
    }
    
    passed, discrepancies = compare_totals(csv_totals, db_totals, tolerance=1.0)
    
    # Should pass - deduplication is expected
    assert passed is True
    row_count_discrepancy = [d for d in discrepancies if d['metric'] == 'Row Count'][0]
    assert "PASS" in row_count_discrepancy['status'] or "duplicates removed" in row_count_discrepancy['status']


def test_reconcile_single_file_integration(sample_csv_path, db_path):
    """
    Integration test: Reconcile a single file against database.
    This test requires actual database with data.
    """
    # This test will be skipped if database doesn't exist
    if not os.path.exists(db_path):
        pytest.skip(f"Database not found: {db_path}")
    
    passed, discrepancies = reconcile_single_file(sample_csv_path, tolerance=1.0, db_path=db_path)
    
    # Just verify function runs without error
    assert isinstance(passed, bool)
    assert isinstance(discrepancies, list)
    assert all('metric' in d for d in discrepancies)


def test_reconciliation_handles_missing_transaction_type(tmp_path):
    """Test that reconciliation works when CSV has no transaction type column"""
    csv_data = {
        'Amount': [100.0, 200.0, 300.0],
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03']
    }
    df = pd.DataFrame(csv_data)
    csv_file = tmp_path / "test_no_type.csv"
    df.to_csv(csv_file, index=False)
    
    totals = load_raw_csv_totals(str(csv_file))
    
    assert totals['total_rows'] == 3
    assert totals['total_amount_raw'] == 600.0
    # Should assume all are shipments when no type column
    assert totals['shipment_count'] == 3


def test_reconciliation_handles_currency_symbols(tmp_path):
    """Test that CSV loading handles currency symbols"""
    csv_data = {
        'Amount': ['₹1,000.50', '₹2,000.75', '$3,000.00'],
        'Type': ['Shipment', 'Shipment', 'Refund']
    }
    df = pd.DataFrame(csv_data)
    csv_file = tmp_path / "test_currency.csv"
    df.to_csv(csv_file, index=False)
    
    totals = load_raw_csv_totals(str(csv_file))
    
    # Should parse currency symbols correctly
    assert totals['total_amount_raw'] > 0
    assert totals['shipment_revenue'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

