"""
Shared test fixtures and configuration for pytest.

This module provides reusable fixtures for all test modules.
"""

import pytest
import pandas as pd
from pathlib import Path
import duckdb
import os
from datetime import datetime
from typing import Dict


@pytest.fixture
def sample_csv_data():
    """
    Generate sample CSV data for testing.
    
    Returns a DataFrame with typical transaction data including:
    - Order IDs (with one duplicate)
    - Dates
    - SKUs
    - Invoice amounts
    - Quantities
    - Transaction types
    """
    return pd.DataFrame({
        'Invoice Number': ['INV001', 'INV002', 'INV003', 'INV002'],  # INV002 duplicate
        'Invoice Date': ['2025-10-01', '2025-10-02', '2025-10-03', '2025-10-02'],
        'SKU': ['PROD-A', 'PROD-B', 'PROD-C', 'PROD-B'],
        'Invoice Amount': [1000.0, 2000.0, -300.0, 2500.0],  # Last one is duplicate with higher amount
        'Qty': [1, 2, 1, 2],
        'Type': ['Shipment', 'Shipment', 'Refund', 'Shipment']
    })


@pytest.fixture
def sample_csv_file(tmp_path, sample_csv_data):
    """
    Create temporary CSV file for testing.
    
    Args:
        tmp_path: pytest temporary directory fixture
        sample_csv_data: DataFrame fixture with sample data
        
    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "test_data.csv"
    sample_csv_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_csv_with_duplicates(tmp_path):
    """
    Create CSV file with multiple types of duplicates for deduplication testing.
    """
    data = pd.DataFrame({
        'Invoice Number': ['INV001', 'INV001', 'INV002', 'INV003', 'INV003'],
        'Invoice Date': ['2025-10-01', '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-03'],
        'SKU': ['PROD-A', 'PROD-A', 'PROD-B', 'PROD-C', 'PROD-C'],
        'Invoice Amount': [1000.0, 1000.0, 2000.0, 3000.0, 3500.0],  # Last INV003 has different amount
        'Type': ['Shipment', 'Shipment', 'Shipment', 'Shipment', 'Shipment']
    })
    csv_path = tmp_path / "test_duplicates.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_csv_invalid_data(tmp_path):
    """
    Create CSV file with invalid data for validation testing.
    """
    data = pd.DataFrame({
        'Invoice Number': ['INV001', 'INV002', 'INV003', 'INV004'],
        'Invoice Date': ['2025-10-01', 'invalid-date', '2025-13-45', '2025-10-04'],  # Invalid dates
        'SKU': ['PROD-A', 'PROD-B', 'PROD-C', 'PROD-D'],
        'Invoice Amount': [1000.0, -500.0, 0.0, 2000.0],  # Negative shipment, zero amount
        'Qty': [1, 2, 0, 1],  # Zero quantity
        'Type': ['Shipment', 'Shipment', 'Shipment', 'Refund']  # Negative Shipment should be flagged
    })
    csv_path = tmp_path / "test_invalid.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def test_database(tmp_path):
    """
    Create temporary test database.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Yields:
        DuckDB connection object
    """
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    yield conn
    conn.close()
    # Clean up
    if os.path.exists(str(db_path)):
        os.remove(str(db_path))


@pytest.fixture
def sample_column_mappings():
    """
    Standard column mappings for testing.
    """
    return {
        'order_date': 'Invoice Date',
        'order_id': 'Invoice Number',
        'sku': 'SKU',
        'revenue_amount': 'Invoice Amount',
        'quantity': 'Qty',
        'transaction_type': 'Type'
    }


@pytest.fixture
def sample_df_with_various_names():
    """
    DataFrame with varying column name conventions to test column mapping.
    """
    return pd.DataFrame({
        'Order Date': ['2025-10-01', '2025-10-02'],
        'Order Number': ['ORD001', 'ORD002'],
        'Product ID': ['PROD-A', 'PROD-B'],
        'Total Amount': [1000.0, 2000.0],
        'Units': [1, 2],
        'Order Type': ['Shipment', 'Shipment']
    })


@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """
    Mock Streamlit components that are not available in test environment.
    This fixture runs automatically for all tests.
    """
    # Mock st.cache_resource to return the decorated function unchanged
    def mock_cache_resource(func):
        return func
    
    # Mock st.cache_data to return the decorated function unchanged
    def mock_cache_data(func):
        return func
    
    # Mock Streamlit session state
    class MockSessionState:
        def __init__(self):
            self.data = {}
        
        def __getitem__(self, key):
            return self.data.get(key)
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def get(self, key, default=None):
            return self.data.get(key, default)
        
        def __contains__(self, key):
            return key in self.data
    
    # Apply mocks
    monkeypatch.setattr('streamlit.cache_resource', mock_cache_resource, raising=False)
    monkeypatch.setattr('streamlit.cache_data', mock_cache_data, raising=False)
    
    # Try to mock streamlit module if available
    try:
        import sys
        mock_st = type('MockStreamlit', (), {
            'cache_resource': staticmethod(mock_cache_resource),
            'cache_data': staticmethod(mock_cache_data),
            'session_state': MockSessionState()
        })()
        sys.modules['streamlit'] = mock_st
    except:
        pass

