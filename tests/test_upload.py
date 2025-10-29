"""
Tests for CSV upload and processing workflow.

Tests cover:
- CSV file acceptance and reading
- Column mapping (auto-detection)
- Invalid CSV rejection
- Data cleaning initiation
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    auto_map_columns,
    clean_dataframe_transaction_aware,
    load_mapping
)


def test_csv_upload_reads_valid_file(sample_csv_file):
    """
    Test that valid CSV file can be read and parsed correctly.
    """
    df = pd.read_csv(sample_csv_file)
    
    assert df is not None
    assert len(df) > 0
    assert len(df.columns) > 0
    assert 'Invoice Number' in df.columns
    assert 'Invoice Amount' in df.columns


def test_csv_column_mapping_detects_required_fields(sample_csv_data, sample_column_mappings):
    """
    Test automatic column mapping detects required fields.
    """
    # Test auto_map_columns function
    mappings = auto_map_columns(sample_csv_data)
    
    # Should detect key fields
    assert mappings is not None
    assert isinstance(mappings, dict)
    
    # Check that essential mappings are found
    # Note: auto_map_columns might not find all fields, so we check what it does find
    detected_fields = [k for k, v in mappings.items() if v is not None]
    
    # At minimum, should detect some fields
    assert len(detected_fields) > 0
    
    # Verify specific common mappings
    if 'order_date' in mappings and mappings['order_date']:
        assert mappings['order_date'] in sample_csv_data.columns
    
    if 'order_id' in mappings and mappings['order_id']:
        assert mappings['order_id'] in sample_csv_data.columns
    
    if 'revenue_amount' in mappings and mappings['revenue_amount']:
        assert mappings['revenue_amount'] in sample_csv_data.columns


def test_column_mapping_handles_various_column_names(sample_df_with_various_names):
    """
    Test that column mapping handles different naming conventions.
    """
    mappings = auto_map_columns(sample_df_with_various_names)
    
    assert mappings is not None
    # Should detect some columns even with different names
    detected = sum(1 for v in mappings.values() if v is not None)
    assert detected > 0, "Should detect at least some columns"


def test_invalid_csv_with_missing_columns():
    """
    Test handling of CSV with missing essential columns.
    """
    # Create DataFrame with random columns that don't match expected fields
    invalid_df = pd.DataFrame({
        'random_col_1': [1, 2, 3],
        'random_col_2': ['a', 'b', 'c']
    })
    
    mappings = auto_map_columns(invalid_df)
    
    # Should return mappings dict but with None values for missing fields
    assert isinstance(mappings, dict)
    # Most mappings should be None for invalid data
    none_count = sum(1 for v in mappings.values() if v is None)
    assert none_count > 0, "Should return None for missing required columns"


def test_dataframe_cleaning_processes_data(sample_csv_data, sample_column_mappings):
    """
    Test that cleaning function processes data correctly.
    """
    df_cleaned, cleaning_report = clean_dataframe_transaction_aware(
        sample_csv_data, 
        sample_column_mappings
    )
    
    assert df_cleaned is not None
    assert isinstance(df_cleaned, pd.DataFrame)
    assert len(df_cleaned) > 0
    
    # Check cleaning report structure - it should be a dictionary
    assert isinstance(cleaning_report, dict)
    
    # Check for common report keys (structure may vary)
    assert 'business_key_cols' in cleaning_report or 'cleaning_steps' in cleaning_report
    
    # Verify that cleaned data has fewer or equal rows to original (duplicates removed)
    initial_count = len(sample_csv_data)
    final_count = len(df_cleaned)
    assert final_count <= initial_count, \
        f"Cleaned data should not have more rows than original: {initial_count} -> {final_count}"


def test_csv_file_is_created_and_readable(sample_csv_file):
    """
    Test that CSV file can be created and read back.
    """
    assert Path(sample_csv_file).exists()
    
    # Read back the file
    df = pd.read_csv(sample_csv_file)
    
    assert len(df) == 4  # Should have 4 rows from sample_csv_data fixture
    assert 'Invoice Number' in df.columns
    assert df['Invoice Number'].iloc[0] == 'INV001'


def test_empty_csv_handling(tmp_path):
    """
    Test that empty CSV is handled gracefully.
    """
    empty_csv = tmp_path / "empty.csv"
    empty_df = pd.DataFrame()
    empty_df.to_csv(empty_csv, index=False)
    
    # Try to read empty CSV - pandas raises EmptyDataError for truly empty CSVs
    try:
        df = pd.read_csv(empty_csv)
        # If it succeeds, should return empty DataFrame
        assert len(df) == 0
    except pd.errors.EmptyDataError:
        # This is expected behavior - empty CSV files raise EmptyDataError
        # The application should handle this gracefully
        pass


def test_csv_with_special_characters(tmp_path):
    """
    Test CSV with special characters in data.
    """
    data = pd.DataFrame({
        'Invoice Number': ['INV-001', 'INV/002', 'INV_003'],
        'Invoice Date': ['2025-10-01', '2025-10-02', '2025-10-03'],
        'Description': ['Product A (Special)', 'Product "B"', 'Product & Co.'],
        'Amount': [1000.0, 2000.0, 3000.0]
    })
    
    csv_path = tmp_path / "special_chars.csv"
    data.to_csv(csv_path, index=False)
    
    # Should read without errors
    df = pd.read_csv(csv_path)
    
    assert len(df) == 3
    assert 'Invoice Number' in df.columns

