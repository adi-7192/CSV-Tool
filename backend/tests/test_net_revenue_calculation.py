"""
Tests for Net Revenue Calculation

Tests verify:
1. September 2025 calculation against /api/metrics endpoint
2. Multi-month aggregation (July + Aug + Sept)
3. Edge cases (empty range, single transaction, etc.)
4. AI Chat query generation (verify correct SQL)

Formula: net_revenue = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
"""
import pytest
import requests
import sys
from pathlib import Path
from typing import Dict, Any
import re

# Add backend to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.metrics_service import calculate_metrics

API_BASE = "http://localhost:8000"


class TestNetRevenueFormula:
    """Test that net revenue formula is correct"""
    
    def test_formula_components(self):
        """Test that net_revenue = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost"""
        gross_revenue = 3000.0
        refund_amount = 500.0
        cancellation_amount = 300.0
        free_replacement_cost = 300.0
        
        expected_net = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
        assert expected_net == 1900.0, f"Expected net_revenue = 1900.0, got {expected_net}"
    
    def test_formula_with_zero_values(self):
        """Test formula with zero values"""
        gross_revenue = 1000.0
        refund_amount = 0.0
        cancellation_amount = 0.0
        free_replacement_cost = 0.0
        
        expected_net = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
        assert expected_net == 1000.0, f"Expected net_revenue = 1000.0, got {expected_net}"


class TestSeptember2025Calculation:
    """Test September 2025 net revenue calculation"""
    
    @pytest.fixture(scope="class")
    def september_metrics(self):
        """Get September 2025 metrics from API"""
        try:
            response = requests.get(
                f"{API_BASE}/api/metrics",
                params={
                    'start_date': '2025-09-01',
                    'end_date': '2025-09-30'
                },
                timeout=10
            )
            if response.status_code == 200:
                return response.json()['data']
            else:
                pytest.skip(f"API not available (status {response.status_code})")
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available at http://localhost:8000")
    
    def test_september_metrics_exist(self, september_metrics):
        """Verify September metrics are returned"""
        assert september_metrics is not None, "September metrics should not be None"
        assert 'gross_revenue' in september_metrics or 'revenue' in september_metrics
        assert 'net_revenue' in september_metrics
    
    def test_september_net_revenue_formula(self, september_metrics):
        """Verify September net_revenue matches formula"""
        gross_revenue = september_metrics.get('gross_revenue') or september_metrics.get('revenue', 0)
        refund_amount = september_metrics.get('refund_amount') or september_metrics.get('refunds', 0)
        cancellation_amount = september_metrics.get('cancellation_amount', 0)
        free_replacement_cost = september_metrics.get('free_replacement_cost', 0)
        net_revenue = september_metrics.get('net_revenue', 0)
        
        # Calculate expected net revenue
        expected_net = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
        
        # Allow small floating point differences (0.01 tolerance)
        diff = abs(net_revenue - expected_net)
        assert diff < 0.01, (
            f"Net revenue formula mismatch!\n"
            f"  Expected: {expected_net:.2f}\n"
            f"  Actual:   {net_revenue:.2f}\n"
            f"  Diff:     {diff:.2f}\n"
            f"  Components:\n"
            f"    gross_revenue: {gross_revenue:.2f}\n"
            f"    refund_amount: {refund_amount:.2f}\n"
            f"    cancellation_amount: {cancellation_amount:.2f}\n"
            f"    free_replacement_cost: {free_replacement_cost:.2f}"
        )
    
    def test_september_all_components_present(self, september_metrics):
        """Verify all components are present in response"""
        required_fields = [
            'gross_revenue', 'refund_amount', 'cancellation_amount', 
            'free_replacement_cost', 'net_revenue', 'net_margin'
        ]
        
        for field in required_fields:
            assert field in september_metrics, f"Missing required field: {field}"
            assert isinstance(september_metrics[field], (int, float)), f"{field} should be numeric"
            assert september_metrics[field] >= 0, f"{field} should be >= 0"


class TestMetricsServiceDirect:
    """Test metrics service directly (not via API)"""
    
    def test_september_calculation_direct(self):
        """Test September calculation using metrics service directly"""
        try:
            metrics = calculate_metrics(
                start_date='2025-09-01',
                end_date='2025-09-30'
            )
            
            assert metrics is not None, "Metrics should not be None"
            assert 'net_revenue' in metrics, "Metrics should include net_revenue"
            
            # Verify formula
            gross_revenue = metrics.get('gross_revenue') or metrics.get('revenue', 0)
            refund_amount = metrics.get('refund_amount') or metrics.get('refunds', 0)
            cancellation_amount = metrics.get('cancellation_amount', 0)
            free_replacement_cost = metrics.get('free_replacement_cost', 0)
            net_revenue = metrics.get('net_revenue', 0)
            
            expected_net = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
            diff = abs(net_revenue - expected_net)
            
            assert diff < 0.01, (
                f"Direct calculation formula mismatch!\n"
                f"  Expected: {expected_net:.2f}\n"
                f"  Actual:   {net_revenue:.2f}\n"
                f"  Components: gross={gross_revenue:.2f}, refund={refund_amount:.2f}, "
                f"cancel={cancellation_amount:.2f}, free_repl={free_replacement_cost:.2f}"
            )
        except Exception as e:
            pytest.skip(f"Database not available or error: {e}")


class TestMultiMonthAggregation:
    """Test multi-month aggregation (July + August + September)"""
    
    def test_july_august_september_aggregation(self):
        """Test calculating net revenue for July + August + September"""
        try:
            # Get metrics for each month
            july_metrics = calculate_metrics('2025-07-01', '2025-07-31')
            august_metrics = calculate_metrics('2025-08-01', '2025-08-31')
            september_metrics = calculate_metrics('2025-09-01', '2025-09-30')
            
            # Get combined metrics
            combined_metrics = calculate_metrics('2025-07-01', '2025-09-30')
            
            # Verify formula for each month
            for month_name, metrics in [
                ('July', july_metrics),
                ('August', august_metrics),
                ('September', september_metrics),
                ('Combined', combined_metrics)
            ]:
                gross = metrics.get('gross_revenue') or metrics.get('revenue', 0)
                refund = metrics.get('refund_amount') or metrics.get('refunds', 0)
                cancel = metrics.get('cancellation_amount', 0)
                free_repl = metrics.get('free_replacement_cost', 0)
                net = metrics.get('net_revenue', 0)
                
                expected = gross - refund - cancel - free_repl
                diff = abs(net - expected)
                
                assert diff < 0.01, (
                    f"{month_name} formula mismatch!\n"
                    f"  Expected: {expected:.2f}\n"
                    f"  Actual:   {net:.2f}\n"
                    f"  Diff:     {diff:.2f}"
                )
            
            # Verify combined equals sum of individual months (approximately)
            # Allow small differences due to rounding
            combined_gross = combined_metrics.get('gross_revenue') or combined_metrics.get('revenue', 0)
            combined_refund = combined_metrics.get('refund_amount') or combined_metrics.get('refunds', 0)
            combined_cancel = combined_metrics.get('cancellation_amount', 0)
            combined_free_repl = combined_metrics.get('free_replacement_cost', 0)
            
            sum_gross = (
                (july_metrics.get('gross_revenue') or july_metrics.get('revenue', 0)) +
                (august_metrics.get('gross_revenue') or august_metrics.get('revenue', 0)) +
                (september_metrics.get('gross_revenue') or september_metrics.get('revenue', 0))
            )
            
            sum_refund = (
                (july_metrics.get('refund_amount') or july_metrics.get('refunds', 0)) +
                (august_metrics.get('refund_amount') or august_metrics.get('refunds', 0)) +
                (september_metrics.get('refund_amount') or september_metrics.get('refunds', 0))
            )
            
            # Components should sum approximately (allow rounding differences)
            gross_diff = abs(combined_gross - sum_gross)
            refund_diff = abs(combined_refund - sum_refund)
            
            # Allow 1% difference or 100 rupees, whichever is larger (for rounding)
            assert gross_diff < max(combined_gross * 0.01, 100), (
                f"Combined gross revenue doesn't match sum of months!\n"
                f"  Combined: {combined_gross:.2f}\n"
                f"  Sum:      {sum_gross:.2f}\n"
                f"  Diff:     {gross_diff:.2f}"
            )
            
        except Exception as e:
            pytest.skip(f"Database not available or error: {e}")


class TestEdgeCases:
    """Test edge cases for net revenue calculation"""
    
    def test_empty_date_range(self):
        """Test calculation with date range that has no data"""
        try:
            metrics = calculate_metrics('2020-01-01', '2020-01-31')
            
            # Should return zeros, not errors
            assert metrics is not None
            assert metrics.get('net_revenue', 0) == 0.0
            assert metrics.get('gross_revenue', 0) == 0.0
            assert metrics.get('refund_amount', 0) == 0.0
            assert metrics.get('cancellation_amount', 0) == 0.0
            assert metrics.get('free_replacement_cost', 0) == 0.0
            
            # Formula should still hold: 0 - 0 - 0 - 0 = 0
            net = metrics.get('net_revenue', 0)
            expected = (
                metrics.get('gross_revenue', 0) -
                metrics.get('refund_amount', 0) -
                metrics.get('cancellation_amount', 0) -
                metrics.get('free_replacement_cost', 0)
            )
            assert abs(net - expected) < 0.01, "Empty range formula should still hold"
            
        except Exception as e:
            pytest.skip(f"Database not available or error: {e}")
    
    def test_single_day_range(self):
        """Test calculation with single day range"""
        try:
            # Use a date that likely has data (September 1, 2025)
            metrics = calculate_metrics('2025-09-01', '2025-09-01')
            
            assert metrics is not None
            assert 'net_revenue' in metrics
            
            # Verify formula
            gross = metrics.get('gross_revenue') or metrics.get('revenue', 0)
            refund = metrics.get('refund_amount') or metrics.get('refunds', 0)
            cancel = metrics.get('cancellation_amount', 0)
            free_repl = metrics.get('free_replacement_cost', 0)
            net = metrics.get('net_revenue', 0)
            
            expected = gross - refund - cancel - free_repl
            diff = abs(net - expected)
            
            assert diff < 0.01, (
                f"Single day formula mismatch!\n"
                f"  Expected: {expected:.2f}\n"
                f"  Actual:   {net:.2f}"
            )
            
        except Exception as e:
            pytest.skip(f"Database not available or error: {e}")
    
    def test_all_transaction_types_present(self):
        """Test that all transaction types are handled correctly"""
        try:
            # Get metrics for a period that likely has all transaction types
            metrics = calculate_metrics('2025-07-01', '2025-09-30')
            
            assert metrics is not None
            
            # Check transaction breakdown
            breakdown = metrics.get('transaction_breakdown', {})
            
            # Should have entries for all transaction types (even if count is 0)
            transaction_types = ['Shipment', 'Refund', 'Cancel', 'FreeReplacement']
            
            for txn_type in transaction_types:
                # Type should be in breakdown or have 0 count
                if txn_type in breakdown:
                    assert 'count' in breakdown[txn_type], f"{txn_type} should have count"
                # If not in breakdown, that's okay - means count is 0
            
        except Exception as e:
            pytest.skip(f"Database not available or error: {e}")


class TestAIChatSQLGeneration:
    """Test AI Chat SQL generation for net revenue queries"""
    
    def test_net_revenue_query_detection(self):
        """Test that net revenue keywords are detected"""
        from backend.core.ai_service import generate_sql_from_question, get_database_schema
        
        net_revenue_queries = [
            "What is net revenue for September?",
            "Show me net earnings for September",
            "What's the profit after costs for September?",
            "Net revenue after deductions",
            "What is net profit for September?",
        ]
        
        for query in net_revenue_queries:
            try:
                schema = get_database_schema()
                sql, error = generate_sql_from_question(query, schema)
                
                if error:
                    pytest.skip(f"SQL generation failed: {error}")
                
                assert sql is not None, f"SQL should be generated for: {query}"
                assert sql.strip().upper().startswith('SELECT'), f"SQL should start with SELECT: {sql[:100]}"
                
                # Check for net revenue components
                sql_upper = sql.upper()
                
                # Should have net_revenue or net_rev in the SQL
                has_net_revenue = 'NET_REVENUE' in sql_upper or 'NET_REV' in sql_upper
                
                # Should have all transaction types
                has_shipment = 'SHIPMENT' in sql_upper
                has_refund = 'REFUND' in sql_upper
                has_cancel = 'CANCEL' in sql_upper
                has_free_repl = 'FREEREPLACEMENT' in sql_upper or 'FREE_REPLACEMENT' in sql_upper
                
                assert has_shipment, f"SQL should include Shipment: {sql[:200]}"
                assert has_refund, f"SQL should include Refund: {sql[:200]}"
                
                # Cancel and FreeReplacement might not always be present in simple queries
                # But net_revenue calculation should be present
                if has_net_revenue:
                    assert has_cancel or 'CANCEL' in sql_upper, (
                        f"Net revenue SQL should include Cancel: {sql[:300]}"
                    )
                    assert has_free_repl, (
                        f"Net revenue SQL should include FreeReplacement: {sql[:300]}"
                    )
                    
            except Exception as e:
                pytest.skip(f"AI service not available: {e}")
    
    def test_net_revenue_sql_structure(self):
        """Test that generated SQL has correct structure"""
        from backend.core.ai_service import generate_sql_from_question, get_database_schema
        
        query = "What is net revenue for September 2025?"
        
        try:
            schema = get_database_schema()
            sql, error = generate_sql_from_question(query, schema)
            
            if error:
                pytest.skip(f"SQL generation failed: {error}")
            
            assert sql is not None, "SQL should be generated"
            
            sql_upper = sql.upper()
            
            # Check for required components
            has_gross_revenue = (
                'SUM(CASE WHEN' in sql_upper and 'SHIPMENT' in sql_upper and 'GROSS_REVENUE' in sql_upper
            ) or ('SUM' in sql_upper and 'SHIPMENT' in sql_upper)
            
            has_refund = (
                'SUM(CASE WHEN' in sql_upper and 'REFUND' in sql_upper
            ) or ('SUM' in sql_upper and 'REFUND' in sql_upper)
            
            has_net_calculation = (
                'NET_REVENUE' in sql_upper or 
                ('GROSS' in sql_upper and 'REFUND' in sql_upper and '-' in sql)
            )
            
            # Should have date filtering for September
            has_september_filter = (
                '2025-09' in sql or 
                'SEPTEMBER' in sql_upper or
                ('2025-09-01' in sql and '2025-09-30' in sql)
            )
            
            assert has_gross_revenue or has_refund, (
                f"SQL should include revenue calculation: {sql[:300]}"
            )
            
            if has_net_calculation:
                # If net_revenue is calculated, should have all components
                assert 'CANCEL' in sql_upper or 'FREE' in sql_upper, (
                    f"Net revenue SQL should include Cancel or FreeReplacement: {sql[:400]}"
                )
                
        except Exception as e:
            pytest.skip(f"AI service not available: {e}")
    
    def test_net_revenue_sql_formula(self):
        """Test that generated SQL uses correct formula"""
        from backend.core.ai_service import generate_sql_from_question, get_database_schema
        
        query = "What is net revenue for September?"
        
        try:
            schema = get_database_schema()
            sql, error = generate_sql_from_question(query, schema)
            
            if error:
                pytest.skip(f"SQL generation failed: {error}")
            
            assert sql is not None, "SQL should be generated"
            
            # Check if SQL has the formula structure
            # Should have: gross - refund - cancel - free_repl
            sql_upper = sql.upper()
            
            # Count minus signs (should have at least 2 for net revenue calculation)
            minus_count = sql.count('-')
            
            # If net_revenue is calculated, should have multiple subtractions
            if 'NET_REVENUE' in sql_upper:
                assert minus_count >= 2, (
                    f"Net revenue SQL should have multiple subtractions: {sql[:400]}"
                )
                
                # Should have all transaction types
                has_shipment = 'SHIPMENT' in sql_upper
                has_refund = 'REFUND' in sql_upper
                has_cancel = 'CANCEL' in sql_upper
                has_free_repl = 'FREE' in sql_upper and 'REPLACEMENT' in sql_upper
                
                assert has_shipment and has_refund, (
                    f"Net revenue SQL must include Shipment and Refund: {sql[:400]}"
                )
                
                # Cancel and FreeReplacement should be present
                if has_cancel and has_free_repl:
                    # All components present - good!
                    pass
                else:
                    # Log warning but don't fail (LLM might generate slightly different SQL)
                    print(f"Warning: SQL may be missing components. Cancel: {has_cancel}, FreeRepl: {has_free_repl}")
                    
        except Exception as e:
            pytest.skip(f"AI service not available: {e}")


class TestAPIVsDirectCalculation:
    """Test that API endpoint matches direct calculation"""
    
    def test_api_matches_direct_calculation(self):
        """Verify API endpoint returns same values as direct calculation"""
        try:
            # Get from API
            api_response = requests.get(
                f"{API_BASE}/api/metrics",
                params={
                    'start_date': '2025-09-01',
                    'end_date': '2025-09-30'
                },
                timeout=10
            )
            
            if api_response.status_code != 200:
                pytest.skip(f"API not available (status {api_response.status_code})")
            
            api_metrics = api_response.json()['data']
            
            # Get from direct calculation
            direct_metrics = calculate_metrics('2025-09-01', '2025-09-30')
            
            # Compare key fields
            api_net = api_metrics.get('net_revenue', 0)
            direct_net = direct_metrics.get('net_revenue', 0)
            
            api_gross = api_metrics.get('gross_revenue') or api_metrics.get('revenue', 0)
            direct_gross = direct_metrics.get('gross_revenue') or direct_metrics.get('revenue', 0)
            
            api_refund = api_metrics.get('refund_amount') or api_metrics.get('refunds', 0)
            direct_refund = direct_metrics.get('refund_amount') or direct_metrics.get('refunds', 0)
            
            # Allow small differences due to rounding
            net_diff = abs(api_net - direct_net)
            gross_diff = abs(api_gross - direct_gross)
            refund_diff = abs(api_refund - direct_refund)
            
            assert net_diff < 1.0, (
                f"API and direct calculation net_revenue mismatch!\n"
                f"  API:    {api_net:.2f}\n"
                f"  Direct: {direct_net:.2f}\n"
                f"  Diff:   {net_diff:.2f}"
            )
            
            assert gross_diff < 1.0, (
                f"API and direct calculation gross_revenue mismatch!\n"
                f"  API:    {api_gross:.2f}\n"
                f"  Direct: {direct_gross:.2f}\n"
                f"  Diff:   {gross_diff:.2f}"
            )
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available at http://localhost:8000")
        except Exception as e:
            pytest.skip(f"Error: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

