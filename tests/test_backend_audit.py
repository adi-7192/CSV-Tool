"""
Backend audit test - Comprehensive verification against legacy Streamlit

Run this before Phase 2 to ensure 100% accuracy

Usage:
    python tests/test_backend_audit.py
"""
import requests
import json
import sys
from typing import Dict, Any

API_BASE = "http://localhost:8000"


def print_section(title: str, width: int = 80):
    """Print formatted section header"""
    print("\n" + "=" * width)
    print(title)
    print("=" * width + "\n")


def print_test_result(test_name: str, status: str, details: Dict[str, Any] = None):
    """Print formatted test result"""
    status_icon = "✅" if status == 'PASS' else ("⚠️" if status == 'WARNING' else "❌")
    print(f"{status_icon} {test_name}: {status}")
    
    if details:
        if 'metrics' in details:
            for key, value in details['metrics'].items():
                if isinstance(value, float):
                    print(f"   {key}: {value:,.2f}")
                elif isinstance(value, int):
                    print(f"   {key}: {value:,}")
                else:
                    print(f"   {key}: {value}")
        
        if 'breakdown' in details and isinstance(details['breakdown'], list):
            print(f"   Breakdown items: {len(details['breakdown'])}")
            if len(details['breakdown']) <= 5:
                for item in details['breakdown']:
                    print(f"     - {item}")
        
        if details.get('warning'):
            print(f"   ⚠️  {details['warning']}")
        
        if details.get('validation'):
            print(f"   📋 {details['validation']}")
        
        if details.get('error'):
            print(f"   ❌ Error: {details['error']}")


def test_full_audit(start_date: str = None, end_date: str = None):
    """Run complete backend audit"""
    print_section("COMPREHENSIVE BACKEND AUDIT")
    
    try:
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        response = requests.get(f"{API_BASE}/api/verification/full-audit", params=params)
        response.raise_for_status()
        audit = response.json()
        
        if 'error' in audit:
            print(f"❌ Error: {audit['error']}")
            if 'message' in audit:
                print(f"   {audit['message']}")
            return False
        
        print(f"📅 Audit Timestamp: {audit['audit_timestamp']}")
        print(f"📊 Date Range: {audit['date_range']['start']} to {audit['date_range']['end']}\n")
        
        print("TEST RESULTS:")
        print("-" * 80)
        
        for i, test in enumerate(audit['tests'], 1):
            print(f"\n{i}. ", end="")
            print_test_result(test['test_name'], test.get('status', 'UNKNOWN'), test)
        
        print_section("SUMMARY")
        summary = audit['summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        
        # Save full report
        with open('backend_audit_report.json', 'w') as f:
            json.dump(audit, f, indent=2)
        print(f"\n📄 Full report saved to: backend_audit_report.json")
        
        return summary['failed'] == 0
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the backend running on http://localhost:8000?")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error running audit: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reconciliation():
    """Test reconciliation with CSV manual calculations"""
    print_section("CSV RECONCILIATION TEST")
    
    try:
        # July reconciliation (using your Excel values)
        print("Testing July 2025...")
        print("Expected values from CSV:")
        print("  Refunds: ₹332,996.41")
        
        response = requests.post(
            f"{API_BASE}/api/verification/reconcile-with-csv",
            params={
                "month": 7,
                "year": 2025,
                "expected_refunds": 332996.41,  # Your Excel value
            }
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"\n📊 Status: {result['status']}")
        print(f"\n📈 Database Metrics:")
        db_metrics = result['database_metrics']
        print(f"  Revenue: ₹{db_metrics.get('revenue', 0):,.2f}")
        print(f"  Refunds: ₹{db_metrics.get('refunds', 0):,.2f}")
        print(f"  Shipping Loss: ₹{db_metrics.get('shipping_loss', 0):,.2f}")
        print(f"  Net Revenue: ₹{db_metrics.get('net_revenue', 0):,.2f}")
        print(f"  Orders: {db_metrics.get('orders', 0):,}")
        
        if result.get('discrepancies'):
            print(f"\n📋 DISCREPANCIES:")
            all_passed = True
            for disc in result['discrepancies']:
                status_icon = "✅" if disc.get('status') == 'PASS' else "❌"
                print(f"  {status_icon} {disc['metric']}:")
                print(f"    Expected: ₹{disc['expected']:,.2f}")
                print(f"    Actual:   ₹{disc['actual']:,.2f}")
                print(f"    Diff:     ₹{disc['difference']:,.2f} ({disc['pct_difference']:.2f}%)")
                if disc.get('status') != 'PASS':
                    all_passed = False
        else:
            print("\n✅ No discrepancies to check")
            all_passed = True
        
        if result.get('diagnosis'):
            print(f"\n🔍 Diagnosis:")
            print(f"   {result['diagnosis']}")
        
        return result['status'] == 'PASS' and all_passed
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the backend running?")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error running reconciliation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_metrics(start_date: str = None, end_date: str = None):
    """Quick comparison endpoint test"""
    print_section("QUICK METRICS COMPARISON")
    
    try:
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        
        response = requests.get(f"{API_BASE}/api/verification/compare-metrics", params=params)
        response.raise_for_status()
        comparison = response.json()
        
        print(f"📅 Date Range: {comparison['date_range']['start']} to {comparison['date_range']['end']}\n")
        print("📊 Key Metrics:")
        for key, value in comparison['key_metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: ₹{value:,.2f}")
            else:
                print(f"  {key}: {value:,}")
        
        print(f"\n🔍 Formula Verification:")
        formula = comparison['formula_verification']
        print(f"  Revenue: ₹{formula['revenue']:,.2f}")
        print(f"  - Refunds: ₹{formula['minus_refunds']:,.2f}")
        print(f"  - Shipping: ₹{formula['minus_shipping']:,.2f}")
        print(f"  - Free Replacement: ₹{formula['minus_free_replacement']:,.2f}")
        print(f"  = Calculated Net: ₹{formula['calculated_net']:,.2f}")
        print(f"  Actual Net: ₹{formula['actual_net']:,.2f}")
        
        if formula['matches']:
            print(f"  ✅ Formula matches!")
        else:
            print(f"  ❌ Formula doesn't match!")
            print(f"     Difference: ₹{abs(formula['calculated_net'] - formula['actual_net']):,.2f}")
        
        print(f"\n📋 Transaction Breakdown:")
        for txn_type, details in comparison['transaction_breakdown'].items():
            print(f"  {txn_type}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, float):
                        print(f"    {key}: ₹{value:,.2f}")
                    elif isinstance(value, int):
                        print(f"    {key}: {value:,}")
                    else:
                        print(f"    {key}: {value}")
        
        return True
    
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the backend running?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_data_quality():
    """Test data quality endpoint"""
    print_section("DATA QUALITY CHECK")
    
    try:
        response = requests.get(f"{API_BASE}/api/health/data-quality")
        response.raise_for_status()
        quality = response.json()
        
        if quality.get('status') == 'no_data':
            print("⚠️  No data available")
            return True
        
        print(f"📊 Status: {quality.get('status', 'unknown')}")
        
        # Duplicates
        duplicates = quality.get('duplicates', {})
        if duplicates.get('found'):
            print(f"\n❌ Duplicates Found: {duplicates.get('count', 0)} duplicate groups")
        else:
            print(f"\n✅ No duplicates found")
        
        # Multi-source data
        multi_source = quality.get('multi_source_data', {})
        if multi_source.get('overlap_detected'):
            print(f"\n⚠️  Overlapping Source Files:")
            for overlap in multi_source.get('overlapping_sources', []):
                print(f"  - {overlap.get('source1')} <-> {overlap.get('source2')}")
        else:
            print(f"\n✅ No overlapping date ranges detected")
        
        # Source files
        source_files = quality.get('source_files', [])
        if source_files:
            print(f"\n📁 Source Files ({len(source_files)}):")
            for file_info in source_files:
                print(f"  - {file_info.get('source_file')}: {file_info.get('row_count', 0):,} rows")
        
        # Summary
        summary = quality.get('summary', {})
        print(f"\n📋 Summary:")
        print(f"  Warnings: {summary.get('total_warnings', 0)}")
        print(f"  Errors: {summary.get('total_errors', 0)}")
        print(f"  Has Duplicates: {summary.get('has_duplicates', False)}")
        print(f"  Has Overlaps: {summary.get('has_overlaps', False)}")
        
        return quality.get('status') != 'issues_found'
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print_section("BACKEND AUDIT TEST SUITE")
    print("This test suite verifies FastAPI backend matches Legacy Streamlit calculations")
    print("Make sure the backend is running on http://localhost:8000\n")
    
    # Run all tests
    results = {}
    
    # Test 1: Data Quality (prerequisite)
    results['data_quality'] = test_data_quality()
    
    # Test 2: Full Audit
    results['full_audit'] = test_full_audit()
    
    # Test 3: Reconciliation (July 2025 specific)
    results['reconciliation'] = test_reconciliation()
    
    # Test 4: Quick Comparison
    results['compare_metrics'] = test_compare_metrics()
    
    # Final Summary
    print_section("FINAL SUMMARY")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Backend ready for Phase 2!\n")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Review audit report and fix issues before Phase 2!\n")
        print("📄 Check backend_audit_report.json for detailed results")
        sys.exit(1)





