"""
Data Integrity Verification Script

Verifies data integrity after CSV re-upload:
1. Count records per source file
2. Check for duplicates
3. Verify Order ID mapping
4. Check revenue calculations
5. Compare with expected values
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_connection, execute_query, table_exists
import pandas as pd
from typing import Dict, Any

def verify_data_integrity() -> Dict[str, Any]:
    """
    Comprehensive data integrity verification
    
    Returns:
        dict: Verification results with issues and recommendations
    """
    results = {
        'table_exists': False,
        'total_records': 0,
        'records_by_file': {},
        'duplicate_check': {},
        'order_id_check': {},
        'revenue_check': {},
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    if not table_exists('sales'):
        results['issues'].append("❌ Table 'sales' does not exist. Database may be empty.")
        return results
    
    results['table_exists'] = True
    
    try:
        conn = get_connection()
        
        # 1. Count total records
        total_query = "SELECT COUNT(*) as total FROM sales"
        total_df = execute_query(total_query)
        results['total_records'] = int(total_df['total'].iloc[0]) if not total_df.empty else 0
        
        print(f"\n{'='*80}")
        print("DATA INTEGRITY VERIFICATION")
        print(f"{'='*80}\n")
        print(f"✅ Table 'sales' exists")
        print(f"📊 Total records: {results['total_records']:,}\n")
        
        # 2. Count records per source file
        print(f"{'='*80}")
        print("1. RECORDS PER SOURCE FILE")
        print(f"{'='*80}\n")
        
        file_count_query = """
        SELECT 
            source_file,
            COUNT(*) as record_count,
            COUNT(DISTINCT order_id) as unique_order_ids,
            MIN(loaded_at) as first_upload,
            MAX(loaded_at) as last_upload
        FROM sales
        GROUP BY source_file
        ORDER BY source_file
        """
        file_counts_df = execute_query(file_count_query)
        
        if not file_counts_df.empty:
            for _, row in file_counts_df.iterrows():
                file_name = row['source_file']
                count = int(row['record_count'])
                unique_ids = int(row['unique_order_ids'])
                first_upload = row['first_upload']
                last_upload = row['last_upload']
                
                results['records_by_file'][file_name] = {
                    'total_records': count,
                    'unique_order_ids': unique_ids,
                    'first_upload': str(first_upload),
                    'last_upload': str(last_upload)
                }
                
                print(f"📄 {file_name}:")
                print(f"   Total records: {count:,}")
                print(f"   Unique order IDs: {unique_ids:,}")
                print(f"   First upload: {first_upload}")
                print(f"   Last upload: {last_upload}")
                
                if count != unique_ids:
                    results['warnings'].append(
                        f"⚠️  {file_name}: {count - unique_ids:,} duplicate order IDs found"
                    )
                    print(f"   ⚠️  WARNING: {count - unique_ids:,} duplicate order IDs!")
                print()
        else:
            results['warnings'].append("⚠️  No source_file information found")
            print("⚠️  No source_file information found\n")
        
        # 3. Check for duplicates (by order_id + transaction_type)
        print(f"{'='*80}")
        print("2. DUPLICATE CHECK (order_id + transaction_type)")
        print(f"{'='*80}\n")
        
        duplicate_check_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT order_id) as unique_order_ids,
            COUNT(DISTINCT CONCAT(order_id, '|', transaction_type)) as unique_business_keys
        FROM sales
        """
        dup_check_df = execute_query(duplicate_check_query)
        
        if not dup_check_df.empty:
            total = int(dup_check_df['total_records'].iloc[0])
            unique_order_ids = int(dup_check_df['unique_order_ids'].iloc[0])
            unique_business_keys = int(dup_check_df['unique_business_keys'].iloc[0])
            
            results['duplicate_check'] = {
                'total_records': total,
                'unique_order_ids': unique_order_ids,
                'unique_business_keys': unique_business_keys,
                'duplicate_order_ids': total - unique_order_ids,
                'duplicate_business_keys': total - unique_business_keys
            }
            
            print(f"Total records: {total:,}")
            print(f"Unique order IDs: {unique_order_ids:,}")
            print(f"Unique business keys (order_id + transaction_type): {unique_business_keys:,}")
            print()
            
            if total > unique_order_ids:
                dup_count = total - unique_order_ids
                results['issues'].append(
                    f"❌ DUPLICATES FOUND: {dup_count:,} records have duplicate order_ids"
                )
                print(f"❌ ISSUE: {dup_count:,} records have duplicate order_ids\n")
                
                # Show sample duplicates
                sample_dup_query = """
                SELECT 
                    order_id,
                    transaction_type,
                    COUNT(*) as duplicate_count,
                    STRING_AGG(DISTINCT source_file, ', ') as source_files
                FROM sales
                GROUP BY order_id, transaction_type
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
                LIMIT 10
                """
                sample_dup_df = execute_query(sample_dup_query)
                if not sample_dup_df.empty:
                    print("Sample duplicate records:")
                    for _, row in sample_dup_df.iterrows():
                        print(f"   Order ID: {row['order_id']}, Type: {row['transaction_type']}, "
                              f"Count: {row['duplicate_count']}, Files: {row['source_files']}")
                    print()
            else:
                print("✅ No duplicate order IDs found\n")
            
            if total > unique_business_keys:
                dup_bk_count = total - unique_business_keys
                results['issues'].append(
                    f"❌ DUPLICATE BUSINESS KEYS: {dup_bk_count:,} records have duplicate (order_id + transaction_type)"
                )
                print(f"❌ ISSUE: {dup_bk_count:,} records have duplicate business keys\n")
            else:
                print("✅ No duplicate business keys found\n")
        
        # 4. Verify Order ID mapping
        print(f"{'='*80}")
        print("3. ORDER ID MAPPING VERIFICATION")
        print(f"{'='*80}\n")
        
        order_id_check_query = """
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN order_id IS NULL OR order_id = '' THEN 1 END) as empty_order_ids,
            COUNT(CASE WHEN order_id LIKE 'UNKNOWN_%' THEN 1 END) as synthetic_order_ids,
            COUNT(CASE WHEN order_id NOT LIKE 'UNKNOWN_%' AND order_id IS NOT NULL AND order_id != '' THEN 1 END) as real_order_ids
        FROM sales
        """
        order_id_df = execute_query(order_id_check_query)
        
        if not order_id_df.empty:
            total = int(order_id_df['total'].iloc[0])
            empty = int(order_id_df['empty_order_ids'].iloc[0])
            synthetic = int(order_id_df['synthetic_order_ids'].iloc[0])
            real = int(order_id_df['real_order_ids'].iloc[0])
            
            results['order_id_check'] = {
                'total_records': total,
                'empty_order_ids': empty,
                'synthetic_order_ids': synthetic,
                'real_order_ids': real
            }
            
            print(f"Total records: {total:,}")
            print(f"Real order IDs: {real:,}")
            print(f"Synthetic order IDs (UNKNOWN_*): {synthetic:,}")
            print(f"Empty order IDs: {empty:,}")
            print()
            
            if empty > 0:
                results['issues'].append(
                    f"❌ {empty:,} records have empty order_ids"
                )
                print(f"❌ ISSUE: {empty:,} records have empty order_ids\n")
            else:
                print("✅ No empty order IDs found\n")
            
            if synthetic > 0:
                results['warnings'].append(
                    f"⚠️  {synthetic:,} records have synthetic order IDs (UNKNOWN_*)"
                )
                print(f"⚠️  WARNING: {synthetic:,} records have synthetic order IDs\n")
                print("   This may indicate missing order IDs in source CSV files\n")
            else:
                print("✅ No synthetic order IDs found\n")
        
        # 5. Check revenue calculations
        print(f"{'='*80}")
        print("4. REVENUE CALCULATIONS")
        print(f"{'='*80}\n")
        
        revenue_query = """
        SELECT 
            transaction_type,
            COUNT(*) as transaction_count,
            SUM(ABS(revenue_amount)) as total_revenue,
            AVG(ABS(revenue_amount)) as avg_revenue,
            MIN(ABS(revenue_amount)) as min_revenue,
            MAX(ABS(revenue_amount)) as max_revenue
        FROM sales
        WHERE revenue_amount IS NOT NULL
        GROUP BY transaction_type
        ORDER BY transaction_type
        """
        revenue_df = execute_query(revenue_query)
        
        if not revenue_df.empty:
            revenue_by_type = {}
            for _, row in revenue_df.iterrows():
                txn_type = row['transaction_type']
                count = int(row['transaction_count'])
                total = float(row['total_revenue'])
                avg = float(row['avg_revenue'])
                
                revenue_by_type[txn_type] = {
                    'count': count,
                    'total_revenue': total,
                    'avg_revenue': avg
                }
                
                print(f"{txn_type}:")
                print(f"   Count: {count:,}")
                print(f"   Total Revenue: ₹{total:,.2f}")
                print(f"   Avg Revenue: ₹{avg:,.2f}")
                print()
            
            results['revenue_check'] = revenue_by_type
            
            # Calculate gross revenue (Shipment only)
            if 'Shipment' in revenue_by_type:
                gross_revenue = revenue_by_type['Shipment']['total_revenue']
                print(f"📊 Gross Revenue (Shipment only): ₹{gross_revenue:,.2f}\n")
                results['revenue_check']['gross_revenue'] = gross_revenue
            else:
                results['warnings'].append("⚠️  No 'Shipment' transactions found")
                print("⚠️  WARNING: No 'Shipment' transactions found\n")
        
        # 6. Summary and recommendations
        print(f"{'='*80}")
        print("5. SUMMARY & RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        if results['issues']:
            print("❌ ISSUES FOUND:\n")
            for issue in results['issues']:
                print(f"   {issue}")
            print()
        
        if results['warnings']:
            print("⚠️  WARNINGS:\n")
            for warning in results['warnings']:
                print(f"   {warning}")
            print()
        
        # Generate recommendations
        if results['issues']:
            if any('DUPLICATE' in issue for issue in results['issues']):
                results['recommendations'].append(
                    "1. CLEAR database: POST /api/upload/reset-database"
                )
                results['recommendations'].append(
                    "2. RE-UPLOAD CSV files (ensure no duplicates in source files)"
                )
                results['recommendations'].append(
                    "3. Verify counts match original after re-upload"
                )
            
            if any('empty order_ids' in issue.lower() for issue in results['issues']):
                results['recommendations'].append(
                    "4. Check source CSV files for missing 'Order Id' values"
                )
                results['recommendations'].append(
                    "5. Ensure 'Order Id' column is properly mapped during upload"
                )
        else:
            results['recommendations'].append("✅ Data integrity verified - no issues found")
        
        if results['recommendations']:
            print("💡 RECOMMENDATIONS:\n")
            for rec in results['recommendations']:
                print(f"   {rec}")
            print()
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        results['issues'].append(f"❌ Error during verification: {str(e)}")
        print(f"❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
    
    return results


if __name__ == "__main__":
    results = verify_data_integrity()
    
    # Return exit code based on issues
    if results['issues']:
        sys.exit(1)
    else:
        sys.exit(0)

