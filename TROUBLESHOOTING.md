# Troubleshooting Guide

## ❌ Issue: Dashboard Shows Zero Values / No Data

### Root Cause
The database table is **EMPTY**. Data was never stored or was cleared.

### Diagnostic Tool
Run this to check your database status:
```bash
./venv/bin/python diagnose.py
```

### Solution: Proper Data Upload Workflow

Follow these steps **in order**:

#### 1️⃣ **Delete Old Database** (if exists)
```bash
rm data.db
```
Or use the "Delete Database & Start Fresh" button on the dashboard.

#### 2️⃣ **Upload CSV File**
- Go to **"📤 Upload CSV"** tab in sidebar
- Click "Browse files" and select your CSV
- CSV must have these columns:
  - `Transaction` (or "Transaction Type")
  - `Invoice Amount` 
  - `Shipping Amount`
  - `Asin`
  - `Sku`
  - `Quantity`
  - `Order Id` (or "Invoice Number")
  - `Invoice Date` (or "Order Date")

#### 3️⃣ **Clean & Validate Data**
- Review the column mapping
- Click **"🧹 Clean & Validate Data"** button
- **CHECK CONSOLE OUTPUT** for cleaning debug messages
- You should see:
  ```
  ================================================================================
  🧹 TRANSACTION-AWARE DATA CLEANING - NO LEGITIMATE DATA DROPPED
  ================================================================================
  Starting with 1,250 rows
  ...
  📊 TRANSACTION SUMMARY:
     Shipment: 1,100 rows
     Refund: 120 rows
  ...
  ```

#### 4️⃣ **Store Data**
- Click **"💾 Store Data"** button
- **WAIT** for confirmation messages:
  - "Storing N rows in replace mode..."
  - "✅ Cleaned data has derived fields"
  - "Total revenue_calc in cleaned data: ₹X"
  - "✅ Verified: N rows now in database"
- **CHECK CONSOLE OUTPUT** for storage debug messages
- You should see:
  ```
  📊 DEBUG: First 5 rows in database with derived fields:
    Row 1: Transaction=Shipment, revenue_calc=1250.0, ...
  ```

#### 5️⃣ **View Dashboard**
- Go to **"📊 Business Dashboard"** tab
- Click **"🔄 Refresh Dashboard"** button
- You should see actual KPI values

---

## 🔍 Debug Checklist

If you still see zero values, check each step:

### ✅ Step 1: Database Exists
```bash
ls -lh data.db
```
Should show a file > 24KB

### ✅ Step 2: Table Has Data
```bash
./venv/bin/python diagnose.py
```
Should show: "Total rows: N" (where N > 0)

### ✅ Step 3: Derived Fields Exist
Look in console output for:
```
✅ Using derived fields (revenue_calc, shipping_loss_calc, units_sold_calc)
```

### ✅ Step 4: Date Range Matches
Console shows:
```
📅 Database date range: 2024-01-01 to 2024-12-31
📅 Requested date range: 2024-01-01 to 2024-12-31
```
These should overlap!

### ✅ Step 5: Transaction Types Present
Console shows:
```
1️⃣ ROWS PER TRANSACTION TYPE:
   Shipment: 1,100 rows
   Refund: 120 rows
```

---

## 🚨 Common Issues

### Issue: "Table is EMPTY"
**Cause**: You clicked "Clean & Validate" but forgot to click "Store Data"
**Solution**: Click the **"💾 Store Data"** button!

### Issue: "All revenue_calc values are zero"
**Cause**: Derived fields weren't created during cleaning
**Solution**: 
1. Delete database
2. Re-upload CSV
3. Check console for cleaning debug output

### Issue: "No data for date range"
**Cause**: Your CSV has dates from 2023, but filter shows 2024
**Solution**: The app now auto-detects date range from your data

### Issue: "Missing derived fields"
**Cause**: Old database schema
**Solution**: Delete `data.db` and re-upload data

---

## 📊 Expected Console Output

### During Cleaning:
```
================================================================================
🧹 TRANSACTION-AWARE DATA CLEANING - NO LEGITIMATE DATA DROPPED
================================================================================
Starting with 1,250 rows

✂️ Trimmed whitespace from string columns
🔤 Normalized Order Id, SKU, ASIN to uppercase
📅 Parsed Invoice Date to YYYY-MM-DD
💰 Cleaned Invoice Amount: 5 NaN values, 120 negative values (preserved)
🚚 Cleaned Shipping Amount
🔄 Normalized Transaction: ['shipment', 'Refund'] → ['Shipment', 'Refund']
📦 Cleaned Quantity
🔄 Removed 12 EXACT duplicates (all columns identical)

📊 Creating derived columns based on Transaction logic...
✅ Derived columns created: revenue_calc, shipping_loss_calc, units_sold_calc

================================================================================
📊 DEBUG SUMMARY AFTER CLEANING
================================================================================

1️⃣ ROWS PER TRANSACTION TYPE:
   Shipment: 1,100 rows
   Refund: 120 rows
   Cancel: 25 rows

2️⃣ SUM OF INVOICE AMOUNT PER TRANSACTION TYPE:
   Shipment:
      Invoice Amount (original): ₹125,450.00
      revenue_calc (derived):    ₹125,450.00
   Refund:
      Invoice Amount (original): ₹-8,500.00
      revenue_calc (derived):    ₹-8,500.00

3️⃣ REVENUE_CALC DISTRIBUTION:
   Positive revenue_calc: 1,100 rows
   Negative revenue_calc: 120 rows
   Zero revenue_calc:     25 rows

4️⃣ SAMPLE ROWS:
   Shipment example (row 0):
      Invoice Amount: 1250.0
      revenue_calc: 1250.0
      Shipping Amount: 50.0
      shipping_loss_calc: 0.0
      Quantity: 2
      units_sold_calc: 2
```

### During Storage:
```
📊 DEBUG: First 5 rows in database with derived fields:
  Row 1: Transaction=Shipment, revenue_calc=1250.0, shipping_loss_calc=0.0, units_sold_calc=2
  Row 2: Transaction=Shipment, revenue_calc=899.0, shipping_loss_calc=0.0, units_sold_calc=1
  Row 3: Transaction=Refund, revenue_calc=-450.0, shipping_loss_calc=30.0, units_sold_calc=0
```

### During KPI Calculation:
```
🔄 calculate_transaction_revenue called with 1,238 rows
Available columns: ['id', 'order_date', 'order_id', 'sku', 'asin', 'revenue_calc', 'shipping_loss_calc', 'units_sold_calc', ...]
✅ Using derived fields (revenue_calc, shipping_loss_calc, units_sold_calc)
  Gross Revenue: ₹125,450.00
  Refunds: ₹8,500.00
  Shipping Loss: ₹2,100.00
  Net Revenue: ₹114,850.00
  Units Sold: 1450
  Orders: 1100
```

---

## 🆘 Still Having Issues?

1. **Check the Streamlit terminal/console** - all debug output goes there
2. **Run the diagnostic tool**: `./venv/bin/python diagnose.py`
3. **Share the console output** with your developer
4. **Check the cleaned CSV** in `data/cleaned/` folder to verify data quality

---

## ✅ Success Indicators

You'll know it's working when you see:

1. ✅ Console shows cleaning debug output with transaction counts
2. ✅ Console shows storage debug output with derived fields
3. ✅ Console shows KPI calculation using derived fields
4. ✅ Dashboard displays non-zero values for:
   - Net Revenue
   - Gross Revenue
   - Orders
   - Units Sold
5. ✅ Transaction breakdown shows counts for each type

---

**Last Updated**: After implementing transaction-aware cleaning and storage pipeline

