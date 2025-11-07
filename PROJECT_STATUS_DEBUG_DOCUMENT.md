# COMPREHENSIVE PROJECT STATUS DOCUMENT
## FastAPI + React Analytics Dashboard - Debug Guide

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX  
**Purpose:** Provide complete context for debugging critical bugs in the analytics dashboard

---

## 1. PROJECT OVERVIEW

### Project Name
**Analytics Dashboard** - Business Intelligence Platform for E-commerce Sales Data

### Purpose
A modern analytics dashboard that processes CSV sales data, calculates business KPIs, and provides AI-powered insights through natural language queries.

### Tech Stack

**Backend:**
- **Framework:** FastAPI (Python 3.9+)
- **Database:** DuckDB (in-process SQL OLAP database)
- **Data Processing:** Pandas
- **AI/LLM:** Ollama (local LLM for SQL generation)
- **Location:** `/Users/adi7192/Documents/Nisarg Project/backend/`

**Frontend:**
- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **State Management:** Zustand
- **UI Library:** Ant Design
- **Charts:** Recharts
- **Routing:** React Router DOM
- **Location:** `/Users/adi7192/Documents/Nisarg Project/frontend/`

### Current Milestone
**Milestone 1:** Core Dashboard Implementation
- ✅ Backend API endpoints operational
- ✅ Frontend dashboard UI complete
- ✅ Data ingestion pipeline functional
- ⚠️ **Two critical bugs preventing full functionality**

### Overall Progress
**~85% Complete** - Core functionality implemented, bug fixes needed

### Timeline
- **Start Date:** Q4 2024
- **Current Phase:** Bug Fixes & Polish
- **Target Completion:** Q1 2025

---

## 2. CURRENT BUGS (CRITICAL)

### Bug #1: Units Sold KPI Showing "0"
**Location:** Dashboard KPI Cards Section  
**Severity:** HIGH  
**Description:** The "Units Sold" metric card displays "0" instead of the actual total units sold count from the database.

**Expected Behavior:**
- Display total quantity/units sold from all Shipment transactions
- Format as a number (e.g., "5,169")

**Actual Behavior:**
- Always displays "0"
- No error messages in console

**Impact:**
- Users cannot see total units sold metric
- Dashboard appears incomplete

---

### Bug #2: Trend % Column Showing "0.00%"
**Location:** SKU Performance Table → "Trend" Column  
**Severity:** HIGH  
**Description:** The "Trend" column in the SKU Performance table displays "0.00%" for all SKUs instead of period-over-period growth percentage.

**Expected Behavior:**
- Calculate period-over-period growth (e.g., current period vs previous period)
- Display as percentage with up/down arrow
- Show green for positive growth, red for negative

**Actual Behavior:**
- All rows show "0.00%" with green arrow (default positive state)
- No actual trend calculation performed

**Impact:**
- Users cannot identify which SKUs are growing/declining
- Trend analysis feature is non-functional

---

## 3. BACKEND CODE

### A. Metrics Endpoint (`/api/metrics`)

**File Location:** `backend/api/routes/metrics.py`

**Endpoint Code:**
```python
@router.get("/")
async def get_metrics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    transaction_type: Optional[str] = Query(None, description="Filter by transaction type"),
    source_file: Optional[str] = Query(None, description="Filter by source file"),
):
    """
    Get core business metrics with transaction-aware calculations
    """
    try:
        # Default to last 30 days if no dates provided
        if not start_date or not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        metrics = calculate_metrics(start_date, end_date, transaction_type, source_file)
        
        return {
            "data": metrics,
            "period": {
                "start_date": start_date,
                "end_date": end_date,
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Service Function:** `backend/services/metrics_service.py` → `calculate_metrics()`

**Key Issue - Units Sold Calculation:**
The `calculate_metrics()` function **DOES NOT** calculate `units_sold` as a top-level metric. It only calculates:
- `gross_revenue`
- `refund_amount`
- `cancellation_amount`
- `free_replacement_cost`
- `net_revenue`
- `orders` (unique order_ids)
- `avg_order_value`
- etc.

**Missing:** `units_sold` field is not included in the return dictionary.

**Database Query Logic:**
```python
# Current query only gets revenue, transaction_type, order_id
all_data_sql = f"""
SELECT 
    transaction_type,
    revenue_amount,
    shipping_amount,
    sku,
    order_id,
    order_date
FROM sales
WHERE 1=1 {date_filter} {txn_filter} {source_filter}
"""
```

**Note:** The query does NOT select the `quantity` column, so units cannot be calculated.

**Expected Response JSON Structure:**
```json
{
  "data": {
    "gross_revenue": 8149751.62,
    "net_revenue": 6656011.40,
    "orders": 5169,
    "avg_order_value": 1576.66,
    "units_sold": 0,  // ❌ MISSING or ZERO
    "refund_amount": 1199000.00,
    "net_margin": 81.67,
    // ... other fields
  },
  "period": {
    "start_date": "2025-07-01",
    "end_date": "2025-09-30"
  }
}
```

---

### B. Top Products Endpoint (`/api/metrics/top-products`)

**File Location:** `backend/api/routes/metrics.py`

**Endpoint Code:**
```python
@router.get("/top-products")
async def get_top_products_endpoint(
    limit: int = Query(50, ge=1, le=100, description="Number of products"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    metric: str = Query('revenue', description="Sort metric (revenue)"),
):
    """
    Get top products by SKU with comprehensive metrics
    """
    try:
        products_df = get_top_products(limit, start_date, end_date, metric)
        
        # Convert DataFrame to list of dicts
        products_list = products_df.to_dict('records')
        
        # Ensure all numeric values are properly formatted
        for product in products_list:
            product['units_sold'] = int(product.get('units_sold', 0))
            product['revenue'] = float(product.get('revenue', 0))
            product['refund_ratio'] = float(product.get('refund_ratio', 0))
            product['rating'] = float(product.get('rating', 0))
            product['trend'] = float(product.get('trend', 0))  // ❌ HARDCODED TO 0
            product['asin'] = str(product.get('asin', ''))
            product['sku'] = str(product.get('sku', ''))
        
        return {
            "data": products_list,
            "count": len(products_list),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Service Function:** `backend/services/metrics_service.py` → `get_top_products()`

**Key Issue - Trend Calculation:**
```python
# Line 1063-1064 in metrics_service.py
# Add trend column (default to 0, can be calculated later if needed)
df['trend'] = 0.0
```

**Problem:** The `trend` field is **hardcoded to 0.0** with a comment saying "can be calculated later". No period-over-period comparison logic exists.

**Expected Response JSON Structure:**
```json
{
  "data": [
    {
      "sku": "SKU-001",
      "asin": "B001234567",
      "units_sold": 1245,
      "revenue": 1956000.50,
      "refund_ratio": 2.3,
      "rating": 4.5,
      "trend": 0  // ❌ Should be calculated as (current_revenue - prev_revenue) / prev_revenue * 100
    }
  ],
  "count": 50
}
```

**What Trend Should Calculate:**
1. Get current period revenue (e.g., July 1 - Sep 30, 2025)
2. Get previous period revenue (e.g., Apr 1 - Jun 30, 2025)
3. Calculate: `((current - previous) / previous) * 100`
4. Return percentage change

---

### C. Database Schema/Models

**Database:** DuckDB (file-based, no ORM)

**Table Name:** `sales`

**Schema Definition** (from `backend/services/upload_service.py`):
```python
STANDARD_SCHEMA = {
    'order_id': 'VARCHAR',          # Unique order/invoice identifier
    'order_date': 'DATE',           # Order placement date
    'revenue_amount': 'DOUBLE',     # Transaction amount (revenue or refund)
    'transaction_type': 'VARCHAR',  # Shipment, Refund, Cancel, FreeReplacement
    'sku': 'VARCHAR',               # Product SKU/ASIN
    'quantity': 'INTEGER',          # Order quantity ⭐ THIS COLUMN EXISTS
    'region': 'VARCHAR',            # Geographic region/city
    'shipping_amount': 'DOUBLE',   # Shipping cost
    
    # Data lineage columns (always added)
    'source_file': 'VARCHAR',       # Original CSV filename
    'ingestion_id': 'VARCHAR',      # Unique upload session ID
    'loaded_at': 'TIMESTAMP',       # When uploaded
    'updated_at': 'TIMESTAMP',      # When last modified
}
```

**SQL to Check Schema:**
```sql
DESCRIBE sales;
```

**Important Columns:**
- `quantity` - INTEGER column that contains units sold per transaction
- `transaction_type` - VARCHAR with values: 'Shipment', 'Refund', 'Cancel', 'FreeReplacement'
- `order_date` - DATE column for date filtering
- `revenue_amount` - DOUBLE for revenue calculations

**Note:** The `quantity` column exists in the schema but is **not being queried** in `calculate_metrics()`.

---

## 4. FRONTEND CODE

### A. KPI Cards Component

**File Location:** `frontend/src/components/MetricCard.tsx`

**Component Usage in Dashboard:**
```tsx
// File: frontend/src/pages/Dashboard.tsx, Line ~220
<Col xs={24} sm={12} lg={8}>
  <MetricCard
    title="Units Sold"
    value={0}  // ❌ HARDCODED TO 0
    format="number"
    trend={5.7}
    trendLabel="vs last month"
    loading={metricsLoading}
  />
</Col>
```

**Problem:** The value is hardcoded to `0` instead of using `metrics?.units_sold`.

**Current Code:**
```tsx
// Line 220-230 in Dashboard.tsx
<MetricCard
  title="Units Sold"
  value={0} // Placeholder, as 'units_sold' is not directly in metrics
  format="number"
  trend={5.7}
  trendLabel="vs last month"
  loading={metricsLoading}
/>
```

**Comment in Code:** `// Placeholder, as 'units_sold' is not directly in metrics`

**What Should Happen:**
```tsx
<MetricCard
  title="Units Sold"
  value={metrics?.units_sold || 0}  // ✅ Use API value
  format="number"
  trend={5.7}
  trendLabel="vs last month"
  loading={metricsLoading}
/>
```

**Data Flow:**
1. API returns: `{ data: { units_sold: 5169, ... } }`
2. Zustand store: `metrics.units_sold`
3. Component: `metrics?.units_sold`

**Current Issue:** API doesn't return `units_sold`, so frontend can't display it.

---

### B. SKU Performance Table Component

**File Location:** `frontend/src/components/PerformanceTable.tsx`

**Table Column Definition:**
```tsx
// Line 163-172 in PerformanceTable.tsx
{
  title: 'Trend',
  dataIndex: 'trend',
  key: 'trend',
  sorter: (a, b) => a.trend - b.trend,
  sortOrder: sortedInfo.columnKey === 'trend' ? sortedInfo.order : null,
  align: 'right',
  render: renderTrend,  // ✅ Renders correctly
  width: 100,
},
```

**Rendering Function:**
```tsx
// Line 70-91 in PerformanceTable.tsx
const renderTrend = (trend: number) => {
  const isPositive = trend >= 0;
  const color = isPositive ? '#10B981' : '#F43F5E';
  const icon = isPositive ? <ArrowUpOutlined /> : <ArrowDownOutlined />;

  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color, fontWeight: '600', justifyContent: 'flex-end' }}>
      {icon}
      {formatPercentage(Math.abs(trend))}  // ✅ Formats correctly
    </span>
  );
};
```

**Data Transformation:**
```tsx
// File: frontend/src/pages/Dashboard.tsx, Line 124-137
const transformSKUData = (skuData: any[] | undefined): PerformanceTableRow[] => {
  if (!skuData || skuData.length === 0) return [];
  return skuData.map((item, index) => ({
    id: item.id || String(index + 1),
    sku: item.sku || `SKU-${String(index + 1).padStart(3, '0')}`,
    asin: item.asin || '',
    unitsSold: item.unitsSold || item.units_sold || 0,
    revenue: item.revenue || 0,
    refundRatio: item.refundRatio || item.refund_ratio || 0,
    rating: item.rating || 4.0,
    trend: item.trend || 0,  // ✅ Receives trend from API
  }));
};
```

**Problem:** The API returns `trend: 0` for all items, so the frontend correctly displays "0.00%".

**Data Flow:**
1. API returns: `{ data: [{ trend: 0, ... }] }`
2. Transform function: `trend: item.trend || 0` → `trend: 0`
3. Table renders: `renderTrend(0)` → Shows "0.00%" with green arrow

**Root Cause:** Backend calculation issue, not frontend rendering issue.

---

## 5. DATA STRUCTURE

### CSV File Format

**Sample CSV Location:** `data/raw/JulyMonthly_raw.csv`

**Expected Columns (from upload service):**
- `Invoice Number` → mapped to `order_id`
- `Invoice Date` → mapped to `order_date`
- `Invoice Amount` → mapped to `revenue_amount`
- `Transaction Type` → mapped to `transaction_type`
- `Sku` → mapped to `sku`
- `Quantity` → mapped to `quantity` ⭐ **THIS COLUMN EXISTS**
- `Ship To City` → mapped to `region`
- `Shipping Amount` → mapped to `shipping_amount`

**Sample Data (First 5 Rows):**
```csv
Invoice Number,Invoice Date,Invoice Amount,Transaction Type,Sku,Quantity,Ship To City,Shipping Amount
INV-001,2025-07-01,1599.00,Shipment,SKU-001,2,Mumbai,50.00
INV-002,2025-07-01,899.00,Shipment,SKU-002,1,Delhi,40.00
INV-003,2025-07-02,2999.00,Refund,SKU-001,1,Bangalore,0.00
INV-004,2025-07-02,1299.00,Shipment,SKU-003,1,Chennai,45.00
INV-005,2025-07-03,599.00,Shipment,SKU-004,3,Pune,55.00
```

**Key Observations:**
- `Quantity` column exists and contains unit counts (1, 2, 3, etc.)
- `Transaction Type` distinguishes Shipment vs Refund
- Data is ingested correctly into database

**How Units Should Be Calculated:**
```sql
SELECT SUM(quantity) as units_sold
FROM sales
WHERE transaction_type = 'Shipment'
  AND order_date >= '2025-07-01'
  AND order_date <= '2025-09-30'
```

---

## 6. DATABASE STATE

### SQL Queries to Verify Data

**Check if Quantity Column Exists:**
```sql
DESCRIBE sales;
-- Should show: quantity INTEGER
```

**Check Sample Data:**
```sql
SELECT 
    order_id,
    order_date,
    transaction_type,
    quantity,
    revenue_amount,
    sku
FROM sales
LIMIT 10;
```

**Check Total Units Sold:**
```sql
SELECT 
    SUM(quantity) as total_units,
    COUNT(*) as total_transactions,
    COUNT(DISTINCT sku) as unique_skus
FROM sales
WHERE transaction_type = 'Shipment'
  AND order_date >= '2025-07-01'
  AND order_date <= '2025-09-30';
```

**Expected Result:**
```
total_units: ~5000-10000 (depends on data)
total_transactions: ~5000-10000
unique_skus: ~50-200
```

**If Query Returns NULL or 0:**
- Check if `quantity` column exists
- Check if data was ingested correctly
- Check if date filter is working

---

## 7. API RESPONSE SAMPLES

### A. `/api/metrics` Response

**Current Response (MISSING units_sold):**
```json
{
  "data": {
    "gross_revenue": 8149751.62,
    "revenue": 8149751.62,
    "refund_amount": 1199000.00,
    "refunds": 1199000.00,
    "cancellation_amount": 0.0,
    "free_replacement_cost": 297740.22,
    "shipping_loss": 0.0,
    "net_revenue": 6656011.40,
    "net_margin": 81.67,
    "refund_rate": 14.71,
    "orders": 5169,
    "avg_order_value": 1576.66,
    "success_rate": 0.0,
    "transaction_breakdown": {
      "Shipment": {
        "count": 5169,
        "revenue": 8149751.62,
        "orders": 5169
      },
      "Refund": {
        "count": 760,
        "refund_amount": 1199000.00,
        "shipping_loss": 0.0
      }
    }
  },
  "period": {
    "start_date": "2025-07-01",
    "end_date": "2025-09-30"
  }
}
```

**Missing Field:** `units_sold` ❌

**Expected Response (SHOULD INCLUDE):**
```json
{
  "data": {
    // ... all existing fields ...
    "units_sold": 5169,  // ✅ ADD THIS
    // ... rest of fields ...
  }
}
```

---

### B. `/api/metrics/top-products` Response

**Current Response (trend = 0):**
```json
{
  "data": [
    {
      "sku": "SKU-001",
      "asin": "B001234567",
      "units_sold": 1245,
      "revenue": 1956000.50,
      "refund_ratio": 2.3,
      "rating": 4.5,
      "trend": 0  // ❌ HARDCODED TO 0
    },
    {
      "sku": "SKU-002",
      "asin": "B002234567",
      "units_sold": 1089,
      "revenue": 1711234.75,
      "refund_ratio": 3.1,
      "rating": 4.3,
      "trend": 0  // ❌ HARDCODED TO 0
    }
  ],
  "count": 50
}
```

**Expected Response (SHOULD CALCULATE TREND):**
```json
{
  "data": [
    {
      "sku": "SKU-001",
      "asin": "B001234567",
      "units_sold": 1245,
      "revenue": 1956000.50,
      "refund_ratio": 2.3,
      "rating": 4.5,
      "trend": 12.5  // ✅ CALCULATED: ((current - previous) / previous) * 100
    },
    {
      "sku": "SKU-002",
      "asin": "B002234567",
      "units_sold": 1089,
      "revenue": 1711234.75,
      "refund_ratio": 3.1,
      "rating": 4.3,
      "trend": -5.2  // ✅ Negative trend = decline
    }
  ],
  "count": 50
}
```

---

## 8. NETWORK/BROWSER DEBUGGING

### Browser Console Errors

**Expected:** No JavaScript errors related to these bugs.

**Check Console:**
```javascript
// Open browser DevTools → Console
// Look for:
- TypeError: Cannot read property 'units_sold' of undefined
- Warning: Missing prop 'units_sold'
```

**Expected Result:** No errors (data is handled gracefully with defaults).

---

### Network Tab - API Calls

**Check Request:**
```
GET /api/metrics?start_date=2025-07-01&end_date=2025-09-30
Status: 200 OK
```

**Check Response Headers:**
```
Content-Type: application/json
```

**Check Response Body:**
- Verify `units_sold` field is missing
- Verify `trend` field is 0 for all items

**Check Request:**
```
GET /api/metrics/top-products?start_date=2025-07-01&end_date=2025-09-30&limit=50
Status: 200 OK
```

**Check Response Body:**
- Verify all `trend` values are `0`

---

## 9. HYPOTHESIS FOR EACH BUG

### Bug #1: Units Sold = 0

**Root Cause Hypothesis:**
1. ✅ **MOST LIKELY:** `calculate_metrics()` function does not query the `quantity` column
2. ✅ **CONFIRMED:** `units_sold` field is not included in the return dictionary
3. ✅ **CONFIRMED:** Frontend hardcodes value to 0 as a placeholder

**Evidence:**
- Database schema includes `quantity INTEGER` column
- `calculate_metrics()` SQL query does NOT select `quantity`
- Return dictionary does NOT include `units_sold`
- Frontend comment: `// Placeholder, as 'units_sold' is not directly in metrics`

**Fix Required:**
1. Modify `calculate_metrics()` to query `quantity` column
2. Sum `quantity` for all `transaction_type = 'Shipment'` records
3. Add `units_sold` to return dictionary
4. Update frontend to use `metrics?.units_sold` instead of hardcoded `0`

**Code Changes Needed:**
```python
# In metrics_service.py → calculate_metrics()
# Add to SQL query:
all_data_sql = f"""
SELECT 
    transaction_type,
    revenue_amount,
    shipping_amount,
    sku,
    order_id,
    order_date,
    quantity  -- ✅ ADD THIS
FROM sales
WHERE 1=1 {date_filter} {txn_filter} {source_filter}
"""

# Add to results:
results['units_sold'] = int(shipment_data['quantity'].sum()) if 'quantity' in shipment_data.columns else 0
```

---

### Bug #2: Trend % = 0

**Root Cause Hypothesis:**
1. ✅ **CONFIRMED:** `trend` field is hardcoded to `0.0` in `get_top_products()`
2. ✅ **CONFIRMED:** No period-over-period comparison logic exists
3. ✅ **CONFIRMED:** Comment says "can be calculated later if needed"

**Evidence:**
- Line 1063-1064 in `metrics_service.py`: `df['trend'] = 0.0`
- No previous period query logic
- No growth calculation formula

**Fix Required:**
1. Calculate previous period revenue for each SKU
2. Compare current period vs previous period
3. Calculate percentage change: `((current - previous) / previous) * 100`
4. Return trend value in response

**Code Changes Needed:**
```python
# In metrics_service.py → get_top_products()
# Replace hardcoded trend with calculation:

# Get previous period dates (e.g., if current is Jul-Sep, previous is Apr-Jun)
prev_end_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
prev_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=92)).strftime('%Y-%m-%d')

# Query previous period revenue by SKU
prev_revenue_sql = f"""
SELECT 
    "{sku_col}" as sku,
    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as prev_revenue
FROM sales
WHERE 1=1 {prev_date_filter}
GROUP BY "{sku_col}"
"""

prev_df = execute_query(prev_revenue_sql)

# Merge and calculate trend
df = df.merge(prev_df, on='sku', how='left')
df['prev_revenue'] = df['prev_revenue'].fillna(0)
df['trend'] = df.apply(
    lambda row: round(((row['revenue'] - row['prev_revenue']) / row['prev_revenue'] * 100) if row['prev_revenue'] > 0 else 0, 1),
    axis=1
)
```

---

## 10. FILE LOCATIONS

### Backend Files

**Main Application:**
- `backend/main.py` - FastAPI app initialization
- `backend/core/config.py` - Configuration settings
- `backend/core/database.py` - DuckDB connection management

**API Routes:**
- `backend/api/routes/metrics.py` - Metrics endpoints ⭐ **BUG FIX LOCATION**
- `backend/api/routes/upload.py` - CSV upload endpoints
- `backend/api/routes/chat.py` - AI chat endpoints
- `backend/api/routes/data_status.py` - Data status endpoint

**Services:**
- `backend/services/metrics_service.py` - KPI calculation logic ⭐ **BUG FIX LOCATION**
- `backend/services/upload_service.py` - CSV processing & schema standardization
- `backend/core/ai_service.py` - LLM SQL generation

**Database:**
- `sales_data.db` - DuckDB database file (location: project root)

---

### Frontend Files

**Main Application:**
- `frontend/src/App.tsx` - React app root with routing
- `frontend/src/main.tsx` - Entry point
- `frontend/src/index.css` - Global styles

**Pages:**
- `frontend/src/pages/Dashboard.tsx` - Main dashboard page ⭐ **BUG FIX LOCATION**

**Components:**
- `frontend/src/components/MetricCard.tsx` - KPI card component ⭐ **BUG FIX LOCATION**
- `frontend/src/components/PerformanceTable.tsx` - SKU performance table ⭐ **BUG FIX LOCATION**
- `frontend/src/components/TrendChart.tsx` - Chart component
- `frontend/src/components/InsightBanner.tsx` - Insight alerts
- `frontend/src/components/TopBar.tsx` - Top navigation bar

**State Management:**
- `frontend/src/store/dataStore.ts` - Zustand store for metrics data
- `frontend/src/store/uiStore.ts` - UI state
- `frontend/src/store/chatStore.ts` - Chat messages state

**Services:**
- `frontend/src/services/api.ts` - API client & service functions

**Utils:**
- `frontend/src/utils/formatters.ts` - Currency/number formatting

---

### Configuration Files

**Backend:**
- `backend/.env` - Environment variables (if exists)
- `backend/pyproject.toml` or `requirements.txt` - Python dependencies

**Frontend:**
- `frontend/package.json` - NPM dependencies
- `frontend/vite.config.ts` - Vite configuration
- `frontend/tsconfig.json` - TypeScript configuration
- `frontend/.env` - Environment variables (`VITE_API_BASE_URL`)

---

## SUMMARY

### Quick Fix Checklist

**Bug #1: Units Sold = 0**
- [ ] Modify `calculate_metrics()` SQL query to include `quantity` column
- [ ] Sum `quantity` for Shipment transactions
- [ ] Add `units_sold` to return dictionary
- [ ] Update Dashboard.tsx to use `metrics?.units_sold` instead of `0`

**Bug #2: Trend % = 0**
- [ ] Calculate previous period dates (3 months before current period)
- [ ] Query previous period revenue by SKU
- [ ] Calculate trend: `((current - previous) / previous) * 100`
- [ ] Replace hardcoded `df['trend'] = 0.0` with calculated values

### Testing Steps

1. **Backend Testing:**
   ```bash
   # Start backend
   cd backend
   uvicorn main:app --reload
   
   # Test API
   curl "http://localhost:8000/api/metrics?start_date=2025-07-01&end_date=2025-09-30"
   # Verify: response includes "units_sold": <non-zero number>
   
   curl "http://localhost:8000/api/metrics/top-products?start_date=2025-07-01&end_date=2025-09-30&limit=10"
   # Verify: response includes "trend": <non-zero number> for each SKU
   ```

2. **Frontend Testing:**
   ```bash
   # Start frontend
   cd frontend
   npm run dev
   
   # Open browser: http://localhost:5173
   # Check:
   # - Units Sold KPI shows actual number (not 0)
   # - SKU Performance table shows Trend % with real values (not 0.00%)
   ```

3. **Database Verification:**
   ```sql
   -- Check if quantity data exists
   SELECT SUM(quantity) FROM sales WHERE transaction_type = 'Shipment';
   
   -- Check if we have previous period data
   SELECT MIN(order_date), MAX(order_date) FROM sales;
   ```

---

## ADDITIONAL CONTEXT

### Data Ingestion Flow

1. CSV uploaded via `/api/upload/csv`
2. `upload_service.py` detects columns and maps to standardized schema
3. Data cleaned and transformed
4. Stored in DuckDB `sales` table with `quantity` column
5. Metrics calculated on-demand via API endpoints

### Known Limitations

- **Trend Calculation:** Requires historical data (previous period). If no previous period exists, trend should be `null` or `0` with a note.
- **Units Sold:** Currently only counts Shipment transactions. Refunds/Cancels don't affect units sold (by design).

### Related Issues

- None identified at this time

---

**END OF DOCUMENT**



