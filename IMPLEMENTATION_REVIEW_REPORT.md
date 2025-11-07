# Implementation Review Report
**Date:** Generated on review  
**Purpose:** Document current state of Movers & Decliners, Revenue by Region, and Region SKUs features

---

## PART A: BACKEND LOGIC REVIEW

### 1. `get_movers_decliners()` Function

**Location:** `backend/services/metrics_service.py` (lines 1486-1809)

#### Date Calculation Logic

**Current Period:**
- Input: `start_date` and `end_date` (e.g., "2025-07-01" to "2025-09-30")
- Calculated: `period_length = (current_end - current_start).days + 1`
- Example: Jul 1 to Sep 30 = 92 days

**Previous Period Calculation:**
```python
prev_start = current_start - timedelta(days=7 + period_length - 1)
prev_end = current_start - timedelta(days=7)
```

**Example:**
- Current: Jul 1 - Sep 30 (92 days)
- Previous: Jun 24 - Sep 23 (92 days)
- Logic: Same duration, starts 1 week before current start date

#### Revenue Calculation

**Current Period Query:**
- **If `revenue_calc` exists:**
  ```sql
  SUM(CASE WHEN transaction_type = 'Shipment' AND revenue_calc > 0 
      THEN revenue_calc ELSE 0 END)
  ```
- **If other revenue column:**
  ```sql
  SUM(CASE WHEN transaction_type = 'Shipment' 
      THEN ABS(revenue_col) ELSE 0 END)
  ```
- **Filters:** Only Shipment transactions, `HAVING current_revenue > 0`
- **Groups by:** SKU

**Previous Period Query:**
- Same logic as current period
- Uses previous date range (`prev_start_str` to `prev_end_str`)
- **No HAVING clause** (includes SKUs with 0 revenue for comparison)

#### Growth Percentage Formula

**Function:** `calculate_growth(row)` (lines 1714-1727)

**Three Cases:**

1. **Normal Case** (`prev_revenue > 0`):
   ```python
   wow_change = ((current_revenue - prev_revenue) / prev_revenue) * 100
   ```
   - Rounded to 1 decimal place
   - Can be positive or negative

2. **New Product** (`prev_revenue = 0` and `current_revenue > 0`):
   ```python
   wow_change = 1000.0  # Placeholder for infinite growth
   ```
   - Later capped at 999% for display
   - Treated as movers

3. **Zero Revenue** (`prev_revenue = 0` and `current_revenue = 0`):
   ```python
   wow_change = 0.0
   ```
   - Shouldn't happen due to HAVING clause

#### Filtering Logic

**Decliners:**
- Initial filter: `wow_change <= -30%` AND `wow_change < 1000` (exclude new products)
- Sorted: Ascending (most negative first)
- Limit: Top `limit` (default 10)
- **Fallback:** If < 5 results, show ANY negative growth (relaxes -30% threshold)

**Movers:**
- Initial filter: `wow_change >= 30%` (includes new products with 1000%)
- Sorted: Descending (highest growth first)
- Limit: Top `limit` (default 10)
- **Fallback:** If < 5 results, show ANY positive growth + new products

#### Edge Cases & Assumptions

1. **New Products:** Products with no previous revenue are assigned 1000% growth, then capped at 999%
2. **Missing Previous Data:** SKUs not in previous period get `prev_revenue = 0` (left merge)
3. **Date Column Type:** Handles VARCHAR (needs CAST) vs DATE types
4. **Revenue Column Priority:** Checks `revenue_calc` first, then `revenue_amount`, `Invoice Amount`, `revenue_in_inr`
5. **Transaction Type:** If `transaction_type` column missing, uses positive revenue amounts
6. **SKU Matching:** Converts SKUs to strings and trims whitespace for proper matching
7. **Growth Capping:** Decliners capped at -100% (can't go below), Movers capped at 999% (new products)

---

### 2. `get_revenue_by_region()` Function

**Location:** `backend/services/metrics_service.py` (lines 1269-1483)

#### Query Logic

**Column Detection:**
- Region columns: `['region', 'Region', 'city', 'City', 'location', 'Location']`
- Revenue columns: `['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']` (priority order)
- Transaction columns: `['Transaction Type', 'transaction_type']`
- Date columns: `['Invoice Date', 'invoice_date', 'order_date', 'Order Date']`

#### SQL Query Structure

**With Transaction Type Column:**

**If `revenue_calc`:**
```sql
SELECT 
    LOWER(TRIM(region_normalized)) as region,
    SUM(revenue) as revenue
FROM (
    SELECT 
        CASE 
            WHEN region_col IS NULL OR TRIM(region_col) = '' THEN NULL
            ELSE TRIM(region_col)
        END as region_normalized,
        CASE WHEN transaction_type = 'Shipment' AND revenue_calc > 0 
            THEN revenue_calc ELSE 0 END as revenue
    FROM sales
    WHERE 1=1 [date_filter]
)
WHERE revenue > 0 
  AND region_normalized IS NOT NULL 
  AND TRIM(region_normalized) != ''
GROUP BY LOWER(TRIM(region_normalized))
ORDER BY revenue DESC
LIMIT {limit}
```

**If other revenue column:**
```sql
-- Same structure but uses ABS(revenue_col) instead
CASE WHEN transaction_type = 'Shipment' 
    THEN ABS(revenue_col) ELSE 0 END
```

**Without Transaction Type Column:**
- Uses `revenue_col > 0` filter instead
- Same grouping and sorting logic

#### Columns Being Summed

- **Revenue:** Sum of shipment transaction amounts per region
- **Grouping:** By normalized region name (lowercase, trimmed)
- **Filtering:** Only Shipment transactions, revenue > 0, non-null regions

#### Sorting and Limiting

- **Sort:** `ORDER BY revenue DESC` (highest revenue first)
- **Limit:** Top `limit` regions (default: 10)
- **Post-processing:** Re-sorts after title-casing region names

#### Region Name Normalization

1. **SQL:** Converts to lowercase and trims: `LOWER(TRIM(region_normalized))`
2. **Python:** Converts to title case: `x[0].upper() + x[1:].lower()`
3. **Purpose:** Groups "mumbai", "Mumbai", "MUMBAI" as same region

---

### 3. `get_skus_by_region()` Function

**Location:** `backend/services/metrics_service.py` (lines 1812-1993)

#### Parameters Accepted

- `region` (required): Region/city name (case-insensitive)
- `start_date` (optional): Start date filter (YYYY-MM-DD)
- `end_date` (optional): End date filter (YYYY-MM-DD)
- `limit` (optional, default: 10): Number of SKUs to return

#### Query Structure

**Region Filter:**
```sql
LOWER(TRIM(region_col)) = LOWER(TRIM('{region}'))
```
- Case-insensitive matching
- Trims whitespace

**Main Query (with transaction_type):**
```sql
SELECT 
    sku_col as sku,
    COALESCE(MAX(asin_col), '') as asin,
    COALESCE(SUM(quantity_col), 0) as units,
    COALESCE(SUM(CASE WHEN transaction_type = 'Shipment' 
        AND revenue_calc > 0 THEN revenue_calc ELSE 0 END), 0) as revenue
FROM sales
WHERE LOWER(TRIM(region_col)) = LOWER(TRIM('{region}'))
  [date_filter]
GROUP BY sku_col
HAVING revenue > 0
ORDER BY units DESC
LIMIT {limit}
```

#### Key Logic Points

1. **ASIN Selection:** Uses `MAX(asin_col)` - takes first ASIN if multiple per SKU
2. **Units Calculation:** Sums `quantity` column, falls back to `COUNT(*)` if no quantity column
3. **Revenue:** Only Shipment transactions, positive amounts
4. **Sorting:** By `units DESC` (highest units sold first)
5. **Filtering:** `HAVING revenue > 0` (excludes zero-revenue SKUs)

#### Return Format

```python
DataFrame with columns: ['sku', 'asin', 'units', 'revenue']
```

---

## PART B: FRONTEND DISPLAY REVIEW

### 1. Fast Movers Table Rendering

**Location:** `frontend/src/pages/Dashboard.tsx` (lines 643-703)

#### Data Reception

**Source:** `moversDecliners.movers` from Zustand store

**Store Flow:**
1. `fetchMoversDecliners(start, end)` called in `useEffect`
2. Calls `moversDeclinersService.getMoversDecliners(start, end, 10)`
3. API: `GET /api/metrics/movers-decliners?start_date={start}&end_date={end}&limit=10`
4. Response: `{movers: [...], decliners: [...]}`
5. Stored in `useDataStore().moversDecliners`

#### Data Structure

```typescript
interface MoverDeclinerItem {
  sku: string;
  revenue: number;
  wow_change: number;  // Percentage (e.g., 35.5 for 35.5%)
}
```

#### Column Display

**Table Configuration:**
```typescript
<Table
  dataSource={moversDecliners.movers}
  columns={[
    {
      title: 'Product',
      dataIndex: 'sku',
      render: (sku: string) => <span>{sku}</span>
    },
    {
      title: 'Revenue',
      dataIndex: 'revenue',
      align: 'right',
      render: (revenue: number) => formatCurrency(revenue)
    },
    {
      title: 'WoW Change',
      dataIndex: 'wow_change',
      align: 'right',
      render: (change: number) => (
        <span style={{ color: '#10B981' }}>
          +{change.toFixed(1)}%
        </span>
      )
    }
  ]}
  pagination={false}
  size="small"
  rowKey="sku"
/>
```

#### Filtering/Sorting Logic

- **No client-side filtering** - uses backend results as-is
- **No client-side sorting** - backend already sorted
- **Display:** Shows "+" prefix for all movers (even if negative somehow)
- **Color:** Green (#10B981) for all values

#### Conditional Rendering

```typescript
{moversDecliners.movers.length > 0 ? (
  <Table ... />
) : (
  <div>No fast movers found</div>
)}
```

---

### 2. Decliners Table Rendering

**Location:** `frontend/src/pages/Dashboard.tsx` (lines 579-640)

#### Data Reception

**Source:** `moversDecliners.decliners` from Zustand store  
**Same flow as Movers** (same API endpoint, different array in response)

#### Column Display

**Table Configuration:**
```typescript
<Table
  dataSource={moversDecliners.decliners}
  columns={[
    {
      title: 'Product',
      dataIndex: 'sku',
      render: (sku: string) => <span>{sku}</span>
    },
    {
      title: 'Revenue',
      dataIndex: 'revenue',
      align: 'right',
      render: (revenue: number) => formatCurrency(revenue)
    },
    {
      title: 'WoW Change',
      dataIndex: 'wow_change',
      align: 'right',
      render: (change: number) => (
        <span style={{ color: '#DC2626' }}>
          {change.toFixed(1)}%  // No "+" prefix
        </span>
      )
    }
  ]}
  pagination={false}
  size="small"
  rowKey="sku"
/>
```

#### Filtering/Sorting Logic

- **No client-side filtering** - uses backend results as-is
- **No client-side sorting** - backend already sorted (ascending)
- **Display:** No "+" prefix (negative values show "-" automatically)
- **Color:** Red (#DC2626) for all values

---

### 3. Revenue by Region Chart

**Location:** `frontend/src/pages/Dashboard.tsx` (lines 435-530)

#### Data Source

**Source:** `regionRevenue` from Zustand store

**Store Flow:**
1. `fetchRegionRevenue(start, end)` called in `useEffect`
2. Calls `regionService.getRevenueByRegion(start, end, 10)`
3. API: `GET /api/metrics/revenue-by-region?start_date={start}&end_date={end}&limit=10`
4. Response: `{data: [{region: "...", revenue: ...}], count: 10}`
5. **Deduplication logic** in store (lines 234-253) - sums revenue for duplicate region names
6. Stored in `useDataStore().regionRevenue`

#### Data Structure

```typescript
interface RegionRevenue {
  region: string;  // e.g., "Mumbai", "Bengaluru"
  revenue: number; // e.g., 5414850.83
}
```

#### Chart Configuration

**Recharts BarChart:**
```typescript
<BarChart
  data={regionRevenue}  // Array of {region, revenue}
  layout="vertical"      // Horizontal bars
  margin={{ top: 20, right: 30, left: 100, bottom: 20 }}
>
  <XAxis
    type="number"
    tickFormatter={(value) => `₹${(value / 100000).toFixed(1)}L`}
    domain={[0, 'dataMax']}
  />
  <YAxis
    type="category"
    dataKey="region"  // Region names on Y-axis
    width={90}
  />
  <Bar
    dataKey="revenue"  // Revenue values for bar length
    onClick={handleRegionClick}
    cursor="pointer"
  >
    {regionRevenue.map((entry, index) => (
      <Cell
        key={`${entry.region}-${index}`}
        fill={selectedRegion === entry.region ? '#6366F1' : '#818CF8'}
        onClick={() => handleRegionClick({ region: entry.region })}
      />
    ))}
  </Bar>
</BarChart>
```

#### Click Handling

**Handler Function:**
```typescript
const handleRegionClick = async (data: { region: string }) => {
  const regionName = data.region;
  setSelectedRegion(regionName);
  setSelectedRegionName(regionName);
  setIsRegionSKUModalOpen(true);
  setRegionSKUsLoading(true);
  
  // Fetch SKUs for this region
  const response = await regionService.getSKUsByRegion(
    regionName,
    dateRange.start,
    dateRange.end,
    10
  );
  
  setRegionSKUs(response?.data || []);
  setRegionSKUsLoading(false);
};
```

**Visual Feedback:**
- Selected region bar: `#6366F1` (darker purple), opacity 1.0
- Other bars: `#818CF8` (lighter purple), opacity 0.8
- Cursor changes to pointer on hover

---

## PART C: DOCUMENTATION REPORT

### MOVERS & DECLINERS

#### Backend Logic

**Function:** `get_movers_decliners(start_date, end_date, limit=10)`

**Date Calculation:**
- Current period: User-provided dates
- Previous period: Same duration, starts 1 week before current start
- Formula: `prev_start = current_start - timedelta(days=7 + period_length - 1)`

**Revenue Calculation:**
- Current period: Sum of Shipment transactions per SKU, `HAVING revenue > 0`
- Previous period: Sum of Shipment transactions per SKU (no HAVING clause)
- Uses `revenue_calc` if available (transaction-aware), otherwise `ABS(revenue_col)`

**Growth Formula:**
- Normal: `((current - previous) / previous) * 100`
- New products: Assigned 1000% (later capped at 999%)
- Zero previous: Returns 0.0

**Filtering:**
- Decliners: `wow_change <= -30%` (excludes new products)
- Movers: `wow_change >= 30%` (includes new products)
- Fallback: If < 5 results, relaxes threshold to show any negative/positive growth

#### Data Flow

1. **Backend:** `get_movers_decliners()` → Returns `{movers: [...], decliners: [...]}`
2. **API:** `GET /api/metrics/movers-decliners` → Returns same structure
3. **Frontend Store:** `fetchMoversDecliners()` → Stores in `moversDecliners` state
4. **Component:** Reads `moversDecliners.movers` and `moversDecliners.decliners`

#### Display Logic

- **Fast Movers:** Green table (#10B981), shows "+" prefix, sorted descending
- **Decliners:** Red table (#DC2626), no "+" prefix, sorted ascending
- **Empty State:** Shows "No fast movers found" or "No decliners found"
- **No client-side filtering/sorting** - uses backend results directly

#### Problems Identified

1. **New Product Display:** Shows "999%" instead of "New" label
2. **Fallback Logic:** May show products with <30% growth if insufficient results
3. **Date Range Assumption:** Assumes 1 week gap is appropriate for all date ranges
4. **No Error Handling:** If API fails, shows empty state without error message
5. **Growth Calculation Edge Case:** Products going from 0 to 0 revenue get 0% (should be excluded)

---

### REVENUE BY REGION

#### Backend Logic

**Function:** `get_revenue_by_region(start_date, end_date, limit=10)`

**Query Logic:**
- Groups by normalized region name (lowercase, trimmed)
- Sums revenue from Shipment transactions only
- Filters: `revenue > 0`, non-null regions
- Sorts: `ORDER BY revenue DESC`
- Limits: Top `limit` regions

**Region Normalization:**
- SQL: `LOWER(TRIM(region_col))` for grouping
- Python: Title case conversion for display

**Revenue Column Priority:**
1. `revenue_calc` (transaction-aware)
2. `revenue_amount`
3. `Invoice Amount`
4. `revenue_in_inr`

#### Data Flow

1. **Backend:** `get_revenue_by_region()` → Returns DataFrame `[region, revenue]`
2. **API:** `GET /api/metrics/revenue-by-region` → Returns `{data: [...], count: N}`
3. **Frontend Store:** `fetchRegionRevenue()` → Deduplicates if needed → Stores in `regionRevenue`
4. **Component:** Reads `regionRevenue` array directly

**Deduplication Logic (Store):**
- Checks for duplicate region names
- If found, sums revenue for duplicates
- Re-sorts by revenue descending

#### Display Logic

- **Chart Type:** Vertical BarChart (horizontal bars)
- **X-Axis:** Revenue (formatted as ₹X.XL for lakhs)
- **Y-Axis:** Region names
- **Colors:** Purple gradient (#818CF8 default, #6366F1 selected)
- **Click Handler:** Opens modal with region SKUs
- **Selected Indicator:** Shows selected region name and revenue below chart

#### Problems Identified

1. **Click Handler Issue:** Bar `onClick` receives data object, but structure may vary - uses both Bar onClick and Cell onClick
2. **Region Name Matching:** Case-insensitive matching in backend, but frontend uses exact match for selection
3. **Empty State:** Shows "No region data available" but doesn't distinguish between loading and no data
4. **Deduplication:** Shouldn't be needed if backend query is correct - indicates potential backend issue
5. **Date Filter:** Optional in backend, but frontend always passes dates - may filter out data if dates don't match

---

### REGION SKUs MODAL

#### Status: ✅ IMPLEMENTED

**Location:** `frontend/src/pages/Dashboard.tsx` (lines 787-866)

#### Implementation Details

**Modal Trigger:**
- Opens when region bar is clicked
- Sets `isRegionSKUModalOpen = true`
- Sets `selectedRegionName` for title

**Data Fetching:**
- Calls `regionService.getSKUsByRegion(region, start, end, 10)`
- API: `GET /api/metrics/region-skus?region={name}&start_date={start}&end_date={end}&limit=10`
- Shows loading skeleton while fetching
- Stores results in `regionSKUs` state

**Table Display:**
- Columns: SKU, ASIN, Units Sold, Revenue
- Sorted: By units DESC (backend)
- Pagination: None (shows top 10)
- Empty State: "No SKU data available for {region}"

**Close Handler:**
- Closes modal
- Resets `selectedRegion` to null

#### Problems Identified

1. **ASIN Display:** Shows "N/A" if ASIN is empty string - may be confusing
2. **Error Handling:** No error message if API call fails - just shows empty state
3. **Loading State:** Only shows skeleton, no error state
4. **Region Name Matching:** Uses exact region name from chart - may fail if backend returns different casing
5. **No Refresh:** If date range changes, modal doesn't refresh data (stale data shown)

---

## SUMMARY OF ISSUES

### Critical Issues

1. **Movers & Decliners:** New products show "999%" instead of "New" label
2. **Revenue by Region:** Potential duplicate regions if backend normalization fails
3. **Region SKUs Modal:** No error handling for failed API calls

### Medium Issues

1. **Date Range Logic:** 1-week gap assumption may not be appropriate for all periods
2. **Fallback Thresholds:** Relaxed thresholds may show irrelevant data
3. **Region Name Matching:** Case sensitivity differences between backend and frontend

### Minor Issues

1. **Empty States:** Generic messages don't distinguish between loading, error, and no data
2. **Visual Feedback:** Selected region indicator persists after modal closes
3. **ASIN Display:** "N/A" for empty ASINs may be confusing

---

## RECOMMENDATIONS

1. **Add error handling** to all API calls with user-friendly messages
2. **Improve new product display** - show "New" instead of "999%"
3. **Fix region name matching** - ensure consistent casing between backend and frontend
4. **Add loading states** for all async operations
5. **Review date range logic** - consider making gap configurable
6. **Add data validation** - verify data structure before rendering

---

**Report Generated:** Implementation Review  
**Files Analyzed:** 
- `backend/services/metrics_service.py`
- `backend/api/routes/metrics.py`
- `frontend/src/pages/Dashboard.tsx`
- `frontend/src/store/dataStore.ts`
- `frontend/src/services/api.ts`


