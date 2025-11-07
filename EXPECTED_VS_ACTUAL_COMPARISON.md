# Expected vs Actual Behavior Comparison

**Date:** Analysis based on code review  
**Purpose:** Identify discrepancies between expected and actual behavior

---

## MOVERS & DECLINERS

| Aspect | Expected Behavior | Actual Behavior (To Be Filled) | Issue Identified | Root Cause (TBD) |
|--------|------------------|-------------------------------|------------------|------------------|
| **WoW Change %** | Realistic percentages:<br>- Positive: 30% to 200%<br>- Negative: -30% to -100%<br>- New products: "New" label | **REPORTED:** All showing 999%<br>**ACTUAL:** [Fill after running logs] | All movers showing 999% instead of actual growth | Backend assigns 1000% to new products, capped at 999% for display. Frontend shows 999% instead of "New" label. |
| **Decliners Shown** | 5-10 products with ≤-30% decline | **REPORTED:** "No decliners found"<br>**ACTUAL:** [Fill after running logs] | No decliners displayed | Possible causes:<br>1. No SKUs with ≤-30% decline<br>2. Fallback logic not working<br>3. Data not reaching frontend |
| **Movers Shown** | 5-10 products with ≥+30% growth | **REPORTED:** "No fast movers found"<br>**ACTUAL:** [Fill after running logs] | No movers displayed | Possible causes:<br>1. No SKUs with ≥+30% growth<br>2. All products are new (999%)<br>3. Data not reaching frontend |
| **Data Source - Current Period** | Sum of Shipment transactions per SKU<br>Date range: User-selected dates<br>Example: Jul 1 - Sep 30, 2025 | **EXPECTED:** Revenue per SKU for current period<br>**ACTUAL:** [Check backend logs] | TBD | Verify SQL query returns correct data |
| **Data Source - Previous Period** | Same duration, 1 week before current start<br>Example: Jun 24 - Sep 23, 2025<br>Sum of Shipment transactions per SKU | **EXPECTED:** Revenue per SKU for previous period<br>**ACTUAL:** [Check backend logs] | TBD | Verify date calculation and query |
| **Growth Calculation Formula** | `((current_revenue - prev_revenue) / prev_revenue) * 100`<br>Rounded to 1 decimal place | **EXPECTED:** Calculated percentage<br>**ACTUAL:** [Check backend logs - BEFORE/AFTER calculation] | TBD | Verify calculation logic in `calculate_growth()` function |
| **New Products Handling** | Products with `prev_revenue = 0` and `current_revenue > 0`<br>Should show as "New" or high growth | **REPORTED:** Shows 999%<br>**ACTUAL:** [Check logs] | Showing 999% instead of "New" | Backend caps at 999%, frontend displays as-is |
| **Filtering Logic** | Decliners: `wow_change <= -30%`<br>Movers: `wow_change >= 30%`<br>Fallback: Show any negative/positive if <5 results | **EXPECTED:** Filtered results<br>**ACTUAL:** [Check backend logs] | TBD | Verify filtering logic and fallback |
| **API Response Structure** | `{movers: [...], decliners: [...]}`<br>Each item: `{sku, revenue, wow_change}` | **EXPECTED:** Correct structure<br>**ACTUAL:** [Check API service logs] | TBD | Verify API endpoint returns correct format |
| **Frontend Display** | Tables show SKU, Revenue, WoW Change %<br>Green for movers, Red for decliners | **REPORTED:** Empty tables<br>**ACTUAL:** [Check frontend logs] | Empty state displayed | Verify data reaches component |

### Backend Logs to Check:
- [ ] `🔍 DEBUG: MOVERS & DECLINERS - BEFORE GROWTH CALCULATION` - First 5 SKUs
- [ ] `🔍 DEBUG: MOVERS & DECLINERS - AFTER GROWTH CALCULATION` - Growth percentages
- [ ] Current period query results (SKU count, sample revenues)
- [ ] Previous period query results (SKU count, sample revenues)
- [ ] Merged dataframe count
- [ ] Filtered movers/decliners count

### Frontend Logs to Check:
- [ ] `🔍 API SERVICE DEBUG: MOVERS & DECLINERS` - API response
- [ ] `🔍 DATASTORE DEBUG: FETCHING MOVERS & DECLINERS` - Store data
- [ ] `🔍 FRONTEND DEBUG: MOVERS & DECLINERS` - Component data

---

## REVENUE BY REGION

| Aspect | Expected Behavior | Actual Behavior (To Be Filled) | Issue Identified | Root Cause (TBD) |
|--------|------------------|-------------------------------|------------------|------------------|
| **Cities Shown** | Top 10 cities by revenue<br>Expected: Bengaluru, Mumbai, Hyderabad, Pune, Chennai, New Delhi, Kolkata, Gurugram, Navi Mumbai, Thane | **REPORTED:** Only 2 cities (Amritsar, Mumbai)<br>**ACTUAL:** [Fill after running logs] | Only 2 cities displayed instead of 10 | Possible causes:<br>1. Query returning only 2 regions<br>2. Data filtering out other regions<br>3. Region names not matching |
| **Data Accuracy** | Revenue amounts match actual CSV transactions<br>Sum of Shipment transactions per region | **EXPECTED:** Accurate revenue per region<br>**ACTUAL:** [Check backend logs] | TBD | Verify SQL query sums correctly |
| **Chart Display** | All 10 bars visible side-by-side<br>Horizontal bars (vertical layout)<br>Full width utilization | **REPORTED:** Only 2 bars visible<br>**ACTUAL:** [Check frontend logs] | Only 2 bars displayed | Data array only contains 2 items |
| **Region Name Normalization** | Backend: Lowercase grouping, title case display<br>Frontend: Exact match for selection | **EXPECTED:** Consistent naming<br>**ACTUAL:** [Check logs] | TBD | Verify normalization logic |
| **Revenue Column Used** | Priority: `revenue_calc` > `revenue_amount` > `Invoice Amount` > `revenue_in_inr` | **EXPECTED:** Using correct column<br>**ACTUAL:** [Check backend logs] | TBD | Verify column detection |
| **Transaction Filter** | Only Shipment transactions included<br>Excludes Refunds, Cancels, FreeReplacements | **EXPECTED:** Only shipments<br>**ACTUAL:** [Check SQL query logs] | TBD | Verify transaction type filter |
| **Date Filter Applied** | Filters by user-selected date range<br>Example: Jul 1 - Sep 30, 2025 | **EXPECTED:** Date range applied<br>**ACTUAL:** [Check backend logs] | TBD | Verify date filter in SQL |
| **Sorting** | `ORDER BY revenue DESC`<br>Highest revenue first | **EXPECTED:** Sorted descending<br>**ACTUAL:** [Check backend logs] | TBD | Verify sorting in query |
| **Limit Applied** | `LIMIT 10`<br>Top 10 regions | **EXPECTED:** Maximum 10 regions<br>**ACTUAL:** [Check backend logs] | TBD | Verify limit in query |
| **Deduplication** | Should NOT be needed if backend query correct<br>Store has deduplication logic as fallback | **EXPECTED:** No duplicates from backend<br>**ACTUAL:** [Check frontend logs] | TBD | Verify if deduplication is triggered |

### Backend Logs to Check:
- [ ] `🔍 DEBUG: REVENUE BY REGION - RAW QUERY RESULTS` - All regions found
- [ ] `🔍 DEBUG: REVENUE BY REGION - FINAL RESULTS` - Final count after cleanup
- [ ] SQL query being executed
- [ ] Column detection (region_col, revenue_col, txn_col)
- [ ] Date filter applied
- [ ] Test query results if empty

### Frontend Logs to Check:
- [ ] `🔍 FRONTEND DEBUG: REVENUE BY REGION CHART` - Full data array
- [ ] Region count and names
- [ ] Duplicate detection
- [ ] Chart configuration

---

## REGION SKUs MODAL

| Aspect | Expected Behavior | Actual Behavior (To Be Filled) | Issue Identified | Root Cause (TBD) |
|--------|------------------|-------------------------------|------------------|------------------|
| **Click Interaction** | Clicking region bar opens modal<br>Cursor changes to pointer on hover<br>Visual feedback (color change) | **EXPECTED:** Modal opens on click<br>**ACTUAL:** [Test and fill] | TBD | Verify onClick handlers work |
| **Modal Opens** | Modal appears with region name in title<br>Example: "Top SKUs - Mumbai" | **EXPECTED:** Modal opens<br>**ACTUAL:** [Test and fill] | TBD | Verify modal state management |
| **Data Fetching** | API call: `GET /api/metrics/region-skus?region=Mumbai&start_date=...&end_date=...`<br>Shows loading skeleton while fetching | **EXPECTED:** API called with correct params<br>**ACTUAL:** [Check API logs] | TBD | Verify API call triggered |
| **Modal Content - SKU List** | Table with columns:<br>- SKU<br>- ASIN<br>- Units Sold<br>- Revenue<br>Top 10 SKUs sorted by units DESC | **EXPECTED:** Table populated<br>**ACTUAL:** [Check backend logs] | TBD | Verify query returns SKUs |
| **ASIN Display** | Shows ASIN ID or "N/A" if empty<br>Monospace font | **EXPECTED:** ASINs displayed<br>**ACTUAL:** [Test and fill] | TBD | Verify ASIN data |
| **Units Display** | Shows units sold per SKU<br>Formatted with commas (e.g., 1,200) | **EXPECTED:** Units displayed<br>**ACTUAL:** [Test and fill] | TBD | Verify units calculation |
| **Revenue Display** | Shows revenue per SKU<br>Formatted as currency (₹) | **EXPECTED:** Revenue displayed<br>**ACTUAL:** [Test and fill] | TBD | Verify revenue calculation |
| **Empty State** | Shows "No SKU data available for {region}" if no data | **EXPECTED:** Empty state if no data<br>**ACTUAL:** [Test and fill] | TBD | Verify empty state handling |
| **Close Button** | Closes modal and resets selected region | **EXPECTED:** Modal closes<br>**ACTUAL:** [Test and fill] | TBD | Verify close handler |
| **Region Name Matching** | Case-insensitive matching in backend<br>Frontend uses exact region name from chart | **EXPECTED:** Correct matching<br>**ACTUAL:** [Check logs] | TBD | Verify region name passed correctly |

### Backend Logs to Check:
- [ ] `🔍 DEBUG: REGION SKUs - Region: {name}` - All SKUs for region
- [ ] Region filter in SQL query
- [ ] SKU count returned
- [ ] Units and revenue per SKU

### Frontend Logs to Check:
- [ ] Click handler triggered
- [ ] API call made with correct region name
- [ ] Modal state changes
- [ ] Data received and displayed

---

## DATA FLOW VERIFICATION

### Movers & Decliners Flow:

```
Backend Calculation
  ↓
get_movers_decliners() → {movers: [...], decliners: [...]}
  ↓
API Endpoint: /api/metrics/movers-decliners
  ↓
Frontend API Service: moversDeclinersService.getMoversDecliners()
  ↓
Data Store: fetchMoversDecliners() → moversDecliners state
  ↓
Dashboard Component: moversDecliners.movers / moversDecliners.decliners
  ↓
Table Display
```

**Checkpoints:**
- [ ] Backend returns data
- [ ] API endpoint returns data
- [ ] API service receives data
- [ ] Data store stores data
- [ ] Component receives data
- [ ] Tables render data

### Revenue by Region Flow:

```
Backend Calculation
  ↓
get_revenue_by_region() → DataFrame [region, revenue]
  ↓
API Endpoint: /api/metrics/revenue-by-region
  ↓
Frontend API Service: regionService.getRevenueByRegion()
  ↓
Data Store: fetchRegionRevenue() → regionRevenue state (with deduplication)
  ↓
Dashboard Component: regionRevenue array
  ↓
BarChart Display
```

**Checkpoints:**
- [ ] Backend returns 10 regions
- [ ] API endpoint returns 10 regions
- [ ] Data store receives data
- [ ] Deduplication not needed (or works correctly)
- [ ] Component receives 10 regions
- [ ] Chart displays 10 bars

### Region SKUs Modal Flow:

```
User Clicks Bar
  ↓
handleRegionClick() → region name extracted
  ↓
API Call: regionService.getSKUsByRegion(region, start, end, 10)
  ↓
Backend: get_skus_by_region(region, start, end, 10)
  ↓
API Endpoint: /api/metrics/region-skus
  ↓
Frontend: Receives {data: [...], count: N}
  ↓
Modal: Displays table with SKUs
```

**Checkpoints:**
- [ ] Click handler triggered
- [ ] Region name extracted correctly
- [ ] API call made
- [ ] Backend query executes
- [ ] SKUs returned
- [ ] Modal displays data

---

## POTENTIAL ROOT CAUSES (To Verify)

### Movers & Decliners Issues:

1. **All showing 999%:**
   - **Possible Cause:** All products are new (no previous revenue)
   - **Verify:** Check backend logs - do SKUs have `prev_revenue = 0`?
   - **Fix:** Display "New" label instead of 999%

2. **No decliners found:**
   - **Possible Cause:** No SKUs with ≤-30% decline
   - **Verify:** Check backend logs - are there any negative growth SKUs?
   - **Fix:** Relax threshold or show any negative growth

3. **No movers found:**
   - **Possible Cause:** No SKUs with ≥+30% growth (except new products)
   - **Verify:** Check backend logs - are growth percentages calculated?
   - **Fix:** Show new products as movers with "New" label

### Revenue by Region Issues:

1. **Only 2 cities shown:**
   - **Possible Cause:** Query only returning 2 regions
   - **Verify:** Check backend logs - how many regions in raw query?
   - **Fix:** Check date filter, transaction filter, region column detection

2. **Wrong cities shown:**
   - **Possible Cause:** Region names not matching expected cities
   - **Verify:** Check backend logs - what region names are returned?
   - **Fix:** Verify region column contains city names, not states/other data

### Region SKUs Modal Issues:

1. **Modal not opening:**
   - **Possible Cause:** Click handler not working
   - **Verify:** Check frontend logs - is click handler triggered?
   - **Fix:** Fix onClick handler in Bar/Cell components

2. **No SKUs in modal:**
   - **Possible Cause:** Query not finding SKUs for region
   - **Verify:** Check backend logs - does query return SKUs?
   - **Fix:** Verify region name matching (case-insensitive)

---

## TESTING CHECKLIST

### Backend Testing:

- [ ] Run backend server
- [ ] Check terminal for debug output
- [ ] Verify SQL queries are executing
- [ ] Check data counts match expectations
- [ ] Verify calculations are correct

### Frontend Testing:

- [ ] Open browser DevTools Console
- [ ] Navigate to Dashboard
- [ ] Check console for debug output
- [ ] Verify API calls are made
- [ ] Check data structure matches expected
- [ ] Test region bar clicks
- [ ] Verify modal opens and displays data

### Integration Testing:

- [ ] Compare backend calculations with frontend display
- [ ] Verify data counts match at each stage
- [ ] Check for data loss between backend and frontend
- [ ] Verify date ranges are applied correctly
- [ ] Test with different date ranges

---

## FILL THIS AFTER RUNNING WITH LOGS

### Actual Backend Output:

**Movers & Decliners:**
```
BEFORE GROWTH CALCULATION:
SKU: [fill]
Current: [fill]
Previous: [fill]
Growth: [fill]

AFTER GROWTH CALCULATION:
[fill actual values]
```

**Revenue by Region:**
```
RAW QUERY RESULTS:
Total regions: [fill]
Regions: [fill list]

FINAL RESULTS:
Total regions: [fill]
Regions: [fill list]
```

**Region SKUs:**
```
Region: [fill]
SKUs found: [fill]
[fill SKU list]
```

### Actual Frontend Output:

**Console Logs:**
```
[fill actual console output]
```

**Data Received:**
```
Movers: [fill count and sample]
Decliners: [fill count and sample]
Regions: [fill count and list]
```

---

## NEXT STEPS

1. **Run application** with debug logging enabled
2. **Capture screenshots** of:
   - Backend terminal output
   - Browser console output
   - Dashboard UI
3. **Fill in "Actual Behavior"** columns in tables above
4. **Compare** expected vs actual
5. **Identify root causes** based on discrepancies
6. **Fix issues** identified

---

**Note:** This document serves as a template. Fill in actual values after running the application with debug logging enabled.


