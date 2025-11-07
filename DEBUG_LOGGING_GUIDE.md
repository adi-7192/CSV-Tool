# Debug Logging Implementation Summary

## Overview
Comprehensive debug logging has been added to identify root causes of issues in Movers & Decliners, Revenue by Region, and Region SKUs features.

---

## PART A: BACKEND LOGGING

### 1. `get_movers_decliners()` Function

**Location:** `backend/services/metrics_service.py`

**Logging Added:**

#### BEFORE Growth Calculation (Lines 1729-1750)
- Prints first 5 SKUs with:
  - SKU name
  - Current period revenue
  - Previous period revenue
  - Manually calculated growth % (for verification)
- Shows total SKUs in merged dataframe

**Output Format:**
```
================================================================================
🔍 DEBUG: MOVERS & DECLINERS - BEFORE GROWTH CALCULATION
================================================================================
SKU: GP10_OM2P_STARTRC
  Current Period Revenue: ₹125,000.00
  Previous Period Revenue: ₹100,000.00
  Calculated Growth %: 25.0%
--------------------------------------------------------------------------------
...
Total SKUs in merged dataframe: 150
================================================================================
```

#### AFTER Growth Calculation (Lines 1758-1764)
- Prints first 5 SKUs with calculated `wow_change` values
- Shows final growth percentages after applying formula

**Output Format:**
```
================================================================================
🔍 DEBUG: MOVERS & DECLINERS - AFTER GROWTH CALCULATION
================================================================================
SKU: GP10_OM2P_STARTRC | Current: ₹125,000.00 | Previous: ₹100,000.00 | Growth: 25.0%
...
================================================================================
```

**Where to See:** Backend terminal/console output

---

### 2. `get_revenue_by_region()` Function

**Location:** `backend/services/metrics_service.py`

**Logging Added:**

#### RAW QUERY RESULTS (Lines 1445-1473)
- Prints ALL regions found BEFORE cleanup
- Shows region name and revenue for each
- Shows total count
- Shows total revenue across all regions
- If empty, runs test query to check data availability

**Output Format:**
```
================================================================================
🔍 DEBUG: REVENUE BY REGION - RAW QUERY RESULTS
================================================================================
Total regions found: 15

All Regions and Revenue:
  1. mumbai: ₹5,414,850.83
  2. bengaluru: ₹4,200,000.00
  3. delhi: ₹3,800,000.00
  ...

Total revenue across all regions: ₹25,000,000.00
================================================================================
```

#### FINAL RESULTS AFTER CLEANUP (Lines 1492-1503)
- Prints final regions AFTER title-casing and filtering
- Shows final count (may differ from raw if duplicates removed)
- Shows final sorted list

**Output Format:**
```
================================================================================
🔍 DEBUG: REVENUE BY REGION - FINAL RESULTS (AFTER CLEANUP)
================================================================================
Final region count: 10

Final Regions and Revenue:
  1. Mumbai: ₹5,414,850.83
  2. Bengaluru: ₹4,200,000.00
  ...
================================================================================
```

**Where to See:** Backend terminal/console output

---

### 3. `get_skus_by_region()` Function

**Location:** `backend/services/metrics_service.py`

**Logging Added:**

#### REGION SKUs DETAILS (Lines 2037-2060)
- Prints selected region name
- Lists ALL SKUs for that region with:
  - SKU name
  - ASIN
  - Units sold
  - Revenue per SKU
- Shows totals (units and revenue)

**Output Format:**
```
================================================================================
🔍 DEBUG: REGION SKUs - Region: Mumbai
================================================================================
Total SKUs found: 8

All SKUs for region 'Mumbai':
  1. SKU: GP10_OM2P_STARTRC
     ASIN: B01234567
     Units Sold: 150
     Revenue: ₹125,000.00
--------------------------------------------------------------------------------
  2. SKU: GP947-L_TLSnk
     ASIN: B09876543
     Units Sold: 200
     Revenue: ₹180,000.00
--------------------------------------------------------------------------------
...

Total Units: 1,200
Total Revenue: ₹1,500,000.00
================================================================================
```

**Where to See:** Backend terminal/console output (when region bar is clicked)

---

## PART B: FRONTEND LOGGING

### 1. Movers & Decliners Section

**Location:** `frontend/src/pages/Dashboard.tsx` (Lines 540-597)

**Logging Added:**
- Full API response (JSON stringified)
- Data structure analysis (types, existence checks)
- Movers count and full array
- Decliners count and full array
- Sample data (first 5 items) with type information
- Warnings if arrays are empty

**Output Format:**
```
================================================================================
🔍 FRONTEND DEBUG: MOVERS & DECLINERS
================================================================================
[Dashboard] Full API Response: {...}
[Dashboard] Data Structure: {hasMovers: true, hasDecliners: true, ...}
[Dashboard] Movers Count: 5
[Dashboard] Decliners Count: 3

[Dashboard] FAST MOVERS DATA:
Number of items: 5
Full movers array: [...]
Sample Movers (first 5): [{sku: "...", revenue: ..., wow_change: ...}, ...]

[Dashboard] DECLINERS DATA:
Number of items: 3
Full decliners array: [...]
Sample Decliners (first 5): [{sku: "...", revenue: ..., wow_change: ...}, ...]
================================================================================
```

**Where to See:** Browser DevTools Console

---

### 2. Revenue by Region Chart

**Location:** `frontend/src/pages/Dashboard.tsx` (Lines 377-429)

**Logging Added:**
- Full region revenue data (JSON stringified)
- Data type check (Array vs other)
- Each region with revenue and type
- Duplicate detection
- Unique regions count
- Total revenue calculation
- Top 10 regions sorted by revenue
- Chart configuration details

**Output Format:**
```
================================================================================
🔍 FRONTEND DEBUG: REVENUE BY REGION CHART
================================================================================
[Dashboard] Chart Data Source: regionRevenue from store
[Dashboard] Full Region Revenue Data: [...]
[Dashboard] Region Revenue Count: 10
[Dashboard] Data Type: Array

[Dashboard] Region Revenue Data Structure:
  1. Region: "Mumbai" | Revenue: 5414850.83 | Type: number
  2. Region: "Bengaluru" | Revenue: 4200000 | Type: number
  ...

[Dashboard] Unique regions: ["Mumbai", "Bengaluru", ...]
[Dashboard] Total vs Unique: 10 vs 10

[Dashboard] Total Revenue across all regions: 25000000

[Dashboard] Top 10 Regions by Revenue:
  1. Mumbai: ₹54,14,851
  2. Bengaluru: ₹42,00,000
  ...

[Dashboard] Chart Configuration:
  - Chart Type: Vertical BarChart (horizontal bars)
  - Data Key: revenue
  - Y-Axis Key: region
  - Number of bars: 10
================================================================================
```

**Where to See:** Browser DevTools Console

---

### 3. Data Store Logging

**Location:** `frontend/src/store/dataStore.ts` (Lines 293-333)

**Logging Added:**
- API request parameters
- Raw API response
- Response structure analysis
- Sample movers/decliners data
- Success/failure status

**Output Format:**
```
================================================================================
🔍 DATASTORE DEBUG: FETCHING MOVERS & DECLINERS
================================================================================
[DataStore] Fetching movers & decliners from 2025-07-01 to 2025-09-30
[DataStore] Raw API Response: {...}
[DataStore] Response Type: object
[DataStore] Has movers: true
[DataStore] Has decliners: true
[DataStore] Movers count: 5
[DataStore] Decliners count: 3
[DataStore] Sample movers: [...]
[DataStore] Sample decliners: [...]
[DataStore] ✅ Successfully stored movers & decliners
================================================================================
```

**Where to See:** Browser DevTools Console

---

### 4. API Service Logging

**Location:** `frontend/src/services/api.ts` (Lines 446-470)

**Logging Added:**
- Request URL and parameters
- Response status code
- Full response data (JSON)
- Response structure analysis
- Sample data from response

**Output Format:**
```
================================================================================
🔍 API SERVICE DEBUG: MOVERS & DECLINERS
================================================================================
[API] Fetching movers & decliners: /api/metrics/movers-decliners
[API] Request params: {start_date: "2025-07-01", end_date: "2025-09-30", limit: 10}
[API] Response status: 200
[API] Full response data: {...}
[API] Response structure: {hasMovers: true, hasDecliners: true, ...}
[API] Sample movers from response: [...]
[API] Sample decliners from response: [...]
================================================================================
```

**Where to See:** Browser DevTools Console

---

## PART C: HOW TO CAPTURE OUTPUT

### Step 1: Start Backend Server

```bash
cd backend
python -m uvicorn main:app --reload
```

**Watch for:** Print statements in terminal output (lines starting with `🔍 DEBUG:`)

### Step 2: Open Frontend in Browser

```bash
cd frontend
npm run dev
```

**Open:** Browser DevTools Console (F12 or Cmd+Option+I)

### Step 3: Navigate to Dashboard

- Go to Dashboard page
- Wait for data to load
- Check console for debug output

### Step 4: Test Interactions

1. **Movers & Decliners:**
   - Check console for movers/decliners debug output
   - Check backend terminal for growth calculation logs

2. **Revenue by Region:**
   - Check console for region chart debug output
   - Check backend terminal for region query logs
   - Click a region bar
   - Check backend terminal for region SKUs logs

### Step 5: Capture Screenshots

**Take screenshots of:**
1. Browser DevTools Console (showing all debug output)
2. Backend terminal (showing print statements)
3. Dashboard UI (showing actual displayed data)

---

## WHAT TO LOOK FOR

### Backend Terminal - Key Indicators:

1. **Movers & Decliners:**
   - Are SKUs being found in current period?
   - Are SKUs being found in previous period?
   - Are growth percentages calculated correctly?
   - Are filters (>=30%, <=-30%) working?

2. **Revenue by Region:**
   - How many regions are found in raw query?
   - Are regions being normalized correctly?
   - Are duplicates being created or removed?
   - Is revenue being summed correctly?

3. **Region SKUs:**
   - Is region name matching correctly?
   - Are SKUs being found for the region?
   - Are units and revenue calculated correctly?

### Browser Console - Key Indicators:

1. **Data Flow:**
   - Is API returning data?
   - Is data structure correct?
   - Are arrays populated?
   - Are data types correct (numbers vs strings)?

2. **Display Logic:**
   - Is data reaching the component?
   - Are conditional renders working?
   - Are empty states showing correctly?

---

## EXPECTED OUTPUT EXAMPLES

### Backend - Movers & Decliners:
```
🔍 DEBUG: MOVERS & DECLINERS - BEFORE GROWTH CALCULATION
SKU: GP10_OM2P_STARTRC
  Current Period Revenue: ₹125,000.00
  Previous Period Revenue: ₹100,000.00
  Calculated Growth %: 25.0%
```

### Backend - Revenue by Region:
```
🔍 DEBUG: REVENUE BY REGION - RAW QUERY RESULTS
Total regions found: 15
All Regions and Revenue:
  1. mumbai: ₹5,414,850.83
  2. bengaluru: ₹4,200,000.00
```

### Frontend Console:
```
🔍 FRONTEND DEBUG: MOVERS & DECLINERS
[Dashboard] Movers Count: 5
[Dashboard] Decliners Count: 3
```

---

## NEXT STEPS

1. **Run the application** with these logs enabled
2. **Capture screenshots** of console and terminal output
3. **Compare** backend calculations with frontend display
4. **Identify discrepancies** between:
   - Backend calculations vs frontend display
   - Expected vs actual data counts
   - Data structure mismatches

---

**Note:** All logging uses clear separators (`===`) and emoji indicators (🔍) for easy identification in logs.


