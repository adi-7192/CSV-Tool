# API Integration Testing Guide

## Overview
This guide helps you test the API integration and verify data flow in the Dashboard.

## Prerequisites
1. ✅ Backend server running on `http://localhost:8000`
2. ✅ Frontend dev server running on `http://localhost:5173`
3. ✅ Database has data loaded (July-September 2025)

## Testing Checklist

### 1. Initial Load Test

**Steps:**
1. Open browser and navigate to `http://localhost:5173/dashboard`
2. Open DevTools (F12) → Network tab
3. Refresh the page (F5)

**Expected Results:**
- ✅ 4 API calls appear in Network tab:
  - `GET /api/metrics?start_date=2025-07-01&end_date=2025-09-30`
  - `GET /api/metrics/trend?start_date=2025-07-01&end_date=2025-09-30&group_by=day`
  - `GET /api/metrics/top-products?start_date=2025-07-01&end_date=2025-09-30&limit=50`
  - `GET /api/data/status` (optional, for insights)
- ✅ All requests return `200 OK` status
- ✅ Response time < 2 seconds
- ✅ KPI cards show values (not 0)
- ✅ Charts display with data points
- ✅ Table shows SKU rows
- ✅ Insights sidebar has alerts

**Validation:**
```javascript
// In browser console, check:
console.log('Metrics:', window.store?.getState()?.metrics);
console.log('Charts:', window.store?.getState()?.chartData);
console.log('SKU:', window.store?.getState()?.skuPerformance);
```

---

### 2. Date Range Change Test

**Steps:**
1. Click date picker in TopBar
2. Change date range (e.g., to August 1-31, 2025)
3. Select new dates

**Expected Results:**
- ✅ New API calls appear in Network tab with updated dates
- ✅ Loading skeletons appear briefly
- ✅ Dashboard updates with new data
- ✅ Charts refresh with new date range
- ✅ Table shows filtered SKU data
- ✅ No console errors

**Network Calls Expected:**
```
GET /api/metrics?start_date=2025-08-01&end_date=2025-08-31
GET /api/metrics/trend?start_date=2025-08-01&end_date=2025-08-31&group_by=day
GET /api/metrics/top-products?start_date=2025-08-01&end_date=2025-08-31&limit=50
```

---

### 3. Error Handling Test

**Test A: Backend Offline**
1. Stop backend server (`Ctrl+C` in backend terminal)
2. Refresh dashboard page
3. Observe error handling

**Expected Results:**
- ✅ Error alert appears at top of dashboard
- ✅ Error message: "Error Loading Data" or "Network Error"
- ✅ Retry button visible
- ✅ App doesn't crash
- ✅ Console shows error logs

**Test B: Retry Functionality**
1. Start backend server again
2. Click "Retry" button in error alert
3. Observe data loading

**Expected Results:**
- ✅ API calls retrigger
- ✅ Loading states show
- ✅ Data loads successfully
- ✅ Error alert disappears

---

### 4. Data Validation Test

**Check Metrics:**
- ✅ `gross_revenue` > 0
- ✅ `net_revenue` > 0
- ✅ `net_revenue` < `gross_revenue` (always true)
- ✅ `orders` > 0
- ✅ `net_margin` between 0-100

**Check Charts:**
- ✅ Revenue trend has data points
- ✅ Refund trend has data points (may be empty)
- ✅ Charts render without errors
- ✅ Tooltips work on hover

**Check Table:**
- ✅ Table shows rows (at least 1)
- ✅ Columns display correctly
- ✅ Sorting works
- ✅ Pagination works

**Check Insights:**
- ✅ Insights appear (may be 0-4 alerts)
- ✅ Different alert types visible
- ✅ Dismiss functionality works

---

### 5. Performance Test

**Metrics to Check:**
1. **Initial Load Time:** < 3 seconds
2. **API Response Time:** < 2 seconds per call
3. **Data Update Time:** < 2 seconds after date change
4. **No Console Errors:** Check Console tab
5. **Memory Usage:** Stable (no leaks)

**Performance Checklist:**
- ✅ First Contentful Paint < 1.5s
- ✅ Time to Interactive < 3s
- ✅ No JavaScript errors in console
- ✅ No memory leaks (check Memory tab)
- ✅ Smooth scrolling
- ✅ No layout shifts

---

## Debugging Guide

### Network Tab Debugging

**Check API Calls:**
1. Open DevTools → Network tab
2. Filter by "Fetch/XHR"
3. Look for calls to `/api/metrics`, `/api/charts`, etc.

**Common Issues:**

**Issue: CORS Error**
```
Error: Access to XMLHttpRequest blocked by CORS policy
```
**Solution:** Check backend CORS settings, ensure `http://localhost:5173` is allowed

**Issue: 404 Not Found**
```
GET http://localhost:8000/api/metrics 404
```
**Solution:** 
- Verify backend is running
- Check endpoint path matches backend routes
- Verify API base URL in `.env` file

**Issue: 500 Internal Server Error**
```
GET http://localhost:8000/api/metrics 500
```
**Solution:**
- Check backend logs for errors
- Verify database connection
- Check date format (should be YYYY-MM-DD)

**Issue: Timeout**
```
Error: timeout of 10000ms exceeded
```
**Solution:**
- Check backend is responding
- Increase timeout in `api.ts` if needed
- Check network connectivity

---

### Console Debugging

**Check Store State:**
```javascript
// In browser console:
import { useDataStore } from './store/dataStore';
const store = useDataStore.getState();
console.log('Store:', store);
```

**Check API Responses:**
```javascript
// In Network tab, click on API call
// Check "Response" tab for data structure
```

**Check Loading States:**
```javascript
// In browser console:
const store = useDataStore.getState();
console.log('Loading:', {
  metrics: store.metricsLoading,
  charts: store.chartsLoading,
  sku: store.skuLoading,
  insights: store.insightsLoading,
});
```

---

## Manual Test Script

Run this in browser console to test API integration:

```javascript
// Test API Integration
async function testAPI() {
  const API_BASE_URL = 'http://localhost:8000';
  const startDate = '2025-07-01';
  const endDate = '2025-09-30';
  
  console.log('🧪 Testing API Integration...');
  
  // Test 1: Metrics
  try {
    const metrics = await fetch(`${API_BASE_URL}/api/metrics?start_date=${startDate}&end_date=${endDate}`);
    const metricsData = await metrics.json();
    console.log('✅ Metrics:', metricsData);
  } catch (error) {
    console.error('❌ Metrics Error:', error);
  }
  
  // Test 2: Trend
  try {
    const trend = await fetch(`${API_BASE_URL}/api/metrics/trend?start_date=${startDate}&end_date=${endDate}&group_by=day`);
    const trendData = await trend.json();
    console.log('✅ Trend:', trendData);
  } catch (error) {
    console.error('❌ Trend Error:', error);
  }
  
  // Test 3: Top Products
  try {
    const products = await fetch(`${API_BASE_URL}/api/metrics/top-products?start_date=${startDate}&end_date=${endDate}&limit=50`);
    const productsData = await products.json();
    console.log('✅ Products:', productsData);
  } catch (error) {
    console.error('❌ Products Error:', error);
  }
}

testAPI();
```

---

## Expected API Response Formats

### Metrics Response
```json
{
  "data": {
    "gross_revenue": 2830655.40,
    "net_revenue": 2370333.02,
    "refund_amount": 366228.57,
    "cancellation_amount": 0,
    "free_replacement_cost": 94093.81,
    "orders": 1802,
    "avg_order_value": 1570.32,
    "net_margin": 83.7,
    "success_rate": 88.96
  },
  "period": {
    "start_date": "2025-07-01",
    "end_date": "2025-09-30"
  }
}
```

### Trend Response
```json
{
  "data": [
    { "date": "2025-07-01", "revenue": 2200000 },
    { "date": "2025-07-02", "revenue": 2250000 }
  ],
  "count": 90
}
```

### Top Products Response
```json
{
  "data": [
    {
      "sku": "SKU-001",
      "asin": "B001234567",
      "revenue": 234567.89,
      "units_sold": 1200,
      "refund_ratio": 2.3
    }
  ],
  "count": 12
}
```

---

## Success Criteria

✅ **All tests pass:**
- Initial load works
- Date range changes trigger API calls
- Error handling works
- Data validation passes
- Performance metrics acceptable

✅ **No errors:**
- No console errors
- No network errors
- No TypeScript errors
- No runtime errors

✅ **Data displays correctly:**
- Metrics show real values
- Charts render with data
- Table shows rows
- Insights appear

---

## Troubleshooting

**Problem:** No data showing
- Check backend is running
- Check database has data
- Check date range matches data range
- Check API responses in Network tab

**Problem:** Charts not rendering
- Check chart data structure
- Check for empty arrays
- Check Recharts library loaded
- Check console for errors

**Problem:** Table empty
- Check SKU performance endpoint
- Check data transformation
- Check table props
- Check for filtering issues

**Problem:** Insights not showing
- Check insights service logic
- Check metrics data available
- Check insight generation criteria
- Check store state

---

## Next Steps

After successful testing:
1. ✅ Document any issues found
2. ✅ Fix any bugs discovered
3. ✅ Optimize performance if needed
4. ✅ Update API documentation
5. ✅ Add automated tests



