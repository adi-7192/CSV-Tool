# AI Chat Limitations (Documented)

## Known Differences from API

### FreeReplacement Cost Calculation

**AI Chat:** Uses `revenue_amount` directly (simplified)
- SQL: `ABS(SUM(CASE WHEN transaction_type = 'FreeReplacement' THEN revenue_amount ELSE 0 END))`
- Result: Often returns `0.0` because FreeReplacement transactions typically have `revenue_amount = 0`

**API (`/api/metrics`):** Uses ASIN-based pricing (accurate)
- Logic: Calculates cost based on average ASIN price from past shipments
- Formula: `Average ASIN Price × 2` (original product + replacement)
- Includes shipping costs if available
- Result: Accurate cost estimation (~₹94,093 per month for September 2025)

**Impact:**
- **Difference:** ~₹94,093 per month
- **Net Revenue Impact:** AI Chat net revenue will be **higher** than actual because it's missing FreeReplacement deductions
- **Example (September 2025):**
  - API Net Revenue: ₹2,370,333.02
  - AI Chat Net Revenue: ₹2,464,426.83
  - Difference: ₹94,093.81 (exactly matches FreeReplacement cost)

**Reason:**
FreeReplacement transactions don't have a `revenue_amount` in the database because they are internal replacements, not sales. The API's `metrics_service.py` uses special logic (`_calculate_free_replacement_cost()`) that:
1. Looks up the ASIN from FreeReplacement transactions
2. Finds average selling price of that ASIN from past Shipment transactions
3. Calculates cost as `2 × Average Price` (original + replacement)
4. Adds shipping costs if available

This calculation cannot be done in simple SQL without:
- Complex subqueries
- ASIN lookup logic
- Historical price averaging

**Recommendation:**
- ✅ **Use `/api/metrics` endpoint for exact figures** when accuracy is critical
- ✅ **Use AI Chat for quick estimates** or when approximate values are acceptable
- ✅ **Document this limitation** in user-facing documentation

---

## Other Limitations

### Date Range Handling
- AI Chat may interpret date ranges differently than API
- Always verify date filters in generated SQL

### Transaction Type Filtering
- AI Chat queries may not always include all transaction types
- Net revenue queries should include all 4 types: Shipment, Refund, Cancel, FreeReplacement

---

## Verification

To verify accuracy, compare:
1. **Manual SQL Calculation** (if database accessible)
2. **API Endpoint** (`/api/metrics`)
3. **AI Chat Response**

Use `MULTI_METHOD_VERIFICATION.py` script to compare all three methods.

---

**Last Updated:** November 2025  
**Status:** Documented limitation, not a bug

