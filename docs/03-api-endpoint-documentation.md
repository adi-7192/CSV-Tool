# API Endpoint Documentation

## Base URL

- **Development:** `http://localhost:8000`
- **Production:** Configure via environment variables

## API Version

**Current Version:** 2.0.0

## Authentication

Currently, the API does not require authentication. This should be added for production deployments.

---

## Endpoints

### Health & Status

#### GET `/api/health/`
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

**Status Codes:**
- `200` - Service is healthy

---

#### GET `/api/health/detailed`
Detailed health check including database and Ollama status.

**Response:**
```json
{
  "api": "healthy",
  "database": "connected",
  "ollama": "connected"
}
```

**Status Codes:**
- `200` - All services healthy
- `503` - One or more services unavailable

---

#### GET `/api/health/data-quality`
Check for data quality issues (duplicates, overlapping sources, etc.).

**Response:**
```json
{
  "status": "ok",
  "duplicates": {
    "found": false,
    "message": "No duplicates found"
  },
  "multi_source_data": {
    "overlap_detected": false,
    "message": "No overlapping date ranges detected"
  },
  "source_files": [],
  "warnings": [],
  "errors": []
}
```

**Status Codes:**
- `200` - Check completed

---

### Metrics

#### GET `/api/metrics/`
Get core business metrics (KPIs).

**Query Parameters:**
- `start_date` (optional, string): Start date in YYYY-MM-DD format
- `end_date` (optional, string): End date in YYYY-MM-DD format
- `transaction_type` (optional, string): Filter by transaction type
- `source_file` (optional, string): Filter by source file

**Default Behavior:**
- If no dates provided, returns last 30 days of data

**Response:**
```json
{
  "gross_revenue": 1500000.00,
  "refund_amount": 50000.00,
  "cancellation_amount": 25000.00,
  "free_replacement_cost": 10000.00,
  "net_revenue": 1415000.00,
  "net_margin": 94.33,
  "shipping_loss": 5000.00,
  "orders": 1500,
  "avg_order_value": 1000.00,
  "success_rate": 96.67,
  "transaction_breakdown": {
    "Shipment": {"count": 1500, "amount": 1500000.00},
    "Refund": {"count": 50, "amount": 50000.00},
    "Cancel": {"count": 25, "amount": 25000.00}
  },
  "revenue": 1500000.00,
  "refunds": 50000.00
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid date range or parameters
- `500` - Server error

---

#### GET `/api/metrics/trend`
Get revenue trend over time.

**Query Parameters:**
- `start_date` (required, string): Start date in YYYY-MM-DD format
- `end_date` (required, string): End date in YYYY-MM-DD format
- `group_by` (required, string): Grouping period - `day`, `week`, or `month`
- `transaction_type` (optional, string): Filter by transaction type

**Response:**
```json
{
  "data": [
    {
      "period": "2025-07-01",
      "revenue": 50000.00,
      "orders": 50
    },
    {
      "period": "2025-07-02",
      "revenue": 55000.00,
      "orders": 55
    }
  ],
  "summary": {
    "total_revenue": 1500000.00,
    "total_orders": 1500,
    "avg_daily_revenue": 50000.00
  }
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid parameters
- `500` - Server error

---

#### GET `/api/metrics/top-products`
Get top performing products by revenue.

**Query Parameters:**
- `limit` (optional, integer): Number of products to return (default: 10, max: 100)
- `start_date` (optional, string): Start date filter
- `end_date` (optional, string): End date filter

**Response:**
```json
{
  "products": [
    {
      "sku": "GP10_OM2P_STARTRC",
      "revenue": 150000.00,
      "orders": 150,
      "avg_order_value": 1000.00
    }
  ],
  "total_products": 2847
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid parameters
- `500` - Server error

---

#### GET `/api/metrics/revenue-by-city`
Get revenue breakdown by city/region.

**Query Parameters:**
- `start_date` (optional, string): Start date filter
- `end_date` (optional, string): End date filter
- `limit` (optional, integer): Number of cities to return

**Response:**
```json
{
  "data": [
    {
      "city": "Bangalore",
      "revenue": 300000.00,
      "orders": 300,
      "percentage": 20.0
    }
  ],
  "total_revenue": 1500000.00
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### GET `/api/metrics/revenue-by-city/skus/{city}`
Get SKUs sold in a specific city.

**Path Parameters:**
- `city` (required, string): City name

**Query Parameters:**
- `limit` (optional, integer): Number of SKUs to return

**Response:**
```json
{
  "city": "Bangalore",
  "skus": [
    {
      "sku": "GP10_OM2P_STARTRC",
      "revenue": 50000.00,
      "orders": 50
    }
  ],
  "total_skus": 150
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid city name
- `500` - Server error

---

#### GET `/api/metrics/movers-decliners`
Get products with significant growth or decline.

**Query Parameters:**
- `start_date` (required, string): Start date for comparison
- `end_date` (required, string): End date for comparison
- `periods` (optional, integer): Number of periods to compare (default: 3)

**Response:**
```json
{
  "movers": [
    {
      "sku": "GP10_OM2P_STARTRC",
      "growth_rate": 45.5,
      "current_period": 15000.00,
      "previous_period": 10300.00
    }
  ],
  "decliners": [
    {
      "sku": "GP947-L_TLSnk",
      "growth_rate": -35.2,
      "current_period": 5000.00,
      "previous_period": 7720.00
    }
  ]
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid parameters
- `500` - Server error

---

#### GET `/api/metrics/top-products-performance`
Get top products performance across multiple periods.

**Query Parameters:**
- `start_date` (required, string): Start date
- `end_date` (required, string): End date
- `limit` (optional, integer): Number of products (default: 10)

**Response:**
```json
{
  "products": [
    {
      "sku": "GP10_OM2P_STARTRC",
      "periods": {
        "Period 1": 15000.00,
        "Period 2": 18000.00,
        "Period 3": 20000.00
      },
      "growth_rate": 33.33
    }
  ]
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid parameters
- `500` - Server error

---

#### GET `/api/metrics/quality-issues/refunds`
Get refund quality issues data.

**Query Parameters:**
- `start_date` (optional, string): Start date filter
- `end_date` (optional, string): End date filter

**Response:**
```json
{
  "total_refunds": 50000.00,
  "refund_percentage": 3.33,
  "severity": "low",
  "thresholds": {
    "high": 15.0,
    "medium": 10.0,
    "low": 5.0
  }
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### GET `/api/metrics/quality-issues/cancellations`
Get cancellation quality issues data.

**Query Parameters:**
- `start_date` (optional, string): Start date filter
- `end_date` (optional, string): End date filter

**Response:**
```json
{
  "total_cancellations": 25000.00,
  "cancellation_percentage": 1.67,
  "severity": "low",
  "thresholds": {
    "high": 20.0,
    "medium": 10.0,
    "low": 5.0
  }
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### GET `/api/metrics/quality-issues/replacements`
Get free replacement quality issues data.

**Query Parameters:**
- `start_date` (optional, string): Start date filter
- `end_date` (optional, string): End date filter

**Response:**
```json
{
  "total_replacements": 100,
  "replacement_cost": 10000.00,
  "replacement_percentage": 6.67
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

### Upload

#### POST `/api/upload/csv`
Upload and process a CSV file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file (max 100MB)

**Response:**
```json
{
  "success": true,
  "rows_processed": 1500,
  "rows_inserted": 1450,
  "validation_warnings": [],
  "ingestion_id": "ing_20251107_123456",
  "filename": "JulyMonthly.csv"
}
```

**Status Codes:**
- `200` - Upload successful
- `400` - Invalid file type or size
- `500` - Processing error

---

#### GET `/api/upload/history`
Get upload history from ingestion log.

**Response:**
```json
{
  "uploads": [
    {
      "ingestion_id": "ing_20251107_123456",
      "filename": "JulyMonthly.csv",
      "uploaded_at": "2025-11-07T12:34:56",
      "rows_inserted": 1450,
      "date_range_start": "2025-07-01",
      "date_range_end": "2025-07-31",
      "validation_status": "passed"
    }
  ]
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### POST `/api/upload/multiple`
Upload multiple CSV files in a single request.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Multiple CSV files

**Response:**
```json
{
  "success": true,
  "uploads": [
    {
      "filename": "JulyMonthly.csv",
      "ingestion_id": "ing_20251107_123456",
      "rows_inserted": 1450,
      "status": "success"
    }
  ],
  "total_files": 2,
  "total_rows_inserted": 2900
}
```

**Status Codes:**
- `200` - All uploads successful
- `400` - Invalid files
- `500` - Processing error

---

#### POST `/api/upload/reset-database`
Reset the database (drop and recreate sales table).

**Response:**
```json
{
  "success": true,
  "message": "Database reset successfully",
  "backup_path": "data/backup_20251107_123456.db"
}
```

**Status Codes:**
- `200` - Reset successful
- `500` - Reset failed

---

#### GET `/api/upload/verify-data-integrity`
Verify data integrity (check for duplicates, missing data, etc.).

**Response:**
```json
{
  "status": "ok",
  "duplicates_found": 0,
  "missing_data": [],
  "warnings": []
}
```

**Status Codes:**
- `200` - Verification complete
- `500` - Verification error

---

#### GET `/api/upload/check-schema`
Check current database schema.

**Response:**
```json
{
  "table": "sales",
  "columns": [
    {
      "name": "order_id",
      "type": "VARCHAR"
    }
  ]
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

### Charts

#### GET `/api/charts/refunds-by-source`
Get refunds breakdown by source file (diagnostic endpoint).

**Query Parameters:**
- `start_date` (optional, string): Start date filter
- `end_date` (optional, string): End date filter

**Response:**
```json
{
  "data": [
    {
      "source_file": "JulyMonthly.csv",
      "refund_count": 50,
      "total_refunds": 50000.00,
      "min_date": "2025-07-01",
      "max_date": "2025-07-31"
    }
  ],
  "total_refunds": 50000.00
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### GET `/api/charts/july-refunds-breakdown`
Detailed breakdown of July 2025 refunds by source file.

**Response:**
```json
{
  "breakdown": [
    {
      "source_file": "JulyMonthly.csv",
      "refund_details": []
    }
  ]
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

### AI Chat

#### POST `/api/chat/ask`
Ask a natural language question about the data.

**Request Body:**
```json
{
  "question": "What is my total revenue?",
  "context": {}
}
```

**Response:**
```json
{
  "answer": "Your total revenue is ₹1,500,000.00",
  "sql": "SELECT SUM(revenue_amount) FROM sales WHERE transaction_type = 'Shipment'",
  "data": {
    "total_revenue": 1500000.00
  },
  "execution_time": 0.123
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid question or Ollama unavailable
- `500` - Server error

---

#### GET `/api/chat/suggestions`
Get suggested questions for the AI chat.

**Response:**
```json
{
  "suggestions": [
    "What is my total revenue?",
    "Which products are selling the most?",
    "What is my refund rate?"
  ]
}
```

**Status Codes:**
- `200` - Success

---

### Data

#### GET `/api/data/transactions`
Get paginated transaction data with filters.

**Query Parameters:**
- `page` (optional, integer): Page number (default: 1, min: 1)
- `limit` (optional, integer): Rows per page (default: 50, min: 1, max: 100)
- `date_from` (optional, string): Start date filter
- `date_to` (optional, string): End date filter
- `sku` (optional, string): SKU filter (exact match)
- `transaction_type` (optional, string): Transaction type filter

**Response:**
```json
{
  "data": [
    {
      "order_id": "ORD123",
      "sku": "GP10_OM2P_STARTRC",
      "transaction_type": "Shipment",
      "amount": 1500.00,
      "date": "2025-07-15",
      "quantity": 2
    }
  ],
  "total": 15507,
  "page": 1,
  "total_pages": 311
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid parameters
- `500` - Server error

---

#### GET `/api/data/skus`
Get list of unique SKUs.

**Response:**
```json
{
  "skus": [
    "GP10_OM2P_STARTRC",
    "GP947-L_TLSnk"
  ],
  "total": 2847
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### GET `/api/data/stats`
Get data statistics (total records, date range, unique SKUs).

**Response:**
```json
{
  "total_records": 15507,
  "date_range": {
    "start": "2025-07-01",
    "end": "2025-09-30"
  },
  "unique_skus": 2847
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### GET `/api/data/export`
Export transactions as CSV.

**Query Parameters:**
- `date_from` (optional, string): Start date filter
- `date_to` (optional, string): End date filter
- `sku` (optional, string): SKU filter
- `transaction_type` (optional, string): Transaction type filter

**Response:**
- Content-Type: `text/csv`
- Body: CSV file download

**Status Codes:**
- `200` - Success
- `400` - Invalid parameters
- `500` - Server error

---

#### POST `/api/data/upload`
Upload CSV file (alias for `/api/upload/csv`).

**Request:** Same as `/api/upload/csv`

**Response:** Same as `/api/upload/csv`

---

#### GET `/api/data/status`
Get data status information.

**Response:**
```json
{
  "database_exists": true,
  "database_path": "data/analytics.duckdb",
  "database_size_mb": 12.5,
  "table_exists": true,
  "row_count": 15507
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

### Verification

#### GET `/api/verification/full-audit`
Perform a full audit of the database.

**Response:**
```json
{
  "status": "complete",
  "audit_results": {
    "duplicates": {},
    "data_quality": {},
    "schema_validation": {}
  }
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

#### POST `/api/verification/reconcile-with-csv`
Reconcile database data with a CSV file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file

**Response:**
```json
{
  "status": "complete",
  "matches": 1450,
  "discrepancies": 0,
  "report": {}
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid file
- `500` - Server error

---

#### GET `/api/verification/compare-metrics`
Compare metrics between different sources.

**Response:**
```json
{
  "comparison": {
    "database": {},
    "csv": {},
    "differences": {}
  }
}
```

**Status Codes:**
- `200` - Success
- `500` - Server error

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "status_code": 400
}
```

**Common Status Codes:**
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (endpoint or resource not found)
- `500` - Internal Server Error
- `503` - Service Unavailable (database or Ollama unavailable)

---

## Rate Limiting

Currently, there is no rate limiting implemented. This should be added for production deployments.

---

## CORS

CORS is enabled for the following origins:
- `http://localhost:3000` (React dev server)
- `http://localhost:5173` (Vite dev server)
- Configured `FRONTEND_URL` from environment

---

## API Documentation

Interactive API documentation is available at:
- **Swagger UI:** `http://localhost:8000/api/docs`
- **ReDoc:** `http://localhost:8000/api/redoc`







