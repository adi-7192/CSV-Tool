# Database Schema Documentation

## Database Technology

**DuckDB** - In-process SQL OLAP database

### Why DuckDB?
- ✅ Fast analytical queries
- ✅ No separate database server needed
- ✅ Pandas integration
- ✅ SQL standard compliance
- ⚠️ Limited transaction support
- ⚠️ Not designed for high concurrency

## Database Location

**Path:** `data/analytics.duckdb` (relative to project root)

**Connection:** Singleton pattern via `core/database.py`

## Tables

### 1. `sales` (Main Transaction Table)

Primary table storing all sales transaction data.

#### Schema

```sql
CREATE TABLE sales (
    -- Business Data
    order_id VARCHAR,              -- Unique order/invoice identifier
    order_date DATE,               -- Order placement date
    revenue_amount DOUBLE,          -- Transaction amount (revenue or refund)
    transaction_type VARCHAR,      -- Shipment, Refund, Cancel, FreeReplacement
    sku VARCHAR,                   -- Product SKU/ASIN
    quantity INTEGER,              -- Order quantity
    region VARCHAR,                -- Geographic region/city
    shipping_amount DOUBLE,        -- Shipping cost
    
    -- Data Lineage
    source_file VARCHAR,           -- Original CSV filename
    ingestion_id VARCHAR,          -- Unique upload session ID
    loaded_at TIMESTAMP,           -- When uploaded
    updated_at TIMESTAMP           -- When last modified
);
```

#### Column Descriptions

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `order_id` | VARCHAR | Unique order/invoice identifier | Business key component |
| `order_date` | DATE | Order placement date | Required for date filtering |
| `revenue_amount` | DOUBLE | Transaction amount in INR | Positive for shipments, negative for refunds |
| `transaction_type` | VARCHAR | Type of transaction | Values: Shipment, Refund, Cancel, FreeReplacement |
| `sku` | VARCHAR | Product SKU/ASIN | Business key component |
| `quantity` | INTEGER | Order quantity | Usually positive |
| `region` | VARCHAR | Geographic region/city | Normalized city names |
| `shipping_amount` | DOUBLE | Shipping cost | Optional |
| `source_file` | VARCHAR | Original CSV filename | Data lineage |
| `ingestion_id` | VARCHAR | Upload session ID | Data lineage |
| `loaded_at` | TIMESTAMP | Upload timestamp | Auto-populated |
| `updated_at` | TIMESTAMP | Last modification timestamp | Auto-populated |

#### Business Keys

**Primary Business Key:** `(order_id, transaction_type, sku)`

This combination uniquely identifies a transaction record. Used for:
- Deduplication during upload
- Data integrity checks
- Reconciliation

#### Indexes

DuckDB automatically creates indexes for:
- Primary keys (if defined)
- Foreign keys (if defined)
- Frequently queried columns

**Note:** No explicit indexes are defined. DuckDB handles optimization automatically.

#### Data Types

- **VARCHAR** - Variable-length strings (no explicit length limit)
- **DATE** - Date values (YYYY-MM-DD format)
- **DOUBLE** - Floating-point numbers (64-bit)
- **INTEGER** - Integer values (32-bit)
- **TIMESTAMP** - Date and time values

#### Constraints

**Current Constraints:**
- None explicitly defined

**Recommended Constraints:**
- `order_id` should not be NULL
- `order_date` should not be NULL
- `transaction_type` should be one of: Shipment, Refund, Cancel, FreeReplacement
- `revenue_amount` should not be NULL

**Note:** DuckDB does not enforce foreign key constraints. Data integrity is maintained at the application level.

#### Sample Data

```sql
SELECT * FROM sales LIMIT 3;
```

| order_id | order_date | revenue_amount | transaction_type | sku | quantity | region | source_file |
|----------|------------|----------------|-----------------|-----|----------|--------|-------------|
| ORD001 | 2025-07-15 | 1500.00 | Shipment | GP10_OM2P_STARTRC | 2 | Bangalore | JulyMonthly.csv |
| ORD001 | 2025-07-20 | -1500.00 | Refund | GP10_OM2P_STARTRC | 2 | Bangalore | JulyMonthly.csv |
| ORD002 | 2025-07-16 | 2000.00 | Shipment | GP947-L_TLSnk | 1 | Mumbai | JulyMonthly.csv |

#### Common Queries

**Get all shipments:**
```sql
SELECT * FROM sales WHERE transaction_type = 'Shipment';
```

**Get revenue by date:**
```sql
SELECT 
    order_date,
    SUM(revenue_amount) as daily_revenue
FROM sales
WHERE transaction_type = 'Shipment'
GROUP BY order_date
ORDER BY order_date;
```

**Get top products:**
```sql
SELECT 
    sku,
    SUM(revenue_amount) as total_revenue,
    COUNT(*) as order_count
FROM sales
WHERE transaction_type = 'Shipment'
GROUP BY sku
ORDER BY total_revenue DESC
LIMIT 10;
```

---

### 2. `ingestion_log` (Upload History Table)

Tracks all CSV file uploads and their processing results.

#### Schema

```sql
CREATE TABLE ingestion_log (
    ingestion_id VARCHAR PRIMARY KEY,
    filename VARCHAR NOT NULL,
    uploaded_at TIMESTAMP NOT NULL,
    rows_raw INTEGER,
    rows_cleaned INTEGER,
    rows_inserted INTEGER,
    date_range_start DATE,
    date_range_end DATE,
    validation_status VARCHAR,
    validation_issues JSON,
    processing_time_seconds FLOAT
);
```

#### Column Descriptions

| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `ingestion_id` | VARCHAR | Unique upload session ID | PRIMARY KEY |
| `filename` | VARCHAR | Original CSV filename | NOT NULL |
| `uploaded_at` | TIMESTAMP | Upload timestamp | NOT NULL |
| `rows_raw` | INTEGER | Number of rows in raw CSV | |
| `rows_cleaned` | INTEGER | Number of rows after cleaning | |
| `rows_inserted` | INTEGER | Number of rows inserted into sales table | |
| `date_range_start` | DATE | Earliest date in uploaded data | |
| `date_range_end` | DATE | Latest date in uploaded data | |
| `validation_status` | VARCHAR | Validation result | Values: passed, failed, warnings |
| `validation_issues` | JSON | Validation issues found | JSON object |
| `processing_time_seconds` | FLOAT | Processing time in seconds | |

#### Primary Key

**PRIMARY KEY:** `ingestion_id`

Unique identifier for each upload session. Format: `ing_YYYYMMDD_HHMMSS`

#### Sample Data

```sql
SELECT * FROM ingestion_log ORDER BY uploaded_at DESC LIMIT 3;
```

| ingestion_id | filename | uploaded_at | rows_raw | rows_cleaned | rows_inserted | date_range_start | date_range_end | validation_status |
|--------------|----------|-------------|----------|--------------|---------------|------------------|----------------|-------------------|
| ing_20251107_123456 | JulyMonthly.csv | 2025-11-07 12:34:56 | 1500 | 1450 | 1450 | 2025-07-01 | 2025-07-31 | passed |
| ing_20251107_120000 | AugMonthly.csv | 2025-11-07 12:00:00 | 2000 | 1950 | 1950 | 2025-08-01 | 2025-08-31 | passed |

#### Common Queries

**Get upload history:**
```sql
SELECT 
    ingestion_id,
    filename,
    uploaded_at,
    rows_inserted,
    date_range_start,
    date_range_end,
    validation_status
FROM ingestion_log
ORDER BY uploaded_at DESC;
```

**Get uploads with issues:**
```sql
SELECT 
    ingestion_id,
    filename,
    validation_status,
    validation_issues
FROM ingestion_log
WHERE validation_status != 'passed';
```

---

## Data Relationships

### Current Relationships

**No explicit foreign key relationships defined.**

**Logical Relationships:**
- `sales.ingestion_id` → `ingestion_log.ingestion_id` (one-to-many)
- `sales.source_file` → `ingestion_log.filename` (many-to-one)

### Data Lineage

Data lineage is tracked through:
1. **`source_file`** - Original CSV filename
2. **`ingestion_id`** - Upload session ID
3. **`loaded_at`** - Upload timestamp

This allows:
- Tracking data source
- Identifying duplicate uploads
- Auditing data changes
- Reconciliation with source files

---

## Data Transformation

### Upload Process

1. **Read CSV** → Raw data with original column names
2. **Detect Columns** → Map CSV columns to standard schema
3. **Transform** → Standardize column names and data types
4. **Validate** → Data quality checks
5. **Deduplicate** → Remove duplicates based on business keys
6. **Insert** → Store in `sales` table
7. **Log** → Record in `ingestion_log` table

### Column Mapping

CSV columns are mapped to standard schema during upload:

| CSV Column (Examples) | Standard Column | Notes |
|----------------------|-----------------|-------|
| Invoice Number, Order ID, OrderID | `order_id` | Normalized |
| Invoice Date, Order Date, Date | `order_date` | Converted to DATE |
| Invoice Amount, Revenue, Amount | `revenue_amount` | Converted to DOUBLE |
| Transaction Type, Type | `transaction_type` | Normalized |
| SKU, Sku, Product SKU | `sku` | Normalized |
| Quantity, Qty | `quantity` | Converted to INTEGER |
| City, Region, Location | `region` | Normalized city names |
| Shipping, Shipping Cost | `shipping_amount` | Converted to DOUBLE |

---

## Data Quality

### Validation Rules

1. **Required Fields:**
   - `order_id` - Must not be NULL
   - `order_date` - Must not be NULL
   - `revenue_amount` - Must not be NULL
   - `transaction_type` - Must not be NULL

2. **Data Type Validation:**
   - `order_date` - Must be valid date format
   - `revenue_amount` - Must be numeric
   - `quantity` - Must be integer

3. **Business Rules:**
   - `transaction_type` - Must be one of: Shipment, Refund, Cancel, FreeReplacement
   - `order_date` - Must not be in the future
   - `revenue_amount` - Should be positive for shipments, negative for refunds

4. **Deduplication:**
   - Records with same `(order_id, transaction_type, sku)` are considered duplicates
   - Duplicates are removed during upload

### Data Quality Checks

**Available via API:**
- `/api/health/data-quality` - Check for duplicates and overlapping sources
- `/api/upload/verify-data-integrity` - Comprehensive integrity check

---

## Performance Considerations

### Query Optimization

1. **Date Filtering:**
   - Always filter by `order_date` when possible
   - Use date ranges to limit data scanned

2. **Transaction Type Filtering:**
   - Filter by `transaction_type` for revenue calculations
   - Use indexes on frequently filtered columns

3. **Aggregations:**
   - Use `GROUP BY` for aggregations
   - Limit result sets with `LIMIT` clause

### Current Limitations

- **Single Connection:** Database uses singleton pattern (one connection)
- **No Connection Pooling:** Not suitable for high concurrency
- **No Caching:** All queries hit the database
- **Synchronous Operations:** No async database operations

### Recommendations

1. **Add Indexes:**
   ```sql
   CREATE INDEX idx_order_date ON sales(order_date);
   CREATE INDEX idx_transaction_type ON sales(transaction_type);
   CREATE INDEX idx_sku ON sales(sku);
   ```

2. **Partitioning:**
   - Consider partitioning by date for large datasets
   - DuckDB supports table partitioning

3. **Materialized Views:**
   - Create materialized views for common aggregations
   - Refresh periodically

---

## Backup and Recovery

### Backup Strategy

**Current Approach:**
- Manual backups via `/api/upload/reset-database` endpoint
- Backups stored in `data/` directory with timestamp

**Backup Format:**
- DuckDB database file (`.duckdb`)
- Includes all tables and data

### Recovery Process

1. **Stop Application**
2. **Restore Database File:**
   ```bash
   cp data/backup_YYYYMMDD_HHMMSS.db data/analytics.duckdb
   ```
3. **Restart Application**

### Recommended Backup Schedule

- **Daily:** Full database backup
- **Before Major Operations:** Backup before bulk uploads or resets
- **Retention:** Keep last 7 days of backups

---

## Migration and Schema Changes

### Current Schema Version

**Version:** 1.0 (as of November 2025)

### Schema Evolution

**Process:**
1. Create migration script
2. Backup database
3. Run migration
4. Verify data integrity
5. Update schema version

### Adding New Columns

**Example:**
```sql
ALTER TABLE sales ADD COLUMN new_column VARCHAR;
```

**Note:** DuckDB supports `ALTER TABLE` for adding columns.

### Schema Reset

**Available via API:**
- `POST /api/upload/reset-database` - Drops and recreates `sales` table

**Warning:** This will delete all data in the `sales` table.

---

## Security Considerations

### Current Security

- **No Authentication:** Database access is not restricted
- **File-based:** Database is a local file
- **No Encryption:** Database file is not encrypted

### Recommendations

1. **File Permissions:**
   - Restrict database file permissions
   - Only application user should have read/write access

2. **Backup Encryption:**
   - Encrypt backup files
   - Store backups in secure location

3. **Access Control:**
   - Implement application-level authentication
   - Restrict database access to authorized users only

---

## Monitoring and Maintenance

### Monitoring

**Key Metrics:**
- Database file size
- Table row counts
- Query performance
- Upload success rate

**Available via API:**
- `/api/data/status` - Database status and size
- `/api/health/data-quality` - Data quality metrics

### Maintenance Tasks

1. **Regular Cleanup:**
   - Remove old backup files
   - Archive old ingestion logs

2. **Performance Tuning:**
   - Monitor slow queries
   - Add indexes as needed
   - Optimize frequent queries

3. **Data Integrity:**
   - Regular data quality checks
   - Verify business key uniqueness
   - Check for data anomalies

---

## Future Enhancements

### Planned Improvements

1. **Connection Pooling:**
   - Support multiple concurrent connections
   - Better performance under load

2. **Caching Layer:**
   - Redis cache for frequent queries
   - Reduce database load

3. **Partitioning:**
   - Partition by date for better performance
   - Easier data archival

4. **Replication:**
   - Read replicas for analytics
   - Better scalability

5. **Encryption:**
   - Encrypt database at rest
   - Secure sensitive data





