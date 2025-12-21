# Backup, Restore, and Monitoring Guide

Complete guide for backing up/restoring data and monitoring the application.

## 📋 Overview

The application uses two data stores:
- **DuckDB**: SQL analytics database (`data/analytics.duckdb`)
- **ChromaDB**: Vector database for RAG embeddings (`data/chromadb/`)

Both can be backed up and restored using the provided scripts.

---

## 🔄 Backup & Restore

### Creating a Backup

**Manual Backup:**
```bash
cd backend
python scripts/backup.py
```

**With custom output directory:**
```bash
python scripts/backup.py --output-dir /path/to/backups
```

**What gets backed up:**
- DuckDB database file (`analytics.duckdb`)
- ChromaDB directory (all collections and embeddings)
- Manifest file (`manifest.json`) with metadata:
  - Timestamp
  - Git SHA
  - App version
  - File sizes

**Backup location:**
- Default: `backend/backups/backup_YYYYMMDD_HHMMSS/`
- Each backup is in its own timestamped directory

**Example output:**
```
Creating backup in: backend/backups/backup_20250115_143022
Backing up DuckDB from: /path/to/data/analytics.duckdb
✅ DuckDB backed up: 45.23 MB
Backing up ChromaDB from: /path/to/data/chromadb
✅ ChromaDB backed up: 12.45 MB
✅ Backup complete: backend/backups/backup_20250115_143022
   Manifest: backend/backups/backup_20250115_143022/manifest.json
```

### Restoring from Backup

**⚠️ IMPORTANT: Stop the application before restoring!**

```bash
# 1. Stop the backend server (Ctrl+C or kill process)

# 2. Restore from backup
cd backend
python scripts/restore.py backups/backup_20250115_143022 --confirm
```

**Safety features:**
- Requires `--confirm` flag to prevent accidental restores
- Creates safety backup of existing data before restore
- Validates backup manifest before proceeding

**Restore process:**
1. Validates backup directory and manifest
2. Creates safety backup of current data (`.pre_restore_backup`)
3. Restores DuckDB file
4. Restores ChromaDB directory
5. Logs completion

**After restore:**
- Restart the backend server
- Verify data integrity via `/api/health/detailed`

### Admin API Backup Endpoint

**Create backup via API (admin only):**
```bash
curl -X POST http://localhost:8000/api/admin/backups/create \
  -H "Authorization: Bearer <admin_jwt_token>"
```

**Response:**
```json
{
  "success": true,
  "backup_path": "/path/to/backups/backup_20250115_143022",
  "message": "Backup created successfully at /path/to/backups/backup_20250115_143022"
}
```

**Use cases:**
- Scheduled backups (cron job calling API)
- Manual backup from admin UI
- Automated backup before major operations

---

## 📊 Monitoring & Logging

### Structured Logging

All requests are logged with structured format including:
- `request_id`: Unique 8-character ID per request
- `user_id`: Authenticated user ID (if available)
- `tenant_id`: User's tenant ID (if available)
- `method`, `path`, `status_code`, `duration_ms`

**Log format example:**
```
2025-01-15 14:30:22 - INFO - request_id=a1b2c3d4 method=GET path=/api/metrics user_id=123 tenant_id=456
2025-01-15 14:30:22 - INFO - request_id=a1b2c3d4 method=GET path=/api/metrics status=200 duration_ms=45.23 user_id=123 tenant_id=456
```

**Log files:**
- `backend/logs/app.log` - General application logs
- `backend/logs/errors.log` - Error logs only
- `backend/logs/api.log` - API request logs
- `backend/logs/database.log` - Database operation logs

**Configuring log level:**
Set `LOG_LEVEL` environment variable:
```bash
# .env file
LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Request ID in responses:**
Every API response includes `X-Request-ID` header for debugging:
```bash
curl -v http://localhost:8000/api/metrics
# Response headers include:
# X-Request-ID: a1b2c3d4
```

### Health Checks

**Basic health check:**
```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

**Detailed health check:**
```bash
curl http://localhost:8000/api/health/detailed
```

**Response:**
```json
{
  "api": "healthy",
  "database": "connected",
  "chromadb": "connected",
  "chromadb_collections": 5,
  "ollama": "connected"
}
```

**Status meanings:**
- `connected`: Service is healthy and accessible
- `unavailable`: Service not configured or not needed
- `error: <message>`: Service check failed with error

**Health check components:**
1. **API**: Always "healthy" if endpoint responds
2. **Database**: DuckDB connection test
3. **ChromaDB**: ChromaDB client connectivity and collection count
4. **Ollama**: LLM service connectivity (optional)

**Monitoring integration:**
- Use `/api/health/detailed` for monitoring tools (Prometheus, Datadog, etc.)
- Returns 503 if critical services (database) are down
- Returns 200 if all services are healthy

### Debugging Multi-Tenant Issues

**Using request_id:**
1. Find request_id from error message or response header
2. Search logs for that request_id:
   ```bash
   grep "request_id=a1b2c3d4" backend/logs/app.log
   ```
3. Trace full request flow with tenant_id context

**Example log search:**
```bash
# Find all requests for a specific tenant
grep "tenant_id=456" backend/logs/app.log

# Find all requests for a specific user
grep "user_id=123" backend/logs/app.log

# Find slow requests (>1 second)
grep "duration_ms=" backend/logs/app.log | awk -F'duration_ms=' '$2 > 1000'
```

---

## 🔧 Maintenance Tasks

### Regular Backups

**Recommended schedule:**
- **Daily backups**: For production data
- **Before major updates**: Always backup before deployments
- **After data imports**: Backup after large CSV uploads

**Automated backup script (cron example):**
```bash
# Add to crontab (crontab -e)
# Daily backup at 2 AM
0 2 * * * cd /path/to/project/backend && python scripts/backup.py >> /var/log/backup.log 2>&1

# Keep only last 7 days of backups
0 3 * * * find /path/to/project/backend/backups -type d -name "backup_*" -mtime +7 -exec rm -rf {} \;
```

### Backup Retention

**Recommended retention:**
- **Daily backups**: Keep 7 days
- **Weekly backups**: Keep 4 weeks
- **Monthly backups**: Keep 12 months

**Cleanup old backups:**
```bash
# Remove backups older than 7 days
find backend/backups -type d -name "backup_*" -mtime +7 -exec rm -rf {} \;
```

### Monitoring Checklist

**Daily checks:**
- [ ] Review error logs: `tail -f backend/logs/errors.log`
- [ ] Check health endpoint: `curl http://localhost:8000/api/health/detailed`
- [ ] Verify disk space for backups

**Weekly checks:**
- [ ] Review slow queries (duration_ms > 1000ms)
- [ ] Check backup directory size
- [ ] Verify ChromaDB collection counts

**Monthly checks:**
- [ ] Test restore procedure on staging
- [ ] Review backup retention policy
- [ ] Audit log file sizes and rotation

---

## 🚨 Troubleshooting

### Backup Issues

**Error: "DuckDB file not found"**
- Check `DATABASE_PATH` in `.env`
- Verify database file exists at configured path
- Ensure backend server is stopped before backup

**Error: "ChromaDB directory not found"**
- Check if ChromaDB has been initialized (first upload triggers it)
- Verify path: `backend/data/chromadb/`

**Error: "Backup script failed"**
- Check file permissions on backup directory
- Ensure sufficient disk space
- Review script output for specific error

### Restore Issues

**Error: "Backup directory not found"**
- Verify backup path is correct
- Check backup directory exists and contains `manifest.json`

**Error: "Cannot restore while app is running"**
- Stop backend server completely
- Close all database connections
- Retry restore

**Data mismatch after restore:**
- Verify backup was created after last data change
- Check manifest.json timestamp
- Compare backup size with current data size

### Logging Issues

**No logs appearing:**
- Check `LOG_LEVEL` in `.env` (set to INFO or DEBUG)
- Verify `backend/logs/` directory exists and is writable
- Check file permissions

**Missing tenant_id in logs:**
- Verify user is authenticated (JWT token present)
- Check JWT token includes tenant_id claim
- Review auth middleware configuration

---

## 📚 Related Documentation

- [Backend README](../README.md) - General backend setup
- [API Documentation](../api/routes/) - API endpoint details
- [Database Schema](../../docs/04-database-schema.md) - Database structure

---

## 🔐 Security Notes

**Backup security:**
- Backup files contain sensitive data (user data, sales data)
- Store backups in secure location
- Encrypt backups if storing off-site
- Limit access to backup directory

**Log security:**
- Logs may contain sensitive information (user IDs, tenant IDs)
- Rotate log files regularly
- Restrict access to log directory
- Consider log encryption for production

---

## 📝 Quick Reference

**Backup commands:**
```bash
# Create backup
python scripts/backup.py

# Restore backup (requires --confirm)
python scripts/restore.py backups/backup_YYYYMMDD_HHMMSS --confirm

# Admin API backup
curl -X POST http://localhost:8000/api/admin/backups/create \
  -H "Authorization: Bearer <token>"
```

**Health checks:**
```bash
# Basic
curl http://localhost:8000/api/health

# Detailed
curl http://localhost:8000/api/health/detailed
```

**Log viewing:**
```bash
# Real-time logs
tail -f backend/logs/app.log

# Search by request_id
grep "request_id=abc123" backend/logs/app.log

# Search by tenant_id
grep "tenant_id=456" backend/logs/app.log
```

