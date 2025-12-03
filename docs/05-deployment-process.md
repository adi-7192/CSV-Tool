# Deployment Process Document

## Overview

This document describes the deployment process for the Analytics Dashboard application, including both development and production environments.

## Prerequisites

### System Requirements

**Backend:**
- Python 3.11 or higher
- 2GB RAM minimum (4GB recommended)
- 500MB disk space for database
- Network access for Ollama (if using AI features)

**Frontend:**
- Node.js 18 or higher
- npm or yarn package manager
- 1GB RAM minimum
- 100MB disk space

**Docker (Optional):**
- Docker Engine 20.10 or higher
- Docker Compose 2.0 or higher
- 4GB RAM minimum
- 10GB disk space

### Software Dependencies

**Backend:**
- Python virtual environment
- All packages from `backend/requirements.txt`

**Frontend:**
- Node.js and npm
- All packages from `frontend/package.json`

**External Services:**
- Ollama (optional, for AI features)
  - Docker image: `ollama/ollama:latest`
  - Or local installation

---

## Development Deployment

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd "Nisarg Project"
```

### Step 2: Backend Setup

#### 2.1 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 2.2 Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### 2.3 Configure Environment Variables

Create `.env` file in project root:

```env
# Database
DATABASE_PATH=data/analytics.duckdb

# Ollama (AI Service)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=30

# Frontend
FRONTEND_URL=http://localhost:3000

# Logging
LOG_LEVEL=INFO
DEBUG=False
```

#### 2.4 Initialize Database

The database is automatically initialized on first startup. No manual setup required.

#### 2.5 Start Backend Server

**Option A: Using startup script (Recommended)**
```bash
cd backend
./start_server.sh
```

**Option B: Manual start**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Option C: From project root**
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify Backend:**
```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

### Step 3: Frontend Setup

#### 3.1 Install Dependencies

```bash
cd frontend
npm install
```

#### 3.2 Configure API Endpoint

Update `frontend/src/services/api.ts` if needed:

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

#### 3.3 Start Frontend Development Server

```bash
npm run dev
```

Frontend will be available at:
- **Vite Dev Server:** `http://localhost:5173`
- **React Dev Server:** `http://localhost:3000` (if configured)

### Step 4: Start Ollama (Optional, for AI Features)

#### Option A: Docker

```bash
docker run -d -p 11434:11434 --name ollama ollama/ollama:latest
```

#### Option B: Local Installation

Follow Ollama installation instructions for your platform:
- https://ollama.ai/download

#### Verify Ollama

```bash
curl http://localhost:11434/api/tags
```

### Step 5: Verify Deployment

1. **Backend Health Check:**
   ```bash
   curl http://localhost:8000/api/health/detailed
   ```

2. **Frontend Access:**
   - Open browser: `http://localhost:5173`
   - Verify dashboard loads

3. **API Documentation:**
   - Swagger UI: `http://localhost:8000/api/docs`
   - ReDoc: `http://localhost:8000/api/redoc`

---

## Production Deployment

### Option 1: Docker Compose (Recommended)

#### Step 1: Prepare Environment

Create `.env` file in project root:

```env
# Database
DATABASE_PATH=data/analytics.duckdb

# Ollama
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=llama3.1:8b

# Frontend
FRONTEND_URL=https://your-domain.com

# Logging
LOG_LEVEL=INFO
DEBUG=False
```

#### Step 2: Build and Start Services

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

#### Step 3: Verify Deployment

```bash
# Backend health check
curl http://localhost:8000/api/health

# Ollama health check
curl http://localhost:11434/api/tags
```

#### Step 4: Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Option 2: Manual Production Deployment

#### Step 1: Server Setup

**Requirements:**
- Linux server (Ubuntu 20.04+ recommended)
- Python 3.11+
- Node.js 18+
- Nginx (for reverse proxy)
- Systemd (for service management)

#### Step 2: Backend Deployment

**2.1 Clone Repository**

```bash
cd /opt
git clone <repository-url> analytics-dashboard
cd analytics-dashboard
```

**2.2 Setup Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate
cd backend
pip install -r requirements.txt
```

**2.3 Create Systemd Service**

Create `/etc/systemd/system/analytics-backend.service`:

```ini
[Unit]
Description=Analytics Dashboard Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/analytics-dashboard/backend
Environment="PATH=/opt/analytics-dashboard/venv/bin"
ExecStart=/opt/analytics-dashboard/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**2.4 Start Service**

```bash
sudo systemctl daemon-reload
sudo systemctl enable analytics-backend
sudo systemctl start analytics-backend
sudo systemctl status analytics-backend
```

#### Step 3: Frontend Deployment

**3.1 Build Frontend**

```bash
cd frontend
npm install
npm run build
```

**3.2 Deploy Static Files**

```bash
# Copy build files to web server
sudo cp -r dist/* /var/www/analytics-dashboard/
```

**3.3 Configure Nginx**

Create `/etc/nginx/sites-available/analytics-dashboard`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/analytics-dashboard;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**3.4 Enable Site**

```bash
sudo ln -s /etc/nginx/sites-available/analytics-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### Step 4: SSL Certificate (Optional)

**Using Let's Encrypt:**

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Database Management

### Backup Database

**Manual Backup:**

```bash
# Stop application
docker-compose down

# Backup database file
cp data/analytics.duckdb data/backup_$(date +%Y%m%d_%H%M%S).db

# Start application
docker-compose up -d
```

**Automated Backup Script:**

Create `scripts/backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="data/backups"
mkdir -p $BACKUP_DIR
cp data/analytics.duckdb $BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).db
# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.db" -mtime +7 -delete
```

**Schedule with Cron:**

```bash
# Add to crontab
0 2 * * * /path/to/scripts/backup.sh
```

### Restore Database

```bash
# Stop application
docker-compose down

# Restore from backup
cp data/backups/backup_YYYYMMDD_HHMMSS.db data/analytics.duckdb

# Start application
docker-compose up -d
```

### Reset Database

**Via API:**

```bash
curl -X POST http://localhost:8000/api/upload/reset-database
```

**Warning:** This will delete all data in the `sales` table.

---

## Monitoring and Logging

### Log Files

**Backend Logs:**
- `backend/logs/app.log` - Application logs
- `backend/logs/api.log` - API request logs
- `backend/logs/errors.log` - Error logs
- `backend/logs/database.log` - Database logs

**View Logs:**

```bash
# Docker
docker-compose logs -f backend

# Systemd
sudo journalctl -u analytics-backend -f

# Direct
tail -f backend/logs/app.log
```

### Health Checks

**Backend Health:**

```bash
curl http://localhost:8000/api/health/detailed
```

**Data Quality:**

```bash
curl http://localhost:8000/api/health/data-quality
```

### Monitoring Scripts

**Database Size:**

```bash
du -h data/analytics.duckdb
```

**Disk Usage:**

```bash
df -h
```

---

## Troubleshooting

### Common Issues

#### 1. Backend Won't Start

**Error:** `Could not import module "main"`

**Solution:**
- Ensure you're in the `backend` directory
- Activate virtual environment
- Check Python path

#### 2. Database Connection Error

**Error:** `Database initialization failed`

**Solution:**
- Check database file permissions
- Verify `DATABASE_PATH` in `.env`
- Ensure `data/` directory exists

#### 3. Ollama Connection Failed

**Error:** `Ollama connection failed`

**Solution:**
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check `OLLAMA_URL` in `.env`
- Restart Ollama service

#### 4. Frontend Can't Connect to Backend

**Error:** CORS error or connection refused

**Solution:**
- Verify backend is running
- Check `FRONTEND_URL` in backend `.env`
- Verify API endpoint in frontend config

#### 5. Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

---

## Security Considerations

### Production Security Checklist

- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set up authentication (if needed)
- [ ] Restrict database file permissions
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Database backups encrypted
- [ ] Log rotation configured

### Environment Variables Security

**Never commit `.env` files to version control!**

**Use `.gitignore`:**
```
.env
.env.local
.env.production
```

**Use secrets management:**
- Docker secrets
- Kubernetes secrets
- AWS Secrets Manager
- HashiCorp Vault

---

## Scaling Considerations

### Current Limitations

- **Single Database Connection:** DuckDB singleton pattern
- **No Caching:** All queries hit database
- **Synchronous Operations:** No async database operations
- **Single Instance:** No load balancing

### Future Enhancements

1. **Connection Pooling:**
   - Multiple DuckDB connections
   - Connection pool manager

2. **Caching Layer:**
   - Redis for query caching
   - Reduce database load

3. **Load Balancing:**
   - Multiple backend instances
   - Nginx load balancer

4. **Database Replication:**
   - Read replicas for analytics
   - Better performance

5. **Background Jobs:**
   - Async task processing
   - Queue system (Celery, RQ)

---

## Rollback Procedure

### Rollback Steps

1. **Stop Application:**
   ```bash
   docker-compose down
   # or
   sudo systemctl stop analytics-backend
   ```

2. **Restore Previous Version:**
   ```bash
   git checkout <previous-version-tag>
   ```

3. **Restore Database (if needed):**
   ```bash
   cp data/backups/backup_YYYYMMDD_HHMMSS.db data/analytics.duckdb
   ```

4. **Rebuild and Restart:**
   ```bash
   docker-compose up --build -d
   # or
   sudo systemctl restart analytics-backend
   ```

5. **Verify:**
   ```bash
   curl http://localhost:8000/api/health
   ```

---

## Maintenance Schedule

### Daily Tasks

- [ ] Check application logs
- [ ] Verify health endpoints
- [ ] Monitor disk usage

### Weekly Tasks

- [ ] Review error logs
- [ ] Check data quality
- [ ] Verify backups

### Monthly Tasks

- [ ] Update dependencies
- [ ] Review security patches
- [ ] Performance optimization
- [ ] Database cleanup

---

## Support and Resources

### Documentation

- API Documentation: `http://localhost:8000/api/docs`
- System Architecture: `docs/01-system-architecture.md`
- Component Dependencies: `docs/02-component-dependency-graph.md`
- API Endpoints: `docs/03-api-endpoint-documentation.md`
- Database Schema: `docs/04-database-schema.md`

### Logs Location

- Application: `backend/logs/app.log`
- API: `backend/logs/api.log`
- Errors: `backend/logs/errors.log`
- Database: `backend/logs/database.log`

### Configuration Files

- Backend Config: `backend/core/config.py`
- Environment: `.env`
- Docker Compose: `docker-compose.yml`
- Backend Dockerfile: `backend/Dockerfile`

---

## Version History

| Version | Date | Changes |
|--------|------|---------|
| 2.0.0 | November 2025 | Initial production deployment documentation |

---

## Appendix

### Quick Reference Commands

**Development:**
```bash
# Backend
cd backend && ./start_server.sh

# Frontend
cd frontend && npm run dev

# Ollama
docker run -d -p 11434:11434 ollama/ollama:latest
```

**Production:**
```bash
# Docker Compose
docker-compose up -d

# Systemd
sudo systemctl start analytics-backend

# Nginx
sudo systemctl reload nginx
```

**Monitoring:**
```bash
# Health check
curl http://localhost:8000/api/health/detailed

# Logs
docker-compose logs -f backend
tail -f backend/logs/app.log
```

**Backup:**
```bash
cp data/analytics.duckdb data/backup_$(date +%Y%m%d_%H%M%S).db
```







