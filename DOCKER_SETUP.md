# Docker Setup Guide

Complete guide for running the CSV Analytics Dashboard using Docker.

## 📋 Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- At least 4GB RAM available
- 5GB+ free disk space (for Ollama model)

## 🚀 Quick Start

### Build and Start Services

```bash
docker-compose up --build
```

This will:
1. Build the Streamlit app container
2. Pull and start Ollama LLM service
3. Download llama3.1:8b model (first time only, ~4.7GB)
4. Start both services

**Access the app:**
- Open browser to: http://localhost:8501
- Ollama API: http://localhost:11434

### Stop Services

```bash
docker-compose down
```

### Stop and Remove Volumes (Clean Slate)

```bash
docker-compose down -v
```

**Warning:** This removes all data, including uploaded CSVs and database.

---

## 🔧 First Time Setup

### Step 1: Copy Environment Template

```bash
cp .env.example .env
```

Edit `.env` if you need to customize settings.

### Step 2: Build Containers

```bash
docker-compose build
```

### Step 3: Start Services in Background

```bash
docker-compose up -d
```

### Step 4: Check Service Status

```bash
docker-compose ps
```

Should show 2 containers:
- `csv-analytics-dashboard` (streamlit-app)
- `ollama-llm` (ollama)

### Step 5: View Logs

```bash
# All services
docker-compose logs -f

# Streamlit app only
docker-compose logs -f streamlit-app

# Ollama only
docker-compose logs -f ollama
```

---

## 📊 Service Details

### Streamlit App
- **Container**: `csv-analytics-dashboard`
- **Port**: 8501
- **Health Check**: http://localhost:8501/_stcore/health
- **Data Volume**: `./data` → `/app/data`
- **Logs Volume**: `./logs` → `/app/logs`

### Ollama LLM
- **Container**: `ollama-llm`
- **Port**: 11434
- **Model**: llama3.1:8b (auto-downloaded)
- **Volume**: `ollama-data` (persists models)

---

## 🔍 Troubleshooting

### Issue: Ollama Model Not Loading

If the model fails to download automatically:

```bash
# Enter Ollama container
docker exec -it ollama-llm sh

# Manually pull model
ollama pull llama3.1:8b

# Exit container
exit
```

### Issue: Data Not Persisting

**Check volumes are mounted:**

```bash
docker-compose ps -a
```

**Verify data directory exists:**

```bash
ls -la data/
```

**Check volume mounts in docker-compose.yml:**

Ensure these lines exist:
```yaml
volumes:
  - ./data:/app/data
```

### Issue: Port Conflicts

If ports 8501 or 11434 are already in use:

1. Edit `docker-compose.yml`
2. Change port mappings:
   ```yaml
   ports:
     - "8502:8501"  # Use different host port
   ```
3. Restart: `docker-compose down && docker-compose up`

### Issue: Container Won't Start

**Check logs for errors:**

```bash
docker-compose logs streamlit-app
```

**Common issues:**

1. **Missing dependencies**: Rebuild container
   ```bash
   docker-compose build --no-cache
   ```

2. **Permission issues**: Fix data directory permissions
   ```bash
   chmod -R 755 data/
   ```

3. **Database locked**: Stop all containers, remove database lock
   ```bash
   docker-compose down
   rm data/analytics.duckdb.*.lock 2>/dev/null || true
   docker-compose up
   ```

### Issue: Ollama Connection Failed

**Check Ollama is running:**

```bash
docker exec ollama-llm curl http://localhost:11434/api/tags
```

**Check network connectivity:**

```bash
docker exec csv-analytics-dashboard ping ollama
```

**Restart Ollama service:**

```bash
docker-compose restart ollama
```

---

## 📦 Data Persistence

### What Persists (Volumes)

✅ **Persisted:**
- Database: `data/analytics.duckdb`
- Uploaded CSVs: `data/raw/` and `data/cleaned/`
- Ollama models: `ollama-data` volume
- Logs: `logs/` directory

❌ **Not Persisted (Lost on Container Removal):**
- Application code changes (unless you rebuild)
- Container filesystem changes

### Backup Data

```bash
# Backup entire data directory
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Backup database only
cp data/analytics.duckdb backup-analytics-$(date +%Y%m%d).duckdb
```

### Restore Data

```bash
# Extract backup
tar -xzf backup-YYYYMMDD.tar.gz

# Or restore database
cp backup-analytics-YYYYMMDD.duckdb data/analytics.duckdb
```

---

## 🔄 Common Operations

### Restart Services

```bash
docker-compose restart
```

### Rebuild After Code Changes

```bash
docker-compose up --build
```

### Update Dependencies

1. Edit `requirements.txt`
2. Rebuild: `docker-compose build --no-cache`
3. Restart: `docker-compose up`

### View Resource Usage

```bash
docker stats
```

### Clean Up Unused Resources

```bash
# Remove stopped containers
docker-compose down

# Remove unused images
docker image prune

# Full cleanup (careful!)
docker system prune -a
```

---

## 🧪 Testing the Setup

### Test Checklist

1. ✅ **Containers Start**
   ```bash
   docker-compose ps
   ```
   Should show 2 containers running

2. ✅ **App Accessible**
   - Open http://localhost:8501
   - Should see dashboard interface

3. ✅ **Ollama Connected**
   - Upload a CSV file
   - Go to Chat tab
   - Ask a question
   - Should get AI response

4. ✅ **Data Persists**
   ```bash
   docker-compose down
   docker-compose up -d
   ```
   - Uploaded data should still be visible

5. ✅ **Health Checks Pass**
   ```bash
   curl http://localhost:8501/_stcore/health
   ```
   Should return 200 OK

---

## 📝 Environment Variables

Edit `.env` file to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `data/analytics.duckdb` | Database file path |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model to use |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEBUG_MODE` | `False` | Debug mode |
| `MAX_UPLOAD_SIZE_MB` | `100` | Max CSV upload size |
| `QUERY_TIMEOUT_SECONDS` | `30` | Query timeout |

---

## 🔐 Security Notes

- **Production Deployment:**
  - Change default ports if exposed publicly
  - Use environment variables for secrets
  - Enable authentication if needed
  - Use reverse proxy (nginx) for HTTPS

- **Data Security:**
  - Keep `.env` file secure (not in git)
  - Back up database regularly
  - Limit file upload sizes

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Deployment](https://docs.streamlit.io/deploy)
- [Ollama Documentation](https://github.com/ollama/ollama)

---

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs`
2. Verify Docker is running: `docker ps`
3. Check disk space: `df -h`
4. Check memory: `free -h`
5. Review this guide's troubleshooting section

---

**Last Updated:** December 26, 2024  
**Docker Compose Version:** 3.8  
**Python Base Image:** 3.11-slim

