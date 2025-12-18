# Tenant Isolation & Rate Limiting Audit Report

**Date:** December 18, 2025  
**Status:** ❌ **FAIL** - Multiple areas need implementation

---

## A) Audit Results

### 1. Rate Limiting: ❌ FAIL

**Current Implementation:**
- **Location:** `backend/utils/rate_limiter.py`
- **Type:** In-memory (dict-based) with thread-safe locks
- **Storage:** `defaultdict(list)` storing request timestamps per key
- **Usage:** Only used in `forgot-password` endpoint
- **Limits:** 5 requests/IP and 3 requests/email per 15 minutes

**Issues:**
- ❌ **Not shared across processes** - Each FastAPI worker has its own in-memory limiter
- ❌ **Resets on server restart** - All rate limit state is lost
- ❌ **Not distributed** - Multiple server instances won't share limits
- ❌ **Memory leak potential** - Old timestamps accumulate (has cleanup but not perfect)

**Files:**
- `backend/utils/rate_limiter.py` - In-memory implementation
- `backend/api/routes/auth.py:270` - Uses `check_forgot_password_rate_limit()`

**Verdict:** ❌ **FAIL** - Needs Redis-based implementation for production

---

### 2. ChromaDB/RAG Tenant Isolation: ❌ FAIL

**Current Implementation:**
- **Location:** `backend/services/embedding_service.py` and `backend/services/rag_service.py`
- **Collection:** Single collection `"sales_context"` shared by all users
- **Strategy:** No tenant isolation whatsoever

**Write Path (Embedding Insert):**
- ❌ **No tenant_id in metadata** when adding documents
  - Location: `embedding_service.py:468-472`
  - Documents added with: `ids`, `documents`, `metadatas` (no tenant_id)
- ❌ **build_schema_documents()** queries ALL data without tenant filter
  - Location: `embedding_service.py:106-114`
  - Query: `SELECT COUNT(*) ... FROM sales` (no WHERE tenant_id)
- ❌ **build_data_insight_documents()** queries ALL data without tenant filter
  - Location: `embedding_service.py:296-425`
  - Calls `get_top_products()`, `get_revenue_by_city()`, `calculate_metrics()` without tenant_id
  - Queries: `SELECT MIN(order_date), MAX(order_date) FROM sales` (no tenant filter)

**Read Path (Retrieval Query):**
- ❌ **No tenant_id filter** in semantic search
  - Location: `embedding_service.py:497-501`
  - Query: `collection.query(query_texts=[query], n_results=...)` (no where filter)
- ❌ **No tenant_id filter** in RAG retrieval
  - Location: `rag_service.py:74`
  - Calls `semantic_search()` which doesn't filter by tenant

**Security Risk:**
- ⚠️ **CRITICAL:** User A can see User B's data insights in RAG context
- ⚠️ **CRITICAL:** Embeddings contain aggregated data from ALL tenants
- ⚠️ **CRITICAL:** No validation that tenant_id is present before querying

**Files:**
- `backend/services/embedding_service.py` - No tenant isolation
- `backend/services/rag_service.py` - No tenant isolation
- `backend/api/routes/chat.py` - Uses RAG but doesn't pass tenant_id

**Verdict:** ❌ **FAIL** - Complete lack of tenant isolation in ChromaDB

---

### 3. End-to-End Tenant Isolation Tests: ❌ FAIL

**Existing Tests:**
- `backend/tests/test_api.py` - Only health check tests
- `backend/tests/test_metrics_service.py` - Unit tests, no tenant isolation
- `backend/tests/test_api_keys.py` - API key tests, no tenant isolation
- No tests for: `/api/metrics`, `/api/charts`, `/api/data/*`, `/api/chat/ask`

**Missing Test Coverage:**
- ❌ No test for "User A cannot access User B's data"
- ❌ No test for metrics endpoint tenant isolation
- ❌ No test for charts endpoint tenant isolation
- ❌ No test for data summary endpoint tenant isolation
- ❌ No test for chat endpoint tenant isolation
- ❌ No test for upload tenant isolation
- ❌ No integration test with two users

**Test Infrastructure:**
- ✅ `tests/conftest.py` exists with fixtures
- ✅ Test database fixture available
- ❌ No test ChromaDB setup
- ❌ No test user creation helpers

**Verdict:** ❌ **FAIL** - No tenant isolation tests exist

---

## B) Implementation Required

### 1. Redis Rate Limiting (Priority: High)

**Files to Create/Modify:**
1. `backend/core/config.py` - Add `REDIS_URL` setting
2. `backend/utils/rate_limiter.py` - Add `RedisRateLimiter` class
3. `backend/api/routes/auth.py` - Use RedisRateLimiter (with fallback)

**Implementation Details:**
- Use Redis `INCR` + `EXPIRE` for atomic rate limiting
- Key format: `ratelimit:{endpoint}:{type}:{identifier}` (e.g., `ratelimit:forgot_password:ip:127.0.0.1`)
- Graceful fallback to in-memory limiter if `REDIS_URL` not set
- Keep same limits: 5/IP, 3/email per 15 minutes

**Dependencies:**
- Add `redis` to `requirements.txt`

---

### 2. ChromaDB Tenant Isolation (Priority: Critical)

**Strategy:** Collection per tenant (Strategy A - Preferred)

**Files to Modify:**
1. `backend/services/embedding_service.py`
   - Change `get_collection()` to accept `tenant_id` parameter
   - Collection name: `f"rag_{tenant_id}"`
   - Update all `get_collection()` calls to pass tenant_id
   - Update `build_schema_documents()` to accept `tenant_id` and filter queries
   - Update `build_data_insight_documents()` to accept `tenant_id` and pass to metrics functions
   - Update `index_all_documents()` to accept `tenant_id` parameter

2. `backend/services/rag_service.py`
   - Update `retrieve_context()` to accept `tenant_id` parameter
   - Pass `tenant_id` to `semantic_search()` and `get_collection()`

3. `backend/api/routes/chat.py`
   - Pass `tenant_id` to RAG functions

**Alternative Strategy B (if preferred):**
- Single collection + metadata filter
- Add `tenant_id` to metadata on every document
- Always query with `where={"tenant_id": tenant_id}`
- Add safety assertion: raise 500 if tenant_id missing

**Recommendation:** Use Strategy A (collection per tenant) - cleaner separation

---

### 3. End-to-End Tenant Isolation Tests (Priority: High)

**Files to Create:**
1. `backend/tests/test_tenant_isolation.py` - New test file

**Test Cases:**
- `test_user_a_cannot_see_user_b_data_in_metrics()`
- `test_user_a_cannot_see_user_b_data_in_charts()`
- `test_user_a_cannot_see_user_b_data_in_data_summary()`
- `test_user_a_cannot_see_user_b_data_in_chat()`
- `test_user_a_cannot_see_user_b_data_in_transactions()`
- `test_upload_isolation()` - User A upload doesn't appear for User B

**Test Setup:**
- Create test database per test run
- Create test ChromaDB path per test run
- Register/login User A and User B
- Upload CSV for User A only
- Assert User A sees data, User B sees empty

**Test Infrastructure:**
- Add fixtures for test users (User A, User B)
- Add fixture for isolated test database
- Add fixture for isolated test ChromaDB

---

## C) Summary

| Area | Status | Priority | Files to Change |
|------|--------|----------|-----------------|
| **Rate Limiting** | ❌ FAIL | High | 3 files |
| **ChromaDB Isolation** | ❌ FAIL | Critical | 3 files |
| **E2E Tests** | ❌ FAIL | High | 1 new file |

**Total Files:** 7 files (3 new/modify, 3 modify, 1 new test file)

---

## D) Implementation Order

1. **ChromaDB Tenant Isolation** (Critical - Security)
2. **End-to-End Tests** (High - Verification)
3. **Redis Rate Limiting** (High - Production readiness)

---

## E) Commands for Testing

### Run Redis Locally (Docker)
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### Run Tests
```bash
# From project root
cd backend
pytest tests/test_tenant_isolation.py -v

# Or run all tests
pytest tests/ -v
```

### Test Redis Connection
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
r.ping()  # Should return True
```

---

## F) Manual Test Plan

### ChromaDB Isolation Test
1. Register User A → Upload CSV → Ask chat question
2. Register User B → Ask same question
3. **Expected:** User B should NOT see User A's data in RAG context

### Rate Limiting Test
1. Make 6 requests to `/api/auth/forgot-password` from same IP
2. **Expected:** 6th request should return rate limit error
3. Wait 15 minutes or restart Redis
4. **Expected:** Rate limit should reset

### Tenant Isolation Test
1. Login User A → Upload CSV → Check `/api/data/summary` → Should show `has_data: true`
2. Login User B → Check `/api/data/summary` → Should show `has_data: false`
3. User B uploads CSV → User B's summary should show `has_data: true`
4. User A's summary should still show only User A's data

