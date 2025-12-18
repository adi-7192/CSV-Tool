# Session Invalidation Audit Report

**Date:** December 18, 2025  
**Status:** ❌ **NOT IMPLEMENTED** - Infrastructure exists but enforcement is missing

---

## (a) What is Already Implemented

### ✅ Database Schema
- **`password_changed_at` column exists** in `users` table
  - Location: `backend/core/database.py:207-210`
  - Column type: `TIMESTAMP`
  - Created during database initialization

### ✅ Password Reset Updates Timestamp
- **`update_user_password()` function updates `password_changed_at`**
  - Location: `backend/services/password_reset_service.py:216-244`
  - Line 237: `SET password_hash = ?, password_changed_at = ?`
  - Called by password reset endpoint

### ✅ Password Reset Endpoint Calls Update
- **`/api/auth/reset-password` endpoint updates password**
  - Location: `backend/api/routes/auth.py:361-362`
  - Calls `update_user_password()` which sets `password_changed_at`

---

## (b) What is Missing

### ❌ Token Version System
- **No `token_version` field** in database
- **No `token_version` claim** in JWT tokens
- **No validation** of token version in `get_current_user()`

### ❌ Password Changed At Validation
- **`password_changed_at` NOT included** in `UserInDB` model
  - Location: `backend/models/user.py:32-37`
  - Model only includes: `id`, `email`, `password_hash`, `role`, `plan`, `onboarded`, `tenant_id`, `created_at`

- **`password_changed_at` NOT retrieved** from database
  - Location: `backend/services/user_service.py:109-138`
  - `get_user_by_id()` and `get_user_by_email()` don't fetch `password_changed_at`

- **No JWT claim comparison** with `password_changed_at`
  - JWT includes `iat` (issued at) but it's never compared
  - Location: `backend/services/auth_service.py:75` - `iat` is set but never validated

- **No validation in `get_current_user()`**
  - Location: `backend/api/deps/auth_deps.py:19-62`
  - Only validates JWT signature and expiration
  - Does NOT check if password was changed after token was issued

### ❌ JWT Token Claims
- **JWT tokens do NOT include `token_version` or `tv` claim**
  - Location: `backend/services/auth_service.py:50-84`
  - Current claims: `sub`, `email`, `role`, `exp`, `iat`, `tenant_id`
  - Missing: `tv` (token version)

---

## (c) Minimal Implementation Required

### Option 1: Token Version (Recommended)

**Files to Change:**

1. **`backend/core/database.py`** (Add column migration)
   - Add `token_version INTEGER NOT NULL DEFAULT 0` to users table
   - Around line 207, after `password_changed_at` migration

2. **`backend/models/user.py`** (Add field to model)
   - Add `token_version: int = 0` to `UserInDB` class
   - Line ~37

3. **`backend/services/user_service.py`** (Retrieve field)
   - Update `get_user_by_id()` to include `token_version` in UserInDB
   - Update `get_user_by_email()` to include `token_version` in UserInDB
   - Lines ~97-106 and ~129-138

4. **`backend/services/auth_service.py`** (Add claim to JWT)
   - Update `create_access_token()` to include `"tv": token_version` in payload
   - Add `token_version` parameter to function signature
   - Lines ~50-84

5. **`backend/api/deps/auth_deps.py`** (Validate token version)
   - In `get_current_user()`, after fetching user, compare `jwt["tv"]` with `user.token_version`
   - If mismatch, raise 401 Unauthorized
   - Lines ~54-62

6. **`backend/api/routes/auth.py`** (Pass token_version to create_access_token)
   - In `/login` endpoint, pass `user.token_version` to `create_access_token()`
   - In `/register` endpoint, pass `0` (new user) to `create_access_token()`
   - Lines ~150-200 (approximate)

7. **`backend/services/password_reset_service.py`** (Increment token_version)
   - Update `update_user_password()` to increment `token_version`
   - Change: `SET password_hash = ?, password_changed_at = ?, token_version = token_version + 1`
   - Line ~237

### Option 2: Password Changed At (Alternative)

**Files to Change:**

1. **`backend/models/user.py`** (Add field to model)
   - Add `password_changed_at: Optional[datetime] = None` to `UserInDB`

2. **`backend/services/user_service.py`** (Retrieve field)
   - Include `password_changed_at` in UserInDB construction

3. **`backend/api/deps/auth_deps.py`** (Validate timestamp)
   - Compare JWT `iat` with `user.password_changed_at`
   - If `password_changed_at > iat`, raise 401

4. **`backend/services/auth_service.py`** (Ensure iat is included)
   - Already includes `iat` - no change needed

**Note:** Option 1 (token_version) is preferred because:
- Simpler comparison (integer vs datetime)
- Works for any password change (not just reset)
- Easier to increment for other invalidation scenarios

---

## Manual Test Plan

### Test: Session Invalidation After Password Reset

**Prerequisites:**
- Two different browsers (or incognito + normal)
- Test user account

**Steps:**

1. **Setup:**
   - Open Browser A (normal)
   - Open Browser B (incognito/private)
   - Log in to the same account in both browsers

2. **Verify Both Sessions Active:**
   - In Browser A: Navigate to `/app/dashboard` - should work
   - In Browser B: Navigate to `/app/dashboard` - should work
   - Both should show the same user data

3. **Reset Password:**
   - In Browser A: Go to `/forgot-password`
   - Enter email and submit
   - Check email for reset link (or check console logs for token)
   - Click reset link and set new password
   - Log in with new password in Browser A

4. **Verify Session Invalidation:**
   - In Browser B: Try to navigate to `/app/dashboard` or any protected route
   - **Expected:** Should receive 401 Unauthorized and be redirected to `/login`
   - **Expected:** Browser B should NOT be able to access protected routes
   - In Browser B: Try calling `/api/auth/me`
   - **Expected:** Should return 401 Unauthorized

5. **Verify New Session Works:**
   - In Browser A: Should still be logged in with new password
   - In Browser A: All protected routes should work normally

**Success Criteria:**
- ✅ Browser B (old session) is logged out after password reset
- ✅ Browser A (new session) continues to work
- ✅ Old JWT tokens are rejected with 401

---

## Implementation Priority

**High Priority** - Security Issue
- Without session invalidation, compromised passwords remain valid until token expiration
- Users expect old sessions to be invalidated after password change
- Industry standard security practice

**Estimated Effort:** 1-2 hours
- Small changes across 7 files
- No frontend changes needed (existing 401 handling works)

---

## Current State Summary

| Component | Status | Location |
|-----------|--------|----------|
| `password_changed_at` column | ✅ Exists | `database.py:209` |
| Password reset updates timestamp | ✅ Works | `password_reset_service.py:237` |
| `password_changed_at` in model | ❌ Missing | `models/user.py` |
| `password_changed_at` retrieval | ❌ Missing | `user_service.py` |
| JWT includes `iat` | ✅ Exists | `auth_service.py:75` |
| JWT includes `tv` | ❌ Missing | `auth_service.py` |
| `token_version` column | ❌ Missing | `database.py` |
| Token validation in `get_current_user` | ❌ Missing | `auth_deps.py` |
| **Session Invalidation** | ❌ **NOT WORKING** | - |

---

## Recommendation

**Implement Option 1 (Token Version)** because:
1. Cleaner implementation (integer comparison)
2. More flexible (can invalidate for other reasons)
3. Standard industry practice
4. Minimal code changes required

