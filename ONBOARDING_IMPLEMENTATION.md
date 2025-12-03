# User Plans, Onboarding Flow, and Session Validation Implementation

## Summary

Implemented user plans, onboarding flow, and proper session validation as specified. All new users start on the "free" plan and are redirected to onboarding after signup.

---

## Backend Changes

### 1. User Model Updates (`backend/models/user.py`)

**Added fields:**
- `plan: Literal["free", "pro", "enterprise"]` (default: "free")
- `onboarded: bool` (default: False)

**Updated models:**
- `UserBase` - includes plan and onboarded
- `UserCreate` - inherits from UserBase
- `UserResponse` - includes plan and onboarded
- `UserInDB` - includes plan and onboarded

### 2. Database Schema (`backend/core/database.py`)

**Added columns to users table:**
- `plan VARCHAR NOT NULL DEFAULT 'free'`
- `onboarded BOOLEAN NOT NULL DEFAULT FALSE`

**Migration:** Automatically adds columns if they don't exist (for existing databases)

### 3. Registration (`backend/api/routes/auth.py`)

**Updated registration handler:**
- All new users get `plan="free"` and `onboarded=False`
- Role field is explicitly ignored (security)

### 4. Auth Endpoints

**GET /api/auth/me** (`backend/api/routes/auth.py`):
- Returns: `{ id, email, role, plan, onboarded, tenant_id, created_at }`
- Requires authentication via `Depends(get_current_user)`
- Used for session validation on frontend

**POST /api/users/onboarded** (`backend/api/routes/users.py`):
- Marks current user as onboarded (`onboarded=True`)
- Returns updated user info
- Requires authentication

### 5. User Service (`backend/services/user_service.py`)

**Updated functions:**
- `create_user()` - Sets plan="free", onboarded=False
- `get_user_by_email()` - Returns plan and onboarded fields
- `get_user_by_id()` - Returns plan and onboarded fields

---

## Frontend Changes

### 1. Auth Store (`frontend/src/store/authStore.ts`)

**Updated User interface:**
```typescript
interface User {
  id: string;
  email: string;
  role: 'user' | 'admin';
  plan: 'free' | 'pro' | 'enterprise';
  onboarded: boolean;
  tenant_id?: string;
  created_at?: string;
}
```

**New functions:**
- `initFromStorage()` - Validates token on app load by calling `/api/auth/me`
  - If token invalid (401): Clears auth and redirects to `/login`
  - If valid: Sets user and token in store
- `markOnboarded()` - Calls `/api/users/onboarded` and updates user state

**Updated functions:**
- `login()` - Returns user with plan and onboarded fields
- `register()` - Returns user with plan="free" and onboarded=False

### 2. Session Validation (`frontend/src/App.tsx`)

**Added:**
- `AppContent` component that calls `initFromStorage()` on mount
- Validates stored token on every app load
- Ensures stale/expired tokens are cleared

### 3. Onboarding Page (`frontend/src/pages/Onboarding.tsx`)

**Features:**
- Welcome message showing current plan ("Free")
- Next steps:
  - "Upload Your First CSV" → navigates to `/app/data-management` and marks onboarded
  - "Explore with Sample Data" → navigates to `/app/dashboard` and marks onboarded
  - "Skip for now" → navigates to `/app/dashboard` and marks onboarded
- Plans section showing:
  - Free (highlighted as current)
  - Pro (disabled, "Coming Soon")
  - Enterprise (disabled, "Coming Soon")

### 4. Login/Signup Redirect Logic

**Updated `LoginPage.tsx` and `SignupPage.tsx`:**

```typescript
// After login/signup:
if (user.role === 'admin') {
  navigate('/admin/users');
} else {
  if (!user.onboarded) {
    navigate('/app/onboarding');  // New users go to onboarding
  } else {
    navigate('/app/dashboard');   // Existing users go to dashboard
  }
}
```

### 5. Settings Page (`frontend/src/pages/Settings.tsx`)

**Added "Plan & Billing" section:**
- Shows current plan (from `authStore.user.plan`)
- Free plan limits displayed
- Pro and Enterprise cards:
  - Feature lists
  - "Coming Soon" buttons (disabled)
  - No payment logic yet

### 6. Routing (`frontend/src/App.tsx`)

**Added route:**
- `/app/onboarding` - Protected route under AppLayout

---

## Flow Summary

### New User Registration Flow:
1. User signs up → `POST /api/auth/register`
2. Backend creates user with `plan="free"`, `onboarded=False`
3. Frontend receives token and user info
4. Redirect logic checks `user.onboarded === false`
5. User redirected to `/app/onboarding`
6. User completes onboarding action (upload CSV, explore sample, or skip)
7. Frontend calls `POST /api/users/onboarded`
8. User redirected to `/app/dashboard`

### Existing User Login Flow:
1. User logs in → `POST /api/auth/login`
2. Frontend receives token and user info (includes `onboarded` status)
3. Redirect logic:
   - If `onboarded === false` → `/app/onboarding`
   - If `onboarded === true` → `/app/dashboard`
   - If `role === 'admin'` → `/admin/users`

### Session Validation Flow:
1. App loads → `App.tsx` calls `initFromStorage()`
2. `initFromStorage()` checks for token in localStorage
3. If token exists → calls `GET /api/auth/me`
4. If 200 OK → Sets user and token in store
5. If 401 Unauthorized → Clears auth, redirects to `/login`

---

## API Endpoints

### Public Endpoints:
- `POST /api/auth/register` - Creates user with plan="free", onboarded=False
- `POST /api/auth/login` - Returns user with plan and onboarded fields

### Protected Endpoints:
- `GET /api/auth/me` - Returns current user info (plan, onboarded, etc.)
- `POST /api/users/onboarded` - Marks user as onboarded

---

## Database Schema

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR NOT NULL UNIQUE,
    password_hash VARCHAR NOT NULL,
    role VARCHAR NOT NULL DEFAULT 'user',
    plan VARCHAR NOT NULL DEFAULT 'free',
    onboarded BOOLEAN NOT NULL DEFAULT FALSE,
    tenant_id VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
)
```

---

## Files Created/Modified

### Backend:
- ✅ `backend/models/user.py` - Added plan and onboarded fields
- ✅ `backend/core/database.py` - Added plan and onboarded columns
- ✅ `backend/services/user_service.py` - Updated to handle plan/onboarded
- ✅ `backend/api/routes/auth.py` - Updated registration and /me endpoint
- ✅ `backend/api/routes/users.py` - NEW: Added /onboarded endpoint
- ✅ `backend/main.py` - Added users router

### Frontend:
- ✅ `frontend/src/store/authStore.ts` - Added plan/onboarded, initFromStorage, markOnboarded
- ✅ `frontend/src/App.tsx` - Added session validation on load, onboarding route
- ✅ `frontend/src/pages/Onboarding.tsx` - NEW: Onboarding page
- ✅ `frontend/src/pages/public/LoginPage.tsx` - Updated redirect logic
- ✅ `frontend/src/pages/public/SignupPage.tsx` - Updated redirect logic
- ✅ `frontend/src/pages/Settings.tsx` - Added Plan & Billing section
- ✅ `frontend/src/services/api.ts` - Added usersService.markOnboarded()

---

## Testing Checklist

- [ ] New user registration creates user with plan="free", onboarded=False
- [ ] New user redirected to /app/onboarding after signup
- [ ] Onboarding page displays correctly
- [ ] Completing onboarding marks user as onboarded
- [ ] Existing user with onboarded=true goes to dashboard
- [ ] Session validation works on app reload
- [ ] Expired token clears auth and redirects to login
- [ ] Admin users bypass onboarding
- [ ] Settings page shows current plan
- [ ] Pro/Enterprise cards show "Coming Soon"

---

## Next Steps (Future)

1. Implement payment/subscription logic for Pro/Enterprise plans
2. Add plan limits enforcement (e.g., 10,000 records for Free)
3. Add upgrade flow when user hits limits
4. Add billing management UI
5. Add plan change notifications

