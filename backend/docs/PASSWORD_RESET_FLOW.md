# Password Reset Flow - Implementation Summary

## Overview

The password reset flow is fully implemented and working. Users can request a password reset via email, receive a secure link, and reset their password through the frontend.

## Flow Diagram

```
1. User clicks "Forgot Password" on login page
   ↓
2. User enters email → POST /api/auth/forgot-password
   ↓
3. Backend creates reset token and sends email
   ↓
4. User receives email with reset link
   ↓
5. User clicks link → Opens frontend /reset-password?token=...
   ↓
6. User enters new password → POST /api/auth/reset-password
   ↓
7. Backend validates token, updates password, invalidates sessions
   ↓
8. User redirected to login → Can login with new password
```

## Backend Implementation

### Configuration

**Environment Variable** (`.env`):
```bash
FRONTEND_URL=http://localhost:5173  # For local development
# In production: FRONTEND_URL=https://yourdomain.com
```

**Default** (`backend/core/config.py`):
- Defaults to `http://localhost:5173` (Vite dev server)

### Email Link Generation

**Location**: `backend/services/email_service.py`

```python
reset_url = f"{settings.FRONTEND_URL.rstrip('/')}/reset-password?token={quote(reset_token)}"
```

**Features**:
- Uses `FRONTEND_URL` from config
- URL-encodes token for safety
- Strips trailing slashes
- Logs URL in development mode

### Endpoints

1. **POST `/api/auth/forgot-password`**
   - Input: `{ "email": "user@example.com" }`
   - Always returns 200 (prevents email enumeration)
   - Creates reset token
   - Sends email via background task
   - Rate limited (per IP and per email)

2. **POST `/api/auth/reset-password`**
   - Input: `{ "token": "...", "new_password": "..." }`
   - Validates token (checks expiry, usage)
   - Validates password strength
   - Updates password hash
   - Increments `token_version` (invalidates all sessions)
   - Marks token as used
   - Returns success message

### Error Codes

Backend returns these error codes (400 Bad Request):
- `"INVALID_TOKEN"` → "Invalid or expired password reset link. Please request a new one."
- `"TOKEN_ALREADY_USED"` → "This password reset link has already been used. Please request a new one."
- `"TOKEN_EXPIRED"` → "This password reset link has expired. Please request a new one."

## Frontend Implementation

### Routes

**File**: `frontend/src/App.tsx`

- `/forgot-password` → `ForgotPasswordPage` (public route)
- `/reset-password` → `ResetPasswordPage` (public route)

Both routes are under `PublicLayout` (no authentication required).

### Pages

1. **ForgotPasswordPage** (`frontend/src/pages/public/ForgotPasswordPage.tsx`)
   - Email input form
   - Always shows success message (security)
   - Links to login page

2. **ResetPasswordPage** (`frontend/src/pages/public/ResetPasswordPage.tsx`)
   - Reads `token` from URL query params
   - Shows error if token missing
   - Password + confirm password form
   - Password strength indicator
   - Submits to `POST /api/auth/reset-password`
   - Shows success message and redirects to login
   - Handles all error cases with specific messages

### API Contract

**Request**:
```typescript
POST /api/auth/reset-password
{
  token: string,
  new_password: string
}
```

**Response** (Success):
```json
{
  "message": "Password has been reset successfully. You can now log in with your new password."
}
```

**Response** (Error - 400):
```json
{
  "detail": "Invalid or expired password reset link. Please request a new one."
}
```

## Security Features

1. **Token Security**:
   - Cryptographically secure random tokens
   - SHA-256 hashed storage (never plaintext)
   - 15-minute expiration
   - Single-use (marked as used after reset)

2. **Email Enumeration Prevention**:
   - Always returns 200 for forgot-password
   - Generic success message
   - Logs actual user existence server-side only

3. **Session Invalidation**:
   - Password change increments `token_version`
   - All existing sessions become invalid
   - User must login again

4. **Rate Limiting**:
   - Per IP: 5 requests per 15 minutes
   - Per email: 3 requests per 15 minutes
   - Uses Redis in production (in-memory fallback in dev)

## Testing

### Manual Test Plan

1. **Start Services**:
   ```bash
   # Backend
   cd backend
   python -m uvicorn main:app --reload
   
   # Frontend
   cd frontend
   npm run dev
   ```

2. **Test Forgot Password**:
   - Go to `/login`
   - Click "Forgot password?"
   - Enter your email
   - Check email inbox (and spam folder)

3. **Verify Email Link**:
   - Link should be: `http://localhost:5173/reset-password?token=...`
   - Click the link
   - Should open Reset Password page

4. **Test Password Reset**:
   - Enter new password (min 8 chars, letter + number)
   - Confirm password
   - Submit
   - Should see success message
   - Should redirect to login after 3 seconds

5. **Test Login with New Password**:
   - Login with new password
   - Should work

6. **Test Token Reuse**:
   - Try using the same reset link again
   - Should show error: "This password reset link has already been used"

### Test Script

Use the test script to verify email sending:
```bash
cd backend
python scripts/test_email_send.py your-email@example.com
```

This will:
- Create a reset token
- Send the email directly
- Display the reset URL
- Show detailed logs

## Troubleshooting

### Email Not Received

1. **Check Backend Logs**:
   - Look for: "Attempting to send password reset email to: ..."
   - Look for: "Email sent successfully to: ..." OR error messages
   - Check if background task executed

2. **Check Spam Folder**:
   - Gmail may filter automated emails
   - Check spam/junk folder

3. **Verify SMTP Configuration**:
   ```bash
   cd backend
   python -c "from core.config import settings; print(f'SMTP_HOST: {settings.SMTP_HOST}')"
   ```

4. **Test Email Sending Directly**:
   ```bash
   cd backend
   python scripts/test_email_send.py your-email@example.com
   ```

### Reset Link Doesn't Work

1. **Check FRONTEND_URL**:
   ```bash
   # In .env file
   FRONTEND_URL=http://localhost:5173  # Should match your frontend URL
   ```

2. **Verify Frontend is Running**:
   - Frontend should be accessible at `FRONTEND_URL`
   - Route `/reset-password` should exist

3. **Check Token in URL**:
   - URL should have `?token=...` parameter
   - Token should not be truncated

4. **Check Browser Console**:
   - Open browser dev tools
   - Check for JavaScript errors
   - Check network tab for API errors

### Password Reset Fails

1. **Check Token Validity**:
   - Token expires after 15 minutes
   - Token can only be used once
   - Check backend logs for validation errors

2. **Check Password Requirements**:
   - Minimum 8 characters
   - Must contain at least one letter
   - Must contain at least one number

3. **Check Backend Logs**:
   - Look for validation errors
   - Check if token was found
   - Check if password update succeeded

## Configuration Checklist

- [x] `FRONTEND_URL` set in `.env` (or uses default)
- [x] SMTP configured (for production)
- [x] Frontend routes exist (`/forgot-password`, `/reset-password`)
- [x] ResetPasswordPage reads token from URL
- [x] ResetPasswordPage submits correct payload format
- [x] Error handling matches backend error codes
- [x] URL encoding for token
- [x] Development logging enabled

## Summary

✅ **Email link generation**: Uses `FRONTEND_URL` with proper encoding  
✅ **Frontend routing**: Routes exist and are public  
✅ **Reset flow**: Complete implementation with error handling  
✅ **Security**: Token expiration, single-use, session invalidation  
✅ **Configuration**: `FRONTEND_URL` in `.env` with sensible defaults  

The password reset flow is production-ready!

