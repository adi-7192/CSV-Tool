# Email Configuration Guide

## Overview

The email service is production-ready with proper configuration, templates, and environment-aware behavior.

## Environment Variables

### Required (Production Mode)

When `ENV=production`, the following SMTP settings are **required**:

```bash
ENV=production
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-password
SMTP_FROM=noreply@example.com
SMTP_FROM_NAME=Datadost Analytics
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

### Optional (Development Mode)

When `ENV=development` (default), SMTP settings are optional. The service will:
- Log emails to console (dry-run mode)
- Not send actual emails
- Return success to prevent breaking the flow

### Backward Compatibility

The following legacy environment variable names are still supported:
- `SMTP_USER` → maps to `SMTP_USERNAME`
- `SMTP_FROM_EMAIL` → maps to `SMTP_FROM`

## Configuration

### Environment Detection

The application detects the environment via the `ENV` variable:
- `development` (default): Dry-run mode, emails logged to console
- `production`: Requires full SMTP configuration, sends real emails

### Startup Validation

On application startup:
- **Production mode**: Validates all SMTP settings are present
- **Development mode**: No validation, allows dry-run operation

If validation fails in production, the app will:
- Log an error message
- Continue to start (to allow debugging)
- Fail email sending with clear error messages

## Email Templates

### Template Location

Templates are stored in `backend/services/email_templates.py`:
- `get_password_reset_template()` - Password reset email
- `get_password_changed_template()` - Password changed notification

### Template Features

All templates include:
- **HTML version**: Styled, responsive email with branding
- **Plain text version**: Fallback for email clients that don't support HTML
- **Required content**: Reset links, expiration messaging, security warnings

### Template Testing

Unit tests verify:
- Reset links are included
- Expiration messaging is present
- Security warnings are included
- Both HTML and text versions are generated

Run tests:
```bash
cd backend
pytest tests/test_email_templates.py -v
```

## Development Mode (Dry-Run)

When `ENV=development` and SMTP is not configured:

1. **Email sending is simulated**
   - No actual SMTP connection is made
   - Email content is logged to console with clear formatting

2. **Console output format**:
   ```
   ================================================================================
   [DEV MODE] Email not configured - DRY RUN (email not sent)
   To: user@example.com
   Subject: Reset Your Password - Datadost Analytics
   --------------------------------------------------------------------------------
   Text Body:
   [Full text content]
   --------------------------------------------------------------------------------
   HTML Body Preview:
   [First 500 characters of HTML]
   ================================================================================
   ```

3. **Function returns `True`**
   - Prevents breaking the application flow
   - Allows testing without SMTP setup

## Production Mode

When `ENV=production`:

1. **SMTP configuration is required**
   - Missing settings will cause validation errors
   - Email sending will fail with clear error messages

2. **Real emails are sent**
   - Uses configured SMTP server
   - Supports both TLS (port 587) and SSL (port 465)
   - Handles authentication errors gracefully

3. **Error handling**
   - SMTP errors are logged but don't crash the application
   - Failed email sends return `False`
   - Background tasks handle email sending (non-blocking)

## Usage Examples

### Password Reset Email

```python
from services.email_service import send_password_reset_email

# Send password reset email
success = send_password_reset_email(
    to_email="user@example.com",
    reset_token="abc123...",
    expires_minutes=15
)
```

### Password Changed Notification

```python
from services.email_service import send_password_changed_email

# Send password changed notification
success = send_password_changed_email(
    to_email="user@example.com"
)
```

## SMTP Providers

### Gmail

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password  # Use App Password, not regular password
SMTP_FROM=your-email@gmail.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

**Note**: Gmail requires an [App Password](https://support.google.com/accounts/answer/185833) for SMTP access.

### SendGrid

```bash
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USERNAME=apikey
SMTP_PASSWORD=your-sendgrid-api-key
SMTP_FROM=noreply@yourdomain.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

### AWS SES

```bash
SMTP_HOST=email-smtp.us-east-1.amazonaws.com
SMTP_PORT=587
SMTP_USERNAME=your-ses-smtp-username
SMTP_PASSWORD=your-ses-smtp-password
SMTP_FROM=noreply@yourdomain.com
SMTP_USE_TLS=true
SMTP_USE_SSL=false
```

## Troubleshooting

### Email not sending in production

1. **Check environment variable**:
   ```bash
   echo $ENV
   ```
   Should be `production` for real email sending.

2. **Verify SMTP settings**:
   ```bash
   python -c "from core.config import settings; print(f'SMTP_HOST: {settings.SMTP_HOST}')"
   ```

3. **Check logs**:
   - Look for SMTP authentication errors
   - Check for connection timeouts
   - Verify firewall/network allows SMTP connections

### Development mode not working

1. **Check ENV variable**:
   ```bash
   echo $ENV
   ```
   Should be `development` or unset.

2. **Check console logs**:
   - Look for `[DEV MODE]` messages
   - Verify email content is being logged

### Template rendering issues

1. **Run template tests**:
   ```bash
   pytest tests/test_email_templates.py -v
   ```

2. **Check template imports**:
   ```python
   from services.email_templates import get_password_reset_template
   html, text = get_password_reset_template("https://example.com/reset?token=test", 15)
   print(html)
   ```

## Security Considerations

1. **Never commit SMTP credentials**
   - Use environment variables
   - Add `.env` to `.gitignore`
   - Use secrets management in production

2. **Use App Passwords**
   - For Gmail and similar providers
   - More secure than regular passwords
   - Can be revoked independently

3. **Rate limiting**
   - Password reset emails are rate-limited
   - Prevents abuse and email enumeration

4. **Token expiration**
   - Reset tokens expire after 15 minutes (configurable)
   - Single-use tokens (marked as used after reset)

## Testing

### Unit Tests

```bash
# Test email templates
pytest tests/test_email_templates.py -v

# Test email service (requires mocking SMTP)
pytest tests/test_email_service.py -v  # If exists
```

### Manual Testing

1. **Development mode**:
   ```bash
   ENV=development python -m uvicorn main:app --reload
   ```
   Request password reset and check console logs.

2. **Production mode** (with test SMTP):
   ```bash
   ENV=production \
   SMTP_HOST=smtp.example.com \
   SMTP_USERNAME=test@example.com \
   SMTP_PASSWORD=test \
   SMTP_FROM=test@example.com \
   python -m uvicorn main:app
   ```

## Migration from Old Configuration

If you're using the old environment variable names:

1. **Option 1: Keep using old names**
   - `SMTP_USER` and `SMTP_FROM_EMAIL` still work
   - Automatically mapped to new names

2. **Option 2: Update to new names** (recommended):
   ```bash
   # Old
   SMTP_USER=user@example.com
   SMTP_FROM_EMAIL=noreply@example.com
   
   # New
   SMTP_USERNAME=user@example.com
   SMTP_FROM=noreply@example.com
   ```

## Summary

- ✅ **Production-ready**: Full SMTP configuration with validation
- ✅ **Development-friendly**: Dry-run mode with console logging
- ✅ **Template-based**: Separated templates for maintainability
- ✅ **Well-tested**: Unit tests for template rendering
- ✅ **Backward compatible**: Supports old env var names
- ✅ **Secure**: Rate limiting, token expiration, single-use tokens

