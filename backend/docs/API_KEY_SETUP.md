# API Key Encryption Setup Guide

This guide explains how to set up and use the secure API key storage system for OpenAI and Anthropic API keys.

## Overview

The system uses **Fernet (AES-256)** symmetric encryption to securely store API keys in the database. Keys are:
- ✅ Encrypted before storage
- ✅ Never stored in plain text
- ✅ Only decrypted on the backend
- ✅ Masked when sent to frontend (first 6, last 4 characters)

## Environment Setup

### 1. Generate Encryption Key

First, generate a secure encryption key:

```bash
# From project root
cd backend
python -m utils.encryption
```

This will output a key like:
```
API_KEY_ENCRYPTION_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### 2. Add to Environment

Add the key to your `.env` file (in project root):

```bash
# .env file
API_KEY_ENCRYPTION_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

**⚠️ CRITICAL SECURITY NOTES:**

1. **Never commit the encryption key to git**
   - Add `.env` to `.gitignore` if not already there
   - Never commit `.env` files with real keys

2. **Use different keys for different environments**
   - Development: Generate one key for local development
   - Production: Use a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)

3. **Rotate keys periodically**
   - If a key is compromised, generate a new one
   - Note: Rotating keys will require re-encrypting all stored API keys

### 3. Verify Setup

Test that encryption is working:

```bash
# Start the backend server
cd backend
uvicorn main:app --reload

# In another terminal, test encryption
curl http://localhost:8000/api-keys/verify/encryption
```

Expected response:
```json
{
  "encryption_working": true,
  "message": "Encryption service is working correctly"
}
```

## API Endpoints

### Create/Update API Key

```bash
POST /api-keys/
Content-Type: application/json

{
  "user_id": "user-123",
  "provider": "openai",
  "api_key": "sk-your-actual-api-key-here"
}
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "uuid-here",
    "user_id": "user-123",
    "provider": "openai",
    "masked_key": "sk-you...here",
    "created_at": "2025-01-01T12:00:00",
    "updated_at": "2025-01-01T12:00:00"
  },
  "message": "API key for openai saved successfully"
}
```

### Get API Key (Masked)

```bash
GET /api-keys/{user_id}/{provider}
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "uuid-here",
    "user_id": "user-123",
    "provider": "openai",
    "masked_key": "sk-you...here",
    "created_at": "2025-01-01T12:00:00",
    "updated_at": "2025-01-01T12:00:00"
  }
}
```

### Get All API Keys

```bash
GET /api-keys/{user_id}
```

Response:
```json
{
  "success": true,
  "data": [
    {
      "id": "uuid-1",
      "user_id": "user-123",
      "provider": "openai",
      "masked_key": "sk-you...here",
      "created_at": "2025-01-01T12:00:00",
      "updated_at": "2025-01-01T12:00:00"
    },
    {
      "id": "uuid-2",
      "user_id": "user-123",
      "provider": "anthropic",
      "masked_key": "sk-ant...here",
      "created_at": "2025-01-01T12:00:00",
      "updated_at": "2025-01-01T12:00:00"
    }
  ],
  "count": 2
}
```

### Delete API Key

```bash
DELETE /api-keys/{user_id}/{provider}
```

Response:
```json
{
  "success": true,
  "data": null,
  "message": "API key for openai deleted successfully"
}
```

## Security Best Practices

### Development

1. **Use a test key for development**
   ```bash
   # Generate a development key
   python -m utils.encryption
   # Add to .env.local (not committed)
   ```

2. **Never commit `.env` files**
   ```bash
   # .gitignore should include:
   .env
   .env.local
   .env.*.local
   ```

### Production

1. **Use a secrets manager**
   - AWS Secrets Manager
   - HashiCorp Vault
   - Azure Key Vault
   - Google Secret Manager

2. **Example: AWS Secrets Manager**
   ```python
   import boto3
   
   def get_encryption_key():
       client = boto3.client('secretsmanager')
       response = client.get_secret_value(SecretId='api-key-encryption-key')
       return response['SecretString']
   ```

3. **Rotate keys periodically**
   - Generate new key
   - Update in secrets manager
   - Re-encrypt all stored keys (requires migration script)

## Database Schema

The `user_api_keys` table structure:

```sql
CREATE TABLE user_api_keys (
    id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    provider VARCHAR NOT NULL CHECK (provider IN ('openai', 'anthropic')),
    encrypted_key VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, provider)
)
```

**Key Points:**
- `encrypted_key` stores the encrypted blob (never plaintext)
- Unique constraint ensures one key per user per provider
- Timestamps track creation and updates

## Testing

Run the test suite:

```bash
# From backend directory
pytest tests/test_api_keys.py -v
```

Tests verify:
- ✅ Encryption/decryption works correctly
- ✅ Masking shows only first 6 and last 4 characters
- ✅ Database stores encrypted blobs (never plaintext)
- ✅ Unique constraint prevents duplicates
- ✅ CRUD operations work correctly

## Troubleshooting

### Error: "API_KEY_ENCRYPTION_KEY environment variable is not set"

**Solution:** Set the environment variable in your `.env` file or export it:
```bash
export API_KEY_ENCRYPTION_KEY=your-key-here
```

### Error: "Encryption service unavailable"

**Solution:** 
1. Verify the key is set correctly
2. Check that the key is valid base64-encoded Fernet key
3. Generate a new key if needed

### Error: "Failed to decrypt API key"

**Solution:**
- The encryption key may have changed
- All stored keys need to be re-encrypted with the new key
- This requires a migration script

## Migration Guide

If you need to rotate the encryption key:

1. **Backup existing keys** (decrypt and store temporarily)
2. **Generate new encryption key**
3. **Re-encrypt all keys** with new key
4. **Update environment variable**
5. **Test thoroughly**

Example migration script:

```python
# scripts/migrate_encryption_key.py
from services.api_key_service import APIKeyService
from core.database import execute_query

# 1. Decrypt all keys with old key
# 2. Set new encryption key
# 3. Re-encrypt all keys
# 4. Update database
```

## Support

For issues or questions:
1. Check the test suite: `tests/test_api_keys.py`
2. Review encryption utility: `utils/encryption.py`
3. Check API documentation: `/api/docs`

