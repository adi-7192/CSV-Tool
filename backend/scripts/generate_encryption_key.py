#!/usr/bin/env python3
"""
Generate Encryption Key Script

Generates a secure Fernet encryption key for API_KEY_ENCRYPTION_KEY.

Usage:
    python backend/scripts/generate_encryption_key.py
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from utils.encryption import generate_encryption_key

if __name__ == "__main__":
    print("=" * 60)
    print("API Key Encryption Key Generator")
    print("=" * 60)
    print()
    print("Generated encryption key:")
    print()
    key = generate_encryption_key()
    print(key)
    print()
    print("=" * 60)
    print("INSTRUCTIONS:")
    print("=" * 60)
    print()
    print("1. Add this key to your .env file:")
    print(f"   API_KEY_ENCRYPTION_KEY={key}")
    print()
    print("2. Make sure .env is in .gitignore (never commit keys!)")
    print()
    print("3. Use different keys for development and production")
    print()
    print("4. In production, use a secrets manager:")
    print("   - AWS Secrets Manager")
    print("   - HashiCorp Vault")
    print("   - Azure Key Vault")
    print()
    print("⚠️  SECURITY WARNING:")
    print("   - Never commit this key to git")
    print("   - Keep it secure and rotate periodically")
    print("   - If compromised, generate a new key immediately")
    print()
    print("=" * 60)

