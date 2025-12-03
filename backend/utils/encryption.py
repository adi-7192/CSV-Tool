"""
Encryption utilities for API key storage

Uses Fernet (AES-256) symmetric encryption to securely store API keys.
The encryption key must be provided via environment variable.
"""

import os
import base64
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Custom exception for encryption errors"""
    pass


class APIKeyEncryption:
    """
    Utility class for encrypting and decrypting API keys
    
    Uses Fernet (AES-256) encryption with a key derived from environment variable.
    """
    
    _fernet: Optional[Fernet] = None
    
    @classmethod
    def _get_encryption_key(cls) -> bytes:
        """
        Get encryption key from environment variable or settings
        
        Raises:
            EncryptionError: If encryption key is not set or invalid
            
        Returns:
            bytes: Encryption key as bytes
        """
        # Try environment variable first, then settings
        encryption_key = os.getenv('API_KEY_ENCRYPTION_KEY')
        if not encryption_key:
            try:
                from core.config import settings
                encryption_key = settings.API_KEY_ENCRYPTION_KEY
            except Exception:
                pass
        
        if not encryption_key:
            raise EncryptionError(
                "API_KEY_ENCRYPTION_KEY environment variable is not set. "
                "Please set it before running the application. "
                "See documentation for how to generate a valid key."
            )
        
        # If key is base64 encoded, decode it
        try:
            # Try to decode as base64 (standard Fernet key format)
            key_bytes = base64.urlsafe_b64decode(encryption_key)
            if len(key_bytes) != 32:
                raise ValueError("Key must be 32 bytes when decoded")
            return base64.urlsafe_b64encode(key_bytes)  # Re-encode to ensure proper format
        except Exception:
            # If not base64, derive key from password using PBKDF2
            logger.warning(
                "API_KEY_ENCRYPTION_KEY is not base64 encoded. "
                "Deriving key using PBKDF2 (less secure). "
                "Consider using a base64-encoded Fernet key instead."
            )
            salt = b'api_key_encryption_salt'  # Fixed salt for consistency
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
            return key
    
    @classmethod
    def _get_fernet(cls) -> Fernet:
        """
        Get or create Fernet instance (singleton pattern)
        
        Returns:
            Fernet: Fernet encryption instance
        """
        if cls._fernet is None:
            try:
                key = cls._get_encryption_key()
                cls._fernet = Fernet(key)
            except Exception as e:
                logger.error(f"Failed to initialize Fernet encryption: {e}")
                raise EncryptionError(f"Encryption initialization failed: {e}")
        
        return cls._fernet
    
    @classmethod
    def encrypt(cls, plaintext: str) -> str:
        """
        Encrypt a plaintext string (API key)
        
        Args:
            plaintext: The API key to encrypt
            
        Returns:
            str: Base64-encoded encrypted string
            
        Raises:
            EncryptionError: If encryption fails
        """
        if not plaintext:
            raise EncryptionError("Cannot encrypt empty string")
        
        try:
            fernet = cls._get_fernet()
            encrypted = fernet.encrypt(plaintext.encode('utf-8'))
            return encrypted.decode('utf-8')
        except EncryptionError:
            raise
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt API key: {e}")
    
    @classmethod
    def decrypt(cls, ciphertext: str) -> str:
        """
        Decrypt an encrypted string (API key)
        
        Args:
            ciphertext: The encrypted API key (base64-encoded)
            
        Returns:
            str: Decrypted plaintext API key
            
        Raises:
            EncryptionError: If decryption fails
        """
        if not ciphertext:
            raise EncryptionError("Cannot decrypt empty string")
        
        try:
            fernet = cls._get_fernet()
            decrypted = fernet.decrypt(ciphertext.encode('utf-8'))
            return decrypted.decode('utf-8')
        except EncryptionError:
            raise
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt API key: {e}")
    
    @classmethod
    def mask_key(cls, api_key: str) -> str:
        """
        Mask an API key for display (shows only first 6 and last 4 characters)
        
        Args:
            api_key: The API key to mask
            
        Returns:
            str: Masked API key (e.g., "sk-123...abcd")
        """
        if not api_key or len(api_key) < 10:
            return "****"
        
        if len(api_key) <= 10:
            return api_key[:2] + "..." + api_key[-2:]
        
        return api_key[:6] + "..." + api_key[-4:]


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key
    
    This function can be used to generate a valid encryption key
    for the API_KEY_ENCRYPTION_KEY environment variable.
    
    Returns:
        str: Base64-encoded Fernet key (32 bytes)
    """
    key = Fernet.generate_key()
    return key.decode('utf-8')


if __name__ == "__main__":
    # Utility script to generate encryption key
    print("=" * 60)
    print("API Key Encryption Key Generator")
    print("=" * 60)
    print()
    print("Generated encryption key:")
    print(generate_encryption_key())
    print()
    print("Add this to your .env file as:")
    print("API_KEY_ENCRYPTION_KEY=<generated_key_above>")
    print()
    print("⚠️  IMPORTANT: Keep this key secure and never commit it to git!")
    print("=" * 60)

