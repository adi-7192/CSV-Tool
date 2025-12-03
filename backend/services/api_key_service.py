"""
API Key Service - Manage user API keys with encryption

Handles CRUD operations for user API keys (OpenAI, Anthropic)
with secure encryption/decryption.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from core.database import get_connection, execute_query
from utils.encryption import APIKeyEncryption, EncryptionError

logger = logging.getLogger(__name__)


class APIKeyService:
    """Service for managing encrypted API keys"""
    
    @staticmethod
    def create_api_key(
        user_id: str,
        provider: str,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Create or update an API key for a user
        
        Args:
            user_id: User identifier
            provider: 'openai' or 'anthropic'
            api_key: The plaintext API key to encrypt and store
            
        Returns:
            dict: Result with success status and key info
            
        Raises:
            ValueError: If provider is invalid
            EncryptionError: If encryption fails
        """
        if provider not in ['openai', 'anthropic']:
            raise ValueError(f"Invalid provider: {provider}. Must be 'openai' or 'anthropic'")
        
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        try:
            # Encrypt the API key
            encrypted_key = APIKeyEncryption.encrypt(api_key.strip())
            
            conn = get_connection()
            
            # Check if key already exists for this user/provider
            existing = execute_query(
                f"""
                SELECT id, encrypted_key, created_at, updated_at
                FROM user_api_keys
                WHERE user_id = '{user_id}' AND provider = '{provider}'
                """
            )
            
            now = datetime.now()
            
            if not existing.empty:
                # Update existing key
                key_id = existing.iloc[0]['id']
                conn.execute(
                    """
                    UPDATE user_api_keys
                    SET encrypted_key = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    [encrypted_key, now, key_id]
                )
                logger.info(f"Updated API key for user {user_id}, provider {provider}")
                
                return {
                    'success': True,
                    'id': key_id,
                    'user_id': user_id,
                    'provider': provider,
                    'masked_key': APIKeyEncryption.mask_key(api_key),
                    'created_at': existing.iloc[0]['created_at'],
                    'updated_at': now.isoformat(),
                }
            else:
                # Create new key
                key_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO user_api_keys (id, user_id, provider, encrypted_key, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [key_id, user_id, provider, encrypted_key, now, now]
                )
                logger.info(f"Created API key for user {user_id}, provider {provider}")
                
                return {
                    'success': True,
                    'id': key_id,
                    'user_id': user_id,
                    'provider': provider,
                    'masked_key': APIKeyEncryption.mask_key(api_key),
                    'created_at': now.isoformat(),
                    'updated_at': now.isoformat(),
                }
        
        except EncryptionError as e:
            logger.error(f"Encryption error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise
    
    @staticmethod
    def get_api_key(user_id: str, provider: str, decrypt: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get API key for a user and provider
        
        Args:
            user_id: User identifier
            provider: 'openai' or 'anthropic'
            decrypt: If True, return decrypted key (backend only). If False, return masked key.
            
        Returns:
            dict: API key info with masked or decrypted key, or None if not found
        """
        try:
            result = execute_query(
                f"""
                SELECT id, user_id, provider, encrypted_key, created_at, updated_at
                FROM user_api_keys
                WHERE user_id = '{user_id}' AND provider = '{provider}'
                """
            )
            
            if result.empty:
                return None
            
            row = result.iloc[0]
            encrypted_key = row['encrypted_key']
            
            # Only decrypt on backend, never send to frontend
            if decrypt:
                try:
                    decrypted_key = APIKeyEncryption.decrypt(encrypted_key)
                    key_display = decrypted_key
                except EncryptionError as e:
                    logger.error(f"Failed to decrypt key: {e}")
                    key_display = None
            else:
                # For frontend, we need to decrypt to mask it, but don't return full key
                try:
                    decrypted_key = APIKeyEncryption.decrypt(encrypted_key)
                    key_display = APIKeyEncryption.mask_key(decrypted_key)
                except EncryptionError:
                    key_display = "****"
            
            return {
                'id': row['id'],
                'user_id': row['user_id'],
                'provider': row['provider'],
                'key': key_display,  # Masked or decrypted based on decrypt flag
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
            }
        
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None
    
    @staticmethod
    def get_all_api_keys(user_id: str) -> List[Dict[str, Any]]:
        """
        Get all API keys for a user (masked)
        
        Args:
            user_id: User identifier
            
        Returns:
            list: List of API key info dicts (all keys masked)
        """
        try:
            result = execute_query(
                f"""
                SELECT id, user_id, provider, encrypted_key, created_at, updated_at
                FROM user_api_keys
                WHERE user_id = '{user_id}'
                ORDER BY provider
                """
            )
            
            if result.empty:
                return []
            
            keys = []
            for _, row in result.iterrows():
                encrypted_key = row['encrypted_key']
                try:
                    decrypted_key = APIKeyEncryption.decrypt(encrypted_key)
                    masked_key = APIKeyEncryption.mask_key(decrypted_key)
                except EncryptionError:
                    masked_key = "****"
                
                keys.append({
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'provider': row['provider'],
                    'masked_key': masked_key,
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                })
            
            return keys
        
        except Exception as e:
            logger.error(f"Failed to get API keys: {e}")
            return []
    
    @staticmethod
    def delete_api_key(user_id: str, provider: str) -> Dict[str, Any]:
        """
        Delete an API key for a user
        
        Args:
            user_id: User identifier
            provider: 'openai' or 'anthropic'
            
        Returns:
            dict: Result with success status
        """
        try:
            conn = get_connection()
            result = conn.execute(
                """
                DELETE FROM user_api_keys
                WHERE user_id = ? AND provider = ?
                """,
                [user_id, provider]
            )
            
            deleted = result.rowcount > 0
            
            if deleted:
                logger.info(f"Deleted API key for user {user_id}, provider {provider}")
                return {
                    'success': True,
                    'message': f'API key for {provider} deleted successfully',
                }
            else:
                return {
                    'success': False,
                    'message': f'No API key found for {provider}',
                }
        
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    @staticmethod
    def verify_encryption() -> bool:
        """
        Verify that encryption is working correctly
        
        Returns:
            bool: True if encryption/decryption works, False otherwise
        """
        try:
            test_key = "test-api-key-12345"
            encrypted = APIKeyEncryption.encrypt(test_key)
            decrypted = APIKeyEncryption.decrypt(encrypted)
            return decrypted == test_key
        except Exception as e:
            logger.error(f"Encryption verification failed: {e}")
            return False

