"""
API Key Service - Manage user API keys with encryption

Handles CRUD operations for user API keys (OpenAI, Anthropic, Gemini)
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
            provider: 'openai', 'anthropic', or 'gemini'
            api_key: The plaintext API key to encrypt and store
            
        Returns:
            dict: Result with success status and key info
            
        Raises:
            ValueError: If provider is invalid
            EncryptionError: If encryption fails
        """
        if provider not in ['openai', 'anthropic', 'gemini']:
            raise ValueError(f"Invalid provider: {provider}. Must be 'openai', 'anthropic', or 'gemini'")
        
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        try:
            # Encrypt the API key
            logger.info(f"Encrypting API key for user {user_id}, provider {provider}")
            encrypted_key = APIKeyEncryption.encrypt(api_key.strip())
            logger.debug(f"API key encrypted successfully (length: {len(encrypted_key)})")
            
            conn = get_connection()
            
            # Check if key already exists for this user/provider
            logger.debug(f"Checking for existing API key for user {user_id}, provider {provider}")
            existing = execute_query(
                f"""
                SELECT id, encrypted_key, enabled, created_at, updated_at
                FROM user_api_keys
                WHERE user_id = '{user_id}' AND provider = '{provider}'
                """
            )
            
            now = datetime.now()
            
            if not existing.empty:
                # Update existing key (preserve enabled status)
                key_id = existing.iloc[0]['id']
                existing_enabled = existing.iloc[0].get('enabled', True)
                logger.info(f"Updating existing API key (id: {key_id}) for user {user_id}, provider {provider}")
                conn.execute(
                    """
                    UPDATE user_api_keys
                    SET encrypted_key = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    [encrypted_key, now, key_id]
                )
                logger.info(f"✅ Updated API key for user {user_id}, provider {provider}")
                
                # Convert timestamps to ISO format strings (DuckDB returns Timestamp objects)
                created_at = existing.iloc[0]['created_at']
                if hasattr(created_at, 'isoformat'):
                    created_at = created_at.isoformat()
                elif not isinstance(created_at, str):
                    created_at = str(created_at)
                
                return {
                    'success': True,
                    'id': key_id,
                    'user_id': user_id,
                    'provider': provider,
                    'masked_key': APIKeyEncryption.mask_key(api_key),
                    'enabled': bool(existing_enabled),
                    'created_at': created_at,
                    'updated_at': now.isoformat(),
                }
            else:
                # Create new key (enabled by default)
                key_id = str(uuid.uuid4())
                logger.info(f"Creating new API key (id: {key_id}) for user {user_id}, provider {provider}")
                conn.execute(
                    """
                    INSERT INTO user_api_keys (id, user_id, provider, encrypted_key, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [key_id, user_id, provider, encrypted_key, True, now, now]
                )
                logger.info(f"Created API key for user {user_id}, provider {provider}")
                
                return {
                    'success': True,
                    'id': key_id,
                    'user_id': user_id,
                    'provider': provider,
                    'masked_key': APIKeyEncryption.mask_key(api_key),
                    'enabled': True,
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
            provider: 'openai', 'anthropic', or 'gemini'
            decrypt: If True, return decrypted key (backend only). If False, return masked key.
            
        Returns:
            dict: API key info with masked or decrypted key, or None if not found
        """
        try:
            logger.debug(f"Retrieving API key for user {user_id}, provider {provider}, decrypt={decrypt}")
            result = execute_query(
                f"""
                SELECT id, user_id, provider, encrypted_key, enabled, created_at, updated_at
                FROM user_api_keys
                WHERE user_id = '{user_id}' AND provider = '{provider}'
                """
            )
            
            if result.empty:
                logger.debug(f"No API key found for user {user_id}, provider {provider}")
                return None
            
            row = result.iloc[0]
            encrypted_key = row['encrypted_key']
            logger.debug(f"Found API key (id: {row['id']}, enabled: {row.get('enabled', True)})")
            
            # Only decrypt on backend, never send to frontend
            if decrypt:
                try:
                    decrypted_key = APIKeyEncryption.decrypt(encrypted_key)
                    key_display = decrypted_key
                    logger.debug("API key decrypted successfully for backend use")
                except EncryptionError as e:
                    logger.error(f"Failed to decrypt key: {e}")
                    key_display = None
            else:
                # For frontend, we need to decrypt to mask it, but don't return full key
                try:
                    decrypted_key = APIKeyEncryption.decrypt(encrypted_key)
                    key_display = APIKeyEncryption.mask_key(decrypted_key)
                    logger.debug(f"API key masked for frontend: {key_display}")
                except EncryptionError as e:
                    logger.error(f"Failed to decrypt key for masking: {e}")
                    key_display = "****"
            
            # Convert timestamps to ISO format strings (DuckDB returns Timestamp objects)
            created_at = row['created_at']
            updated_at = row['updated_at']
            
            # Handle various timestamp types
            if hasattr(created_at, 'isoformat'):
                created_at = created_at.isoformat()
            elif not isinstance(created_at, str):
                created_at = str(created_at)
                
            if hasattr(updated_at, 'isoformat'):
                updated_at = updated_at.isoformat()
            elif not isinstance(updated_at, str):
                updated_at = str(updated_at)
            
            return {
                'id': row['id'],
                'user_id': row['user_id'],
                'provider': row['provider'],
                'key': key_display,  # Masked or decrypted based on decrypt flag
                'enabled': bool(row.get('enabled', True)),
                'created_at': created_at,
                'updated_at': updated_at,
            }
        
        except Exception as e:
            logger.error(f"Failed to get API key: {e}", exc_info=True)
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
                SELECT id, user_id, provider, encrypted_key, enabled, created_at, updated_at
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
                    'enabled': bool(row.get('enabled', True)),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                })
            
            return keys
        
        except Exception as e:
            logger.error(f"Failed to get API keys: {e}")
            return []
    
    @staticmethod
    def update_enabled_status(user_id: str, provider: str, enabled: bool) -> Dict[str, Any]:
        """
        Update the enabled status of an API key
        
        Args:
            user_id: User identifier
            provider: 'openai', 'anthropic', or 'gemini'
            enabled: Whether the key should be enabled
            
        Returns:
            dict: Result with success status
        """
        try:
            conn = get_connection()
            result = conn.execute(
                """
                UPDATE user_api_keys
                SET enabled = ?, updated_at = ?
                WHERE user_id = ? AND provider = ?
                """,
                [enabled, datetime.now(), user_id, provider]
            )
            
            if result.rowcount > 0:
                logger.info(f"Updated enabled status for user {user_id}, provider {provider}: {enabled}")
                return {
                    'success': True,
                    'message': f'API key enabled status updated successfully',
                }
            else:
                return {
                    'success': False,
                    'message': f'No API key found for {provider}',
                }
        
        except Exception as e:
            logger.error(f"Failed to update enabled status: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    @staticmethod
    def delete_api_key(user_id: str, provider: str) -> Dict[str, Any]:
        """
        Delete an API key for a user
        
        Args:
            user_id: User identifier
            provider: 'openai', 'anthropic', or 'gemini'
            
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

