"""
Tests for API key encryption and management

Tests encryption/decryption, masking, and database operations.
"""

import pytest
import os
from datetime import datetime
from utils.encryption import APIKeyEncryption, EncryptionError, generate_encryption_key
from services.api_key_service import APIKeyService
from core.database import get_connection, execute_query, init_database


@pytest.fixture(scope="function")
def setup_encryption_key():
    """Set up encryption key for testing"""
    # Generate a test key
    test_key = generate_encryption_key()
    os.environ['API_KEY_ENCRYPTION_KEY'] = test_key
    yield test_key
    # Cleanup
    if 'API_KEY_ENCRYPTION_KEY' in os.environ:
        del os.environ['API_KEY_ENCRYPTION_KEY']


@pytest.fixture(scope="function")
def reset_encryption():
    """Reset encryption singleton between tests"""
    APIKeyEncryption._fernet = None


@pytest.fixture(scope="function")
def test_user_id():
    """Test user ID"""
    return "test-user-123"


class TestEncryption:
    """Test encryption utilities"""
    
    def test_generate_encryption_key(self):
        """Test that encryption key generation works"""
        key = generate_encryption_key()
        assert key is not None
        assert len(key) > 0
        assert isinstance(key, str)
    
    def test_encrypt_decrypt(self, setup_encryption_key, reset_encryption):
        """Test that encryption and decryption work correctly"""
        plaintext = "sk-test-api-key-12345"
        encrypted = APIKeyEncryption.encrypt(plaintext)
        
        # Encrypted should be different from plaintext
        assert encrypted != plaintext
        assert len(encrypted) > 0
        
        # Decrypt should return original
        decrypted = APIKeyEncryption.decrypt(encrypted)
        assert decrypted == plaintext
    
    def test_encrypt_empty_string(self, setup_encryption_key, reset_encryption):
        """Test that encrypting empty string raises error"""
        with pytest.raises(EncryptionError):
            APIKeyEncryption.encrypt("")
    
    def test_decrypt_empty_string(self, setup_encryption_key, reset_encryption):
        """Test that decrypting empty string raises error"""
        with pytest.raises(EncryptionError):
            APIKeyEncryption.decrypt("")
    
    def test_encrypt_without_key(self, reset_encryption):
        """Test that encryption fails without encryption key"""
        if 'API_KEY_ENCRYPTION_KEY' in os.environ:
            del os.environ['API_KEY_ENCRYPTION_KEY']
        
        with pytest.raises(EncryptionError) as exc_info:
            APIKeyEncryption.encrypt("test-key")
        
        assert "API_KEY_ENCRYPTION_KEY" in str(exc_info.value)
    
    def test_mask_key_short(self, setup_encryption_key, reset_encryption):
        """Test masking short keys"""
        short_key = "sk-123"
        masked = APIKeyEncryption.mask_key(short_key)
        assert masked == "****"
    
    def test_mask_key_medium(self, setup_encryption_key, reset_encryption):
        """Test masking medium keys"""
        medium_key = "sk-1234567890"
        masked = APIKeyEncryption.mask_key(medium_key)
        assert "..." in masked
        assert len(masked) < len(medium_key)
    
    def test_mask_key_long(self, setup_encryption_key, reset_encryption):
        """Test masking long keys (first 6, last 4)"""
        long_key = "sk-test-api-key-1234567890abcdef"
        masked = APIKeyEncryption.mask_key(long_key)
        
        # Should show first 6 and last 4 characters
        assert masked.startswith("sk-tes")
        assert masked.endswith("cdef")
        assert "..." in masked
        assert len(masked) < len(long_key)
    
    def test_mask_key_exact_10_chars(self, setup_encryption_key, reset_encryption):
        """Test masking key with exactly 10 characters"""
        key = "sk-1234567"
        masked = APIKeyEncryption.mask_key(key)
        # Should show first 2 and last 2
        assert masked.startswith("sk")
        assert masked.endswith("67")
        assert "..." in masked


class TestAPIKeyService:
    """Test API key service operations"""
    
    def test_create_api_key(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test creating an API key"""
        init_database()
        
        result = APIKeyService.create_api_key(
            user_id=test_user_id,
            provider="openai",
            api_key="sk-test-key-12345"
        )
        
        assert result['success'] is True
        assert result['user_id'] == test_user_id
        assert result['provider'] == "openai"
        assert result['masked_key'] == "sk-tes...2345"
        assert 'id' in result
        assert 'created_at' in result
        assert 'updated_at' in result
    
    def test_create_duplicate_key_updates(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test that creating duplicate key updates existing"""
        init_database()
        
        # Create first key
        result1 = APIKeyService.create_api_key(
            user_id=test_user_id,
            provider="openai",
            api_key="sk-first-key"
        )
        key_id_1 = result1['id']
        
        # Create second key (should update)
        result2 = APIKeyService.create_api_key(
            user_id=test_user_id,
            provider="openai",
            api_key="sk-second-key"
        )
        key_id_2 = result2['id']
        
        # Should have same ID (updated, not created new)
        assert key_id_1 == key_id_2
        assert result2['masked_key'] == "sk-sec...d-key"
    
    def test_get_api_key_masked(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test getting API key (masked)"""
        init_database()
        
        # Create key
        APIKeyService.create_api_key(
            user_id=test_user_id,
            provider="openai",
            api_key="sk-test-key-12345"
        )
        
        # Get key (masked)
        key_info = APIKeyService.get_api_key(test_user_id, "openai", decrypt=False)
        
        assert key_info is not None
        assert key_info['user_id'] == test_user_id
        assert key_info['provider'] == "openai"
        assert key_info['key'] == "sk-tes...2345"  # Masked
        assert len(key_info['key']) < 20  # Should be masked, not full key
    
    def test_get_api_key_decrypted(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test getting API key (decrypted - backend only)"""
        init_database()
        
        original_key = "sk-test-key-12345"
        
        # Create key
        APIKeyService.create_api_key(
            user_id=test_user_id,
            provider="openai",
            api_key=original_key
        )
        
        # Get key (decrypted)
        key_info = APIKeyService.get_api_key(test_user_id, "openai", decrypt=True)
        
        assert key_info is not None
        assert key_info['key'] == original_key  # Full key, not masked
    
    def test_get_nonexistent_key(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test getting non-existent key"""
        init_database()
        
        key_info = APIKeyService.get_api_key(test_user_id, "openai", decrypt=False)
        assert key_info is None
    
    def test_get_all_api_keys(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test getting all API keys for a user"""
        init_database()
        
        # Create multiple keys
        APIKeyService.create_api_key(test_user_id, "openai", "sk-openai-key")
        APIKeyService.create_api_key(test_user_id, "anthropic", "sk-ant-key-123")
        
        # Get all keys
        keys = APIKeyService.get_all_api_keys(test_user_id)
        
        assert len(keys) == 2
        assert all(key['masked_key'] != key['masked_key'].replace('...', '') for key in keys)  # All masked
        providers = [key['provider'] for key in keys]
        assert "openai" in providers
        assert "anthropic" in providers
    
    def test_delete_api_key(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test deleting an API key"""
        init_database()
        
        # Create key
        APIKeyService.create_api_key(test_user_id, "openai", "sk-test-key")
        
        # Delete key
        result = APIKeyService.delete_api_key(test_user_id, "openai")
        
        assert result['success'] is True
        
        # Verify deleted
        key_info = APIKeyService.get_api_key(test_user_id, "openai", decrypt=False)
        assert key_info is None
    
    def test_delete_nonexistent_key(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test deleting non-existent key"""
        init_database()
        
        result = APIKeyService.delete_api_key(test_user_id, "openai")
        assert result['success'] is False
    
    def test_verify_encryption(self, setup_encryption_key, reset_encryption):
        """Test encryption verification"""
        is_working = APIKeyService.verify_encryption()
        assert is_working is True
    
    def test_invalid_provider(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test that invalid provider raises error"""
        init_database()
        
        with pytest.raises(ValueError):
            APIKeyService.create_api_key(test_user_id, "invalid", "sk-key")
    
    def test_empty_api_key(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test that empty API key raises error"""
        init_database()
        
        with pytest.raises(ValueError):
            APIKeyService.create_api_key(test_user_id, "openai", "")


class TestDatabaseStorage:
    """Test that database stores encrypted blobs"""
    
    def test_database_stores_encrypted(self, setup_encryption_key, reset_encryption, test_user_id):
        """Verify that database only stores encrypted keys, never plaintext"""
        init_database()
        
        original_key = "sk-test-api-key-12345"
        
        # Create key
        APIKeyService.create_api_key(test_user_id, "openai", original_key)
        
        # Query database directly
        result = execute_query(
            f"SELECT encrypted_key FROM user_api_keys WHERE user_id = '{test_user_id}' AND provider = 'openai'"
        )
        
        assert not result.empty
        encrypted_key = result.iloc[0]['encrypted_key']
        
        # Encrypted key should NOT contain original key
        assert original_key not in encrypted_key
        assert encrypted_key != original_key
        
        # Encrypted key should be base64-like (Fernet format)
        assert len(encrypted_key) > 0
        # Fernet tokens are URL-safe base64, typically 100+ chars
        assert len(encrypted_key) >= 50
    
    def test_unique_constraint(self, setup_encryption_key, reset_encryption, test_user_id):
        """Test that unique constraint prevents duplicate user/provider combinations"""
        init_database()
        
        # Create first key
        APIKeyService.create_api_key(test_user_id, "openai", "sk-key-1")
        
        # Try to create another with same user/provider (should update, not fail)
        result = APIKeyService.create_api_key(test_user_id, "openai", "sk-key-2")
        
        # Should succeed (updates existing)
        assert result['success'] is True
        
        # Verify only one key exists
        keys = APIKeyService.get_all_api_keys(test_user_id)
        openai_keys = [k for k in keys if k['provider'] == 'openai']
        assert len(openai_keys) == 1

