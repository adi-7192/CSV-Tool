/**
 * API Key Service
 * 
 * Service for managing user LLM API keys (OpenAI, Anthropic)
 */

import apiClient from './api';

// Note: In production, get userId from authentication context
// For now, using a placeholder that should be replaced

export interface APIKeyInfo {
  id: string;
  user_id: string;
  provider: 'openai' | 'anthropic' | 'gemini';
  masked_key: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface APIKeyResponse {
  success: boolean;
  data: APIKeyInfo | null;
  message?: string;
}

export interface CreateAPIKeyRequest {
  provider: 'openai' | 'anthropic' | 'gemini';
  api_key: string;
}

class APIKeyService {
  /**
   * Get API key for a provider
   * Note: User ID is extracted from JWT token by backend, but we keep it for backward compatibility
   */
  async getAPIKey(provider: 'openai' | 'anthropic' | 'gemini', userId: string): Promise<APIKeyInfo | null> {
    try {
      // JWT token is automatically attached by apiClient interceptor
      const response = await apiClient.get<APIKeyResponse>(
        `/api/user/api-key?provider=${provider}`
      );
      
      if (response.data.success && response.data.data) {
        return response.data.data;
      }
      return null;
    } catch (error: any) {
      if (error.response?.status === 404) {
        return null; // Key not found
      }
      throw error;
    }
  }

  /**
   * Create or update API key
   * Simplified with shorter timeout (validation is now fast)
   * Note: User ID is extracted from JWT token by backend
   */
  async createAPIKey(
    provider: 'openai' | 'anthropic' | 'gemini',
    apiKey: string,
    userId: string
  ): Promise<APIKeyInfo> {
    // 15 second timeout should be plenty - validation takes ~1-2 seconds
    const timeout = 15000;
    
    // JWT token is automatically attached by apiClient interceptor
    const response = await apiClient.post<APIKeyResponse>(
      '/api/user/api-key',
      {
        provider,
        api_key: apiKey,
      },
      {
        timeout: timeout,
      }
    );

    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to save API key');
    }

    return response.data.data;
  }

  /**
   * Delete API key
   * Note: User ID is extracted from JWT token by backend
   */
  async deleteAPIKey(provider: 'openai' | 'anthropic' | 'gemini', userId: string): Promise<void> {
    // JWT token is automatically attached by apiClient interceptor
    await apiClient.delete(`/api/user/api-key?provider=${provider}`);
  }

  /**
   * Update enabled status of an API key
   * Note: User ID is extracted from JWT token by backend
   */
  async updateEnabledStatus(
    provider: 'openai' | 'anthropic' | 'gemini',
    enabled: boolean,
    userId: string
  ): Promise<APIKeyInfo> {
    // JWT token is automatically attached by apiClient interceptor
    const response = await apiClient.patch<APIKeyResponse>(
      '/api/user/api-key/enable',
      {
        provider,
        enabled,
      }
    );

    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to update enabled status');
    }

    return response.data.data;
  }
}

export const apiKeyService = new APIKeyService();

