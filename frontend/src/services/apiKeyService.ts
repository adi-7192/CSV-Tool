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
  provider: 'openai' | 'anthropic';
  masked_key: string;
  created_at: string;
  updated_at: string;
}

export interface APIKeyResponse {
  success: boolean;
  data: APIKeyInfo | null;
  message?: string;
}

export interface CreateAPIKeyRequest {
  provider: 'openai' | 'anthropic';
  api_key: string;
}

class APIKeyService {
  /**
   * Get API key for a provider
   */
  async getAPIKey(provider: 'openai' | 'anthropic', userId: string): Promise<APIKeyInfo | null> {
    try {
      const response = await apiClient.get<APIKeyResponse>(
        `/api/user/api-key?provider=${provider}`,
        {
          headers: {
            'X-User-ID': userId,
          },
        }
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
   */
  async createAPIKey(
    provider: 'openai' | 'anthropic',
    apiKey: string,
    userId: string
  ): Promise<APIKeyInfo> {
    const response = await apiClient.post<APIKeyResponse>(
      '/api/user/api-key',
      {
        provider,
        api_key: apiKey,
      },
      {
        headers: {
          'X-User-ID': userId,
        },
      }
    );

    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.message || 'Failed to save API key');
    }

    return response.data.data;
  }

  /**
   * Delete API key
   */
  async deleteAPIKey(provider: 'openai' | 'anthropic', userId: string): Promise<void> {
    await apiClient.delete(`/api/user/api-key?provider=${provider}`, {
      headers: {
        'X-User-ID': userId,
      },
    });
  }
}

export const apiKeyService = new APIKeyService();

