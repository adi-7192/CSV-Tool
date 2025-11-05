import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const metricsService = {
  getMetrics: async (startDate?: string, endDate?: string) => {
    const params: Record<string, string> = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    
    const response = await apiClient.get('/api/metrics', { params });
    return response.data;
  },
};

export const chatService = {
  askQuestion: async (question: string) => {
    const response = await apiClient.post('/api/chat/ask', { question });
    return response.data;
  },
};

export const dataService = {
  getDataStatus: async () => {
    const response = await apiClient.get('/api/data/status');
    return response.data;
  },
};

export default apiClient;
