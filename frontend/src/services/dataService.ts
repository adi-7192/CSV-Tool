/**
 * Data Service - Raw transaction data API calls
 */
import axios, { AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Transaction {
  order_id: string;
  sku: string;
  transaction_type: string;
  amount: number;
  date: string;
  quantity: number;
}

export interface TransactionsResponse {
  data: Transaction[];
  total: number;
  page: number;
  total_pages: number;
}

export interface SKUsResponse {
  skus: string[];
}

export interface DataStatisticsResponse {
  total_records: number;
  date_range: {
    start: string | null;
    end: string | null;
  };
  unique_skus: number;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  rows_inserted: number;
  filename: string;
  ingestion_id?: string;
}

/**
 * Get paginated transaction data with optional filters
 * @param page - Page number (1-indexed)
 * @param limit - Number of rows per page
 * @param dateFrom - Start date filter (YYYY-MM-DD)
 * @param dateTo - End date filter (YYYY-MM-DD)
 * @param sku - SKU filter (exact match)
 * @param transactionType - Transaction type filter
 * @returns Transaction data with pagination info
 */
export const getTransactions = async (
  page: number = 1,
  limit: number = 50,
  dateFrom?: string,
  dateTo?: string,
  sku?: string,
  transactionType?: string
): Promise<TransactionsResponse | null> => {
  try {
    const params: Record<string, any> = {
      page,
      limit,
    };
    
    if (dateFrom) params.date_from = dateFrom;
    if (dateTo) params.date_to = dateTo;
    if (sku) params.sku = sku;
    if (transactionType) params.transaction_type = transactionType;
    
    const response = await apiClient.get<TransactionsResponse>('/api/data/transactions', {
      params,
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching transactions:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    return null;
  }
};

/**
 * Get list of unique SKUs for autocomplete
 * @returns List of SKU strings
 */
export const getUniqueSKUs = async (): Promise<string[] | null> => {
  try {
    const response = await apiClient.get<SKUsResponse>('/api/data/skus');
    return response.data.skus;
  } catch (error) {
    console.error('Error fetching SKUs:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    return null;
  }
};

/**
 * Get data statistics
 * @returns Statistics including total records, date range, unique SKUs
 */
export const getDataStatistics = async (): Promise<DataStatisticsResponse | null> => {
  try {
    const response = await apiClient.get<DataStatisticsResponse>('/api/data/stats');
    return response.data;
  } catch (error) {
    console.error('Error fetching statistics:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    return null;
  }
};

/**
 * Export filtered transactions as CSV
 * @param dateFrom - Start date filter (YYYY-MM-DD)
 * @param dateTo - End date filter (YYYY-MM-DD)
 * @param sku - SKU filter (exact match)
 * @param transactionType - Transaction type filter
 */
export const downloadTransactionsCSV = async (
  dateFrom?: string,
  dateTo?: string,
  sku?: string,
  transactionType?: string
): Promise<void> => {
  try {
    const params: Record<string, any> = {};
    
    if (dateFrom) params.date_from = dateFrom;
    if (dateTo) params.date_to = dateTo;
    if (sku) params.sku = sku;
    if (transactionType) params.transaction_type = transactionType;
    
    const response = await apiClient.get('/api/data/export', {
      params,
      responseType: 'blob',
    });
    
    // Create blob and download
    const blob = new Blob([response.data], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    
    // Generate filename with current date
    const today = new Date().toISOString().split('T')[0];
    link.download = `transactions_export_${today}.csv`;
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error exporting CSV:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    throw error;
  }
};

/**
 * Upload CSV file
 * @param file - File object to upload
 * @returns Upload response with status and row count
 */
export const uploadCSV = async (file: File): Promise<UploadResponse | null> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post<UploadResponse>('/api/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error uploading CSV:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    return null;
  }
};

