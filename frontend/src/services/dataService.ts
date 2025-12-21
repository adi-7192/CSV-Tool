/**
 * Data Service - Raw transaction data API calls
 * 
 * Uses the shared apiClient from api.ts which includes JWT authentication interceptor.
 * All endpoints in this service require authentication.
 */
import { AxiosError } from 'axios';
import { apiClient } from './api';

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

export interface UploadFile {
  ingestion_id: string;
  filename: string;
  uploaded_at: string;
  rows_inserted: number;
  date_range_start: string | null;
  date_range_end: string | null;
  validation_status: string | null;
  file_id?: string; // Alias for ingestion_id for compatibility
  file_size?: number; // File size in bytes
  upload_timestamp?: string; // Alias for uploaded_at
  row_count?: number; // Alias for rows_inserted
}

export interface UploadDetailsResponse {
  ingestion_id: string;
  filename: string;
  uploaded_at: string;
  rows_inserted: number;
  date_range_start: string | null;
  date_range_end: string | null;
  validation_status: string | null;
  total_records?: number;
  unique_skus?: number;
  file_size?: number;
  upload_timestamp?: string;
  row_count?: number;
  date_range?: {
    start: string | null;
    end: string | null;
  };
  column_names?: string[];
}

export interface DataDateRangeResponse {
  has_data: boolean;
  start_date: string | null;
  end_date: string | null;
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
export interface DuplicateUploadError {
  error: 'DUPLICATE_UPLOAD';
  message: string;
  existing_ingestion_id: string;
  existing_filename: string;
  existing_uploaded_at: string;
  existing_rows: number;
}

export interface RequiredColumnsError {
  error: 'REQUIRED_COLUMNS_MISSING';
  message: string;
  missing_columns: string[];
  expected_schema: Record<string, string>;
  csv_columns: string[];
  detected_mapping: Record<string, string>;
}

export const uploadCSV = async (file: File): Promise<UploadResponse | DuplicateUploadError | RequiredColumnsError | null> => {
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
      const responseData = error.response?.data;
      
      // Handle duplicate upload (409)
      if (error.response?.status === 409 && responseData?.error === 'DUPLICATE_UPLOAD') {
        return {
          error: 'DUPLICATE_UPLOAD',
          message: responseData.message || 'This file has already been uploaded',
          existing_ingestion_id: responseData.existing_ingestion_id,
          existing_filename: responseData.existing_filename,
          existing_uploaded_at: responseData.existing_uploaded_at,
          existing_rows: responseData.existing_rows || 0,
        } as DuplicateUploadError;
      }
      
      // Handle required columns missing (400)
      if (error.response?.status === 400 && responseData?.error === 'REQUIRED_COLUMNS_MISSING') {
        return {
          error: 'REQUIRED_COLUMNS_MISSING',
          message: responseData.message || 'Required columns are missing',
          missing_columns: responseData.missing_columns || [],
          expected_schema: responseData.expected_schema || {},
          csv_columns: responseData.csv_columns || [],
          detected_mapping: responseData.detected_mapping || {},
        } as RequiredColumnsError;
      }
      
      console.error('Response:', responseData);
    }
    return null;
  }
};

export interface BatchUploadResult {
  filename: string;
  success: boolean;
  rows_inserted: number;
  ingestion_id?: string;
  message: string;
  error?: string;
  existing_ingestion_id?: string;
  existing_filename?: string;
  existing_uploaded_at?: string;
  existing_rows?: number;
  missing_columns?: string[];
  expected_schema?: Record<string, any>;
  csv_columns?: string[];
  detected_mapping?: Record<string, string>;
}

export interface BatchUploadResponse {
  results: BatchUploadResult[];
  total_files: number;
  successful: number;
  failed: number;
}

/**
 * Upload multiple CSV files in a batch
 * @param files - Array of File objects to upload
 * @returns Batch upload response with per-file results
 */
export const uploadMultipleCSV = async (files: File[]): Promise<BatchUploadResponse | null> => {
  try {
    const formData = new FormData();
    
    // Append all files to FormData
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    const response = await apiClient.post<BatchUploadResponse>('/api/data/upload/multiple', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error uploading multiple CSVs:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    return null;
  }
};

/**
 * Get data date range from statistics
 * @returns Date range response with has_data flag and start/end dates
 */
export const getDataDateRange = async (): Promise<DataDateRangeResponse | null> => {
  try {
    const stats = await getDataStatistics();
    if (!stats) {
      return {
        has_data: false,
        start_date: null,
        end_date: null,
      };
    }
    return {
      has_data: stats.total_records > 0,
      start_date: stats.date_range.start,
      end_date: stats.date_range.end,
    };
  } catch (error) {
    console.error('Error fetching date range:', error);
    return null;
  }
};

/**
 * Get list of uploaded files
 * @returns List of uploaded files with metadata
 */
export const getUploads = async (): Promise<UploadFile[] | null> => {
  try {
    const response = await apiClient.get<{ uploads: UploadFile[] }>('/api/upload/history');
    // Map ingestion_id to file_id for compatibility
    return response.data.uploads.map(upload => ({
      ...upload,
      file_id: upload.ingestion_id,
    }));
  } catch (error) {
    console.error('Error fetching uploads:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    return null;
  }
};

export interface DeleteUploadResponse {
  success: boolean;
  message: string;
  deleted_rows?: number;
}

/**
 * Delete a specific upload by ingestion ID
 * @param ingestionId - Ingestion ID to delete
 * @returns Delete response with success status and message
 */
export const deleteUpload = async (ingestionId: string): Promise<DeleteUploadResponse> => {
  try {
    const response = await apiClient.delete<DeleteUploadResponse>(`/api/upload/${ingestionId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting upload:', error);
    if (error instanceof AxiosError) {
      return {
        success: false,
        message: error.response?.data?.detail || 'Failed to delete upload',
      };
    }
    return {
      success: false,
      message: 'Failed to delete upload',
    };
  }
};

/**
 * Delete all uploads
 * @returns Delete response with success status and message
 */
export const deleteAllUploads = async (): Promise<DeleteUploadResponse> => {
  try {
    const response = await apiClient.delete<DeleteUploadResponse>('/api/upload/all');
    return response.data;
  } catch (error) {
    console.error('Error deleting all uploads:', error);
    if (error instanceof AxiosError) {
      return {
        success: false,
        message: error.response?.data?.detail || 'Failed to delete all uploads',
      };
    }
    return {
      success: false,
      message: 'Failed to delete all uploads',
    };
  }
};

/**
 * Download a file by ingestion ID
 * @param ingestionId - Ingestion ID of the file to download
 * @param filename - Optional filename for the download
 */
export const downloadFile = async (ingestionId: string, filename?: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/api/upload/download/${ingestionId}`, {
      responseType: 'blob',
    });
    
    // Create blob and download
    const blob = new Blob([response.data], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || `download_${ingestionId}.csv`;
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Error downloading file:', error);
    if (error instanceof AxiosError) {
      throw new Error(error.response?.data?.detail || 'Failed to download file');
    }
    throw error;
  }
};

/**
 * Get details for a specific upload
 * @param fileId - File/ingestion ID
 * @returns Upload details response
 */
export const getUploadDetails = async (fileId: string): Promise<UploadDetailsResponse | null> => {
  try {
    // Get all uploads and find the one matching fileId
    const uploads = await getUploads();
    if (!uploads) {
      return null;
    }
    
    const upload = uploads.find(u => u.ingestion_id === fileId || u.file_id === fileId);
    if (!upload) {
      return null;
    }
    
    // Get additional statistics if available
    const stats = await getDataStatistics();
    
    return {
      ingestion_id: upload.ingestion_id,
      filename: upload.filename,
      uploaded_at: upload.uploaded_at,
      rows_inserted: upload.rows_inserted,
      date_range_start: upload.date_range_start,
      date_range_end: upload.date_range_end,
      validation_status: upload.validation_status,
      total_records: stats?.total_records,
      unique_skus: stats?.unique_skus,
    };
  } catch (error) {
    console.error('Error fetching upload details:', error);
    if (error instanceof AxiosError) {
      console.error('Response:', error.response?.data);
    }
    return null;
  }
};

