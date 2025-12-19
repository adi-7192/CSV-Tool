import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Table, Card, Pagination, Spin, Alert, DatePicker, Select, Button, Row, Col, Upload, Statistic, message, Modal, Collapse, List, Tag, Space } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { ReloadOutlined, DownloadOutlined, InboxOutlined, CheckCircleOutlined, CloseCircleOutlined, ClockCircleOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import dayjs, { Dayjs } from 'dayjs';
import { getTransactions, getUniqueSKUs, getDataStatistics, downloadTransactionsCSV, uploadCSV, uploadMultipleCSV, Transaction, DuplicateUploadError, RequiredColumnsError, BatchUploadResponse, BatchUploadResult } from '@/services/dataService';
import ColumnMappingModal from '@/components/ColumnMappingModal';
import BatchUploadProgress, { FileProgress } from '@/components/BatchUploadProgress';
import UploadSuccessModal from '@/components/UploadSuccessModal';
import { dataService } from '@/services/api';
import { formatCurrency } from '@/utils/formatters';
import { useAuthStore } from '@/store/authStore';
import { useDataStore } from '@/store/dataStore';

const { RangePicker } = DatePicker;
const { Dragger } = Upload;

const Workspace: React.FC = () => {
  const navigate = useNavigate();
  const { user, loading: authLoading } = useAuthStore();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [total, setTotal] = useState(0);
  // Note: totalPages is calculated but not currently used in UI
  const [, setTotalPages] = useState(0);
  const pageSize = 50;
  
  // Data availability state
  const [hasData, setHasData] = useState<boolean | null>(null);
  const [checkingData, setCheckingData] = useState(true);

  // Filter states
  const [dateRange, setLocalDateRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [selectedSKU, setSelectedSKU] = useState<string | undefined>(undefined);
  const [selectedTransactionType, setSelectedTransactionType] = useState<string | undefined>('all');
  const [skuOptions, setSkuOptions] = useState<string[]>([]);
  const [loadingSKUs, setLoadingSKUs] = useState(false);
  
  // Statistics state
  const [statistics, setStatistics] = useState<{
    total_records: number;
    date_range: { start: string | null; end: string | null };
    unique_skus: number;
  } | null>(null);
  const [loadingStats, setLoadingStats] = useState(false);
  
  // Upload state
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    success: boolean;
    message: string;
    rowsInserted?: number;
  } | null>(null);
  const [duplicateUpload, setDuplicateUpload] = useState<DuplicateUploadError | null>(null);
  const [requiredColumnsError, setRequiredColumnsError] = useState<RequiredColumnsError | null>(null);
  const [columnMappingModalVisible, setColumnMappingModalVisible] = useState(false);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [detectedMapping, setDetectedMapping] = useState<Record<string, string>>({});
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [uploadRowErrors, setUploadRowErrors] = useState<Array<{row: number; column: string; reason: string}>>([]);
  
  // Batch upload state
  const [isBatchUploading, setIsBatchUploading] = useState(false);
  const [batchFiles, setBatchFiles] = useState<FileProgress[]>([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [batchCancelled, setBatchCancelled] = useState(false);
  const [batchSummaryVisible, setBatchSummaryVisible] = useState(false);
  const [batchSummary, setBatchSummary] = useState<{successful: number; failed: number; totalRows: number} | null>(null);
  const [pendingMappingFile, setPendingMappingFile] = useState<{file: File; index: number} | null>(null);
  
  // Upload success modal state
  const [uploadSuccessVisible, setUploadSuccessVisible] = useState(false);
  const [uploadSuccessData, setUploadSuccessData] = useState<{filesUploaded: number; totalRows: number} | null>(null);
  
  // Data store for date range (global date range for dashboard)
  const { setDateRange: setGlobalDateRange } = useDataStore();

  // Check data availability on mount (only when authenticated)
  useEffect(() => {
    if (authLoading || !user) {
      return; // Wait for auth to be ready
    }
    
    const checkDataAvailability = async () => {
      setCheckingData(true);
      try {
        const summary = await dataService.getSummary();
        setHasData(summary.has_data);
      } catch (err) {
        console.error('Error checking data availability:', err);
        setHasData(false);
      } finally {
        setCheckingData(false);
      }
    };
    checkDataAvailability();
  }, [user, authLoading]);

  // Load SKU options on mount (only when authenticated)
  useEffect(() => {
    if (authLoading || !user) {
      return; // Wait for auth to be ready
    }
    
    const fetchSKUs = async () => {
      setLoadingSKUs(true);
      const skus = await getUniqueSKUs();
      if (skus) {
        setSkuOptions(skus);
      }
      setLoadingSKUs(false);
    };
    fetchSKUs();
  }, [user, authLoading]);

  // Load statistics on mount (only when authenticated)
  useEffect(() => {
    if (authLoading || !user) {
      return; // Wait for auth to be ready
    }
    
    const fetchStats = async () => {
      setLoadingStats(true);
      const stats = await getDataStatistics();
      if (stats) {
        setStatistics(stats);
      }
      setLoadingStats(false);
    };
    fetchStats();
  }, [user, authLoading]);

  // Fetch transactions when filters or page changes (only when authenticated)
  useEffect(() => {
    if (authLoading || !user) {
      return; // Wait for auth to be ready
    }
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      const result = await getTransactions(
        currentPage,
        pageSize,
        dateRange && dateRange[0] ? dateRange[0].format('YYYY-MM-DD') : undefined,
        dateRange && dateRange[1] ? dateRange[1].format('YYYY-MM-DD') : undefined,
        selectedSKU,
        selectedTransactionType === 'all' ? undefined : selectedTransactionType
      );
      
      if (result) {
        setTransactions(result.data);
        setTotal(result.total);
        setTotalPages(result.total_pages);
      } else {
        setError('Failed to load transaction data');
      }
      
      setLoading(false);
    };

    fetchData();
  }, [currentPage, dateRange, selectedSKU, selectedTransactionType, user, authLoading]);

  // Handle page change
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  // Handle filter changes (reset to page 1)
  const handleDateRangeChange = (dates: [Dayjs | null, Dayjs | null] | null) => {
    setLocalDateRange(dates);
    setCurrentPage(1);
  };

  const handleSKUChange = (value: string | undefined) => {
    setSelectedSKU(value);
    setCurrentPage(1);
  };

  const handleTransactionTypeChange = (value: string | undefined) => {
    // If "All Transactions" is selected, set to undefined to show all types
    if (value === 'all') {
      setSelectedTransactionType(undefined);
    } else {
      setSelectedTransactionType(value);
    }
    setCurrentPage(1);
  };

  // Clear all filters
  const handleClearFilters = () => {
    setLocalDateRange(null);
    setSelectedSKU(undefined);
    setSelectedTransactionType('all');
    setCurrentPage(1);
  };

  // Handle export CSV
  const handleExportCSV = async () => {
    try {
      await downloadTransactionsCSV(
        dateRange && dateRange[0] ? dateRange[0].format('YYYY-MM-DD') : undefined,
        dateRange && dateRange[1] ? dateRange[1].format('YYYY-MM-DD') : undefined,
        selectedSKU,
        selectedTransactionType === 'all' ? undefined : selectedTransactionType
      );
      message.success('CSV export started');
    } catch (error) {
      message.error('Failed to export CSV');
    }
  };

  // Handle file upload - auto-upload by default, show mapping only if needed
  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setUploadStatus(null);
    setUploadRowErrors([]); // Clear previous row errors
    setPendingFile(file);
    
    try {
      const result = await uploadCSV(file);
      
      // Extract row_errors from response if present (both success and error cases)
      const rowErrors = (result && 'row_errors' in result && Array.isArray(result.row_errors)) 
        ? result.row_errors.slice(0, 20) // Limit to 20 as per backend
        : [];
      setUploadRowErrors(rowErrors);
      
      // Handle duplicate upload
      if (result && 'error' in result && result.error === 'DUPLICATE_UPLOAD') {
        setDuplicateUpload(result as DuplicateUploadError);
        setUploading(false);
        return;
      }
      
      // Handle required columns missing - show mapping modal
      if (result && 'error' in result && result.error === 'REQUIRED_COLUMNS_MISSING') {
        const errorData = result as RequiredColumnsError;
        // Read CSV to get columns for mapping modal
        const reader = new FileReader();
        reader.onload = (e) => {
          const text = e.target?.result as string;
          const lines = text.split('\n');
          if (lines.length > 0) {
            const detectedCols = lines[0].split(',').map((col) => col.trim().replace(/"/g, ''));
            setCsvColumns(detectedCols);
            // Use backend's detected mapping as starting point
            setDetectedMapping(errorData.detected_mapping || {});
            setColumnMappingModalVisible(true);
          }
        };
        reader.readAsText(file);
        setUploading(false);
        return;
      }
      
      // Success
      if (result && 'success' in result && result.success) {
        setUploadStatus({
          success: true,
          message: result.message,
          rowsInserted: result.rows_inserted,
        });
        
        // Refresh statistics to get date range
        const fetchStats = async () => {
          const stats = await getDataStatistics();
          if (stats) {
            setStatistics(stats);
            
            // Auto-detect and set date range from uploaded data
            if (stats.date_range.start && stats.date_range.end) {
              setGlobalDateRange(stats.date_range.start, stats.date_range.end);
            }
          }
        };
        
        // Refresh data after upload
        const fetchData = async () => {
          setLoading(true);
          const result = await getTransactions(
            currentPage,
            pageSize,
            dateRange && dateRange[0] ? dateRange[0].format('YYYY-MM-DD') : undefined,
            dateRange && dateRange[1] ? dateRange[1].format('YYYY-MM-DD') : undefined,
            selectedSKU,
            selectedTransactionType
          );
          
          if (result) {
            setTransactions(result.data);
            setTotal(result.total);
            setTotalPages(result.total_pages);
          }
          setLoading(false);
        };
        
        // Update hasData state
        setHasData(true);
        
        // Show success modal immediately
        setUploadSuccessData({
          filesUploaded: 1,
          totalRows: result.rows_inserted || 0,
        });
        setUploadSuccessVisible(true);
        
        // Refresh data in background (non-blocking)
        Promise.all([fetchData(), fetchStats()]);
      } else {
        const errorMsg = (result && 'message' in result) ? result.message : 'Upload failed. Please try again.';
        setUploadStatus({
          success: false,
          message: errorMsg,
        });
        message.error(errorMsg);
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      let errorMessage = 'Upload failed. ';
      
      // Extract row_errors from error response if present
      const responseData = error.response?.data;
      const rowErrors = (responseData && 'row_errors' in responseData && Array.isArray(responseData.row_errors))
        ? responseData.row_errors.slice(0, 20) // Limit to 20 as per backend
        : [];
      setUploadRowErrors(rowErrors);
      
      if (error.code === 'ERR_NETWORK' || !error.response) {
        errorMessage += 'Unable to connect to server. Please check if the backend is running.';
      } else if (error.response?.status === 401) {
        errorMessage += 'Authentication failed. Please log in again.';
      } else if (error.response?.status === 413) {
        errorMessage += 'File too large. Maximum size is 100MB.';
      } else if (error.response?.status === 400) {
        // Check if it's a mapping error
        if (responseData?.error === 'REQUIRED_COLUMNS_MISSING') {
          // Handle mapping error - show modal
          const errorData: RequiredColumnsError = {
            error: 'REQUIRED_COLUMNS_MISSING',
            message: responseData.message || 'Required columns are missing',
            missing_columns: responseData.missing_columns || [],
            expected_schema: responseData.expected_schema || {},
            csv_columns: responseData.csv_columns || [],
            detected_mapping: responseData.detected_mapping || {},
          };
          setRequiredColumnsError(errorData);
          // Read CSV to get columns for mapping modal
          const reader = new FileReader();
          reader.onload = (e) => {
            const text = e.target?.result as string;
            const lines = text.split('\n');
            if (lines.length > 0) {
              const detectedCols = lines[0].split(',').map((col) => col.trim().replace(/"/g, ''));
              setCsvColumns(detectedCols);
              setDetectedMapping(errorData.detected_mapping || {});
              setColumnMappingModalVisible(true);
            }
          };
          reader.readAsText(file);
          setUploading(false);
          return;
        }
        errorMessage += responseData?.detail || 'Invalid file format.';
      } else if (error.response?.status === 500) {
        errorMessage += 'Server error. Please try again later.';
      } else {
        errorMessage += error.response?.data?.detail || error.message || 'Unknown error occurred.';
      }
      
      setUploadStatus({
        success: false,
        message: errorMessage,
      });
      message.error(errorMessage);
    } finally {
      setUploading(false);
      setPendingFile(null);
    }
  };

  // Handle column mapping confirmation (when mapping modal is shown)
  const handleColumnMappingConfirm = async (_mapping: Record<string, string>) => {
    // Check if this is a batch upload mapping or single file mapping
    if (pendingMappingFile) {
      await handleBatchColumnMappingConfirm(_mapping);
      return;
    }
    
    if (!pendingFile) return;
    // Note: mapping parameter is reserved for future use when backend accepts custom mappings
    // For now, we re-upload with the same file and let backend auto-detect again
    // In future, we can pass custom mapping to backend
    
    setColumnMappingModalVisible(false);
    setRequiredColumnsError(null);
    
    // Re-upload the file (backend will auto-detect again)
    // TODO: In future, pass custom mapping to backend
    await handleFileUpload(pendingFile);
  };

  // Process a single file in batch context
  const processBatchFile = async (file: File, index: number, allFiles: File[]): Promise<{shouldPause: boolean; rowsInserted?: number; success?: boolean}> => {
    setCurrentFileIndex(index);
    
    // Update status to processing
    setBatchFiles((prev) => {
      const updated = [...prev];
      updated[index] = { ...updated[index], status: 'processing' };
      return updated;
    });
    
    try {
      const result = await uploadCSV(file);
      
      if (result && 'error' in result && result.error === 'REQUIRED_COLUMNS_MISSING') {
        // Pause batch for column mapping
        const errorData = result as RequiredColumnsError;
        setPendingMappingFile({ file, index });
        
        // Read CSV to get columns for mapping modal
        const reader = new FileReader();
        reader.onload = (e) => {
          const text = e.target?.result as string;
          const lines = text.split('\n');
          if (lines.length > 0) {
            const detectedCols = lines[0].split(',').map((col) => col.trim().replace(/"/g, ''));
            setCsvColumns(detectedCols);
            setDetectedMapping(errorData.detected_mapping || {});
            setRequiredColumnsError(errorData);
            setColumnMappingModalVisible(true);
          }
        };
        reader.readAsText(file);
        
        return { shouldPause: true }; // Signal to pause batch
      }
      
      if (result && 'error' in result && result.error === 'DUPLICATE_UPLOAD') {
        // Mark as skipped (don't count as success or failure)
        setBatchFiles((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            status: 'skipped',
            result: {
              message: result.message || 'File already uploaded',
              error: 'DUPLICATE_UPLOAD',
            },
          };
          return updated;
        });
        return { shouldPause: false, success: false };
      }
      
      if (result && 'success' in result && result.success) {
        // Success
        const rowsInserted = result.rows_inserted || 0;
        
        setBatchFiles((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            status: 'success',
            result: {
              rows_inserted: rowsInserted,
              message: result.message || 'Upload successful',
            },
          };
          return updated;
        });
        return { shouldPause: false, rowsInserted, success: true };
      } else {
        // Failed
        setBatchFiles((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            status: 'error',
            result: {
              message: (result && 'message' in result) ? result.message : 'Upload failed',
              error: 'PROCESSING_ERROR',
            },
          };
          return updated;
        });
        return { shouldPause: false, success: false };
      }
    } catch (error: any) {
      // Error occurred
      setBatchFiles((prev) => {
        const updated = [...prev];
        updated[index] = {
          ...updated[index],
          status: 'error',
          result: {
            message: error.message || 'Upload failed',
            error: 'EXCEPTION',
          },
        };
        return updated;
      });
      return { shouldPause: false, success: false };
    }
  };

  // Handle batch file upload
  const handleBatchUpload = async (files: File[], startIndex: number = 0) => {
    if (files.length === 0) return;
    
    // If single file, use existing single upload flow
    if (files.length === 1 && startIndex === 0) {
      await handleFileUpload(files[0]);
      return;
    }
    
    // Initialize batch upload state (only on first call)
    if (startIndex === 0) {
      const fileProgress: FileProgress[] = files.map((file) => ({
        file,
        status: 'pending' as const,
      }));
      
      setBatchFiles(fileProgress);
      setIsBatchUploading(true);
      setBatchCancelled(false);
      setCurrentFileIndex(0);
      setBatchSummary(null);
      setUploadSuccessVisible(false); // Reset success modal
    }
    
    // Process files sequentially and track results
    let successfulCount = 0;
    let failedCount = 0;
    let totalRowsCount = 0;
    
    for (let i = startIndex; i < files.length; i++) {
      if (batchCancelled) {
        // Mark remaining files as skipped
        for (let j = i; j < files.length; j++) {
          setBatchFiles((prev) => {
            const updated = [...prev];
            updated[j] = { ...updated[j], status: 'skipped' };
            return updated;
          });
        }
        break;
      }
      
      const file = files[i];
      const result = await processBatchFile(file, i, files);
      
      // If column mapping needed, pause and return
      if (result.shouldPause) {
        return; // Will resume from handleBatchColumnMappingConfirm
      }
      
      // Track results directly from the processBatchFile return value
      if (result.success && result.rowsInserted !== undefined) {
        successfulCount++;
        totalRowsCount += result.rowsInserted;
      }
      // Note: We don't track failures here because skipped files also return success: false
      // We'll count failures from the final state
    }
    
    // Wait a tiny bit for all state updates to complete, then calculate final counts from state
    // Use a promise to ensure we read the state after React has processed all updates
    await new Promise<void>((resolve) => {
      setTimeout(() => {
        // Read current batchFiles state to get accurate final counts
        // We need to use a callback approach since we can't directly read state
        setBatchFiles((prev) => {
          const finalSuccessful = prev.filter((f) => f.status === 'success').length;
          const finalFailed = prev.filter((f) => f.status === 'error').length;
          const finalTotalRows = prev.reduce((sum, f) => sum + (f.result?.rows_inserted || 0), 0);
          
          // Update our local counters with the accurate final counts
          successfulCount = finalSuccessful;
          totalRowsCount = finalTotalRows;
          failedCount = finalFailed;
          
          resolve();
          return prev;
        });
      }, 50);
    });
    
    // Now show the modal with the accurate counts
    setIsBatchUploading(false);
    
    if (successfulCount > 0) {
      setHasData(true);
      
      // Show success modal immediately with correct counts
      setUploadSuccessData({
        filesUploaded: successfulCount,
        totalRows: totalRowsCount,
      });
      setUploadSuccessVisible(true);
      
      // Also show batch summary for detailed view (non-blocking)
      setBatchSummary({
        successful: successfulCount,
        failed: failedCount,
        totalRows: totalRowsCount,
      });
      setBatchSummaryVisible(true);
      
      // Refresh data in background (non-blocking)
      const fetchData = async () => {
        setLoading(true);
        const result = await getTransactions(
          currentPage,
          pageSize,
          dateRange && dateRange[0] ? dateRange[0].format('YYYY-MM-DD') : undefined,
          dateRange && dateRange[1] ? dateRange[1].format('YYYY-MM-DD') : undefined,
          selectedSKU,
          selectedTransactionType
        );
        
        if (result) {
          setTransactions(result.data);
          setTotal(result.total);
          setTotalPages(result.total_pages);
        }
        setLoading(false);
      };
      
      const fetchStats = async () => {
        const stats = await getDataStatistics();
        if (stats) {
          setStatistics(stats);
          
          // Auto-detect and set date range from uploaded data
          if (stats.date_range.start && stats.date_range.end) {
            setGlobalDateRange(stats.date_range.start, stats.date_range.end);
          }
        }
      };
      
      Promise.all([fetchData(), fetchStats()]);
    } else {
      // No successful uploads - just show summary
      setBatchSummary({
        successful: successfulCount,
        failed: failedCount,
        totalRows: totalRowsCount,
      });
      setBatchSummaryVisible(true);
    }
  };

  // Handle column mapping confirmation during batch upload
  const handleBatchColumnMappingConfirm = async (mapping: Record<string, string>) => {
    if (!pendingMappingFile) return;
    
    setColumnMappingModalVisible(false);
    setRequiredColumnsError(null);
    
    const { file, index } = pendingMappingFile;
    setPendingMappingFile(null);
    
    // Note: For now, we re-upload with auto-detection
    // TODO: In future, pass custom mapping to backend
    
    try {
      const result = await uploadCSV(file);
      
      if (result && 'success' in result && result.success) {
        const rowsInserted = result.rows_inserted || 0;
        setBatchFiles((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            status: 'success',
            result: {
              rows_inserted: rowsInserted,
              message: result.message || 'Upload successful',
            },
          };
          return updated;
        });
      } else {
        setBatchFiles((prev) => {
          const updated = [...prev];
          updated[index] = {
            ...updated[index],
            status: 'error',
            result: {
              message: (result && 'message' in result) ? result.message : 'Upload failed',
              error: 'PROCESSING_ERROR',
            },
          };
          return updated;
        });
      }
      
      // Resume batch processing from next file
      const allFiles = batchFiles.map((fp) => fp.file);
      await handleBatchUpload(allFiles, index + 1);
    } catch (error: any) {
      setBatchFiles((prev) => {
        const updated = [...prev];
        updated[index] = {
          ...updated[index],
          status: 'error',
          result: {
            message: error.message || 'Upload failed',
            error: 'EXCEPTION',
          },
        };
        return updated;
      });
      
      // Still resume batch processing from next file
      const allFiles = batchFiles.map((fp) => fp.file);
      await handleBatchUpload(allFiles, index + 1);
    }
  };

  // Upload props - support both single and multiple files
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: true, // Enable multiple file selection
    accept: '.csv',
    beforeUpload: () => {
      // Prevent default upload (we handle it manually via onChange)
      return false;
    },
    onChange: (info) => {
      // When files are selected, process them
      const fileList = info.fileList;
      
      // Get all files that are ready (not in error state)
      const readyFiles = fileList
        .filter((f) => f.status !== 'error' && f.originFileObj)
        .map((f) => f.originFileObj as File);
      
      if (readyFiles.length === 0) return;
      
      // If only one file, use single upload flow
      if (readyFiles.length === 1) {
        handleFileUpload(readyFiles[0]);
      } else {
        // Multiple files - use batch upload
        handleBatchUpload(readyFiles);
      }
    },
    showUploadList: false,
  };

  // Table columns definition
  const columns: ColumnsType<Transaction> = [
    {
      title: 'Order ID',
      dataIndex: 'order_id',
      key: 'order_id',
      width: 150,
      render: (text: string, record: Transaction) => {
        // Display order_id value directly from API response, no transformation
        const orderId = record.order_id || text || '';
        return (
          <span style={{ fontFamily: 'monospace', fontSize: '13px' }}>{orderId}</span>
        );
      },
    },
    {
      title: 'SKU',
      dataIndex: 'sku',
      key: 'sku',
      width: 180,
      render: (text: string) => (
        <span style={{ fontWeight: '500', color: '#030712' }}>{text}</span>
      ),
    },
    {
      title: 'Transaction Type',
      dataIndex: 'transaction_type',
      key: 'transaction_type',
      width: 140,
      render: (type: string) => {
        const colorMap: Record<string, string> = {
          Shipment: '#10B981',
          Refund: '#F43F5E',
          Cancel: '#F59E0B',
          FreeReplacement: '#6366F1',
        };
        const color = colorMap[type] || '#64748B';
        return (
          <span
            style={{
              color,
              fontWeight: '600',
              fontSize: '13px',
            }}
          >
            {type}
          </span>
        );
      },
    },
    {
      title: 'Amount',
      dataIndex: 'amount',
      key: 'amount',
      width: 130,
      align: 'right',
      render: (amount: number) => formatCurrency(amount),
      sorter: (a, b) => a.amount - b.amount,
    },
    {
      title: 'Date',
      dataIndex: 'date',
      key: 'date',
      width: 120,
      render: (date: string) => {
        if (!date) return '-';
        try {
          const d = new Date(date);
          return d.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
          });
        } catch {
          return date;
        }
      },
    },
    {
      title: 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 100,
      align: 'right',
      render: (quantity: number) => quantity.toLocaleString(),
    },
  ];

  // Show loading state while checking auth or data availability
  if (authLoading || checkingData) {
    return (
      <div style={{ padding: '24px', backgroundColor: '#FFFFFF', minHeight: '100vh' }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          <Spin size="large" />
        </div>
      </div>
    );
  }
  
  // If not authenticated, show nothing (ProtectedRoute will handle redirect)
  if (!user) {
    return null;
  }

  // Show prominent empty state when no data exists
  if (hasData === false) {
    return (
      <div style={{ padding: '24px', backgroundColor: '#FFFFFF', minHeight: '100vh' }}>
        <div style={{ marginBottom: '24px' }}>
          <h1
            style={{
              fontSize: '28px',
              fontWeight: '700',
              color: '#030712',
              marginBottom: '8px',
            }}
          >
            Data Workspace
          </h1>
          <p style={{ color: '#64748B', fontSize: '14px' }}>
            Upload your CSV data to get started with analytics
          </p>
        </div>

        {/* Prominent Empty State with Upload */}
        <Card
          style={{
            marginBottom: '24px',
            borderRadius: '12px',
            border: '2px dashed #6366F1',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          }}
        >
          <div style={{ textAlign: 'center', padding: '48px 24px' }}>
            <div style={{ fontSize: '64px', marginBottom: '24px' }}>📊</div>
            <h2 style={{ fontSize: '24px', fontWeight: '600', marginBottom: '12px', color: '#030712' }}>
              No data yet
            </h2>
            <p style={{ fontSize: '16px', color: '#64748B', marginBottom: '32px', maxWidth: '500px', margin: '0 auto 32px' }}>
              Upload a CSV file to generate your dashboard and chat insights. Your data is private and isolated to your account.
            </p>
            
            <Dragger {...uploadProps} disabled={uploading} style={{ maxWidth: '500px', margin: '0 auto' }}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined style={{ fontSize: '48px', color: '#6366F1' }} />
              </p>
              <p className="ant-upload-text" style={{ color: '#030712', fontWeight: '500' }}>
                Click or drag CSV file to upload
              </p>
              <p className="ant-upload-hint" style={{ color: '#64748B' }}>
                Support for CSV files up to 100MB. Column mapping is automatic.
              </p>
            </Dragger>
            
            {/* Advanced: Review/Change Mapping (collapsed by default) */}
            <div style={{ marginTop: '16px', maxWidth: '500px', margin: '16px auto 0' }}>
              <Collapse
                ghost
                items={[
                  {
                    key: '1',
                    label: (
                      <span style={{ color: '#64748B', fontSize: '14px' }}>
                        Advanced: Review/Change Column Mapping
                      </span>
                    ),
                    children: (
                      <div style={{ padding: '8px 0' }}>
                        <p style={{ color: '#64748B', fontSize: '13px', marginBottom: '12px' }}>
                          By default, columns are automatically detected and mapped. If your CSV has non-standard column names, you can manually review and change the mapping before uploading.
                        </p>
                        <Button
                          type="link"
                          size="small"
                          onClick={() => {
                            const input = document.createElement('input');
                            input.type = 'file';
                            input.accept = '.csv';
                            input.onchange = (e) => {
                              const file = (e.target as HTMLInputElement).files?.[0];
                              if (file) {
                                const reader = new FileReader();
                                reader.onload = (event) => {
                                  const text = event.target?.result as string;
                                  const lines = text.split('\n');
                                  if (lines.length > 0) {
                                    const detectedCols = lines[0].split(',').map((col) => col.trim().replace(/"/g, ''));
                                    setCsvColumns(detectedCols);
                                    setDetectedMapping({});
                                    setPendingFile(file);
                                    setColumnMappingModalVisible(true);
                                  }
                                };
                                reader.readAsText(file);
                              }
                            };
                            input.click();
                          }}
                        >
                          Select CSV to Review Mapping
                        </Button>
                      </div>
                    ),
                  },
                ]}
              />
            </div>
            
            {uploading && (
              <div style={{ marginTop: '16px', textAlign: 'center' }}>
                <Spin /> <span style={{ marginLeft: '8px', color: '#64748B' }}>Uploading...</span>
              </div>
            )}
            
            {uploadStatus && (
              <Alert
                message={uploadStatus.success ? 'Upload Successful' : 'Upload Failed'}
                description={
                  uploadStatus.success
                    ? `${uploadStatus.message}${uploadStatus.rowsInserted ? ` (${uploadStatus.rowsInserted.toLocaleString()} rows)` : ''}`
                    : uploadStatus.message
                }
                type={uploadStatus.success ? 'success' : 'error'}
                showIcon
                icon={uploadStatus.success ? <CheckCircleOutlined /> : undefined}
                closable
                onClose={() => {
                  setUploadStatus(null);
                  setUploadRowErrors([]);
                }}
                style={{ marginTop: '16px', maxWidth: '500px', margin: '16px auto 0' }}
              />
            )}
            
            {/* Row-level errors display */}
            {uploadRowErrors.length > 0 && (
              <Card
                style={{
                  marginTop: '16px',
                  maxWidth: '500px',
                  margin: '16px auto 0',
                  border: '1px solid #FEE2E2',
                  backgroundColor: '#FEF2F2',
                }}
              >
                <div style={{ marginBottom: '12px' }}>
                  <h4 style={{ margin: 0, color: '#991B1B', fontSize: '16px', fontWeight: '600' }}>
                    Row-level issues found
                  </h4>
                  <p style={{ margin: '4px 0 0 0', color: '#7F1D1D', fontSize: '13px' }}>
                    Fix these rows and re-upload.
                  </p>
                </div>
                <Table
                  dataSource={uploadRowErrors.map((err, idx) => ({ ...err, key: idx }))}
                  columns={[
                    {
                      title: 'Row',
                      dataIndex: 'row',
                      key: 'row',
                      width: 80,
                      render: (row: number) => <strong>{row}</strong>,
                    },
                    {
                      title: 'Column',
                      dataIndex: 'column',
                      key: 'column',
                      width: 120,
                    },
                    {
                      title: 'Reason',
                      dataIndex: 'reason',
                      key: 'reason',
                      ellipsis: true,
                    },
                  ]}
                  pagination={false}
                  size="small"
                  scroll={{ y: 200 }}
                  style={{ fontSize: '13px' }}
                />
              </Card>
            )}
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px', backgroundColor: '#FFFFFF', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1
          style={{
            fontSize: '28px',
            fontWeight: '700',
            color: '#030712',
            marginBottom: '8px',
          }}
        >
          Data Workspace
        </h1>
        <p style={{ color: '#64748B', fontSize: '14px' }}>
          View and explore raw transaction data from your sales database
        </p>
      </div>

      {error && (
        <Alert
          message="Error Loading Data"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* Upload Section */}
      <Card
        style={{
          marginBottom: '24px',
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div style={{ marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: '#030712' }}>
          Upload CSV File
        </div>
        <Dragger {...uploadProps} disabled={uploading}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: '48px', color: '#6366F1' }} />
          </p>
          <p className="ant-upload-text" style={{ color: '#030712', fontWeight: '500' }}>
            Click or drag CSV file(s) to this area to upload
          </p>
          <p className="ant-upload-hint" style={{ color: '#64748B' }}>
            Support for CSV files. You can upload multiple files at once. Column mapping is automatic.
          </p>
        </Dragger>
        
        {/* Advanced: Review/Change Mapping (collapsed by default) */}
        <Collapse
          ghost
          style={{ marginTop: '8px' }}
          items={[
            {
              key: '1',
              label: (
                <span style={{ color: '#64748B', fontSize: '13px' }}>
                  Advanced: Review/Change Column Mapping
                </span>
              ),
              children: (
                <div style={{ padding: '8px 0' }}>
                  <p style={{ color: '#64748B', fontSize: '12px', marginBottom: '12px' }}>
                    By default, columns are automatically detected. If your CSV has non-standard column names, you can manually review and change the mapping before uploading.
                  </p>
                  <Button
                    type="link"
                    size="small"
                    onClick={() => {
                      const input = document.createElement('input');
                      input.type = 'file';
                      input.accept = '.csv';
                      input.onchange = (e) => {
                        const file = (e.target as HTMLInputElement).files?.[0];
                        if (file) {
                          const reader = new FileReader();
                          reader.onload = (event) => {
                            const text = event.target?.result as string;
                            const lines = text.split('\n');
                            if (lines.length > 0) {
                              const detectedCols = lines[0].split(',').map((col) => col.trim().replace(/"/g, ''));
                              setCsvColumns(detectedCols);
                              setDetectedMapping({});
                              setPendingFile(file);
                              setColumnMappingModalVisible(true);
                            }
                          };
                          reader.readAsText(file);
                        }
                      };
                      input.click();
                    }}
                  >
                    Select CSV to Review Mapping
                  </Button>
                </div>
              ),
            },
          ]}
        />
        
        {uploading && !isBatchUploading && (
          <div style={{ marginTop: '16px', textAlign: 'center' }}>
            <Spin /> <span style={{ marginLeft: '8px', color: '#64748B' }}>Uploading...</span>
          </div>
        )}
        
        {isBatchUploading && batchFiles.length > 0 && (
          <BatchUploadProgress
            files={batchFiles}
            currentIndex={currentFileIndex}
            onCancel={() => {
              setBatchCancelled(true);
              setIsBatchUploading(false);
              message.info('Batch upload cancelled');
            }}
            totalRowsInserted={batchFiles.reduce((sum, f) => sum + (f.result?.rows_inserted || 0), 0)}
          />
        )}
        
        {uploadStatus && !isBatchUploading && (
          <Alert
            message={uploadStatus.success ? 'Upload Successful' : 'Upload Failed'}
            description={
              uploadStatus.success
                ? `${uploadStatus.message}${uploadStatus.rowsInserted ? ` (${uploadStatus.rowsInserted.toLocaleString()} rows)` : ''}`
                : uploadStatus.message
            }
            type={uploadStatus.success ? 'success' : 'error'}
            showIcon
            icon={uploadStatus.success ? <CheckCircleOutlined /> : undefined}
            closable
            onClose={() => {
              setUploadStatus(null);
              setUploadRowErrors([]);
            }}
            style={{ marginTop: '16px' }}
          />
        )}
        
        {/* Row-level errors display */}
        {uploadRowErrors.length > 0 && (
          <Card
            style={{
              marginTop: '16px',
              border: '1px solid #FEE2E2',
              backgroundColor: '#FEF2F2',
            }}
          >
            <div style={{ marginBottom: '12px' }}>
              <h4 style={{ margin: 0, color: '#991B1B', fontSize: '16px', fontWeight: '600' }}>
                Row-level issues found
              </h4>
              <p style={{ margin: '4px 0 0 0', color: '#7F1D1D', fontSize: '13px' }}>
                Fix these rows and re-upload.
              </p>
            </div>
            <Table
              dataSource={uploadRowErrors.map((err, idx) => ({ ...err, key: idx }))}
              columns={[
                {
                  title: 'Row',
                  dataIndex: 'row',
                  key: 'row',
                  width: 80,
                  render: (row: number) => <strong>{row}</strong>,
                },
                {
                  title: 'Column',
                  dataIndex: 'column',
                  key: 'column',
                  width: 120,
                },
                {
                  title: 'Reason',
                  dataIndex: 'reason',
                  key: 'reason',
                  ellipsis: true,
                },
              ]}
              pagination={false}
              size="small"
              scroll={{ y: 200 }}
              style={{ fontSize: '13px' }}
            />
          </Card>
        )}
      </Card>

      {/* Statistics Card */}
      <Card
        style={{
          marginBottom: '24px',
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        }}
      >
        {loadingStats ? (
          <div style={{ textAlign: 'center', padding: '24px' }}>
            <Spin />
          </div>
        ) : (
          <Row gutter={[24, 24]}>
            <Col xs={24} sm={6}>
              <Statistic
                title="Total Records"
                value={statistics?.total_records || 0}
                valueStyle={{ color: '#6366F1', fontSize: '24px', fontWeight: '700' }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="Date Range"
                value={
                  statistics?.date_range.start && statistics?.date_range.end
                    ? `${dayjs(statistics.date_range.start).format('MMM D, YYYY')} - ${dayjs(statistics.date_range.end).format('MMM D, YYYY')}`
                    : 'N/A'
                }
                valueStyle={{ fontSize: '16px', fontWeight: '500', color: '#030712' }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <Statistic
                title="Unique SKUs"
                value={statistics?.unique_skus || 0}
                valueStyle={{ color: '#10B981', fontSize: '24px', fontWeight: '700' }}
              />
            </Col>
            <Col xs={24} sm={6}>
              <div style={{ textAlign: 'center', paddingTop: '8px' }}>
                <div style={{ marginBottom: '8px', fontSize: '14px', color: '#64748B', fontWeight: '500' }}>
                  Export Data
                </div>
                <Button
                  type="primary"
                  icon={<DownloadOutlined />}
                  onClick={handleExportCSV}
                  size="large"
                  style={{
                    backgroundColor: '#6366F1',
                    borderColor: '#6366F1',
                    fontWeight: '500',
                    width: '100%',
                  }}
                >
                  Export CSV
                </Button>
              </div>
            </Col>
          </Row>
        )}
      </Card>

      {/* Filters Section */}
      <Card
        style={{
          marginBottom: '24px',
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        }}
      >
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={12} md={6}>
            <div style={{ marginBottom: '8px', fontSize: '13px', fontWeight: '500', color: '#030712' }}>
              Date Range
            </div>
            <RangePicker
              disabledDate={(current) => {
                // Disable future dates
                return current && current > dayjs().endOf('day');
              }}
              value={dateRange}
              onChange={handleDateRangeChange}
              format="MMM D, YYYY"
              style={{ width: '100%' }}
              allowClear={true}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <div style={{ marginBottom: '8px', fontSize: '13px', fontWeight: '500', color: '#030712' }}>
              SKU
            </div>
            <Select
              placeholder="Select SKU"
              value={selectedSKU}
              onChange={handleSKUChange}
              allowClear
              showSearch
              filterOption={(input, option) =>
                (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
              }
              style={{ width: '100%' }}
              loading={loadingSKUs}
              options={skuOptions.map(sku => ({ label: sku, value: sku }))}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <div style={{ marginBottom: '8px', fontSize: '13px', fontWeight: '500', color: '#030712' }}>
              Transaction Type
            </div>
            <Select
              placeholder="Select Type"
              value={selectedTransactionType}
              onChange={handleTransactionTypeChange}
              allowClear
              style={{ width: '100%' }}
              options={[
                { label: 'All Transactions', value: 'all' },
                { label: 'Shipment', value: 'Shipment' },
                { label: 'Refund', value: 'Refund' },
                { label: 'Cancellation', value: 'Cancel' },
              ]}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <div style={{ marginBottom: '8px', fontSize: '13px', fontWeight: '500', color: 'transparent' }}>
              Actions
            </div>
            <Button
              icon={<ReloadOutlined />}
              onClick={handleClearFilters}
              style={{ width: '100%' }}
            >
              Clear Filters
            </Button>
          </Col>
        </Row>
      </Card>

      {/* Table Section */}
      <Card
        style={{
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        }}
      >
        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px' }}>
            <Spin size="large" />
            <p style={{ marginTop: '16px', color: '#64748B' }}>Loading transactions...</p>
          </div>
        ) : (
          <>
            <Table
              columns={columns}
              dataSource={transactions}
              rowKey={(record) => `${record.order_id}-${record.transaction_type}-${record.sku}`}
              pagination={false}
              scroll={{ x: 'max-content' }}
              style={{
                backgroundColor: '#FFFFFF',
              }}
            />
            <div
              style={{
                marginTop: '24px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div style={{ color: '#64748B', fontSize: '14px' }}>
                Showing {transactions.length > 0 ? (currentPage - 1) * pageSize + 1 : 0} -{' '}
                {Math.min(currentPage * pageSize, total)} of {total.toLocaleString()} transactions
              </div>
              <Pagination
                current={currentPage}
                total={total}
                pageSize={pageSize}
                onChange={handlePageChange}
                showSizeChanger={false}
                showTotal={(total, range) =>
                  `${range[0]}-${range[1]} of ${total} items`
                }
              />
            </div>
          </>
        )}
      </Card>

      {/* Duplicate Upload Modal */}
      <Modal
        title="File Already Uploaded"
        open={!!duplicateUpload}
        onCancel={() => setDuplicateUpload(null)}
        footer={[
          <Button key="cancel" onClick={() => setDuplicateUpload(null)}>
            Cancel
          </Button>,
          <Button
            key="keep"
            onClick={() => {
              setDuplicateUpload(null);
              message.info('Keeping existing upload. You can upload a different file.');
            }}
          >
            Keep Both
          </Button>,
          <Button
            key="replace"
            type="primary"
            danger
            onClick={async () => {
              if (!duplicateUpload) return;
              
              try {
                // Delete existing ingestion
                const { deleteUpload } = await import('@/services/dataService');
                await deleteUpload(duplicateUpload.existing_ingestion_id);
                message.success('Deleted existing upload. Please upload the file again.');
                setDuplicateUpload(null);
              } catch (error) {
                message.error('Failed to delete existing upload. Please try again.');
              }
            }}
          >
            Replace (Delete Old)
          </Button>,
        ]}
      >
        <p>This file has already been uploaded:</p>
        <ul>
          <li><strong>File:</strong> {duplicateUpload?.existing_filename}</li>
          <li><strong>Uploaded:</strong> {duplicateUpload?.existing_uploaded_at ? new Date(duplicateUpload.existing_uploaded_at).toLocaleString() : 'Unknown'}</li>
          <li><strong>Rows:</strong> {duplicateUpload?.existing_rows.toLocaleString()}</li>
        </ul>
        <p>Would you like to replace the existing upload or keep both?</p>
      </Modal>

      {/* Required Columns Error Modal */}
      <Modal
        title="Required Columns Missing"
        open={!!requiredColumnsError}
        onCancel={() => setRequiredColumnsError(null)}
        footer={[
          <Button key="close" type="primary" onClick={() => setRequiredColumnsError(null)}>
            Close
          </Button>,
        ]}
        width={600}
      >
        <Alert
          message="Missing Required Columns"
          description={`The following required columns were not detected: ${requiredColumnsError?.missing_columns.join(', ')}`}
          type="error"
          style={{ marginBottom: '16px' }}
        />
        
        <div>
          <h4>Expected Column Names:</h4>
          <ul>
            {requiredColumnsError?.missing_columns.map((col) => (
              <li key={col}>
                <strong>{col}:</strong> {requiredColumnsError?.expected_schema[col] || 'Any column with similar name'}
              </li>
            ))}
          </ul>
          
          <h4 style={{ marginTop: '16px' }}>Detected Columns:</h4>
          <p>{requiredColumnsError?.csv_columns.join(', ') || 'None'}</p>
        </div>
      </Modal>

      {/* Batch Upload Summary Modal - Show briefly, then auto-close to show success modal */}
      <Modal
        title="Batch Upload Complete"
        open={batchSummaryVisible && !uploadSuccessVisible}
        onCancel={() => {
          setBatchSummaryVisible(false);
          setBatchFiles([]);
          setBatchSummary(null);
        }}
        footer={[
          <Button
            key="close"
            type="primary"
            onClick={() => {
              setBatchSummaryVisible(false);
              setBatchFiles([]);
              setBatchSummary(null);
            }}
          >
            Close
          </Button>,
        ]}
        width={600}
        closable={true}
      >
        {batchSummary && (
          <div>
            <div style={{ marginBottom: '24px' }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="Total Files"
                    value={batchFiles.length}
                    prefix={<CheckCircleOutlined />}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Successful"
                    value={batchSummary.successful}
                    valueStyle={{ color: '#16A34A' }}
                    prefix={<CheckCircleOutlined />}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="Failed"
                    value={batchSummary.failed}
                    valueStyle={{ color: '#DC2626' }}
                    prefix={<CloseCircleOutlined />}
                  />
                </Col>
              </Row>
              {batchSummary.totalRows > 0 && (
                <div style={{ marginTop: '16px', textAlign: 'center' }}>
                  <Text strong style={{ fontSize: '18px', color: '#2563EB' }}>
                    {batchSummary.totalRows.toLocaleString()} total rows inserted
                  </Text>
                </div>
              )}
            </div>
            
            <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <List
                size="small"
                dataSource={batchFiles}
                renderItem={(item) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={
                        item.status === 'success' ? (
                          <CheckCircleOutlined style={{ color: '#16A34A', fontSize: '20px' }} />
                        ) : item.status === 'error' ? (
                          <CloseCircleOutlined style={{ color: '#DC2626', fontSize: '20px' }} />
                        ) : (
                          <ClockCircleOutlined style={{ color: '#64748B', fontSize: '20px' }} />
                        )
                      }
                      title={
                        <Space>
                          <Text strong>{item.file.name}</Text>
                          {item.status === 'success' && item.result?.rows_inserted && (
                            <Tag color="success">{item.result.rows_inserted.toLocaleString()} rows</Tag>
                          )}
                          {item.status === 'error' && (
                            <Tag color="error">Failed</Tag>
                          )}
                          {item.status === 'skipped' && (
                            <Tag color="default">Skipped</Tag>
                          )}
                        </Space>
                      }
                      description={
                        item.result?.message || item.result?.error || (
                          item.status === 'pending' ? 'Pending' : 'Processing...'
                        )
                      }
                    />
                  </List.Item>
                )}
              />
            </div>
          </div>
        )}
      </Modal>

      {/* Upload Success Modal */}
      {uploadSuccessData && (
        <UploadSuccessModal
          visible={uploadSuccessVisible}
          filesUploaded={uploadSuccessData.filesUploaded}
          totalRows={uploadSuccessData.totalRows}
          onViewDashboard={() => {
            setUploadSuccessVisible(false);
            navigate('/app/dashboard');
          }}
          onViewFiles={() => {
            setUploadSuccessVisible(false);
            navigate('/app/data-management');
          }}
          onClose={() => {
            setUploadSuccessVisible(false);
          }}
        />
      )}

      {/* Column Mapping Modal */}
      <ColumnMappingModal
        open={columnMappingModalVisible}
        csvColumns={csvColumns}
        detectedMapping={detectedMapping}
        onConfirm={handleColumnMappingConfirm}
        onCancel={() => {
          setColumnMappingModalVisible(false);
          setPendingFile(null);
          setPendingMappingFile(null);
        }}
      />
    </div>
  );
};

export default Workspace;

