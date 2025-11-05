import React, { useState, useEffect } from 'react';
import { Table, Card, Pagination, Spin, Alert, DatePicker, Select, Button, Row, Col, Upload, Statistic, message } from 'antd';
import type { ColumnsType } from 'antd/es/table';
import { ReloadOutlined, DownloadOutlined, InboxOutlined, CheckCircleOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import dayjs, { Dayjs } from 'dayjs';
import { getTransactions, getUniqueSKUs, getDataStatistics, downloadTransactionsCSV, uploadCSV, Transaction } from '@/services/dataService';
import { formatCurrency } from '@/utils/formatters';

const { RangePicker } = DatePicker;
const { Dragger } = Upload;

const Workspace: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const pageSize = 50;

  // Filter states
  const [dateRange, setDateRange] = useState<[Dayjs | null, Dayjs | null] | null>(null);
  const [selectedSKU, setSelectedSKU] = useState<string | undefined>(undefined);
  const [selectedTransactionType, setSelectedTransactionType] = useState<string | undefined>(undefined);
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

  // Load SKU options on mount
  useEffect(() => {
    const fetchSKUs = async () => {
      setLoadingSKUs(true);
      const skus = await getUniqueSKUs();
      if (skus) {
        setSkuOptions(skus);
      }
      setLoadingSKUs(false);
    };
    fetchSKUs();
  }, []);

  // Load statistics on mount
  useEffect(() => {
    const fetchStats = async () => {
      setLoadingStats(true);
      const stats = await getDataStatistics();
      if (stats) {
        setStatistics(stats);
      }
      setLoadingStats(false);
    };
    fetchStats();
  }, []);

  // Fetch transactions when filters or page changes
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
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
      } else {
        setError('Failed to load transaction data');
      }
      
      setLoading(false);
    };

    fetchData();
  }, [currentPage, dateRange, selectedSKU, selectedTransactionType]);

  // Handle page change
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  // Handle filter changes (reset to page 1)
  const handleDateRangeChange = (dates: [Dayjs | null, Dayjs | null] | null) => {
    setDateRange(dates);
    setCurrentPage(1);
  };

  const handleSKUChange = (value: string | undefined) => {
    setSelectedSKU(value);
    setCurrentPage(1);
  };

  const handleTransactionTypeChange = (value: string | undefined) => {
    setSelectedTransactionType(value);
    setCurrentPage(1);
  };

  // Clear all filters
  const handleClearFilters = () => {
    setDateRange(null);
    setSelectedSKU(undefined);
    setSelectedTransactionType(undefined);
    setCurrentPage(1);
  };

  // Handle export CSV
  const handleExportCSV = async () => {
    try {
      await downloadTransactionsCSV(
        dateRange && dateRange[0] ? dateRange[0].format('YYYY-MM-DD') : undefined,
        dateRange && dateRange[1] ? dateRange[1].format('YYYY-MM-DD') : undefined,
        selectedSKU,
        selectedTransactionType
      );
      message.success('CSV export started');
    } catch (error) {
      message.error('Failed to export CSV');
    }
  };

  // Handle file upload
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv',
    beforeUpload: async (file) => {
      setUploading(true);
      setUploadStatus(null);
      
      const result = await uploadCSV(file);
      
      if (result?.success) {
        setUploadStatus({
          success: true,
          message: result.message,
          rowsInserted: result.rows_inserted,
        });
        message.success(`Successfully uploaded ${result.rows_inserted} rows`);
        
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
        
        // Refresh statistics
        const fetchStats = async () => {
          const stats = await getDataStatistics();
          if (stats) {
            setStatistics(stats);
          }
        };
        
        await Promise.all([fetchData(), fetchStats()]);
      } else {
        setUploadStatus({
          success: false,
          message: result?.message || 'Upload failed',
        });
        message.error(result?.message || 'Upload failed');
      }
      
      setUploading(false);
      return false; // Prevent auto upload
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
      render: (text: string) => (
        <span style={{ fontFamily: 'monospace', fontSize: '13px' }}>{text}</span>
      ),
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
            Click or drag CSV file to this area to upload
          </p>
          <p className="ant-upload-hint" style={{ color: '#64748B' }}>
            Support for CSV files. Files will be processed and added to the database.
          </p>
        </Dragger>
        
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
            onClose={() => setUploadStatus(null)}
            style={{ marginTop: '16px' }}
          />
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
    </div>
  );
};

export default Workspace;

