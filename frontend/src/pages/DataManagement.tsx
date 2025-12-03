/**
 * Data Management Page
 * 
 * Premium file management UI with modern SaaS design.
 * Features file grid layout, status indicators, and polished action controls.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Button, Modal, Input, Select, message, Skeleton, Row, Col, Space } from 'antd';
import {
  CloudUploadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  InboxOutlined,
  ExclamationCircleOutlined,
  SearchOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import {
  getUploads,
  getDataStatistics,
  deleteUpload,
  deleteAllUploads,
  getUploadDetails,
  UploadFile,
  UploadDetailsResponse,
} from '@/services/dataService';
import { MetricCard, FileCard, EmptyState, DataCard } from '@/components/common';
import { SPACING, BORDER_RADIUS, COLORS, TYPOGRAPHY } from '@/styles/design-tokens';
import './DataManagement.css';

dayjs.extend(relativeTime);

const { Search } = Input;
const { Option } = Select;

const DataManagement: React.FC = () => {
  const navigate = useNavigate();

  // State
  const [uploads, setUploads] = useState<UploadFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [statistics, setStatistics] = useState<{
    total_records: number;
    date_range: { start: string | null; end: string | null };
    unique_skus: number;
  } | null>(null);
  const [loadingStats, setLoadingStats] = useState(false);

  // Modal states
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [deleteAllModalVisible, setDeleteAllModalVisible] = useState(false);
  const [selectedFile, setSelectedFile] = useState<UploadFile | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleting, setDeleting] = useState(false);
  const [viewDetailsModalVisible, setViewDetailsModalVisible] = useState(false);
  const [uploadDetails, setUploadDetails] = useState<UploadDetailsResponse | null>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // Search and sort
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'size'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Fetch uploads
  const fetchUploads = async () => {
    setLoading(true);
    try {
      const files = await getUploads();
      if (files) {
        setUploads(files);
      }
    } catch (error) {
      console.error('Error fetching uploads:', error);
      message.error('Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  // Fetch statistics
  const fetchStatistics = async () => {
    setLoadingStats(true);
    try {
      const stats = await getDataStatistics();
      if (stats) {
        setStatistics(stats);
      }
    } catch (error) {
      console.error('Error fetching statistics:', error);
    } finally {
      setLoadingStats(false);
    }
  };

  // Load data on mount
  useEffect(() => {
    fetchUploads();
    fetchStatistics();
  }, []);

  // Calculate summary stats
  const summaryStats = useMemo(() => {
    const totalFiles = uploads.length;
    const totalRecords = statistics?.total_records || 0;
    const totalSize = uploads.reduce((sum, file) => sum + (file.file_size || 0), 0);

    return {
      totalFiles,
      totalRecords,
      totalSize,
    };
  }, [uploads, statistics]);

  // Filtered and sorted files
  const filteredAndSortedFiles = useMemo(() => {
    let filtered = uploads;

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter((file) =>
        file.filename.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply sort
    const sorted = [...filtered].sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'name':
          comparison = a.filename.localeCompare(b.filename);
          break;
        case 'date':
          comparison = new Date(a.upload_timestamp || a.uploaded_at || '').getTime() - new Date(b.upload_timestamp || b.uploaded_at || '').getTime();
          break;
        case 'size':
          comparison = (a.file_size || 0) - (b.file_size || 0);
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return sorted;
  }, [uploads, searchQuery, sortBy, sortOrder]);

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  // Format date (relative)
  const formatDate = (dateString: string): string => {
    const date = dayjs(dateString);
    const now = dayjs();
    const diffDays = now.diff(date, 'day');

    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.format('MMM D, YYYY');
    }
  };

  // Handle delete file
  const handleDeleteFile = (file: UploadFile) => {
    setSelectedFile(file);
    setDeleteModalVisible(true);
  };

  // Confirm delete file
  const confirmDeleteFile = async () => {
    if (!selectedFile) return;

    setDeleting(true);
    try {
      const fileId = selectedFile.file_id || selectedFile.ingestion_id;
      if (!fileId) {
        message.error('File ID not found');
        return;
      }
      const result = await deleteUpload(fileId);
      if (result?.success) {
        message.success(`Successfully deleted ${(result.deleted_rows || 0).toLocaleString()} rows from ${selectedFile.filename}`);
        setDeleteModalVisible(false);
        setSelectedFile(null);
        await fetchUploads();
        await fetchStatistics();
      } else {
        message.error(result?.message || 'Failed to delete file');
      }
    } catch (error: any) {
      console.error('Error deleting file:', error);
      if (error?.response?.status === 404) {
        message.error('File not found. It may have already been deleted.');
      } else {
        message.error('Failed to delete file. Please try again.');
      }
    } finally {
      setDeleting(false);
    }
  };

  // Handle delete all
  const handleDeleteAll = () => {
    setDeleteAllModalVisible(true);
    setDeleteConfirmText('');
  };

  // Confirm delete all
  const confirmDeleteAll = async () => {
    if (deleteConfirmText !== 'DELETE ALL') {
      message.error('Please type "DELETE ALL" to confirm');
      return;
    }

    setDeleting(true);
    try {
      const result = await deleteAllUploads();
      if (result?.success) {
        message.success('All data deleted successfully');
        setDeleteAllModalVisible(false);
        setDeleteConfirmText('');
        setUploads([]);
        setStatistics(null);
      } else {
        message.error(result?.message || 'Failed to delete all data');
      }
    } catch (error) {
      console.error('Error deleting all data:', error);
      message.error('Failed to delete all data. Please try again.');
    } finally {
      setDeleting(false);
    }
  };

  // Handle view details
  const handleViewDetails = async (file: UploadFile) => {
    setSelectedFile(file);
    setViewDetailsModalVisible(true);
    setLoadingDetails(true);
    try {
      const fileId = file.file_id || file.ingestion_id;
      if (!fileId) {
        message.error('File ID not found');
        return;
      }
      const details = await getUploadDetails(fileId);
      if (details) {
        setUploadDetails(details);
      } else {
        message.error('Failed to load file details');
      }
    } catch (error) {
      console.error('Error fetching file details:', error);
      message.error('Failed to load file details');
    } finally {
      setLoadingDetails(false);
    }
  };

  // Handle download
  const handleDownload = async (file: UploadFile) => {
    try {
      // Navigate to workspace with file filter
      navigate('/workspace', { state: { fileId: file.file_id } });
      message.info('Opening workspace with file filter');
    } catch (error) {
      console.error('Error downloading file:', error);
      message.error('Failed to download file');
    }
  };

  // Handle upload new file
  const handleUploadNew = () => {
    navigate('/workspace', { state: { tab: 'upload' } });
  };

  return (
    <div
      style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: SPACING.xl,
        backgroundColor: COLORS.backgroundSecondary,
        minHeight: 'calc(100vh - 64px)',
      }}
    >
      {/* Header Section */}
      <div style={{ marginBottom: SPACING.xl }}>
        <h1
          style={{
            ...TYPOGRAPHY.heading1,
            color: COLORS.neutralDark,
            marginBottom: SPACING.sm,
            marginTop: 0,
          }}
        >
          Data Management
        </h1>
        <p
          style={{
            ...TYPOGRAPHY.body,
            color: COLORS.secondary,
            margin: 0,
          }}
        >
          Manage your uploaded data files, view metadata, and organize your analytics data
        </p>
      </div>

      {/* Summary Stats */}
      <Row gutter={[SPACING.lg, SPACING.lg]} style={{ marginBottom: SPACING.xl }}>
        <Col xs={24} sm={8}>
          <MetricCard
            title="Total Files"
            value={summaryStats.totalFiles}
            icon={<FileTextOutlined />}
            loading={loadingStats}
          />
        </Col>
        <Col xs={24} sm={8}>
          <MetricCard
            title="Total Records"
            value={summaryStats.totalRecords.toLocaleString()}
            icon={<FileTextOutlined />}
            loading={loadingStats}
          />
        </Col>
        <Col xs={24} sm={8}>
          <MetricCard
            title="Storage Used"
            value={formatFileSize(summaryStats.totalSize)}
            icon={<FileTextOutlined />}
            loading={loadingStats}
          />
        </Col>
      </Row>

      {/* Action Bar */}
      <DataCard
        style={{ marginBottom: SPACING.lg }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: SPACING.md,
          }}
        >
          <div
            style={{
              display: 'flex',
              gap: SPACING.md,
              flexWrap: 'wrap',
              flex: 1,
            }}
          >
            <Search
              placeholder="Search files..."
              allowClear
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ width: 300, maxWidth: '100%' }}
              prefix={<SearchOutlined />}
            />
            <Select
              value={sortBy}
              onChange={setSortBy}
              style={{ width: 150 }}
            >
              <Option value="name">Name</Option>
              <Option value="date">Date</Option>
              <Option value="size">Size</Option>
            </Select>
            <Select
              value={sortOrder}
              onChange={setSortOrder}
              style={{ width: 120 }}
            >
              <Option value="desc">Descending</Option>
              <Option value="asc">Ascending</Option>
            </Select>
          </div>
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                fetchUploads();
                fetchStatistics();
                message.success('Refreshed');
              }}
              loading={loading}
            >
              Refresh
            </Button>
            <Button
              type="primary"
              icon={<CloudUploadOutlined />}
              onClick={handleUploadNew}
            >
              Upload New File
            </Button>
            {uploads.length > 0 && (
              <Button
                danger
                icon={<DeleteOutlined />}
                onClick={handleDeleteAll}
              >
                Delete All
              </Button>
            )}
          </Space>
        </div>
      </DataCard>

      {/* Files Grid */}
      {loading ? (
        <Row gutter={[SPACING.lg, SPACING.lg]}>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Col key={i} xs={24} sm={12} lg={8}>
              <Skeleton active paragraph={{ rows: 4 }} />
            </Col>
          ))}
        </Row>
      ) : filteredAndSortedFiles.length === 0 ? (
        <EmptyState
          icon={<InboxOutlined />}
          title={searchQuery ? 'No files found' : 'No files uploaded yet'}
          description={
            searchQuery
              ? 'Try adjusting your search query'
              : 'Upload your first CSV file to get started with analytics'
          }
          action={
            !searchQuery
              ? {
                text: 'Upload File',
                onClick: handleUploadNew,
              }
              : undefined
          }
        />
      ) : (
        <Row gutter={[SPACING.lg, SPACING.lg]}>
          {filteredAndSortedFiles.map((file, index) => (
            <Col key={file.file_id} xs={24} sm={12} lg={8}>
              <div
                className="file-card-wrapper"
                style={{
                  animationDelay: `${index * 50}ms`,
                }}
              >
                <FileCard
                  fileId={file.file_id || file.ingestion_id || ''}
                  filename={file.filename}
                  fileSize={file.file_size || 0}
                  uploadDate={file.upload_timestamp || file.uploaded_at || ''}
                  recordCount={file.row_count || file.rows_inserted || 0}
                  status="active"
                  formattedSize={formatFileSize(file.file_size || 0)}
                  formattedDate={formatDate(file.upload_timestamp || file.uploaded_at || '')}
                  onView={() => handleViewDetails(file)}
                  onDownload={() => handleDownload(file)}
                  onDelete={() => handleDeleteFile(file)}
                  loading={deleting && selectedFile?.file_id === file.file_id}
                />
              </div>
            </Col>
          ))}
        </Row>
      )}

      {/* Delete File Modal */}
      <Modal
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: SPACING.xs }}>
            <ExclamationCircleOutlined style={{ color: COLORS.danger, fontSize: '20px' }} />
            <span>Delete File?</span>
          </div>
        }
        open={deleteModalVisible}
        onCancel={() => {
          setDeleteModalVisible(false);
          setSelectedFile(null);
        }}
        footer={[
          <Button
            key="cancel"
            onClick={() => {
              setDeleteModalVisible(false);
              setSelectedFile(null);
            }}
            disabled={deleting}
          >
            Cancel
          </Button>,
          <Button
            key="delete"
            type="primary"
            danger
            icon={<DeleteOutlined />}
            onClick={confirmDeleteFile}
            loading={deleting}
          >
            Delete
          </Button>,
        ]}
        width={500}
      >
        {selectedFile && (
          <div>
            <p style={{ ...TYPOGRAPHY.body, marginBottom: SPACING.md }}>
              Are you sure you want to delete <strong>{selectedFile.filename}</strong>?
            </p>
            <p style={{ ...TYPOGRAPHY.body, color: COLORS.secondary }}>
              This will remove {(selectedFile.row_count || selectedFile.rows_inserted || 0).toLocaleString()} records from your database.
            </p>
          </div>
        )}
      </Modal>

      {/* Delete All Modal */}
      <Modal
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: SPACING.xs }}>
            <ExclamationCircleOutlined style={{ color: COLORS.danger, fontSize: '20px' }} />
            <span>Delete All Files?</span>
          </div>
        }
        open={deleteAllModalVisible}
        onCancel={() => {
          setDeleteAllModalVisible(false);
          setDeleteConfirmText('');
        }}
        footer={[
          <Button
            key="cancel"
            onClick={() => {
              setDeleteAllModalVisible(false);
              setDeleteConfirmText('');
            }}
            disabled={deleting}
          >
            Cancel
          </Button>,
          <Button
            key="delete"
            type="primary"
            danger
            icon={<DeleteOutlined />}
            onClick={confirmDeleteAll}
            loading={deleting}
            disabled={deleteConfirmText !== 'DELETE ALL'}
          >
            Delete All
          </Button>,
        ]}
        width={500}
      >
        <div>
          <p style={{ ...TYPOGRAPHY.body, marginBottom: SPACING.md }}>
            This will permanently delete all uploaded files and data. This action cannot be undone.
          </p>
          <p style={{ ...TYPOGRAPHY.body, marginBottom: SPACING.sm }}>
            To confirm, please type <strong style={{ color: COLORS.danger }}>DELETE ALL</strong> in the box below:
          </p>
          <Input
            value={deleteConfirmText}
            onChange={(e) => setDeleteConfirmText(e.target.value)}
            placeholder="Type DELETE ALL to confirm"
            style={{ marginBottom: SPACING.sm }}
          />
          <p style={{ ...TYPOGRAPHY.caption, color: COLORS.secondary }}>
            This will delete all {uploads.length} uploaded file{uploads.length !== 1 ? 's' : ''} and all associated data.
          </p>
        </div>
      </Modal>

      {/* View Details Modal */}
      <Modal
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: SPACING.xs }}>
            <FileTextOutlined style={{ color: COLORS.primary, fontSize: '20px' }} />
            <span>File Details</span>
          </div>
        }
        open={viewDetailsModalVisible}
        onCancel={() => {
          setViewDetailsModalVisible(false);
          setSelectedFile(null);
          setUploadDetails(null);
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setViewDetailsModalVisible(false);
              setSelectedFile(null);
              setUploadDetails(null);
            }}
          >
            Close
          </Button>,
        ]}
        width={700}
      >
        {loadingDetails ? (
          <div style={{ textAlign: 'center', padding: SPACING.xl }}>
            <Skeleton active paragraph={{ rows: 4 }} />
          </div>
        ) : uploadDetails ? (
          <div>
            <Row gutter={[SPACING.md, SPACING.md]}>
              <Col span={12}>
                <strong>Filename:</strong>
                <div>{uploadDetails.filename}</div>
              </Col>
              <Col span={12}>
                <strong>Row Count:</strong>
                <div>{(uploadDetails.row_count || uploadDetails.rows_inserted || 0).toLocaleString()} rows</div>
              </Col>
              <Col span={12}>
                <strong>File Size:</strong>
                <div>{formatFileSize(uploadDetails.file_size || 0)}</div>
              </Col>
              <Col span={12}>
                <strong>Upload Time:</strong>
                <div>{dayjs(uploadDetails.upload_timestamp || uploadDetails.uploaded_at || '').format('MMM D, YYYY h:mm A')}</div>
              </Col>
              {uploadDetails.date_range?.start && uploadDetails.date_range?.end && (
                <Col span={24}>
                  <strong>Date Range:</strong>
                  <div>
                    {dayjs(uploadDetails.date_range.start).format('MMM D, YYYY')} -{' '}
                    {dayjs(uploadDetails.date_range.end).format('MMM D, YYYY')}
                  </div>
                </Col>
              )}
              {uploadDetails.column_names && uploadDetails.column_names.length > 0 && (
                <Col span={24}>
                  <strong>Columns ({uploadDetails.column_names.length}):</strong>
                  <div style={{ marginTop: SPACING.xs }}>
                    <Space wrap>
                      {uploadDetails.column_names.map((col: string, idx: number) => (
                        <span
                          key={idx}
                          style={{
                            padding: '4px 8px',
                            backgroundColor: `${COLORS.primary}15`,
                            color: COLORS.primary,
                            borderRadius: BORDER_RADIUS.sm,
                            fontSize: '12px',
                          }}
                        >
                          {col}
                        </span>
                      ))}
                    </Space>
                  </div>
                </Col>
              )}
            </Row>
          </div>
        ) : (
          <EmptyState
            icon={<InboxOutlined />}
            title="No Details Available"
            description="File details could not be loaded"
          />
        )}
      </Modal>
    </div>
  );
};

export default DataManagement;

