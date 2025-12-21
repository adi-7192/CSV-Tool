/**
 * BatchUploadProgress Component
 * 
 * Displays progress for multiple file uploads with per-file status tracking.
 */
import React from 'react';
import { Card, Progress, Tag, Space, Button, List, Typography } from 'antd';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  ClockCircleOutlined,
  StopOutlined,
} from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS } from '@/styles/designTokens';

const { Text } = Typography;

export interface FileProgress {
  file: File;
  status: 'pending' | 'processing' | 'success' | 'error' | 'skipped';
  result?: {
    rows_inserted?: number;
    message?: string;
    error?: string;
  };
}

interface BatchUploadProgressProps {
  files: FileProgress[];
  currentIndex: number;
  onCancel?: () => void;
  totalRowsInserted?: number;
}

const BatchUploadProgress: React.FC<BatchUploadProgressProps> = ({
  files,
  currentIndex,
  onCancel,
  totalRowsInserted = 0,
}) => {
  const completed = files.filter((f) => f.status === 'success' || f.status === 'error' || f.status === 'skipped').length;
  const successful = files.filter((f) => f.status === 'success').length;
  const failed = files.filter((f) => f.status === 'error').length;
  const skipped = files.filter((f) => f.status === 'skipped').length;
  const total = files.length;
  const progressPercent = total > 0 ? Math.round((completed / total) * 100) : 0;

  const getStatusIcon = (status: FileProgress['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircleOutlined style={{ color: COLORS.success }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: COLORS.danger }} />;
      case 'processing':
        return <LoadingOutlined style={{ color: COLORS.primary }} spin />;
      case 'skipped':
        return <ClockCircleOutlined style={{ color: COLORS.secondary }} />;
      default:
        return <ClockCircleOutlined style={{ color: COLORS.secondary }} />;
    }
  };

  const getStatusTag = (status: FileProgress['status']) => {
    switch (status) {
      case 'success':
        return <Tag color="success">Success</Tag>;
      case 'error':
        return <Tag color="error">Error</Tag>;
      case 'processing':
        return <Tag color="processing">Processing...</Tag>;
      case 'skipped':
        return <Tag color="default">Skipped</Tag>;
      default:
        return <Tag color="default">Pending</Tag>;
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <Card
      style={{
        marginTop: SPACING.md,
        borderRadius: BORDER_RADIUS.md,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
      }}
    >
      <div style={{ marginBottom: SPACING.md }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: SPACING.sm }}>
          <Text strong style={{ fontSize: '16px' }}>
            Batch Upload Progress
          </Text>
          {onCancel && (
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={onCancel}
              disabled={completed === total}
            >
              Cancel
            </Button>
          )}
        </div>
        
        <Progress
          percent={progressPercent}
          status={completed === total ? (failed === 0 ? 'success' : 'exception') : 'active'}
          format={() => `${completed} of ${total} files`}
          style={{ marginBottom: SPACING.sm }}
        />
        
        <Space size="large" style={{ marginBottom: SPACING.md }}>
          <Text type="secondary">
            <CheckCircleOutlined style={{ color: COLORS.success, marginRight: '4px' }} />
            {successful} successful
          </Text>
          {failed > 0 && (
            <Text type="danger">
              <CloseCircleOutlined style={{ marginRight: '4px' }} />
              {failed} failed
            </Text>
          )}
          {skipped > 0 && (
            <Text type="secondary">
              <ClockCircleOutlined style={{ marginRight: '4px' }} />
              {skipped} skipped
            </Text>
          )}
          {totalRowsInserted > 0 && (
            <Text strong style={{ color: COLORS.primary }}>
              {totalRowsInserted.toLocaleString()} rows inserted
            </Text>
          )}
        </Space>
      </div>

      <List
        size="small"
        dataSource={files}
        renderItem={(item, index) => (
          <List.Item
            style={{
              padding: SPACING.sm,
              borderLeft: index === currentIndex ? `3px solid ${COLORS.primary}` : '3px solid transparent',
              backgroundColor: index === currentIndex ? `${COLORS.primary}08` : 'transparent',
            }}
          >
            <List.Item.Meta
              avatar={getStatusIcon(item.status)}
              title={
                <Space>
                  <Text strong style={{ fontSize: '14px' }}>
                    {item.file.name}
                  </Text>
                  {getStatusTag(item.status)}
                </Space>
              }
              description={
                <div>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {formatFileSize(item.file.size)}
                  </Text>
                  {item.result && (
                    <div style={{ marginTop: '4px' }}>
                      {item.result.rows_inserted !== undefined && item.result.rows_inserted > 0 && (
                        <Text type="success" style={{ fontSize: '12px' }}>
                          {item.result.rows_inserted.toLocaleString()} rows inserted
                        </Text>
                      )}
                      {item.result.message && (
                        <Text type="secondary" style={{ fontSize: '12px', display: 'block', marginTop: '2px' }}>
                          {item.result.message}
                        </Text>
                      )}
                      {item.result.error && (
                        <Text type="danger" style={{ fontSize: '12px', display: 'block', marginTop: '2px' }}>
                          {item.result.error}
                        </Text>
                      )}
                    </div>
                  )}
                </div>
              }
            />
          </List.Item>
        )}
      />
    </Card>
  );
};

export default BatchUploadProgress;

