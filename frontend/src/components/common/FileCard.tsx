/**
 * FileCard Component
 * 
 * Premium file card component for displaying file metadata with actions.
 * Used in Data Management page for file grid layout.
 * 
 * Features:
 * - Clean white background with subtle shadow
 * - File icon and metadata display
 * - Status badge with color coding
 * - Action buttons (View, Download, Delete)
 * - Hover effects with elevation
 * - Smooth transitions
 */

import React from 'react';
import { Button, Tooltip, Tag } from 'antd';
import { FileOutlined, EyeOutlined, DownloadOutlined, DeleteOutlined } from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS, TYPOGRAPHY } from '@/styles/design-tokens';

export interface FileCardProps {
  /**
   * File ID
   */
  fileId: string;
  
  /**
   * Filename
   */
  filename: string;
  
  /**
   * File size in bytes
   */
  fileSize: number;
  
  /**
   * Upload timestamp (ISO string)
   */
  uploadDate: string;
  
  /**
   * Record/row count
   */
  recordCount?: number;
  
  /**
   * Status: 'active', 'processing', 'error'
   */
  status?: 'active' | 'processing' | 'error';
  
  /**
   * Formatted file size string (e.g., "2.5 MB")
   */
  formattedSize?: string;
  
  /**
   * Formatted upload date string (e.g., "2 days ago")
   */
  formattedDate?: string;
  
  /**
   * Callback when View button is clicked
   */
  onView?: () => void;
  
  /**
   * Callback when Download button is clicked
   */
  onDownload?: () => void;
  
  /**
   * Callback when Delete button is clicked
   */
  onDelete?: () => void;
  
  /**
   * Loading state for actions
   */
  loading?: boolean;
  
  /**
   * Optional className
   */
  className?: string;
}

/**
 * FileCard - Premium file card component
 */
const FileCard: React.FC<FileCardProps> = ({
  filename,
  fileSize,
  uploadDate,
  recordCount,
  status = 'active',
  formattedSize,
  formattedDate,
  onView,
  onDownload,
  onDelete,
  loading = false,
  className,
}) => {
  // Format filename (truncate if too long)
  const displayFilename = filename.length > 30 ? `${filename.substring(0, 30)}...` : filename;
  
  // Get status badge color
  const getStatusColor = () => {
    switch (status) {
      case 'active':
        return COLORS.success;
      case 'processing':
        return COLORS.warning;
      case 'error':
        return COLORS.danger;
      default:
        return COLORS.secondary;
    }
  };
  
  const getStatusText = () => {
    switch (status) {
      case 'active':
        return 'Active';
      case 'processing':
        return 'Processing';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  return (
    <div
      className={`card-enter card-hoverable ${className || ''}`}
      style={{
        backgroundColor: COLORS.background,
        borderRadius: BORDER_RADIUS.lg,
        padding: SPACING.lg,
        boxShadow: SHADOWS.md,
        border: '1px solid transparent',
        transition: 'all 0.2s ease',
        display: 'flex',
        flexDirection: 'column',
        gap: SPACING.md,
        cursor: 'default',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = SHADOWS.lg;
        e.currentTarget.style.transform = 'scale(1.02)';
        e.currentTarget.style.borderColor = COLORS.primary;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = SHADOWS.md;
        e.currentTarget.style.transform = 'scale(1)';
        e.currentTarget.style.borderColor = 'transparent';
      }}
    >
      {/* Header: Icon and Status */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: SPACING.sm,
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: SPACING.sm,
            flex: 1,
            minWidth: 0,
          }}
        >
          <div
            style={{
              width: '48px',
              height: '48px',
              borderRadius: BORDER_RADIUS.md,
              backgroundColor: `${COLORS.primary}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            <FileOutlined
              style={{
                fontSize: '24px',
                color: COLORS.primary,
              }}
            />
          </div>
          <div
            style={{
              flex: 1,
              minWidth: 0,
            }}
          >
            <Tooltip title={filename}>
              <h4
                style={{
                  ...TYPOGRAPHY.body,
                  fontWeight: 600,
                  color: COLORS.neutralDark,
                  margin: 0,
                  marginBottom: SPACING.xs,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {displayFilename}
              </h4>
            </Tooltip>
            <Tag
              color={getStatusColor()}
              style={{
                margin: 0,
                fontSize: '11px',
                padding: '2px 8px',
                borderRadius: BORDER_RADIUS.sm,
              }}
            >
              {getStatusText()}
            </Tag>
          </div>
        </div>
      </div>

      {/* Metadata */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: SPACING.xs,
          flex: 1,
        }}
      >
        {recordCount !== undefined && (
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span style={{ ...TYPOGRAPHY.caption, color: COLORS.secondary }}>
              Records:
            </span>
            <span style={{ ...TYPOGRAPHY.body, fontWeight: 600, color: COLORS.neutralDark }}>
              {recordCount.toLocaleString()}
            </span>
          </div>
        )}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span style={{ ...TYPOGRAPHY.caption, color: COLORS.secondary }}>
            Size:
          </span>
          <span style={{ ...TYPOGRAPHY.body, fontWeight: 500, color: COLORS.neutralDark }}>
            {formattedSize || formatFileSize(fileSize)}
          </span>
        </div>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span style={{ ...TYPOGRAPHY.caption, color: COLORS.secondary }}>
            Uploaded:
          </span>
          <span style={{ ...TYPOGRAPHY.body, fontWeight: 500, color: COLORS.neutralDark }}>
            {formattedDate || formatDate(uploadDate)}
          </span>
        </div>
      </div>

      {/* Action Buttons */}
      <div
        style={{
          display: 'flex',
          gap: SPACING.xs,
          paddingTop: SPACING.sm,
          borderTop: '1px solid #E5E7EB',
        }}
      >
        {onView && (
          <Tooltip title="View Details">
            <Button
              icon={<EyeOutlined />}
              onClick={onView}
              loading={loading}
              style={{
                flex: 1,
                transition: 'all 0.2s ease',
              }}
            >
              View
            </Button>
          </Tooltip>
        )}
        {onDownload && (
          <Tooltip title="Download">
            <Button
              icon={<DownloadOutlined />}
              onClick={onDownload}
              loading={loading}
              style={{
                flex: 1,
                transition: 'all 0.2s ease',
              }}
            >
              Download
            </Button>
          </Tooltip>
        )}
        {onDelete && (
          <Tooltip title="Delete">
            <Button
              danger
              icon={<DeleteOutlined />}
              onClick={onDelete}
              loading={loading}
              style={{
                flex: 1,
                transition: 'all 0.2s ease',
              }}
            >
              Delete
            </Button>
          </Tooltip>
        )}
      </div>
    </div>
  );
};

/**
 * Format file size helper
 */
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format date helper (relative or absolute)
 */
function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  
  if (diffDays === 0) {
    return 'Today';
  } else if (diffDays === 1) {
    return 'Yesterday';
  } else if (diffDays < 7) {
    return `${diffDays} days ago`;
  } else {
    // Format as "Nov 17, 2025"
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  }
}

export default FileCard;

