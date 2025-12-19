/**
 * UploadSuccessModal Component
 * 
 * Beautiful success modal shown after file uploads complete.
 * Displays upload confirmation with animations and action buttons.
 */
import React, { useEffect, useState } from 'react';
import { Modal, Button, Space, Typography, Divider } from 'antd';
import {
  CheckCircleOutlined,
  FileTextOutlined,
  BarChartOutlined,
  FolderOutlined,
} from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS } from '@/styles/designTokens';
import './UploadSuccessModal.css';

const { Title, Text, Paragraph } = Typography;

export interface UploadSuccessModalProps {
  visible: boolean;
  filesUploaded: number;
  totalRows: number;
  onViewDashboard: () => void;
  onViewFiles: () => void;
  onClose: () => void;
}

const UploadSuccessModal: React.FC<UploadSuccessModalProps> = ({
  visible,
  filesUploaded,
  totalRows,
  onViewDashboard,
  onViewFiles,
  onClose,
}) => {
  const [showAnimation, setShowAnimation] = useState(false);

  useEffect(() => {
    if (visible) {
      // Trigger animation immediately when modal becomes visible
      setShowAnimation(true);
    } else {
      setShowAnimation(false);
    }
  }, [visible]);

  return (
    <Modal
      open={visible}
      onCancel={onClose}
      footer={null}
      closable={true}
      centered
      width={520}
      className="upload-success-modal"
      maskClosable={false}
      zIndex={2000}
      maskStyle={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}
    >
      <div style={{ textAlign: 'center', padding: SPACING.xl }}>
        {/* Success Icon with Animation */}
        <div
          className={`success-icon-container ${showAnimation ? 'animate' : ''}`}
          style={{
            marginBottom: SPACING.lg,
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              backgroundColor: `${COLORS.success}15`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
            }}
          >
            <CheckCircleOutlined
              style={{
                fontSize: '48px',
                color: COLORS.success,
                zIndex: 1,
              }}
            />
            {/* Animated ring */}
            <div
              className={`success-ring ${showAnimation ? 'expand' : ''}`}
              style={{
                position: 'absolute',
                width: '80px',
                height: '80px',
                borderRadius: '50%',
                border: `3px solid ${COLORS.success}`,
                opacity: 0,
              }}
            />
          </div>
        </div>

        {/* Title */}
        <Title level={3} style={{ marginBottom: SPACING.sm, color: COLORS.neutralDark }}>
          Upload Successful!
        </Title>

        {/* Description */}
        <Paragraph
          style={{
            fontSize: '16px',
            color: COLORS.secondary,
            marginBottom: SPACING.lg,
            lineHeight: '1.6',
          }}
        >
          Your files have been uploaded, cleaned, and loaded successfully.
        </Paragraph>

        {/* Stats */}
        <div
          style={{
            backgroundColor: COLORS.backgroundSecondary,
            borderRadius: BORDER_RADIUS.md,
            padding: SPACING.lg,
            marginBottom: SPACING.lg,
          }}
        >
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Space>
                <FileTextOutlined style={{ fontSize: '20px', color: COLORS.primary }} />
                <Text strong style={{ fontSize: '16px' }}>
                  Files Uploaded
                </Text>
              </Space>
              <Text strong style={{ fontSize: '18px', color: COLORS.primary }}>
                {filesUploaded}
              </Text>
            </div>
            <Divider style={{ margin: 0 }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Space>
                <BarChartOutlined style={{ fontSize: '20px', color: COLORS.primary }} />
                <Text strong style={{ fontSize: '16px' }}>
                  Total Rows
                </Text>
              </Space>
              <Text strong style={{ fontSize: '18px', color: COLORS.primary }}>
                {totalRows.toLocaleString()}
              </Text>
            </div>
          </Space>
        </div>

        {/* Action Buttons */}
        <Space size="middle" style={{ width: '100%', justifyContent: 'center' }}>
          <Button
            type="primary"
            size="large"
            icon={<BarChartOutlined />}
            onClick={onViewDashboard}
            style={{
              height: '44px',
              fontSize: '16px',
              fontWeight: '600',
              paddingLeft: SPACING.lg,
              paddingRight: SPACING.lg,
              borderRadius: BORDER_RADIUS.md,
            }}
          >
            View Dashboard
          </Button>
          <Button
            size="large"
            icon={<FolderOutlined />}
            onClick={onViewFiles}
            style={{
              height: '44px',
              fontSize: '16px',
              fontWeight: '500',
              paddingLeft: SPACING.lg,
              paddingRight: SPACING.lg,
              borderRadius: BORDER_RADIUS.md,
            }}
          >
            View Files
          </Button>
        </Space>
      </div>
    </Modal>
  );
};

export default UploadSuccessModal;

