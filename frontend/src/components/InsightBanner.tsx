import React, { useState } from 'react';
import { Button } from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  CloseOutlined,
} from '@ant-design/icons';

export interface InsightBannerProps {
  title: string;
  description: string;
  type: 'success' | 'warning' | 'error' | 'info';
  icon?: React.ReactNode;
  action?: {
    label: string;
    onClick: () => void;
  };
  dismissible?: boolean;
  onDismiss?: () => void;
}

const InsightBanner: React.FC<InsightBannerProps> = ({
  title,
  description,
  type,
  icon,
  action,
  dismissible = false,
  onDismiss,
}) => {
  const [isVisible, setIsVisible] = useState(true);

  // Color scheme by type
  const colorSchemes = {
    success: {
      bg: '#F0FDF4',
      border: '#DCFCE7',
      iconColor: '#16A34A',
      textColor: '#15803D',
      actionBg: '#16A34A',
      actionHover: '#15803D',
    },
    warning: {
      bg: '#FFFBEB',
      border: '#FEF3C7',
      iconColor: '#D97706',
      textColor: '#B45309',
      actionBg: '#D97706',
      actionHover: '#B45309',
    },
    error: {
      bg: '#FDF2F2',
      border: '#FECACA',
      iconColor: '#DC2626',
      textColor: '#B91C1C',
      actionBg: '#DC2626',
      actionHover: '#B91C1C',
    },
    info: {
      bg: '#F0F9FF',
      border: '#BFDBFE',
      iconColor: '#2563EB',
      textColor: '#1D4ED8',
      actionBg: '#2563EB',
      actionHover: '#1D4ED8',
    },
  };

  // Default icons by type
  const defaultIcons = {
    success: <CheckCircleOutlined style={{ fontSize: '20px' }} />,
    warning: <ExclamationCircleOutlined style={{ fontSize: '20px' }} />,
    error: <CloseCircleOutlined style={{ fontSize: '20px' }} />,
    info: <InfoCircleOutlined style={{ fontSize: '20px' }} />,
  };

  const colors = colorSchemes[type];
  const displayIcon = icon || defaultIcons[type];

  const handleDismiss = () => {
    setIsVisible(false);
    if (onDismiss) {
      // Call onDismiss after animation completes
      setTimeout(() => {
        onDismiss();
      }, 300);
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px',
        padding: '16px',
        backgroundColor: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: '8px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        position: 'relative',
        animation: 'fadeIn 0.3s ease-in',
        opacity: isVisible ? 1 : 0,
        transition: 'opacity 0.3s ease-out',
      }}
    >
      {/* Icon */}
      <div
        style={{
          color: colors.iconColor,
          display: 'flex',
          alignItems: 'center',
          flexShrink: 0,
          marginTop: '2px',
        }}
      >
        {displayIcon}
      </div>

      {/* Content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Title */}
        <div
          style={{
            fontWeight: '600',
            fontSize: '14px',
            color: colors.textColor,
            marginBottom: '4px',
            lineHeight: '1.5',
          }}
        >
          {title}
        </div>

        {/* Description */}
        <div
          style={{
            fontSize: '13px',
            color: '#64748B',
            marginBottom: action ? '12px' : '0',
            lineHeight: '1.5',
          }}
        >
          {description}
        </div>

        {/* Action Button */}
        {action && (
          <Button
            type="primary"
            size="small"
            onClick={action.onClick}
            style={{
              backgroundColor: colors.actionBg,
              borderColor: colors.actionBg,
              marginTop: '8px',
              fontSize: '12px',
              height: '28px',
              padding: '0 12px',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = colors.actionHover;
              e.currentTarget.style.borderColor = colors.actionHover;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = colors.actionBg;
              e.currentTarget.style.borderColor = colors.actionBg;
            }}
          >
            {action.label}
          </Button>
        )}
      </div>

      {/* Dismiss Button */}
      {dismissible && (
        <button
          onClick={handleDismiss}
          style={{
            position: 'absolute',
            top: '12px',
            right: '12px',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: '#94A3B8',
            padding: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: '4px',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.05)';
            e.currentTarget.style.color = '#64748B';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'transparent';
            e.currentTarget.style.color = '#94A3B8';
          }}
          aria-label="Dismiss"
        >
          <CloseOutlined style={{ fontSize: '14px' }} />
        </button>
      )}

      <style>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-8px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
};

export default InsightBanner;

