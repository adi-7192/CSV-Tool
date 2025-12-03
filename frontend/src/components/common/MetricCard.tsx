/**
 * MetricCard Component
 * 
 * Premium metric card component with Basedash-inspired design.
 * Displays key metrics with trend indicators, icons, and loading states.
 * 
 * Features:
 * - Clean white background with subtle shadow
 * - Icon badge with brand colors
 * - Large prominent value display
 * - Trend indicator with up/down arrows
 * - Smooth hover elevation effect
 * - Loading skeleton state
 * - Fully responsive
 */

import React from 'react';
import { Skeleton } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, MinusOutlined } from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS, TYPOGRAPHY } from '@/styles/design-tokens';

export interface MetricCardProps {
  /**
   * Card title/label
   */
  title: string;
  
  /**
   * Main metric value to display
   */
  value: string | number;
  
  /**
   * Percentage change (can be positive or negative)
   */
  change?: number;
  
  /**
   * Trend direction: 'up' for positive, 'down' for negative, 'neutral' for no change
   */
  trend?: 'up' | 'down' | 'neutral';
  
  /**
   * Optional icon to display in colored badge
   */
  icon?: React.ReactNode;
  
  /**
   * Loading state - shows skeleton
   */
  loading?: boolean;
  
  /**
   * Optional className for custom styling
   */
  className?: string;
}

/**
 * MetricCard - Premium metric display card
 */
const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  trend,
  icon,
  loading = false,
  className,
}) => {
  // Determine trend color and icon
  const getTrendConfig = () => {
    if (trend === 'up' || (change !== undefined && change > 0)) {
      return {
        color: COLORS.success,
        icon: <ArrowUpOutlined />,
        displayValue: change !== undefined ? `${Math.abs(change).toFixed(1)}%` : '',
      };
    }
    if (trend === 'down' || (change !== undefined && change < 0)) {
      return {
        color: COLORS.danger,
        icon: <ArrowDownOutlined />,
        displayValue: change !== undefined ? `${Math.abs(change).toFixed(1)}%` : '',
      };
    }
    if (trend === 'neutral' || change === 0) {
      return {
        color: COLORS.secondary,
        icon: <MinusOutlined />,
        displayValue: '0%',
      };
    }
    return null;
  };

  const trendConfig = getTrendConfig();

  if (loading) {
    return (
      <div
        className={className}
        style={{
          backgroundColor: COLORS.background,
          borderRadius: BORDER_RADIUS.lg,
          padding: SPACING.lg,
          boxShadow: SHADOWS.md,
          transition: 'all 0.2s ease',
        }}
      >
        <Skeleton active paragraph={{ rows: 2 }} />
      </div>
    );
  }

  return (
    <div
      className={`card-enter card-hoverable ${className || ''}`}
      style={{
        backgroundColor: COLORS.background,
        borderRadius: BORDER_RADIUS.lg,
        padding: SPACING.lg,
        boxShadow: SHADOWS.md,
        transition: 'all 0.2s ease',
        border: '1px solid transparent',
        cursor: 'pointer',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = SHADOWS.lg;
        e.currentTarget.style.transform = 'translateY(-2px)';
        e.currentTarget.style.borderColor = '#E5E7EB';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = SHADOWS.md;
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.borderColor = 'transparent';
      }}
    >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: SPACING.md,
        }}
      >
        {/* Header: Icon and Title */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: SPACING.sm,
          }}
        >
          {icon && (
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: BORDER_RADIUS.full,
                backgroundColor: `${COLORS.primary}15`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: COLORS.primary,
                fontSize: '18px',
                flexShrink: 0,
              }}
            >
              {icon}
            </div>
          )}
          <div
            style={{
              ...TYPOGRAPHY.body,
              color: COLORS.secondary,
              fontWeight: 500,
              flex: 1,
            }}
          >
            {title}
          </div>
        </div>

        {/* Value */}
        <div
          style={{
            ...TYPOGRAPHY.heading2,
            color: COLORS.neutralDark,
            fontWeight: 700,
            lineHeight: 1.2,
          }}
        >
          {typeof value === 'number' ? value.toLocaleString() : value}
        </div>

        {/* Trend Indicator */}
        {trendConfig && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: SPACING.xs,
              ...TYPOGRAPHY.caption,
              color: trendConfig.color,
              fontWeight: 600,
            }}
          >
            {trendConfig.icon}
            <span>{trendConfig.displayValue}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default MetricCard;

