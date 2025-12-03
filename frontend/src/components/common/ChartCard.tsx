/**
 * ChartCard Component
 * 
 * Premium chart container card with Basedash-inspired design.
 * Provides consistent styling for charts with headers, actions, and loading states.
 * 
 * Features:
 * - Clean white background with subtle shadow
 * - Header with title and optional action buttons
 * - Optional subtitle for context
 * - Proper padding for chart content
 * - Hover state with subtle border highlight
 * - Loading skeleton state
 */

import React from 'react';
import { Skeleton } from 'antd';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS, TYPOGRAPHY } from '@/styles/design-tokens';

export interface ChartCardProps {
  /**
   * Card title
   */
  title: string;

  /**
   * Optional subtitle for additional context
   */
  subtitle?: string;

  /**
   * Chart content (children)
   */
  children: React.ReactNode;

  /**
   * Optional action buttons in header
   */
  actions?: React.ReactNode | React.ReactNode[];

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
 * ChartCard - Premium chart container card
 */
const ChartCard: React.FC<ChartCardProps> = ({
  title,
  subtitle,
  children,
  actions,
  loading = false,
  className,
}) => {
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
        display: 'flex',
        flexDirection: 'column',
        gap: SPACING.lg,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#E5E7EB';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'transparent';
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: SPACING.md,
          flexWrap: 'wrap',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: SPACING.xs,
            flex: 1,
            minWidth: 0,
          }}
        >
          <h3
            style={{
              ...TYPOGRAPHY.heading2,
              color: COLORS.neutralDark,
              margin: 0,
              fontWeight: 600,
            }}
          >
            {title}
          </h3>
          {subtitle && (
            <p
              style={{
                ...TYPOGRAPHY.body,
                color: COLORS.secondary,
                margin: 0,
                fontWeight: 400,
              }}
            >
              {subtitle}
            </p>
          )}
        </div>

        {/* Actions */}
        {actions && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: SPACING.sm,
              flexShrink: 0,
            }}
          >
            {Array.isArray(actions) ? (
              actions.map((action, index) => (
                <React.Fragment key={index}>{action}</React.Fragment>
              ))
            ) : (
              actions
            )}
          </div>
        )}
      </div>

      {/* Content */}
      {loading ? (
        <Skeleton active paragraph={{ rows: 4 }} />
      ) : (
        <div
          style={{
            flex: 1,
            minHeight: 0,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
};

export default ChartCard;

