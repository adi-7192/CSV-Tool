/**
 * DataCard Component
 * 
 * Generic content card for data displays with Basedash-inspired design.
 * Flexible container for tables, lists, or custom content.
 * 
 * Features:
 * - Clean white background with subtle shadow
 * - Optional title and subtitle
 * - Flexible content area
 * - Optional footer section for actions/metadata
 * - Hoverable state with elevation effect
 * - Consistent spacing and typography
 */

import React from 'react';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS, TYPOGRAPHY } from '@/styles/design-tokens';

export interface DataCardProps {
  /**
   * Card title
   */
  title?: string | React.ReactNode;
  
  /**
   * Optional subtitle for additional context
   */
  subtitle?: string;
  
  /**
   * Card content (children)
   */
  children: React.ReactNode;
  
  /**
   * Optional action buttons in header
   */
  actions?: React.ReactNode | React.ReactNode[];
  
  /**
   * Optional footer content (actions, metadata, etc.)
   */
  footer?: React.ReactNode;
  
  /**
   * Enable hover effect with elevation
   */
  hoverable?: boolean;
  
  /**
   * Optional className for custom styling
   */
  className?: string;
  
  /**
   * Optional inline style
   */
  style?: React.CSSProperties;
}

/**
 * DataCard - Generic content card for data displays
 */
const DataCard: React.FC<DataCardProps> = ({
  title,
  subtitle,
  children,
  actions,
  footer,
  hoverable = false,
  className,
  style,
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
        ...(hoverable && {
          cursor: 'pointer',
        }),
        ...style,
      }}
      onMouseEnter={
        hoverable
          ? (e) => {
              e.currentTarget.style.boxShadow = SHADOWS.lg;
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.borderColor = '#E5E7EB';
            }
          : undefined
      }
      onMouseLeave={
        hoverable
          ? (e) => {
              e.currentTarget.style.boxShadow = SHADOWS.md;
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.borderColor = 'transparent';
            }
          : undefined
      }
    >
      {/* Header */}
      {(title || subtitle || actions) && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
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
            }}
          >
            {title && (
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
            )}
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
          {actions && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: SPACING.sm,
                flexWrap: 'wrap',
              }}
            >
              {Array.isArray(actions) ? actions.map((action, index) => (
                <React.Fragment key={index}>{action}</React.Fragment>
              )) : actions}
            </div>
          )}
        </div>
      )}

      {/* Content */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
        }}
      >
        {children}
      </div>

      {/* Footer */}
      {footer && (
        <div
          style={{
            paddingTop: SPACING.md,
            borderTop: '1px solid #E5E7EB',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: SPACING.md,
            flexWrap: 'wrap',
          }}
        >
          {footer}
        </div>
      )}
    </div>
  );
};

export default DataCard;

