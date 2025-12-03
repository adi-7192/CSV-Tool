/**
 * EmptyState Component
 * 
 * Premium empty state component with Basedash-inspired design.
 * Used to display helpful messages when there's no data or content to show.
 * 
 * Features:
 * - Centered layout with generous white space
 * - Icon display with optional colored background
 * - Clear typography hierarchy
 * - Optional action button
 * - Clean, minimalist design
 */

import React from 'react';
import { Button } from 'antd';
import { COLORS, SPACING, BORDER_RADIUS, TYPOGRAPHY } from '@/styles/design-tokens';

export interface EmptyStateProps {
  /**
   * Icon to display (React node)
   */
  icon: React.ReactNode;
  
  /**
   * Main title text
   */
  title: string;
  
  /**
   * Description text below title
   */
  description?: string;
  
  /**
   * Optional action button
   */
  action?: {
    text: string;
    onClick: () => void;
  };
  
  /**
   * Optional className for custom styling
   */
  className?: string;
}

/**
 * EmptyState - Premium empty state component
 */
const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  className,
}) => {
  return (
    <div
      className={`empty-state-enter ${className || ''}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: SPACING['2xl'],
        textAlign: 'center',
        minHeight: '300px',
      }}
    >
      {/* Icon */}
      <div
        className="empty-state-icon"
        style={{
          width: '64px',
          height: '64px',
          borderRadius: BORDER_RADIUS.full,
          backgroundColor: `${COLORS.primary}10`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: COLORS.primary,
          fontSize: '32px',
          marginBottom: SPACING.lg,
          flexShrink: 0,
        }}
      >
        {icon}
      </div>

      {/* Title */}
      <h3
        style={{
          ...TYPOGRAPHY.heading2,
          color: COLORS.neutralDark,
          margin: 0,
          marginBottom: SPACING.sm,
          fontWeight: 600,
        }}
      >
        {title}
      </h3>

      {/* Description */}
      {description && (
        <p
          style={{
            ...TYPOGRAPHY.body,
            color: COLORS.secondary,
            margin: 0,
            marginBottom: action ? SPACING.lg : 0,
            maxWidth: '400px',
            fontWeight: 400,
          }}
        >
          {description}
        </p>
      )}

      {/* Action Button */}
      {action && (
        <Button
          type="primary"
          onClick={action.onClick}
          style={{
            marginTop: SPACING.md,
            borderRadius: BORDER_RADIUS.md,
            height: '40px',
            paddingLeft: SPACING.lg,
            paddingRight: SPACING.lg,
            fontWeight: 500,
          }}
        >
          {action.text}
        </Button>
      )}
    </div>
  );
};

export default EmptyState;

