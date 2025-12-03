/**
 * EmptyState Component
 * 
 * Premium empty state component with Stripe-inspired design.
 * Used to display helpful messages when there's no data or content to show.
 * 
 * Features:
 * - Centered layout with generous white space
 * - Gradient icon container with breathing animation
 * - Clear typography hierarchy
 * - Primary and optional secondary action buttons
 * - Smooth fade-in animation
 * - Premium SaaS styling
 */

import React, { useEffect, useState } from 'react';
import { Button } from 'antd';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS } from '@/styles/designTokens';

export interface EmptyStateProps {
  /**
   * Icon to display in the gradient circle
   */
  icon: React.ReactNode;
  
  /**
   * Main title text
   */
  title: string;
  
  /**
   * Description text below title (can be string or ReactNode for custom content)
   */
  description: string | React.ReactNode;
  
  /**
   * Primary action button
   */
  primaryButton: {
    text: string;
    onClick: () => void;
  };
  
  /**
   * Optional secondary action button
   */
  secondaryButton?: {
    text: string;
    onClick: () => void;
  };
}

const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  primaryButton,
  secondaryButton,
}) => {
  const [isVisible, setIsVisible] = useState(false);

  // Fade in animation on mount
  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px',
        width: '100%',
        padding: SPACING.lg,
        opacity: isVisible ? 1 : 0,
        transition: 'opacity 0.4s ease',
      }}
    >
      <div
        style={{
          maxWidth: '400px',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        {/* Icon Container */}
        <div
          className="empty-state-icon-container"
          style={{
            width: '120px',
            height: '120px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: SPACING.md, // 24px
            position: 'relative',
          }}
        >
          <div
            style={{
              fontSize: '48px',
              color: COLORS.primary, // #3B82F6
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </div>
        </div>

        {/* Title */}
        <h2
          style={{
            fontSize: '24px',
            fontWeight: 700,
            color: COLORS.neutralDark, // #1F2937
            marginBottom: SPACING.sm, // 12px
            marginTop: 0,
            lineHeight: 1.3,
            textAlign: 'center',
          }}
        >
          {title}
        </h2>

        {/* Description */}
        <div
          style={{
            fontSize: '16px',
            fontWeight: 400,
            color: '#6B7280',
            lineHeight: 1.6,
            marginBottom: SPACING.lg, // 32px
            marginTop: 0,
            textAlign: 'center',
            maxWidth: '400px',
            width: '100%',
          }}
        >
          {typeof description === 'string' ? (
            <p style={{ margin: 0 }}>{description}</p>
          ) : (
            description
          )}
        </div>

        {/* Action Buttons */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: SPACING.sm, // 16px
            width: '100%',
            maxWidth: '300px',
            alignItems: 'stretch',
          }}
        >
          {/* Primary Button */}
          <Button
            type="primary"
            size="large"
            onClick={primaryButton.onClick}
            style={{
              height: '48px',
              padding: `0 ${SPACING.lg}`, // 0 32px
              background: `linear-gradient(135deg, ${COLORS.primary} 0%, ${COLORS.secondary} 100%)`,
              border: 'none',
              borderRadius: BORDER_RADIUS.sm, // 8px
              color: '#FFFFFF',
              fontSize: '16px',
              fontWeight: 600,
              boxShadow: SHADOWS.card,
              transition: 'all 0.2s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = SHADOWS.cardHover;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = SHADOWS.card;
            }}
          >
            {primaryButton.text}
          </Button>

          {/* Secondary Button (Optional) */}
          {secondaryButton && (
            <Button
              size="large"
              onClick={secondaryButton.onClick}
              style={{
                height: '44px',
                padding: `0 ${SPACING.lg}`, // 0 32px
                backgroundColor: '#FFFFFF',
                border: '1px solid #E5E7EB',
                borderRadius: BORDER_RADIUS.sm, // 8px
                color: '#6B7280',
                fontSize: '16px',
                fontWeight: 600,
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#D1D5DB';
                e.currentTarget.style.color = COLORS.neutralDark;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#E5E7EB';
                e.currentTarget.style.color = '#6B7280';
              }}
            >
              {secondaryButton.text}
            </Button>
          )}
        </div>
      </div>

      {/* Styles for breathing animation */}
      <style>{`
        @keyframes emptyStateBreathing {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.05);
          }
        }

        .empty-state-icon-container {
          animation: emptyStateBreathing 2s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
};

export default EmptyState;

