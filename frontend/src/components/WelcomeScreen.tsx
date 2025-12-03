/**
 * WelcomeScreen Component
 * 
 * Premium onboarding modal that appears on first visit after authentication.
 * 
 * Display Conditions:
 * - Only shows after user is authenticated (Firebase)
 * - Only shows on first visit (checks localStorage 'hasSeenWelcome')
 * - Full-screen modal overlay
 * 
 * Design: Premium SaaS style inspired by Notion's onboarding modals
 */

import React, { useState, useEffect } from 'react';
import { Modal, Button, message } from 'antd';
import { RocketOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS, TYPOGRAPHY } from '@/styles/designTokens';

interface WelcomeScreenProps {
  /**
   * Whether user is authenticated
   * TODO: Replace with Firebase auth check when implemented
   */
  isAuthenticated?: boolean;
  
  /**
   * Product name to display in welcome message
   */
  productName?: string;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  isAuthenticated = true, // TODO: Replace with actual Firebase auth check
  productName = 'Analytics Dashboard',
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const navigate = useNavigate();

  // Check if welcome screen should be shown
  useEffect(() => {
    // Only show if:
    // 1. User is authenticated
    // 2. User hasn't seen welcome screen before
    if (isAuthenticated) {
      const hasSeenWelcome = localStorage.getItem('hasSeenWelcome');
      if (!hasSeenWelcome) {
        // Small delay for smooth animation
        setTimeout(() => {
          setIsVisible(true);
        }, 300);
      }
    }
  }, [isAuthenticated]);

  // Handle primary button click
  const handleGetStarted = () => {
    // Set flag to prevent showing again
    localStorage.setItem('hasSeenWelcome', 'true');
    
    // Close modal
    setIsVisible(false);
    
    // Navigate to workspace
    navigate('/workspace');
  };

  // Handle secondary link click
  const handleTrySample = () => {
    message.info('Sample data coming soon!', 3);
  };

  // Handle modal close (if user clicks outside or X button)
  const handleClose = () => {
    localStorage.setItem('hasSeenWelcome', 'true');
    setIsVisible(false);
  };

  // Don't render if not authenticated or already seen
  if (!isAuthenticated) {
    return null;
  }

  return (
    <>
      <Modal
        open={isVisible}
        onCancel={handleClose}
        footer={null}
        closable={true}
        maskClosable={true}
        width={600}
        centered
        styles={{
          // Custom modal styles
          content: {
            padding: SPACING.xl, // 48px
            borderRadius: BORDER_RADIUS.lg, // 16px
            boxShadow: SHADOWS.cardHover,
            backgroundColor: '#FFFFFF',
            position: 'relative',
            maxWidth: '600px',
          },
          mask: {
            backgroundColor: 'rgba(0, 0, 0, 0.6)', // 60% opacity dark overlay
            backdropFilter: 'blur(4px)',
          },
          body: {
            padding: 0,
          },
        }}
        // Remove default modal styling
        className="welcome-screen-modal"
        wrapClassName="welcome-screen-wrapper"
      >
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        {/* Icon */}
        <div
          style={{
            marginBottom: SPACING.md, // 24px
            animation: 'fadeInScale 0.4s ease-out 0.1s both',
          }}
        >
          <RocketOutlined
            style={{
              fontSize: '64px',
              color: COLORS.primary, // #3B82F6
            }}
          />
        </div>

        {/* Headline */}
        <h1
          style={{
            ...TYPOGRAPHY.headingLarge,
            color: COLORS.neutralDark, // #1F2937
            marginBottom: SPACING.sm, // 16px
            marginTop: 0,
            animation: 'fadeInUp 0.4s ease-out 0.2s both',
          }}
        >
          Welcome to {productName}! 👋
        </h1>

        {/* Subheadline */}
        <p
          style={{
            fontSize: '18px',
            fontWeight: 400,
            lineHeight: 1.6,
            color: '#6B7280',
            marginBottom: SPACING.lg, // 32px
            marginTop: 0,
            animation: 'fadeInUp 0.4s ease-out 0.3s both',
          }}
        >
          Let's get your first insights in 2 minutes
        </p>

        {/* Primary Button */}
        <Button
          type="primary"
          size="large"
          onClick={handleGetStarted}
          block
          style={{
            height: '48px',
            fontSize: '16px',
            fontWeight: 700,
            borderRadius: BORDER_RADIUS.md, // 12px
            background: `linear-gradient(135deg, ${COLORS.primary} 0%, ${COLORS.secondary} 100%)`,
            border: 'none',
            boxShadow: '0 4px 12px rgba(59, 130, 246, 0.4)',
            transition: 'all 0.2s ease',
            marginBottom: SPACING.sm, // 16px
            animation: 'fadeInUp 0.4s ease-out 0.4s both',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 6px 16px rgba(59, 130, 246, 0.5)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.4)';
          }}
        >
          Upload Your First File
        </Button>

        {/* Secondary Link */}
        <button
          onClick={handleTrySample}
          style={{
            background: 'none',
            border: 'none',
            color: '#6B7280',
            fontSize: '14px',
            cursor: 'pointer',
            padding: `${SPACING.xs} ${SPACING.sm}`,
            textDecoration: 'none',
            transition: 'all 0.2s ease',
            animation: 'fadeInUp 0.4s ease-out 0.5s both',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = COLORS.primary;
            e.currentTarget.style.textDecoration = 'underline';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = '#6B7280';
            e.currentTarget.style.textDecoration = 'none';
          }}
        >
          Try with sample data
        </button>
      </div>

      </Modal>
      
      {/* Global styles for animations */}
      <style>{`
        /* Modal fade in animation */
        .welcome-screen-wrapper .ant-modal-mask {
          animation: welcomeFadeIn 0.3s ease-out;
        }

        @keyframes welcomeFadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        /* Modal content slide up animation */
        .welcome-screen-modal .ant-modal-content {
          animation: welcomeSlideUp 0.3s ease-out;
        }

        @keyframes welcomeSlideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        /* Content animations */
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes fadeInScale {
          from {
            opacity: 0;
            transform: scale(0.9);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
      `}</style>
    </>
  );
};

export default WelcomeScreen;

