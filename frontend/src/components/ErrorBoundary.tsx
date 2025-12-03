/**
 * Error Boundary Component
 * 
 * Catches React errors in child component tree and displays a friendly fallback UI.
 * Prevents the entire app from crashing when an error occurs.
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Button } from 'antd';
import { ExclamationCircleOutlined, ReloadOutlined, DashboardOutlined } from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS, TYPOGRAPHY } from '@/styles/designTokens';

// ============================================================================
// TYPES
// ============================================================================

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

// ============================================================================
// ERROR BOUNDARY COMPONENT
// ============================================================================

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error details to console
    console.error('Error Boundary caught an error:', error);
    console.error('Error Info:', errorInfo);
    console.error('Component Stack:', errorInfo.componentStack);

    // Store error details in state for potential error reporting
    this.setState({
      error,
      errorInfo,
    });

    // Call optional error callback
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // TODO: Send error to error tracking service (e.g., Sentry, LogRocket)
    // Example:
    // if (window.Sentry) {
    //   window.Sentry.captureException(error, {
    //     contexts: {
    //       react: {
    //         componentStack: errorInfo.componentStack,
    //       },
    //     },
    //   });
    // }
  }

  handleReload = () => {
    // Reload the page to recover from error
    window.location.href = '/';
  };

  handleGoToDashboard = () => {
    // Navigate to dashboard
    window.location.href = '/dashboard';
  };

  render() {
    if (this.state.hasError) {
      // If custom fallback is provided, use it
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Otherwise, show default fallback UI
      return <ErrorFallbackUI onReload={this.handleReload} onGoToDashboard={this.handleGoToDashboard} />;
    }

    return this.props.children;
  }
}

// ============================================================================
// ERROR FALLBACK UI COMPONENT
// ============================================================================

interface ErrorFallbackUIProps {
  onReload: () => void;
  onGoToDashboard: () => void;
}

const ErrorFallbackUI: React.FC<ErrorFallbackUIProps> = ({ onReload, onGoToDashboard }) => {
  const [isVisible, setIsVisible] = React.useState(false);

  React.useEffect(() => {
    // Fade-in animation on mount
    setTimeout(() => setIsVisible(true), 100);
  }, []);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        width: '100%',
        padding: SPACING.lg,
        backgroundColor: '#FFFFFF',
        transition: 'opacity 0.3s ease',
        opacity: isVisible ? 1 : 0,
      }}
    >
      <div
        style={{
          maxWidth: '500px',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        {/* Error Icon */}
        <div
          style={{
            marginBottom: SPACING.md,
            animation: 'errorIconPulse 2s ease-in-out infinite',
          }}
        >
          <ExclamationCircleOutlined
            style={{
              fontSize: '48px',
              color: COLORS.danger,
            }}
          />
        </div>

        {/* Heading */}
        <h1
          style={{
            ...TYPOGRAPHY.headingMedium,
            color: COLORS.neutralDark,
            marginBottom: SPACING.sm,
            marginTop: 0,
            lineHeight: 1.3,
          }}
        >
          Oops! Something went wrong
        </h1>

        {/* Description */}
        <p
          style={{
            ...TYPOGRAPHY.body,
            color: '#6B7280',
            marginBottom: SPACING.lg,
            marginTop: 0,
            lineHeight: 1.6,
            maxWidth: '400px',
          }}
        >
          We've been notified and are working on a fix. Try reloading the page.
        </p>

        {/* Action Buttons */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: SPACING.sm,
            width: '100%',
            maxWidth: '300px',
            alignItems: 'stretch',
          }}
        >
          {/* Primary Button: Reload Page */}
          <Button
            type="primary"
            size="large"
            icon={<ReloadOutlined />}
            onClick={onReload}
            style={{
              height: '48px',
              padding: `0 ${SPACING.lg}`,
              background: `linear-gradient(135deg, ${COLORS.primary} 0%, ${COLORS.secondary} 100%)`,
              border: 'none',
              borderRadius: BORDER_RADIUS.sm,
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
            Reload Page
          </Button>

          {/* Secondary Button: Go to Dashboard */}
          <Button
            size="large"
            icon={<DashboardOutlined />}
            onClick={onGoToDashboard}
            style={{
              height: '44px',
              padding: `0 ${SPACING.lg}`,
              backgroundColor: '#FFFFFF',
              border: '1px solid #E5E7EB',
              borderRadius: BORDER_RADIUS.sm,
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
            Go to Dashboard
          </Button>
        </div>
      </div>

      {/* CSS Animations */}
      <style>{`
        @keyframes errorIconPulse {
          0%, 100% {
            transform: scale(1);
            opacity: 1;
          }
          50% {
            transform: scale(1.05);
            opacity: 0.9;
          }
        }
      `}</style>
    </div>
  );
};

export default ErrorBoundary;

