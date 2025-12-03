/**
 * DevHelper Component
 * 
 * Development testing helper for resetting first-time experience flags.
 * Only shows in development mode.
 * 
 * Features:
 * - Reset First-Time Experience: Clears welcome screen and tooltip flags
 * - Clear All Data: Clears all localStorage data
 * - Fixed position: bottom-left on AI Analyst page, bottom-right elsewhere
 * - Only visible in development mode
 */

import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Button, Tooltip } from 'antd';
import { ReloadOutlined, DeleteOutlined, BugOutlined } from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS } from '@/styles/designTokens';

const DevHelper: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  // Only show in development mode
  const isDevelopment = import.meta.env.DEV || import.meta.env.MODE === 'development';

  if (!isDevelopment) {
    return null;
  }

  // Check if we're on the AI Analyst page
  const isAIAnalystPage = location.pathname === '/analyst' || location.pathname === '/ai-analyst';

  // Reset first-time experience flags
  const handleResetFirstTimeExperience = () => {
    // Clear all first-time experience flags
    localStorage.removeItem('hasSeenWelcome');
    localStorage.removeItem('hasSeenDashboardTips');
    localStorage.removeItem('hasSeenTopBarTip');
    
    // Reload page to apply changes
    window.location.reload();
  };

  // Clear all localStorage data
  const handleClearAllData = () => {
    if (window.confirm('Are you sure you want to clear all localStorage data? This will reset all user preferences.')) {
      localStorage.clear();
      window.location.reload();
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        bottom: isAIAnalystPage ? '80px' : SPACING.md,
        left: isAIAnalystPage ? '16px' : 'auto',
        right: isAIAnalystPage ? 'auto' : SPACING.md,
        zIndex: 100,
        display: 'flex',
        flexDirection: 'column',
        gap: SPACING.xs,
        alignItems: isAIAnalystPage ? 'flex-start' : 'flex-end',
      }}
    >
      {isOpen && (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: SPACING.xs,
            backgroundColor: '#FFFFFF',
            borderRadius: BORDER_RADIUS.md,
            padding: SPACING.sm,
            boxShadow: SHADOWS.cardHover,
            border: `1px solid #E2E8F0`,
            minWidth: '200px',
            marginBottom: SPACING.xs,
          }}
        >
          <Button
            type="default"
            icon={<ReloadOutlined />}
            onClick={handleResetFirstTimeExperience}
            style={{
              width: '100%',
              textAlign: 'left',
              fontSize: '13px',
              height: '36px',
            }}
          >
            Reset First-Time Experience
          </Button>
          <Button
            type="default"
            danger
            icon={<DeleteOutlined />}
            onClick={handleClearAllData}
            style={{
              width: '100%',
              textAlign: 'left',
              fontSize: '13px',
              height: '36px',
            }}
          >
            Clear All Data
          </Button>
        </div>
      )}
      <Tooltip title={isOpen ? 'Close Dev Helper' : 'Open Dev Helper'}>
        <Button
          type="primary"
          icon={<BugOutlined />}
          onClick={() => setIsOpen(!isOpen)}
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            backgroundColor: COLORS.primary,
            borderColor: COLORS.primary,
            boxShadow: SHADOWS.cardHover,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'scale(1.1)';
            e.currentTarget.style.boxShadow = SHADOWS.cardHover;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.boxShadow = SHADOWS.card;
          }}
        />
      </Tooltip>
    </div>
  );
};

export default DevHelper;




