/**
 * UploadWizard Component
 * 
 * Premium step-by-step upload wizard with visual progress indicators.
 * 
 * Design: Premium SaaS style inspired by Linear's progress indicators
 * 
 * Features:
 * - Responsive layout (horizontal on desktop, vertical on mobile)
 * - Custom styled step icons and connectors
 * - Clear visual status indicators
 * - Clean, spacious design
 */

import React from 'react';
import { Steps } from 'antd';
import {
  UploadOutlined,
  CheckCircleOutlined,
  DashboardOutlined,
} from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS } from '@/styles/designTokens';


interface UploadWizardProps {
  /**
   * Current step index (0, 1, or 2)
   * 0 = Upload Data
   * 1 = Review Mapping
   * 2 = Dashboard Ready
   */
  currentStep: number;

  /**
   * Optional callback when step is clicked
   */
  onStepClick?: (step: number) => void;
}

const UploadWizard: React.FC<UploadWizardProps> = ({
  currentStep,
  onStepClick,
}) => {
  // Define steps configuration
  const steps = [
    {
      title: 'Upload Data',
      description: 'Choose your CSV file',
      icon: <UploadOutlined />,
      status: currentStep === 0 ? 'process' : currentStep > 0 ? 'finish' : 'wait',
    },
    {
      title: 'Review Mapping',
      description: 'Confirm column mapping',
      icon: <CheckCircleOutlined />,
      status: currentStep === 1 ? 'process' : currentStep > 1 ? 'finish' : 'wait',
    },
    {
      title: 'Dashboard Ready',
      description: 'View your insights',
      icon: <DashboardOutlined />,
      status: currentStep === 2 ? 'process' : currentStep > 2 ? 'finish' : 'wait',
    },
  ];

  // Handle step click
  const handleStepClick = (step: number) => {
    if (onStepClick) {
      onStepClick(step);
    }
  };

  return (
    <div
      style={{
        width: '100%',
        marginBottom: SPACING.lg, // 32px
        backgroundColor: '#FFFFFF',
        borderRadius: BORDER_RADIUS.md, // 12px
        padding: SPACING.md, // 24px
        boxShadow: SHADOWS.card,
      }}
    >
      <Steps
        current={currentStep}
        direction="horizontal"
        responsive={true}
        style={{
          width: '100%',
        }}
        items={steps.map((step) => ({
          title: step.title,
          description: step.description,
          icon: step.icon,
          status: step.status as 'wait' | 'process' | 'finish' | 'error',
        }))}
        className="upload-wizard-steps"
        onChange={onStepClick ? (current) => handleStepClick(current) : undefined}
        // Disable default badges/notifications
        progressDot={false}
      />

      {/* Custom styles for step components */}
      <style>{`
        /* Step container */
        .upload-wizard-steps .ant-steps-item {
          cursor: ${onStepClick ? 'pointer' : 'default'};
          transition: all 0.2s ease;
        }

        .upload-wizard-steps .ant-steps-item:hover {
          opacity: ${onStepClick ? '0.8' : '1'};
        }

        /* Step icon container */
        .upload-wizard-steps .ant-steps-item-icon {
          width: 32px !important;
          height: 32px !important;
          font-size: 16px !important;
          border-radius: 50% !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
          border: 2px solid !important;
          transition: all 0.2s ease !important;
        }

        /* Remove any badges or notification dots */
        .upload-wizard-steps .ant-steps-item-icon .ant-badge {
          display: none !important;
        }

        .upload-wizard-steps .ant-steps-item-icon .ant-badge-dot {
          display: none !important;
        }

        .upload-wizard-steps .ant-steps-item-icon .ant-badge-count {
          display: none !important;
        }

        /* Wait state (inactive) */
        .upload-wizard-steps .ant-steps-item-wait .ant-steps-item-icon {
          background-color: #FFFFFF !important;
          border-color: #D1D5DB !important;
          color: #D1D5DB !important;
        }

        /* Process state (active) */
        .upload-wizard-steps .ant-steps-item-process .ant-steps-item-icon {
          background-color: ${COLORS.primary} !important;
          border-color: ${COLORS.primary} !important;
          color: #FFFFFF !important;
        }

        /* Finish state (completed) */
        .upload-wizard-steps .ant-steps-item-finish .ant-steps-item-icon {
          background-color: ${COLORS.success} !important;
          border-color: ${COLORS.success} !important;
          color: #FFFFFF !important;
        }

        /* Step title */
        .upload-wizard-steps .ant-steps-item-title {
          font-size: 16px !important;
          font-weight: 600 !important;
          line-height: 1.5 !important;
          color: ${COLORS.neutralDark} !important;
        }

        .upload-wizard-steps .ant-steps-item-wait .ant-steps-item-title {
          color: #9CA3AF !important;
        }

        /* Step description */
        .upload-wizard-steps .ant-steps-item-description {
          font-size: 14px !important;
          font-weight: 400 !important;
          line-height: 1.5 !important;
          color: #6B7280 !important;
          margin-top: 4px !important;
        }

        .upload-wizard-steps .ant-steps-item-wait .ant-steps-item-description {
          color: #9CA3AF !important;
        }

        /* Connector line */
        .upload-wizard-steps .ant-steps-item-tail {
          top: 16px !important;
          height: 2px !important;
          background-color: #D1D5DB !important;
          border: none !important;
        }

        .upload-wizard-steps .ant-steps-item-finish .ant-steps-item-tail {
          background-color: ${COLORS.success} !important;
        }

        .upload-wizard-steps .ant-steps-item-process .ant-steps-item-tail {
          background-color: #D1D5DB !important;
        }

        /* Responsive: Vertical on mobile */
        @media (max-width: 768px) {
          .upload-wizard-steps {
            flex-direction: column !important;
          }

          .upload-wizard-steps .ant-steps-item {
            width: 100% !important;
            margin-bottom: ${SPACING.md} !important;
          }

          .upload-wizard-steps .ant-steps-item-tail {
            display: none !important;
          }
        }

        /* Step icon hover effect */
        .upload-wizard-steps .ant-steps-item:not(.ant-steps-item-wait):hover .ant-steps-item-icon {
          transform: scale(1.05);
        }
      `}</style>
    </div>
  );
};

export default UploadWizard;

