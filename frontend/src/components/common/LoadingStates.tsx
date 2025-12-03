/**
 * LoadingStates Component
 * 
 * Premium loading state components with consistent styling.
 * Uses design tokens and animations for a polished experience.
 */

import React from 'react';
import { Spin, Skeleton } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import { COLORS, SPACING, BORDER_RADIUS, TYPOGRAPHY } from '@/styles/design-tokens';
import './LoadingStates.css';

interface PageLoaderProps {
  /**
   * Optional loading text
   */
  text?: string;
  
  /**
   * Optional custom size
   */
  size?: 'small' | 'default' | 'large';
}

/**
 * PageLoader - Full-page loading spinner
 */
export const PageLoader: React.FC<PageLoaderProps> = ({ 
  text = 'Loading...', 
  size = 'large' 
}) => {
  const [showLoader, setShowLoader] = React.useState(false);

  React.useEffect(() => {
    // Delay showing loader to avoid flash
    const timer = setTimeout(() => setShowLoader(true), 200);
    return () => clearTimeout(timer);
  }, []);

  if (!showLoader) return null;

  return (
    <div className="page-loader">
      <Spin
        indicator={
          <LoadingOutlined
            style={{
              fontSize: size === 'large' ? 48 : size === 'default' ? 32 : 24,
              color: COLORS.primary,
            }}
            spin
          />
        }
        size={size}
      />
      {text && (
        <p
          style={{
            marginTop: SPACING.lg,
            ...TYPOGRAPHY.body,
            color: COLORS.secondary,
          }}
        >
          {text}
        </p>
      )}
    </div>
  );
};

interface CardLoaderProps {
  /**
   * Show title skeleton
   */
  showTitle?: boolean;
  
  /**
   * Number of content lines
   */
  lines?: number;
  
  /**
   * Show action buttons skeleton
   */
  showActions?: boolean;
  
  /**
   * Custom height
   */
  height?: string;
}

/**
 * CardLoader - Skeleton loader for card content
 */
export const CardLoader: React.FC<CardLoaderProps> = ({
  showTitle = true,
  lines = 3,
  showActions = false,
  height,
}) => {
  return (
    <div
      className="card-loader"
      style={{
        height,
        padding: SPACING.lg,
        borderRadius: BORDER_RADIUS.lg,
        backgroundColor: COLORS.background,
      }}
    >
      {showTitle && (
        <Skeleton
          active
          title={{ width: '60%' }}
          paragraph={false}
          className="card-loader-title"
        />
      )}
      <Skeleton
        active
        paragraph={{ rows: lines }}
        title={false}
        className="card-loader-content"
      />
      {showActions && (
        <div
          style={{
            display: 'flex',
            gap: SPACING.sm,
            marginTop: SPACING.md,
          }}
        >
          <Skeleton.Button active size="default" style={{ width: 100 }} />
          <Skeleton.Button active size="default" style={{ width: 100 }} />
        </div>
      )}
    </div>
  );
};

interface TableLoaderProps {
  /**
   * Number of skeleton rows
   */
  rows?: number;
  
  /**
   * Number of columns
   */
  columns?: number;
}

/**
 * TableLoader - Skeleton loader for table rows
 */
export const TableLoader: React.FC<TableLoaderProps> = ({
  rows = 5,
  columns = 4,
}) => {
  return (
    <div className="table-loader">
      {/* Header skeleton */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${columns}, 1fr)`,
          gap: SPACING.md,
          padding: SPACING.md,
          borderBottom: `1px solid #E5E7EB`,
          marginBottom: SPACING.sm,
        }}
      >
        {Array.from({ length: columns }).map((_, i) => (
          <Skeleton
            key={`header-${i}`}
            active
            title={{ width: '80%' }}
            paragraph={false}
          />
        ))}
      </div>
      
      {/* Row skeletons */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div
          key={`row-${rowIndex}`}
          className="table-loader-row"
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${columns}, 1fr)`,
            gap: SPACING.md,
            padding: SPACING.md,
            borderBottom: `1px solid #F3F4F6`,
          }}
        >
          {Array.from({ length: columns }).map((_, colIndex) => (
            <Skeleton
              key={`cell-${rowIndex}-${colIndex}`}
              active
              title={{ width: colIndex === 0 ? '100%' : '60%' }}
              paragraph={false}
            />
          ))}
        </div>
      ))}
    </div>
  );
};

interface ChartLoaderProps {
  /**
   * Chart width
   */
  width?: string;
  
  /**
   * Chart height
   */
  height?: string;
  
  /**
   * Show axis placeholders
   */
  showAxes?: boolean;
}

/**
 * ChartLoader - Skeleton loader for chart areas
 */
export const ChartLoader: React.FC<ChartLoaderProps> = ({
  width = '100%',
  height = '300px',
  showAxes = true,
}) => {
  return (
    <div
      className="chart-loader"
      style={{
        width,
        height,
        position: 'relative',
        padding: SPACING.lg,
      }}
    >
      {/* Main chart area */}
      <Skeleton
        active
        title={false}
        paragraph={{ rows: 0 }}
        className="chart-loader-main"
        style={{
          width: '100%',
          height: '100%',
          borderRadius: BORDER_RADIUS.md,
        }}
      />
      
      {/* Axis placeholders */}
      {showAxes && (
        <>
          <div
            className="chart-loader-axis-x"
            style={{
              position: 'absolute',
              bottom: SPACING.lg,
              left: SPACING.lg,
              right: SPACING.lg,
              height: '2px',
              backgroundColor: '#E5E7EB',
              borderRadius: '1px',
            }}
          />
          <div
            className="chart-loader-axis-y"
            style={{
              position: 'absolute',
              top: SPACING.lg,
              bottom: SPACING.lg,
              left: SPACING.lg,
              width: '2px',
              backgroundColor: '#E5E7EB',
              borderRadius: '1px',
            }}
          />
        </>
      )}
    </div>
  );
};

/**
 * InlineLoader - Small inline loading spinner
 */
export const InlineLoader: React.FC<{ size?: number }> = ({ size = 16 }) => {
  return (
    <Spin
      indicator={
        <LoadingOutlined
          style={{
            fontSize: size,
            color: COLORS.primary,
          }}
          spin
        />
      }
      size="small"
    />
  );
};

export default {
  PageLoader,
  CardLoader,
  TableLoader,
  ChartLoader,
  InlineLoader,
};

