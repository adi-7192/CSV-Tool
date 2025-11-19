import React from 'react';
import { Card, Skeleton } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { formatCurrency, formatNumber, formatPercentage } from '@/utils/formatters';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

export interface MetricCardProps {
  title: string;              // "Gross Revenue"
  value: number;              // 2830655.40
  format: 'currency' | 'number' | 'percentage';
  trend?: number;             // 5.2 (percentage)
  trendLabel?: string;        // "vs last week"
  sparklineData?: number[];   // [100, 120, 115, 130, ...]
  loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  format,
  trend,
  trendLabel,
  sparklineData,
  loading = false,
}) => {
  // Format value based on format type
  const formatValue = () => {
    switch (format) {
      case 'currency':
        return formatCurrency(value);
      case 'percentage':
        return formatPercentage(value);
      case 'number':
        return formatNumber(value);
      default:
        return value.toString();
    }
  };

  // Determine trend color and icon
  const getTrendConfig = () => {
    if (trend === undefined || trend === null) return null;
    
    const isPositive = trend >= 0;
    return {
      color: isPositive ? '#10B981' : '#F43F5E', // Emerald 500 or Rose 500
      icon: isPositive ? <ArrowUpOutlined /> : <ArrowDownOutlined />,
      displayValue: `${Math.abs(trend).toFixed(1)}%`,
    };
  };

  // Convert sparkline data array to Recharts format
  const sparklineChartData = sparklineData?.map((val, index) => ({
    value: val,
    index,
  })) || [];

  const trendConfig = getTrendConfig();

  if (loading) {
    return (
      <Card
        style={{
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          transition: 'box-shadow 0.2s',
          height: '100%',
        }}
      >
        <Skeleton active paragraph={{ rows: 2 }} />
      </Card>
    );
  }

  return (
    <Card
      style={{
        borderRadius: '8px',
        border: '1px solid #E2E8F0',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        padding: '20px',
        backgroundColor: '#FFFFFF',
        transition: 'box-shadow 0.2s',
        height: '100%',
      }}
      hoverable
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
      }}
      bodyStyle={{ padding: 0 }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {/* Title */}
        <div
          style={{
            fontSize: '14px',
            fontWeight: '500',
            color: '#64748B', // Slate 500
            lineHeight: '1.4',
          }}
        >
          {title}
        </div>

        {/* Value and Trend */}
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: '8px',
            flexWrap: 'wrap',
          }}
        >
          <span
            style={{
              fontSize: '28px',
              fontWeight: '700',
              color: '#030712', // Slate 950
              lineHeight: '1.2',
            }}
          >
            {formatValue()}
          </span>

          {trendConfig && (
            <span
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '14px',
                fontWeight: '600',
                color: trendConfig.color,
              }}
            >
              {trendConfig.icon}
              {trendConfig.displayValue}
            </span>
          )}
        </div>

        {/* Sparkline Chart */}
        {sparklineData && sparklineData.length > 0 && (
          <div
            style={{
              marginTop: '8px',
              marginBottom: '4px',
              height: '40px',
              width: '100%',
            }}
          >
            <ResponsiveContainer width="100%" height={40}>
              <LineChart data={sparklineChartData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#6366F1"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Trend Label */}
        {trendLabel && (
          <div
            style={{
              fontSize: '12px',
              color: '#94A3B8', // Slate 400
              lineHeight: '1.4',
              marginTop: trendConfig ? '0' : '4px',
            }}
          >
            {trendLabel}
          </div>
        )}
      </div>
    </Card>
  );
};

export default MetricCard;
