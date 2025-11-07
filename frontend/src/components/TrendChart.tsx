import React from 'react';
import { Card, Skeleton } from 'antd';
import { LineChart, AreaChart, XAxis, YAxis, CartesianGrid, Tooltip, Line, Area, ResponsiveContainer } from 'recharts';
import { formatCurrency } from '@/utils/formatters';

export interface TrendChartProps {
  title: string;              // "Revenue Trend"
  type: 'line' | 'area';     // Chart type
  data: Array<{
    date: string;             // "Jul 1"
    value: number;            // 1234567
  }>;
  color?: string;             // Indigo 500 default
  loading?: boolean;
  error?: string;
  height?: number;            // Chart height in pixels (default: 400)
}

const TrendChart: React.FC<TrendChartProps> = ({
  title,
  type,
  data,
  color = '#6366F1', // Indigo 500 default
  loading = false,
  error,
  height = 400, // Default height
}) => {
  // Format Y-axis tick values
  const formatYAxisTick = (value: number) => {
    if (value >= 1000000) {
      return `₹${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `₹${(value / 1000).toFixed(1)}K`;
    }
    return `₹${value.toLocaleString()}`;
  };

  // Format tooltip values
  const formatTooltipValue = (value: number) => {
    return formatCurrency(value);
  };

  // Format tooltip label
  const formatTooltipLabel = (label: string) => {
    return `Date: ${label}`;
  };

  if (loading) {
    return (
      <Card
        style={{
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          backgroundColor: '#FFFFFF',
          padding: '20px',
        }}
      >
        <Skeleton active paragraph={{ rows: 4 }} />
      </Card>
    );
  }

  if (error) {
    return (
      <Card
        style={{
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          backgroundColor: '#FFFFFF',
          padding: '20px',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '300px',
            color: '#F43F5E', // Rose 500
            fontSize: '14px',
            fontWeight: '500',
          }}
        >
          {error || 'Failed to load chart'}
        </div>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card
        style={{
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          backgroundColor: '#FFFFFF',
          padding: '20px',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '300px',
            color: '#64748B', // Slate 500
            fontSize: '14px',
          }}
        >
          No data available
        </div>
      </Card>
    );
  }

  // Calculate total card height: title (16px) + gap (16px) + chart height
  const cardHeight = height + 16 + 16 + 20; // height + title + gap + padding
  
  return (
    <Card
      style={{
        borderRadius: '8px',
        border: '1px solid #E2E8F0',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        backgroundColor: '#FFFFFF',
        padding: '20px',
        height: `${cardHeight}px`,
      }}
      bodyStyle={{ padding: 0 }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Title */}
        <div
          style={{
            fontSize: '16px',
            fontWeight: '600',
            color: '#030712', // Slate 950
            lineHeight: '1.4',
            textAlign: 'center',
            width: '100%',
          }}
        >
          {title}
        </div>

        {/* Chart */}
        <div
          style={{
            width: '100%',
            minWidth: '300px',
            height: `${height}px`,
          }}
        >
          <ResponsiveContainer width="100%" height={height}>
            {type === 'line' ? (
              <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#E2E8F0"
                  strokeWidth={1}
                />
                <XAxis
                  dataKey="date"
                  stroke="#94A3B8"
                  fontSize={12}
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                />
                <YAxis
                  stroke="#94A3B8"
                  fontSize={12}
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tickFormatter={formatYAxisTick}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E2E8F0',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                  }}
                  labelStyle={{
                    color: '#64748B',
                    fontSize: '12px',
                    fontWeight: '500',
                    marginBottom: '4px',
                  }}
                  formatter={(value: number) => formatTooltipValue(value)}
                  labelFormatter={formatTooltipLabel}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: color }}
                  isAnimationActive={true}
                  animationDuration={300}
                />
              </LineChart>
            ) : (
              <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id={`gradient-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={color} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#E2E8F0"
                  strokeWidth={1}
                />
                <XAxis
                  dataKey="date"
                  stroke="#94A3B8"
                  fontSize={12}
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                />
                <YAxis
                  stroke="#94A3B8"
                  fontSize={12}
                  tickLine={false}
                  axisLine={{ stroke: '#E2E8F0' }}
                  tickFormatter={formatYAxisTick}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E2E8F0',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                  }}
                  labelStyle={{
                    color: '#64748B',
                    fontSize: '12px',
                    fontWeight: '500',
                    marginBottom: '4px',
                  }}
                  formatter={(value: number) => formatTooltipValue(value)}
                  labelFormatter={formatTooltipLabel}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke={color}
                  strokeWidth={2}
                  fill={`url(#gradient-${color.replace('#', '')})`}
                  dot={false}
                  activeDot={{ r: 4, fill: color }}
                />
              </AreaChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  );
};

export default TrendChart;

