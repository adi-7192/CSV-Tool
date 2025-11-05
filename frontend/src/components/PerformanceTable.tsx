import React, { useState } from 'react';
import { Table, Skeleton } from 'antd';
import type { ColumnsType, TableProps } from 'antd/es/table';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { formatCurrency, formatPercentage } from '@/utils/formatters';

export interface PerformanceTableRow {
  id: string;
  sku: string;
  asin: string;
  unitsSold: number;
  revenue: number;
  refundRatio: number; // 0-100 percentage
  rating: number; // 1-5 stars
  trend: number; // percentage change
}

export interface PerformanceTableProps {
  data: PerformanceTableRow[];
  loading?: boolean;
  error?: string;
  onRowClick?: (row: PerformanceTableRow) => void;
}

const PerformanceTable: React.FC<PerformanceTableProps> = ({
  data,
  loading = false,
  error,
  onRowClick,
}) => {
  const [sortedInfo, setSortedInfo] = useState<{
    order?: 'ascend' | 'descend';
    columnKey?: string;
  }>({});

  // Render rating as stars
  const renderRating = (rating: number) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

    return (
      <span style={{ color: '#F59E0B', fontSize: '14px' }}>
        {'★'.repeat(fullStars)}
        {hasHalfStar && '☆'}
        {'☆'.repeat(emptyStars)}
        <span style={{ color: '#64748B', marginLeft: '4px', fontSize: '12px' }}>
          {rating.toFixed(1)}
        </span>
      </span>
    );
  };

  // Render refund ratio with color coding
  const renderRefundRatio = (ratio: number) => {
    let color = '#10B981'; // Green (good)
    if (ratio >= 3 && ratio <= 5) {
      color = '#F59E0B'; // Amber (warning)
    } else if (ratio > 5) {
      color = '#F43F5E'; // Rose (bad)
    }

    return (
      <span style={{ color, fontWeight: '600' }}>
        {formatPercentage(ratio)}
      </span>
    );
  };

  // Render trend with arrow and color
  const renderTrend = (trend: number) => {
    const isPositive = trend >= 0;
    const color = isPositive ? '#10B981' : '#F43F5E';
    const icon = isPositive ? <ArrowUpOutlined /> : <ArrowDownOutlined />;

    return (
      <span
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          color,
          fontWeight: '600',
          justifyContent: 'flex-end',
        }}
      >
        {icon}
        {formatPercentage(Math.abs(trend))}
      </span>
    );
  };

  // Handle table change (sorting, pagination)
  const handleTableChange: TableProps<PerformanceTableRow>['onChange'] = (
    _pagination,
    _filters,
    sorter
  ) => {
    if (sorter && 'order' in sorter) {
      setSortedInfo({
        order: sorter.order === 'ascend' ? 'ascend' : 'descend',
        columnKey: sorter.columnKey as string,
      });
    } else {
      setSortedInfo({});
    }
  };

  const columns: ColumnsType<PerformanceTableRow> = [
    {
      title: 'SKU + ASIN',
      key: 'sku-asin',
      dataIndex: 'sku',
      sorter: (a, b) => a.sku.localeCompare(b.sku),
      sortOrder: sortedInfo.columnKey === 'sku' ? sortedInfo.order : null,
      render: (_, record) => (
        <div>
          <div style={{ fontWeight: '600', color: '#030712' }}>{record.sku}</div>
          <div style={{ fontSize: '12px', color: '#64748B' }}>{record.asin}</div>
        </div>
      ),
      width: 150,
    },
    {
      title: 'Units Sold',
      dataIndex: 'unitsSold',
      key: 'unitsSold',
      sorter: (a, b) => a.unitsSold - b.unitsSold,
      sortOrder: sortedInfo.columnKey === 'unitsSold' ? sortedInfo.order : null,
      align: 'right',
      render: (value: number) => value.toLocaleString(),
      width: 120,
    },
    {
      title: 'Revenue',
      dataIndex: 'revenue',
      key: 'revenue',
      sorter: (a, b) => a.revenue - b.revenue,
      sortOrder: sortedInfo.columnKey === 'revenue' ? sortedInfo.order : null,
      align: 'right',
      render: (value: number) => formatCurrency(value),
      width: 130,
    },
    {
      title: 'Refund %',
      dataIndex: 'refundRatio',
      key: 'refundRatio',
      sorter: (a, b) => a.refundRatio - b.refundRatio,
      sortOrder: sortedInfo.columnKey === 'refundRatio' ? sortedInfo.order : null,
      align: 'right',
      render: renderRefundRatio,
      width: 110,
    },
    {
      title: 'Rating',
      dataIndex: 'rating',
      key: 'rating',
      sorter: (a, b) => a.rating - b.rating,
      sortOrder: sortedInfo.columnKey === 'rating' ? sortedInfo.order : null,
      render: renderRating,
      width: 120,
    },
    {
      title: 'Trend',
      dataIndex: 'trend',
      key: 'trend',
      sorter: (a, b) => a.trend - b.trend,
      sortOrder: sortedInfo.columnKey === 'trend' ? sortedInfo.order : null,
      align: 'right',
      render: renderTrend,
      width: 100,
    },
  ];

  if (loading) {
    return (
      <div
        style={{
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          backgroundColor: '#FFFFFF',
          padding: '20px',
        }}
      >
        <Skeleton active paragraph={{ rows: 5 }} />
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          borderRadius: '8px',
          border: '1px solid #E2E8F0',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          backgroundColor: '#FFFFFF',
          padding: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '200px',
          color: '#F43F5E',
          fontSize: '14px',
          fontWeight: '500',
        }}
      >
        {error || 'Failed to load table data'}
      </div>
    );
  }

  return (
    <div
      style={{
        borderRadius: '8px',
        border: '1px solid #E2E8F0',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        backgroundColor: '#FFFFFF',
        overflow: 'hidden',
      }}
    >
      <Table
        columns={columns}
        dataSource={data}
        rowKey="id"
        pagination={{
          pageSize: 10,
          showSizeChanger: false,
          showTotal: (total) => `Total ${total} items`,
          style: { padding: '16px' },
        }}
        onChange={handleTableChange}
        onRow={(record) => ({
          onClick: () => {
            if (onRowClick) {
              onRowClick(record);
            }
          },
          style: {
            cursor: onRowClick ? 'pointer' : 'default',
          },
        })}
        scroll={{ x: 'max-content' }}
        style={{
          backgroundColor: '#FFFFFF',
        }}
        components={{
          header: {
            cell: (props: any) => (
              <th
                {...props}
                style={{
                  ...props.style,
                  backgroundColor: '#F8FAFC',
                  borderBottom: '2px solid #E2E8F0',
                  fontWeight: '600',
                  color: '#030712',
                  fontSize: '13px',
                  padding: '12px 16px',
                }}
              />
            ),
          },
        }}
        size="middle"
      />
      <style>{`
        .ant-table-tbody > tr:hover > td {
          background-color: #F8FAFC !important;
        }
        .ant-table-tbody > tr > td {
          border-bottom: 1px solid #F1F5F9;
          padding: 12px 16px;
          color: #030712;
        }
        .ant-table-thead > tr > th {
          background-color: #F8FAFC;
          border-bottom: 2px solid #E2E8F0;
        }
        .ant-pagination {
          margin: 16px;
        }
        .ant-table-pagination {
          margin: 16px 0;
        }
      `}</style>
    </div>
  );
};

export default PerformanceTable;

