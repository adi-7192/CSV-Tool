import React, { useState, useEffect } from 'react';
import { Row, Col, Button } from 'antd';
import { LeftOutlined, RightOutlined } from '@ant-design/icons';
import MetricCard from '@/components/MetricCard';
import TrendChart from '@/components/TrendChart';
import PerformanceTable, { PerformanceTableRow } from '@/components/PerformanceTable';
import InsightBanner from '@/components/InsightBanner';
import { useDataStore } from '@/store';
import './Dashboard.css';

const Dashboard: React.FC = () => {
  const { dateRange } = useDataStore();
  const [loading, setLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Mock metrics data
  const mockMetrics = {
    grossRevenue: 2830655.40,
    netRevenue: 2370333.02,
    orderCount: 1802,
    unitsSold: 5234,
    avgSellingPrice: 1570.32,
    netMargin: 83.7,
  };

  // Mock chart data (90 days)
  const generateMockChartData = () => {
    const data = [];
    const startDate = new Date('2025-07-01');
    const baseValue = 2200000;
    
    for (let i = 0; i < 90; i++) {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i);
      const variation = (Math.random() - 0.5) * 200000;
      const trend = i * 15000; // Gradual upward trend
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        value: baseValue + variation + trend,
      });
    }
    return data;
  };

  const mockChartData = generateMockChartData();
  
  // Mock refund trend data
  const generateMockRefundData = () => {
    const data = [];
    const startDate = new Date('2025-07-01');
    const baseValue = 50000;
    
    for (let i = 0; i < 90; i++) {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i);
      const variation = (Math.random() - 0.5) * 10000;
      const trend = i * 200; // Gradual upward trend
      data.push({
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        value: Math.max(0, baseValue + variation + trend),
      });
    }
    return data;
  };

  const mockRefundData = generateMockRefundData();

  // Mock SKU data
  const mockSKUData: PerformanceTableRow[] = [
    {
      id: '1',
      sku: 'SKU-001',
      asin: 'B001234567',
      unitsSold: 1200,
      revenue: 234567.89,
      refundRatio: 2.3,
      rating: 4.5,
      trend: 12.3,
    },
    {
      id: '2',
      sku: 'SKU-002',
      asin: 'B002345678',
      unitsSold: 890,
      revenue: 156234.56,
      refundRatio: 5.1,
      rating: 4.2,
      trend: -3.2,
    },
    {
      id: '3',
      sku: 'SKU-003',
      asin: 'B003456789',
      unitsSold: 2340,
      revenue: 489012.34,
      refundRatio: 1.8,
      rating: 4.8,
      trend: 8.7,
    },
    {
      id: '4',
      sku: 'SKU-004',
      asin: 'B004567890',
      unitsSold: 560,
      revenue: 89012.45,
      refundRatio: 6.5,
      rating: 3.9,
      trend: -5.4,
    },
    {
      id: '5',
      sku: 'SKU-005',
      asin: 'B005678901',
      unitsSold: 1780,
      revenue: 345678.90,
      refundRatio: 3.8,
      rating: 4.6,
      trend: 15.2,
    },
    {
      id: '6',
      sku: 'SKU-006',
      asin: 'B006789012',
      unitsSold: 945,
      revenue: 189234.67,
      refundRatio: 2.9,
      rating: 4.4,
      trend: 7.1,
    },
    {
      id: '7',
      sku: 'SKU-007',
      asin: 'B007890123',
      unitsSold: 1234,
      revenue: 267890.12,
      refundRatio: 4.2,
      rating: 4.3,
      trend: -1.8,
    },
    {
      id: '8',
      sku: 'SKU-008',
      asin: 'B008901234',
      unitsSold: 2100,
      revenue: 456789.01,
      refundRatio: 1.5,
      rating: 4.9,
      trend: 22.5,
    },
    {
      id: '9',
      sku: 'SKU-009',
      asin: 'B009012345',
      unitsSold: 678,
      revenue: 123456.78,
      refundRatio: 7.2,
      rating: 3.7,
      trend: -8.3,
    },
    {
      id: '10',
      sku: 'SKU-010',
      asin: 'B010123456',
      unitsSold: 1567,
      revenue: 298765.43,
      refundRatio: 2.1,
      rating: 4.7,
      trend: 9.4,
    },
    {
      id: '11',
      sku: 'SKU-011',
      asin: 'B011234567',
      unitsSold: 789,
      revenue: 145678.90,
      refundRatio: 3.5,
      rating: 4.1,
      trend: 4.2,
    },
    {
      id: '12',
      sku: 'SKU-012',
      asin: 'B012345678',
      unitsSold: 2345,
      revenue: 512345.67,
      refundRatio: 1.2,
      rating: 4.8,
      trend: 18.9,
    },
  ];

  // Mock insights
  const mockInsights = [
    {
      id: '1',
      type: 'warning' as const,
      title: 'Refund Spike Detected',
      description: `Refunds increased 15% this week for SKU-002. Review product quality.`,
      action: {
        label: 'View Details',
        onClick: () => console.log('Navigate to SKU-002'),
      },
    },
    {
      id: '2',
      type: 'success' as const,
      title: 'Revenue Target Achieved',
      description: `September revenue exceeded target by 12%. Great performance!`,
      action: {
        label: 'View Report',
        onClick: () => console.log('View report'),
      },
    },
    {
      id: '3',
      type: 'info' as const,
      title: 'Free Replacement Cost Impact',
      description: `Free replacement costs are impacting margin by 3.2%. Consider quality improvements.`,
      action: {
        label: 'Analyze',
        onClick: () => console.log('Analyze free replacement'),
      },
    },
    {
      id: '4',
      type: 'error' as const,
      title: 'High Refund Rate Alert',
      description: `SKU-009 has refund rate of 7.2% (above 5% threshold). Immediate action required.`,
      action: {
        label: 'Investigate',
        onClick: () => console.log('Investigate SKU-009'),
      },
    },
  ];

  const [insights, setInsights] = useState(mockInsights);

  useEffect(() => {
    // Simulate loading when date range changes
    setLoading(true);
    const timer = setTimeout(() => {
      setLoading(false);
      // TODO: Fetch real data based on dateRange
      console.log('Date range changed:', dateRange);
    }, 500);
    return () => clearTimeout(timer);
  }, [dateRange]);

  const handleDismissInsight = (id: string) => {
    setInsights((prev) => prev.filter((insight) => insight.id !== id));
  };

  const handleRowClick = (row: PerformanceTableRow) => {
    console.log('SKU clicked:', row);
    // TODO: Navigate to SKU detail page or show modal
  };

  return (
    <div className="dashboard">
      <div className="dashboard-container">
        {/* Main Content Area */}
        <div className="dashboard-main-layout">
          {/* Left Column - Main Content */}
          <div className="dashboard-main-content">
            {/* KPI Row */}
            <div className="dashboard-section">
              <h2 className="dashboard-section-title">Key Performance Indicators</h2>
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} lg={8}>
                  <MetricCard
                    title="Gross Revenue"
                    value={mockMetrics.grossRevenue}
                    format="currency"
                    trend={12.5}
                    trendLabel="vs last month"
                    loading={loading}
                    sparklineData={mockChartData.slice(-7).map((d) => d.value)}
                  />
                </Col>
                <Col xs={24} sm={12} lg={8}>
                  <MetricCard
                    title="Net Revenue"
                    value={mockMetrics.netRevenue}
                    format="currency"
                    trend={8.3}
                    trendLabel="vs last month"
                    loading={loading}
                    sparklineData={mockChartData.slice(-7).map((d) => d.value)}
                  />
                </Col>
                <Col xs={24} sm={12} lg={8}>
                  <MetricCard
                    title="Total Orders"
                    value={mockMetrics.orderCount}
                    format="number"
                    trend={-2.1}
                    trendLabel="vs last month"
                    loading={loading}
                  />
                </Col>
                <Col xs={24} sm={12} lg={8}>
                  <MetricCard
                    title="Units Sold"
                    value={mockMetrics.unitsSold}
                    format="number"
                    trend={5.7}
                    trendLabel="vs last month"
                    loading={loading}
                  />
                </Col>
                <Col xs={24} sm={12} lg={8}>
                  <MetricCard
                    title="Avg Selling Price"
                    value={mockMetrics.avgSellingPrice}
                    format="currency"
                    trend={3.2}
                    trendLabel="vs last month"
                    loading={loading}
                  />
                </Col>
                <Col xs={24} sm={12} lg={8}>
                  <MetricCard
                    title="Net Margin %"
                    value={mockMetrics.netMargin}
                    format="percentage"
                    trend={0.5}
                    trendLabel="vs last month"
                    loading={loading}
                  />
                </Col>
              </Row>
            </div>

            {/* Charts Section */}
            <div className="dashboard-section">
              <h2 className="dashboard-section-title">Revenue Trends</h2>
              <Row gutter={[16, 16]}>
                <Col xs={24} lg={12}>
                  <TrendChart
                    title="Revenue Trend"
                    type="line"
                    data={mockChartData.map((d) => ({ date: d.date, value: d.value }))}
                    color="#6366F1"
                    loading={loading}
                  />
                </Col>
                <Col xs={24} lg={12}>
                  <TrendChart
                    title="Refund Trend"
                    type="area"
                    data={mockRefundData.map((d) => ({ date: d.date, value: d.value }))}
                    color="#F43F5E"
                    loading={loading}
                  />
                </Col>
              </Row>
            </div>

            {/* Table Section */}
            <div className="dashboard-section">
              <h2 className="dashboard-section-title">SKU Performance</h2>
              <PerformanceTable
                data={mockSKUData}
                loading={loading}
                onRowClick={handleRowClick}
              />
            </div>
          </div>

          {/* Right Column - Insights Sidebar */}
          {!sidebarCollapsed && (
            <div className="dashboard-sidebar">
              <div className="dashboard-sidebar-content">
                <div className="dashboard-sidebar-header">
                  <h2 className="dashboard-section-title">Insights</h2>
                  <Button
                    type="text"
                    icon={<LeftOutlined />}
                    onClick={() => setSidebarCollapsed(true)}
                    className="dashboard-sidebar-toggle"
                    aria-label="Collapse sidebar"
                  />
                </div>
                {insights.length > 0 ? (
                  <div className="dashboard-insights-list">
                    {insights.map((insight) => (
                      <InsightBanner
                        key={insight.id}
                        type={insight.type}
                        title={insight.title}
                        description={insight.description}
                        action={insight.action}
                        dismissible={true}
                        onDismiss={() => handleDismissInsight(insight.id)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="dashboard-empty-insights">
                    No insights available
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Collapsed Sidebar Toggle */}
          {sidebarCollapsed && (
            <div className="dashboard-sidebar-toggle-floating">
              <Button
                type="primary"
                icon={<RightOutlined />}
                onClick={() => setSidebarCollapsed(false)}
                className="dashboard-sidebar-toggle-button"
                aria-label="Expand sidebar"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
