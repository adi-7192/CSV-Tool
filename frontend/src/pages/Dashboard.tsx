import React, { useState, useEffect, Component, ErrorInfo, ReactNode } from 'react';
import { Row, Col, Button, Alert, Skeleton, Card } from 'antd';
import { LeftOutlined, RightOutlined, ReloadOutlined } from '@ant-design/icons';
import MetricCard from '@/components/MetricCard';
import TrendChart from '@/components/TrendChart';
import PerformanceTable, { PerformanceTableRow } from '@/components/PerformanceTable';
import InsightBanner from '@/components/InsightBanner';
import { useDataStore } from '@/store/dataStore';
import './Dashboard.css';

// ============================================================================
// ERROR BOUNDARY COMPONENT
// ============================================================================

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Dashboard Error:', error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-container">
          <div className="error-boundary-content">
            <h2 className="error-boundary-title">Something went wrong</h2>
            <p className="error-boundary-message">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <Button
              type="primary"
              size="large"
              icon={<ReloadOutlined />}
              onClick={this.handleReload}
              className="error-boundary-reload-button"
            >
              Reload Page
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// ============================================================================
// DASHBOARD COMPONENT
// ============================================================================

const Dashboard: React.FC = () => {
  const {
    metrics,
    chartData,
    skuPerformance,
    insights,
    dateRange,
    metricsLoading,
    chartsLoading,
    skuLoading,
    insightsLoading,
    error,
    fetchMetrics,
    fetchChartData,
    fetchSKUPerformance,
    fetchInsights,
  } = useDataStore();

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Fetch all data when date range changes
  useEffect(() => {
    fetchMetrics(dateRange.start, dateRange.end);
    fetchChartData(dateRange.start, dateRange.end);
    fetchSKUPerformance(dateRange.start, dateRange.end);
    fetchInsights(dateRange.start, dateRange.end);
  }, [dateRange.start, dateRange.end]);

  // Handle retry on error
  const handleRetry = () => {
    fetchMetrics(dateRange.start, dateRange.end);
    fetchChartData(dateRange.start, dateRange.end);
    fetchSKUPerformance(dateRange.start, dateRange.end);
    fetchInsights(dateRange.start, dateRange.end);
  };

  // Transform chart data for TrendChart component
  const transformChartData = (trendData: any[] | undefined) => {
    if (!trendData || trendData.length === 0) return [];
    return trendData.map((point) => ({
      date: point.date || point.period || '',
      value: point.value || point.revenue || 0,
    }));
  };

  // Transform SKU performance data for PerformanceTable
  const transformSKUData = (skuData: any[] | undefined): PerformanceTableRow[] => {
    if (!skuData || skuData.length === 0) return [];
    return skuData.map((item, index) => ({
      id: item.id || String(index + 1),
      sku: item.sku || `SKU-${String(index + 1).padStart(3, '0')}`,
      asin: item.asin || '',
      unitsSold: item.unitsSold || item.units_sold || 0,
      revenue: item.revenue || 0,
      refundRatio: item.refundRatio || item.refund_ratio || 0,
      rating: item.rating || 4.0,
    }));
  };

  const handleDismissInsight = (id: string) => {
    // Note: Insights are managed by the store, so we'd need to update the store
    // For now, we'll just log it (can be enhanced later)
    console.log('Dismiss insight:', id);
  };

  const handleRowClick = (row: PerformanceTableRow) => {
    console.log('SKU clicked:', row);
    // TODO: Navigate to SKU detail page or show modal
  };

  return (
    <ErrorBoundary>
      <div className="dashboard">
        <div className="dashboard-container">
          {/* Error Alert */}
          {error && (
            <Alert
              message="Error Loading Data"
              description={error}
              type="error"
              showIcon
              closable
              action={
                <Button size="small" icon={<ReloadOutlined />} onClick={handleRetry}>
                  Retry
                </Button>
              }
              style={{ marginBottom: '24px' }}
              className="dashboard-error-alert"
            />
          )}

          {/* Main Content Area */}
          <div className="dashboard-main-layout">
            {/* Left Column - Main Content */}
            <div className="dashboard-main-content">
              {/* KPI Row */}
              <div className="dashboard-section">
                <h2 className="dashboard-section-title">Key Performance Indicators</h2>
                {metricsLoading && !metrics ? (
                  <Row gutter={[16, 16]}>
                    {[1, 2, 3, 4, 5, 6].map((i) => (
                      <Col xs={24} sm={12} lg={8} key={i}>
                        <Card>
                          <Skeleton active paragraph={{ rows: 2 }} />
                        </Card>
                      </Col>
                    ))}
                  </Row>
                ) : (
                  <Row gutter={[16, 16]}>
                    <Col xs={24} sm={12} lg={8}>
                      <MetricCard
                        title="Gross Revenue"
                        value={metrics?.gross_revenue || metrics?.revenue || 0}
                        format="currency"
                        trend={12.5}
                        trendLabel="vs last month"
                        loading={metricsLoading}
                        sparklineData={
                          chartData?.revenue_trend
                            ?.slice(-7)
                            .map((d) => d.value) || []
                        }
                      />
                    </Col>
                    <Col xs={24} sm={12} lg={8}>
                      <MetricCard
                        title="Net Revenue"
                        value={metrics?.net_revenue || 0}
                        format="currency"
                        trend={8.3}
                        trendLabel="vs last month"
                        loading={metricsLoading}
                        sparklineData={
                          chartData?.revenue_trend
                            ?.slice(-7)
                            .map((d) => d.value) || []
                        }
                      />
                    </Col>
                    <Col xs={24} sm={12} lg={8}>
                      <MetricCard
                        title="Total Orders"
                        value={metrics?.orders || 0}
                        format="number"
                        trend={-2.1}
                        trendLabel="vs last month"
                        loading={metricsLoading}
                      />
                    </Col>
                    <Col xs={24} sm={12} lg={8}>
                      <MetricCard
                        title="Units Sold"
                        value={metrics?.units_sold || 0}
                        format="number"
                        trend={5.7}
                        trendLabel="vs last month"
                        loading={metricsLoading}
                      />
                    </Col>
                    <Col xs={24} sm={12} lg={8}>
                      <MetricCard
                        title="Avg Selling Price"
                        value={metrics?.avg_order_value || 0}
                        format="currency"
                        trend={3.2}
                        trendLabel="vs last month"
                        loading={metricsLoading}
                      />
                    </Col>
                    <Col xs={24} sm={12} lg={8}>
                      <MetricCard
                        title="Net Margin %"
                        value={metrics?.net_margin || 0}
                        format="percentage"
                        trend={0.5}
                        trendLabel="vs last month"
                        loading={metricsLoading}
                      />
                    </Col>
                  </Row>
                )}
              </div>

              {/* Charts Section */}
              <div className="dashboard-section">
                <h2 className="dashboard-section-title">Revenue Trends</h2>
                {chartsLoading && !chartData ? (
                  <Row gutter={[16, 16]}>
                    <Col xs={24} lg={12}>
                      <Card>
                        <Skeleton active paragraph={{ rows: 6 }} />
                      </Card>
                    </Col>
                    <Col xs={24} lg={12}>
                      <Card>
                        <Skeleton active paragraph={{ rows: 6 }} />
                      </Card>
                    </Col>
                  </Row>
                ) : (
                  <Row gutter={[16, 16]}>
                    <Col xs={24} lg={12}>
                      <TrendChart
                        title="Revenue Trend"
                        type="line"
                        data={transformChartData(chartData?.revenue_trend)}
                        color="#6366F1"
                        loading={chartsLoading}
                      />
                    </Col>
                    <Col xs={24} lg={12}>
                      <TrendChart
                        title="Refund Trend"
                        type="area"
                        data={transformChartData(chartData?.refund_trend)}
                        color="#F43F5E"
                        loading={chartsLoading}
                      />
                    </Col>
                  </Row>
                )}
              </div>

              {/* Table Section */}
              <div className="dashboard-section">
                <h2 className="dashboard-section-title">SKU Performance</h2>
                {skuLoading && (!skuPerformance || skuPerformance.length === 0) ? (
                  <Card>
                    <Skeleton active paragraph={{ rows: 8 }} />
                  </Card>
                ) : (
                  <PerformanceTable
                    data={transformSKUData(skuPerformance || undefined)}
                    loading={skuLoading}
                    onRowClick={handleRowClick}
                  />
                )}
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
                  {insightsLoading && (!insights || insights.length === 0) ? (
                    <div className="dashboard-loading-insights">
                      <Skeleton active paragraph={{ rows: 2 }} />
                    </div>
                  ) : insights && insights.length > 0 ? (
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
    </ErrorBoundary>
  );
};

export default Dashboard;
