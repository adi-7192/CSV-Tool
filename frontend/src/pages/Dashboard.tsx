import React, { useState, useEffect, Component, ErrorInfo, ReactNode } from 'react';
import { Row, Col, Button, Alert, Skeleton, Card, Table, Modal } from 'antd';
import { ReloadOutlined, ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import MetricCard from '@/components/MetricCard';
import TrendChart from '@/components/TrendChart';
import PerformanceTable, { PerformanceTableRow } from '@/components/PerformanceTable';
import InsightBanner from '@/components/InsightBanner';
import { useDataStore } from '@/store/dataStore';
import { regionService, RegionSKU } from '@/services/api';
import { formatCurrency } from '@/utils/formatters';
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
    regionRevenue,
    moversDecliners,
    dateRange,
    metricsLoading,
    chartsLoading,
    skuLoading,
    insightsLoading,
    regionLoading,
    moversDeclinersLoading,
    error,
    fetchMetrics,
    fetchChartData,
    fetchSKUPerformance,
    fetchInsights,
    fetchRegionRevenue,
    fetchMoversDecliners,
  } = useDataStore();

  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [isInsightsModalOpen, setIsInsightsModalOpen] = useState(false);
  const [isRegionSKUModalOpen, setIsRegionSKUModalOpen] = useState(false);
  const [selectedRegionName, setSelectedRegionName] = useState<string>('');
  const [regionSKUs, setRegionSKUs] = useState<RegionSKU[]>([]);
  const [regionSKUsLoading, setRegionSKUsLoading] = useState(false);

  // Fetch all data when date range changes
  useEffect(() => {
    fetchMetrics(dateRange.start, dateRange.end);
    fetchChartData(dateRange.start, dateRange.end);
    fetchSKUPerformance(dateRange.start, dateRange.end);
    fetchInsights(dateRange.start, dateRange.end);
    fetchRegionRevenue(dateRange.start, dateRange.end);
    fetchMoversDecliners(dateRange.start, dateRange.end);
  }, [dateRange.start, dateRange.end]);

  // Handle retry on error
  const handleRetry = () => {
    fetchMetrics(dateRange.start, dateRange.end);
    fetchChartData(dateRange.start, dateRange.end);
    fetchSKUPerformance(dateRange.start, dateRange.end);
    fetchInsights(dateRange.start, dateRange.end);
    fetchRegionRevenue(dateRange.start, dateRange.end);
    fetchMoversDecliners(dateRange.start, dateRange.end);
  };

  // Handle region bar click
  const handleRegionClick = async (data: any) => {
    if (data && data.region) {
      const cityName = data.region;
      console.log('\n' + '='.repeat(80));
      console.log('[Dashboard] 🔍 CITY CLICK DEBUG');
      console.log('='.repeat(80));
      console.log('[Dashboard] City clicked:', cityName);
      console.log('[Dashboard] City name type:', typeof cityName);
      console.log('[Dashboard] City name length:', cityName?.length);
      console.log('[Dashboard] City name trimmed:', cityName?.trim());
      
      setSelectedRegion(cityName);
      setSelectedRegionName(cityName);
      setIsRegionSKUModalOpen(true);
      setRegionSKUsLoading(true);
      
      // Fetch SKUs for this city (all-time data, no date filters)
      try {
        console.log(`[Dashboard] Fetching SKUs for city: "${cityName}"`);
        const response = await regionService.getSKUsByCity(cityName, 10);
        
        console.log('[Dashboard] API Response:', response);
        console.log('[Dashboard] Response data type:', typeof response?.data);
        console.log('[Dashboard] Response data is array:', Array.isArray(response?.data));
        console.log('[Dashboard] Response data length:', response?.data?.length);
        
        if (response && response.data && Array.isArray(response.data) && response.data.length > 0) {
          console.log('[Dashboard] ✅ City SKUs fetched successfully:', response.data.length, 'SKUs');
          console.log('[Dashboard] Sample SKUs:', response.data.slice(0, 3));
          setRegionSKUs(response.data);
        } else {
          console.warn('[Dashboard] ⚠️  No SKUs data returned for city:', cityName);
          console.warn('[Dashboard] Full response:', JSON.stringify(response, null, 2));
          if (response && response.data && Array.isArray(response.data) && response.data.length === 0) {
            console.warn('[Dashboard] ⚠️  API returned empty array - city might not have SKU data');
            console.warn('[Dashboard] ⚠️  This could be a normalization mismatch issue');
            console.warn('[Dashboard] ⚠️  Check backend logs for normalization details');
          }
          setRegionSKUs([]);
        }
      } catch (error) {
        console.error('[Dashboard] ❌ Error fetching city SKUs:', error);
        if (error instanceof Error) {
          console.error('[Dashboard] Error message:', error.message);
          console.error('[Dashboard] Error stack:', error.stack);
        }
        setRegionSKUs([]);
      } finally {
        setRegionSKUsLoading(false);
        console.log('='.repeat(80) + '\n');
      }
    } else {
      console.warn('[Dashboard] ⚠️  handleRegionClick called with invalid data:', data);
    }
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
            {/* KPI Row with Minimized Insights Panel - Side by Side */}
            <Row gutter={16} style={{ marginBottom: '24px' }}>
              {/* Left Column - KPI Cards */}
              <Col xs={24} md={14}>
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
              </Col>

              {/* Right Column - Minimized Insights Panel */}
              <Col xs={24} md={10}>
                <div className="dashboard-section">
                  <h2 className="dashboard-section-title">Insights</h2>
                  <div className="dashboard-insights-minimized">
                    {insightsLoading && (!insights || insights.length === 0) ? (
                      <div className="dashboard-loading-insights">
                        <Skeleton active paragraph={{ rows: 2 }} />
                      </div>
                    ) : insights && insights.length > 0 ? (
                      <>
                        {/* Show all insights */}
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
                        {/* View All button */}
                        <div style={{ marginTop: '12px', textAlign: 'center' }}>
                          <Button
                            type="link"
                            size="small"
                            onClick={() => setIsInsightsModalOpen(true)}
                            style={{
                              fontSize: '12px',
                              padding: '0',
                              height: 'auto',
                              color: '#6366F1',
                            }}
                          >
                            View All Insights →
                          </Button>
                        </div>
                      </>
                    ) : (
                      <div className="dashboard-empty-insights">
                        No insights available
                      </div>
                    )}
                  </div>
                </div>
              </Col>
            </Row>

            {/* Revenue Analysis Section - Full Width */}
            <Row gutter={16} style={{ marginTop: '20px', marginBottom: '32px' }}>
              <Col xs={24}>
                <div className="dashboard-section">
                <h2 className="dashboard-section-title">Revenue Analysis</h2>
                {/* Debug logging */}
                {(() => {
                  console.log('\n' + '='.repeat(80));
                  console.log('🔍 FRONTEND DEBUG: REVENUE BY REGION CHART');
                  console.log('='.repeat(80));
                  
                  if (regionRevenue) {
                    console.log('[Dashboard] Chart Data Source: regionRevenue from store');
                    console.log('[Dashboard] Full Region Revenue Data:', JSON.stringify(regionRevenue, null, 2));
                    console.log('[Dashboard] Region Revenue Count:', regionRevenue.length);
                    console.log('[Dashboard] Data Type:', Array.isArray(regionRevenue) ? 'Array' : typeof regionRevenue);
                    
                    console.log('\n[Dashboard] Region Revenue Data Structure:');
                    regionRevenue.forEach((r, idx) => {
                      console.log(`  ${idx + 1}. Region: "${r.region}" | Revenue: ${r.revenue} | Type: ${typeof r.revenue}`);
                    });
                    
                    // Check for duplicates
                    const regionNames = regionRevenue.map(r => r.region);
                    const duplicates = regionNames.filter((name, index) => regionNames.indexOf(name) !== index);
                    if (duplicates.length > 0) {
                      console.warn('\n[Dashboard] ⚠️  DUPLICATE REGIONS FOUND:', duplicates);
                    }
                    
                    // Log unique regions
                    const uniqueRegions = [...new Set(regionNames)];
                    console.log('\n[Dashboard] Unique regions:', uniqueRegions);
                    console.log('[Dashboard] Total vs Unique:', regionNames.length, 'vs', uniqueRegions.length);
                    
                    // Log revenue totals
                    const totalRevenue = regionRevenue.reduce((sum, r) => sum + (Number(r.revenue) || 0), 0);
                    console.log('\n[Dashboard] Total Revenue across all regions:', totalRevenue);
                    
                    // Log top 10 regions
                    const sortedRegions = [...regionRevenue].sort((a, b) => (Number(b.revenue) || 0) - (Number(a.revenue) || 0));
                    console.log('\n[Dashboard] Top 10 Regions by Revenue:');
                    sortedRegions.slice(0, 10).forEach((r, idx) => {
                      console.log(`  ${idx + 1}. ${r.region}: ₹${r.revenue?.toLocaleString('en-IN') || 0}`);
                    });
                    
                    console.log('\n[Dashboard] Chart Configuration:');
                    console.log('  - Chart Type: Vertical BarChart (horizontal bars)');
                    console.log('  - Data Key: revenue');
                    console.log('  - Y-Axis Key: region');
                    console.log('  - Number of bars:', regionRevenue.length);
                  } else {
                    console.warn('[Dashboard] ⚠️  Region Revenue Data is null or undefined');
                    console.log('[Dashboard] Loading state:', regionLoading);
                    console.log('[Dashboard] regionRevenue value:', regionRevenue);
                  }
                  
                  console.log('='.repeat(80) + '\n');
                  return null;
                })()}
                
                {/* Loading State */}
                {(chartsLoading && !chartData) || (regionLoading && !regionRevenue) ? (
                  <Row gutter={[16, 16]} style={{ width: '100%', marginLeft: 0, marginRight: 0 }}>
                    <Col xs={24} md={12}>
                      <Card>
                        <Skeleton active paragraph={{ rows: 6 }} />
                      </Card>
                    </Col>
                    <Col xs={24} md={12}>
                      <Card>
                        <Skeleton active paragraph={{ rows: 6 }} />
                      </Card>
                    </Col>
                  </Row>
                ) : (
                  <Row gutter={[16, 16]} style={{ width: '100%', marginLeft: 0, marginRight: 0 }}>
                    {/* Revenue Trend Chart - Left Column */}
                    <Col xs={24} md={12}>
                      <TrendChart
                        title="Revenue Trend"
                        type="line"
                        data={transformChartData(chartData?.revenue_trend)}
                        color="#6366F1"
                        loading={chartsLoading}
                        height={400}
                      />
                    </Col>
                    
                    {/* Revenue by Region Chart - Right Column */}
                    <Col xs={24} md={12}>
                      {regionRevenue && regionRevenue.length > 0 ? (
                        <Card style={{ height: '452px', display: 'flex', flexDirection: 'column', padding: '16px', paddingLeft: '0px', justifyContent: 'flex-start' }}>
                          <div style={{ marginBottom: '16px', textAlign: 'center', width: '100%', paddingLeft: '16px' }}>
                            <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600', color: '#030712' }}>
                              Revenue by Region
                            </h3>
                          </div>
                          <div style={{ height: '400px', width: '100%', flexShrink: 0, marginLeft: '-16px', paddingLeft: '8px' }}>
                            <ResponsiveContainer width="100%" height={400}>
                            <BarChart
                              data={regionRevenue}
                              layout="vertical"
                              margin={{ top: 20, right: 20, left: 0, bottom: 20 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                              <XAxis
                                type="number"
                                tickFormatter={(value) => `₹${(value / 100000).toFixed(1)}L`}
                                stroke="#64748B"
                                fontSize={12}
                                domain={[0, 'dataMax']}
                              />
                              <YAxis
                                type="category"
                                dataKey="region"
                                stroke="#64748B"
                                fontSize={13}
                                width={120}
                                tick={{ fill: '#64748B', fontSize: 13 }}
                                interval={0}
                              />
                              <Tooltip
                                formatter={(value: number) => `₹${value.toLocaleString('en-IN')}`}
                                contentStyle={{
                                  backgroundColor: '#FFFFFF',
                                  border: '1px solid #E2E8F0',
                                  borderRadius: '8px',
                                  padding: '12px',
                                }}
                              />
                              <Bar
                                dataKey="revenue"
                                radius={[0, 8, 8, 0]}
                                cursor="pointer"
                                onClick={(data: any, index: number) => {
                                  // Handle click on bar - data contains the region info
                                  console.log('[Dashboard] Bar clicked - data:', data, 'index:', index);
                                  if (data && data.region) {
                                    handleRegionClick({ region: data.region });
                                  } else if (regionRevenue && regionRevenue[index]) {
                                    // Fallback: use index to get region
                                    handleRegionClick({ region: regionRevenue[index].region });
                                  }
                                }}
                              >
                                {regionRevenue.map((entry, index) => {
                                  // Ensure unique keys - use region name + index
                                  const uniqueKey = `${entry.region}-${index}`;
                                  return (
                                    <Cell
                                      key={uniqueKey}
                                      fill={selectedRegion === entry.region ? '#6366F1' : '#818CF8'}
                                      style={{
                                        opacity: selectedRegion === entry.region ? 1 : 0.8,
                                        cursor: 'pointer',
                                      }}
                                      onClick={(e: any) => {
                                        e.stopPropagation();
                                        console.log('[Dashboard] Cell clicked for region:', entry.region);
                                        handleRegionClick({ region: entry.region });
                                      }}
                                    />
                                  );
                                })}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                          </div>
                          {selectedRegion && (
                            <div style={{ marginTop: '16px', padding: '12px', backgroundColor: '#F3F4F6', borderRadius: '8px' }}>
                              <strong>Selected:</strong> {selectedRegion}
                              {' - '}
                              ₹{regionRevenue.find((r) => r.region === selectedRegion)?.revenue.toLocaleString('en-IN') || 0}
                            </div>
                          )}
                        </Card>
                      ) : (
                        <Card style={{ width: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', minHeight: '400px' }}>
                          <div style={{ textAlign: 'center', padding: '48px', color: '#64748B' }}>
                            <h3 style={{ margin: 0, marginBottom: '8px', fontSize: '16px', fontWeight: '600', color: '#030712' }}>
                              Revenue by Region
                            </h3>
                            <div>No region data available</div>
                            {regionRevenue && regionRevenue.length === 0 && (
                              <div style={{ marginTop: '8px', fontSize: '12px', color: '#94A3B8' }}>
                                (API returned empty array)
                              </div>
                            )}
                          </div>
                        </Card>
                      )}
                    </Col>
                  </Row>
                )}
                </div>
              </Col>
            </Row>

            {/* Movers & Decliners Section - Full Width */}
            <Row gutter={16} style={{ marginTop: '32px', marginBottom: '32px' }}>
              <Col xs={24}>
                {/* Movers & Decliners Section */}
                <div className="dashboard-section">
                  <h2 className="dashboard-section-title">Movers & Decliners</h2>
                {/* Debug logging */}
                {(() => {
                  console.log('\n' + '='.repeat(80));
                  console.log('🔍 FRONTEND DEBUG: MOVERS & DECLINERS');
                  console.log('='.repeat(80));
                  
                  if (moversDecliners) {
                    console.log('[Dashboard] Full API Response:', JSON.stringify(moversDecliners, null, 2));
                    console.log('[Dashboard] Data Structure:', {
                      hasMovers: !!moversDecliners.movers,
                      hasDecliners: !!moversDecliners.decliners,
                      moversType: typeof moversDecliners.movers,
                      declinersType: typeof moversDecliners.decliners,
                    });
                    console.log('[Dashboard] Movers Count:', moversDecliners.movers?.length || 0);
                    console.log('[Dashboard] Decliners Count:', moversDecliners.decliners?.length || 0);
                    
                    if (moversDecliners.movers && moversDecliners.movers.length > 0) {
                      console.log('\n[Dashboard] FAST MOVERS DATA:');
                      console.log('Number of items:', moversDecliners.movers.length);
                      console.log('Full movers array:', moversDecliners.movers);
                      console.log('Sample Movers (first 5):', moversDecliners.movers.slice(0, 5).map(m => ({
                        sku: m.sku,
                        revenue: m.revenue,
                        wow_change: m.wow_change,
                        revenueType: typeof m.revenue,
                        wowChangeType: typeof m.wow_change,
                      })));
                    } else {
                      console.warn('[Dashboard] ⚠️  Movers array is EMPTY or undefined');
                      console.log('[Dashboard] Movers value:', moversDecliners.movers);
                    }
                    
                    if (moversDecliners.decliners && moversDecliners.decliners.length > 0) {
                      console.log('\n[Dashboard] DECLINERS DATA:');
                      console.log('Number of items:', moversDecliners.decliners.length);
                      console.log('Full decliners array:', moversDecliners.decliners);
                      console.log('Sample Decliners (first 5):', moversDecliners.decliners.slice(0, 5).map(d => ({
                        sku: d.sku,
                        revenue: d.revenue,
                        wow_change: d.wow_change,
                        revenueType: typeof d.revenue,
                        wowChangeType: typeof d.wow_change,
                      })));
                    } else {
                      console.warn('[Dashboard] ⚠️  Decliners array is EMPTY or undefined');
                      console.log('[Dashboard] Decliners value:', moversDecliners.decliners);
                    }
                  } else if (!moversDeclinersLoading) {
                    console.warn('[Dashboard] ⚠️  Movers & Decliners Data is null or undefined');
                    console.log('[Dashboard] Loading state:', moversDeclinersLoading);
                    console.log('[Dashboard] moversDecliners value:', moversDecliners);
                  } else {
                    console.log('[Dashboard] Still loading movers & decliners data...');
                  }
                  
                  console.log('='.repeat(80) + '\n');
                  return null;
                })()}
                {moversDeclinersLoading && !moversDecliners ? (
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
                ) : moversDecliners ? (
                  <Row gutter={[16, 16]}>
                    {/* Decliners Column */}
                    <Col xs={24} lg={12}>
                      <Card
                        style={{
                          border: '1px solid #FEE2E2',
                          backgroundColor: '#FEF2F2',
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginBottom: '16px',
                          }}
                        >
                          <ArrowDownOutlined style={{ color: '#DC2626', fontSize: '20px' }} />
                          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#DC2626' }}>
                            Decliners
                          </h3>
                        </div>
                        {moversDecliners.decliners.length > 0 ? (
                          <Table
                            dataSource={moversDecliners.decliners}
                            columns={[
                              {
                                title: 'Product',
                                dataIndex: 'sku',
                                key: 'sku',
                                render: (sku: string) => (
                                  <span style={{ fontWeight: '500', color: '#030712' }}>{sku}</span>
                                ),
                              },
                              {
                                title: 'Revenue',
                                dataIndex: 'revenue',
                                key: 'revenue',
                                align: 'right',
                                render: (revenue: number) => formatCurrency(revenue),
                              },
                              {
                                title: 'WoW Change',
                                dataIndex: 'wow_change',
                                key: 'wow_change',
                                align: 'right',
                                render: (change: number) => (
                                  <span style={{ color: '#DC2626', fontWeight: '600' }}>
                                    {change.toFixed(1)}%
                                  </span>
                                ),
                              },
                            ]}
                            pagination={false}
                            size="small"
                            rowKey="sku"
                          />
                        ) : (
                          <div style={{ textAlign: 'center', padding: '24px', color: '#64748B' }}>
                            No decliners found
                          </div>
                        )}
                      </Card>
                    </Col>

                    {/* Fast Movers Column */}
                    <Col xs={24} lg={12}>
                      <Card
                        style={{
                          border: '1px solid #D1FAE5',
                          backgroundColor: '#F0FDF4',
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginBottom: '16px',
                          }}
                        >
                          <ArrowUpOutlined style={{ color: '#10B981', fontSize: '20px' }} />
                          <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#10B981' }}>
                            Fast Movers
                          </h3>
                        </div>
                        {moversDecliners.movers.length > 0 ? (
                          <Table
                            dataSource={moversDecliners.movers}
                            columns={[
                              {
                                title: 'Product',
                                dataIndex: 'sku',
                                key: 'sku',
                                render: (sku: string) => (
                                  <span style={{ fontWeight: '500', color: '#030712' }}>{sku}</span>
                                ),
                              },
                              {
                                title: 'Revenue',
                                dataIndex: 'revenue',
                                key: 'revenue',
                                align: 'right',
                                render: (revenue: number) => formatCurrency(revenue),
                              },
                              {
                                title: 'WoW Change',
                                dataIndex: 'wow_change',
                                key: 'wow_change',
                                align: 'right',
                                render: (change: number) => (
                                  <span style={{ color: '#10B981', fontWeight: '600' }}>
                                    +{change.toFixed(1)}%
                                  </span>
                                ),
                              },
                            ]}
                            pagination={false}
                            size="small"
                            rowKey="sku"
                          />
                        ) : (
                          <div style={{ textAlign: 'center', padding: '24px', color: '#64748B' }}>
                            No fast movers found
                          </div>
                        )}
                      </Card>
                    </Col>
                  </Row>
                ) : (
                  <Row gutter={[16, 16]}>
                    <Col xs={24} lg={12}>
                      <Card>
                        <div style={{ textAlign: 'center', padding: '48px', color: '#64748B' }}>
                          No movers & decliners data available
                        </div>
                      </Card>
                    </Col>
                    <Col xs={24} lg={12}>
                      <Card>
                        <div style={{ textAlign: 'center', padding: '48px', color: '#64748B' }}>
                          No movers & decliners data available
                        </div>
                      </Card>
                    </Col>
                  </Row>
                )}
              </div>
              </Col>
            </Row>

            {/* SKU Performance Section - Full Width */}
            <Row gutter={16} style={{ marginTop: '32px' }}>
              <Col xs={24}>
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
              </Col>
            </Row>
          </div>

          {/* All Insights Modal */}
          <Modal
            title="All Insights"
            open={isInsightsModalOpen}
            onCancel={() => setIsInsightsModalOpen(false)}
            footer={[
              <Button key="close" onClick={() => setIsInsightsModalOpen(false)}>
                Close
              </Button>,
            ]}
            width={600}
            centered
            closable
          >
            <div style={{ maxHeight: '60vh', overflowY: 'auto', paddingRight: '8px' }}>
              {insightsLoading && (!insights || insights.length === 0) ? (
                <div className="dashboard-loading-insights">
                  <Skeleton active paragraph={{ rows: 2 }} />
                </div>
              ) : insights && insights.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
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
          </Modal>

          {/* Region SKUs Modal */}
          <Modal
            title={`Top SKUs - ${selectedRegionName}`}
            open={isRegionSKUModalOpen}
            onCancel={() => {
              setIsRegionSKUModalOpen(false);
              setSelectedRegion(null);
            }}
            footer={[
              <Button key="close" onClick={() => {
                setIsRegionSKUModalOpen(false);
                setSelectedRegion(null);
              }}>
                Close
              </Button>,
            ]}
            width={800}
            centered
            closable
          >
            {regionSKUsLoading ? (
              <div style={{ padding: '24px' }}>
                <Skeleton active paragraph={{ rows: 5 }} />
              </div>
            ) : regionSKUs && regionSKUs.length > 0 ? (
              <Table
                dataSource={regionSKUs}
                columns={[
                  {
                    title: 'SKU',
                    dataIndex: 'sku',
                    key: 'sku',
                    render: (sku: string) => (
                      <span style={{ fontWeight: '500', color: '#030712' }}>{sku}</span>
                    ),
                  },
                  {
                    title: 'ASIN',
                    dataIndex: 'asin',
                    key: 'asin',
                    render: (asin: string) => (
                      <span style={{ color: '#64748B', fontFamily: 'monospace' }}>
                        {asin || 'N/A'}
                      </span>
                    ),
                  },
                  {
                    title: 'Units Sold',
                    dataIndex: 'units',
                    key: 'units',
                    align: 'right',
                    render: (units: number) => (
                      <span style={{ fontWeight: '500', color: '#030712' }}>
                        {units.toLocaleString('en-IN')}
                      </span>
                    ),
                  },
                  {
                    title: 'Revenue',
                    dataIndex: 'revenue',
                    key: 'revenue',
                    align: 'right',
                    render: (revenue: number) => (
                      <span style={{ fontWeight: '600', color: '#6366F1' }}>
                        {formatCurrency(revenue)}
                      </span>
                    ),
                  },
                ]}
                pagination={false}
                size="small"
                rowKey="sku"
                style={{ marginTop: '16px' }}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: '48px', color: '#64748B' }}>
                No SKU data available for {selectedRegionName}
              </div>
            )}
          </Modal>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default Dashboard;
