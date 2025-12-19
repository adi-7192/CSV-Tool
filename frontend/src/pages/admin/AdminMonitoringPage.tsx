import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Tag,
  Button,
  Input,
  Space,
  Alert,
  Spin,
  Modal,
  message,
  Tabs,
  Select,
  DatePicker,
  Statistic,
  Row,
  Col,
  Switch,
  Typography,
} from 'antd';
import {
  ReloadOutlined,
  EyeOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import { adminMonitoring, MonitoringEvent, MonitoringSummaryResponse, MonitoringHealthResponse } from '@/services/api';
import dayjs, { Dayjs } from 'dayjs';

const { TabPane } = Tabs;
const { Text } = Typography;
const { RangePicker } = DatePicker;

const AdminMonitoringPage: React.FC = () => {
  const [events, setEvents] = useState<MonitoringEvent[]>([]);
  const [summary, setSummary] = useState<MonitoringSummaryResponse | null>(null);
  const [health, setHealth] = useState<MonitoringHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isAuthError, setIsAuthError] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<MonitoringEvent | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);

  // Filters
  const [levelFilter, setLevelFilter] = useState<string | undefined>(undefined);
  const [categoryFilter, setCategoryFilter] = useState<string | undefined>(undefined);
  const [endpointFilter, setEndpointFilter] = useState<string>('');
  const [tenantIdFilter, setTenantIdFilter] = useState<string>('');
  const [timeRange, setTimeRange] = useState<[Dayjs, Dayjs] | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      setIsAuthError(false);

      // Build query params
      const params: any = {
        limit: 200,
      };

      if (levelFilter) params.level = levelFilter;
      if (categoryFilter) params.category = categoryFilter;
      if (endpointFilter) params.endpoint = endpointFilter;
      if (tenantIdFilter) params.tenant_id = tenantIdFilter;
      if (timeRange && timeRange[0] && timeRange[1]) {
        params.from = timeRange[0].format('YYYY-MM-DD HH:mm:ss');
        params.to = timeRange[1].format('YYYY-MM-DD HH:mm:ss');
      }

      const [eventsData, summaryData, healthData] = await Promise.all([
        adminMonitoring.getEvents(params),
        adminMonitoring.getSummary(),
        adminMonitoring.getHealth(),
      ]);

      setEvents(eventsData.events || []);
      setSummary(summaryData);
      setHealth(healthData);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || 'Failed to load monitoring data';

      if (status === 401 || status === 403) {
        setIsAuthError(true);
        setError(status === 401
          ? 'Not authenticated. Please log in again.'
          : 'Access denied. Admin privileges required.');
      } else {
        setError(detail);
      }
      console.error('Error fetching monitoring data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchData();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, levelFilter, categoryFilter, endpointFilter, tenantIdFilter, timeRange]);

  const handleViewDetails = (event: MonitoringEvent) => {
    setSelectedEvent(event);
    setDetailModalVisible(true);
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR':
        return 'red';
      case 'WARN':
        return 'orange';
      case 'INFO':
        return 'blue';
      default:
        return 'default';
    }
  };

  const getStatusColor = (statusCode: number | null) => {
    if (!statusCode) return 'default';
    if (statusCode >= 500) return 'red';
    if (statusCode >= 400) return 'orange';
    return 'green';
  };

  const columns = [
    {
      title: 'Time',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (text: string) => dayjs(text).format('YYYY-MM-DD HH:mm:ss'),
      sorter: (a: MonitoringEvent, b: MonitoringEvent) =>
        dayjs(a.created_at).unix() - dayjs(b.created_at).unix(),
    },
    {
      title: 'Level',
      dataIndex: 'level',
      key: 'level',
      width: 100,
      render: (level: string) => (
        <Tag color={getLevelColor(level)}>{level}</Tag>
      ),
      filters: [
        { text: 'INFO', value: 'INFO' },
        { text: 'WARN', value: 'WARN' },
        { text: 'ERROR', value: 'ERROR' },
      ],
      onFilter: (value: any, record: MonitoringEvent) => record.level === value,
    },
    {
      title: 'Category',
      dataIndex: 'category',
      key: 'category',
      width: 120,
    },
    {
      title: 'Endpoint',
      key: 'endpoint',
      width: 200,
      render: (_: any, record: MonitoringEvent) => (
        <div>
          {record.method && <Tag>{record.method}</Tag>}
          <Text ellipsis style={{ maxWidth: 150 }}>
            {record.endpoint || '-'}
          </Text>
        </div>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status_code',
      key: 'status_code',
      width: 100,
      render: (statusCode: number | null) =>
        statusCode ? (
          <Tag color={getStatusColor(statusCode)}>{statusCode}</Tag>
        ) : (
          '-'
        ),
    },
    {
      title: 'Duration (ms)',
      dataIndex: 'duration_ms',
      key: 'duration_ms',
      width: 120,
      render: (duration: number | null) =>
        duration !== null ? (
          <Tag color={duration > 1000 ? 'orange' : 'default'}>
            {duration.toLocaleString()}
          </Tag>
        ) : (
          '-'
        ),
      sorter: (a: MonitoringEvent, b: MonitoringEvent) =>
        (a.duration_ms || 0) - (b.duration_ms || 0),
    },
    {
      title: 'Tenant',
      dataIndex: 'tenant_id',
      key: 'tenant_id',
      width: 120,
      render: (tenantId: string | null) =>
        tenantId ? <Text code>{tenantId.substring(0, 8)}...</Text> : '-',
    },
    {
      title: 'Message',
      dataIndex: 'message',
      key: 'message',
      ellipsis: true,
      render: (text: string) => <Text ellipsis>{text}</Text>,
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      render: (_: any, record: MonitoringEvent) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => handleViewDetails(record)}
        >
          View
        </Button>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
          <WarningOutlined style={{ marginRight: '8px' }} />
          System Monitoring
        </h1>
        <Space>
          <span>Auto-refresh:</span>
          <Switch checked={autoRefresh} onChange={setAutoRefresh} />
          <Button icon={<ReloadOutlined />} onClick={fetchData} loading={loading}>
            Refresh
          </Button>
        </Space>
      </div>

      {error && (
        <Alert
          message={isAuthError ? 'Authentication Error' : 'Error'}
          description={error}
          type={isAuthError ? 'warning' : 'error'}
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      <Tabs defaultActiveKey="overview">
        <TabPane tab="Overview" key="overview">
          <Spin spinning={loading}>
            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Errors (24h)"
                    value={summary?.counts_by_level?.ERROR || 0}
                    prefix={<CloseCircleOutlined style={{ color: '#ff4d4f' }} />}
                    valueStyle={{ color: '#ff4d4f' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Warnings (24h)"
                    value={summary?.counts_by_level?.WARN || 0}
                    prefix={<WarningOutlined style={{ color: '#faad14' }} />}
                    valueStyle={{ color: '#faad14' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Info (24h)"
                    value={summary?.counts_by_level?.INFO || 0}
                    prefix={<CheckCircleOutlined style={{ color: '#1890ff' }} />}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="Database"
                    value={health?.db_ok ? 'OK' : 'ERROR'}
                    prefix={
                      health?.db_ok ? (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      ) : (
                        <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                      )
                    }
                    valueStyle={{ color: health?.db_ok ? '#52c41a' : '#ff4d4f' }}
                  />
                </Card>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={12}>
                <Card title="Top Error Endpoints (24h)" size="small">
                  <Table
                    dataSource={summary?.top_error_endpoints || []}
                    columns={[
                      { title: 'Endpoint', dataIndex: 'endpoint', key: 'endpoint' },
                      { title: 'Method', dataIndex: 'method', key: 'method', width: 100 },
                      {
                        title: 'Errors',
                        dataIndex: 'error_count',
                        key: 'error_count',
                        width: 100,
                        render: (count: number) => <Tag color="red">{count}</Tag>,
                      },
                    ]}
                    pagination={false}
                    size="small"
                    rowKey={(record) => `${record.endpoint}-${record.method}`}
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="Top Slow Endpoints (24h)" size="small">
                  <Table
                    dataSource={summary?.top_slow_endpoints || []}
                    columns={[
                      { title: 'Endpoint', dataIndex: 'endpoint', key: 'endpoint' },
                      { title: 'Method', dataIndex: 'method', key: 'method', width: 100 },
                      {
                        title: 'Avg Duration (ms)',
                        dataIndex: 'avg_duration_ms',
                        key: 'avg_duration_ms',
                        width: 150,
                        render: (ms: number) => (
                          <Tag color={ms > 1000 ? 'orange' : 'default'}>
                            {ms.toFixed(0)}
                          </Tag>
                        ),
                      },
                      {
                        title: 'Requests',
                        dataIndex: 'request_count',
                        key: 'request_count',
                        width: 100,
                      },
                    ]}
                    pagination={false}
                    size="small"
                    rowKey={(record) => `${record.endpoint}-${record.method}`}
                  />
                </Card>
              </Col>
            </Row>

            <Card title="System Health">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <strong>Database:</strong>{' '}
                  <Tag color={health?.db_ok ? 'green' : 'red'}>
                    {health?.db_ok ? 'Connected' : 'Disconnected'}
                  </Tag>
                </div>
                <div>
                  <strong>Redis:</strong>{' '}
                  {health?.redis_ok === null ? (
                    <Tag>Not Configured</Tag>
                  ) : (
                    <Tag color={health?.redis_ok ? 'green' : 'red'}>
                      {health?.redis_ok ? 'Connected' : 'Disconnected'}
                    </Tag>
                  )}
                </div>
                <div>
                  <strong>App Version:</strong> {health?.app_version || 'Unknown'}
                </div>
              </Space>
            </Card>
          </Spin>
        </TabPane>

        <TabPane tab="Events" key="events">
          <Card>
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              <Space wrap>
                <Select
                  placeholder="Filter by Level"
                  allowClear
                  style={{ width: 150 }}
                  value={levelFilter}
                  onChange={setLevelFilter}
                >
                  <Select.Option value="INFO">INFO</Select.Option>
                  <Select.Option value="WARN">WARN</Select.Option>
                  <Select.Option value="ERROR">ERROR</Select.Option>
                </Select>

                <Select
                  placeholder="Filter by Category"
                  allowClear
                  style={{ width: 150 }}
                  value={categoryFilter}
                  onChange={setCategoryFilter}
                >
                  <Select.Option value="http">HTTP</Select.Option>
                  <Select.Option value="auth">Auth</Select.Option>
                  <Select.Option value="upload">Upload</Select.Option>
                  <Select.Option value="export">Export</Select.Option>
                  <Select.Option value="chat">Chat</Select.Option>
                  <Select.Option value="db">Database</Select.Option>
                  <Select.Option value="redis">Redis</Select.Option>
                  <Select.Option value="exception">Exception</Select.Option>
                </Select>

                <Input
                  placeholder="Search endpoint"
                  style={{ width: 200 }}
                  value={endpointFilter}
                  onChange={(e) => setEndpointFilter(e.target.value)}
                  allowClear
                />

                <Input
                  placeholder="Filter by Tenant ID"
                  style={{ width: 200 }}
                  value={tenantIdFilter}
                  onChange={(e) => setTenantIdFilter(e.target.value)}
                  allowClear
                />

                <RangePicker
                  showTime
                  value={timeRange}
                  onChange={(dates) => setTimeRange(dates as [Dayjs, Dayjs] | null)}
                  format="YYYY-MM-DD HH:mm:ss"
                />

                <Button onClick={fetchData} loading={loading}>
                  Apply Filters
                </Button>
              </Space>

              <Table
                columns={columns}
                dataSource={events}
                rowKey="id"
                loading={loading}
                pagination={{
                  pageSize: 50,
                  showSizeChanger: true,
                  showTotal: (total) => `Total ${total} events`,
                }}
                scroll={{ x: 1200 }}
              />
            </Space>
          </Card>
        </TabPane>
      </Tabs>

      <Modal
        title="Event Details"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            Close
          </Button>,
        ]}
        width={800}
      >
        {selectedEvent && (
          <div>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <strong>Time:</strong> {dayjs(selectedEvent.created_at).format('YYYY-MM-DD HH:mm:ss')}
              </div>
              <div>
                <strong>Level:</strong> <Tag color={getLevelColor(selectedEvent.level)}>{selectedEvent.level}</Tag>
              </div>
              <div>
                <strong>Category:</strong> {selectedEvent.category}
              </div>
              <div>
                <strong>Endpoint:</strong> {selectedEvent.method} {selectedEvent.endpoint || '-'}
              </div>
              <div>
                <strong>Status Code:</strong>{' '}
                {selectedEvent.status_code ? (
                  <Tag color={getStatusColor(selectedEvent.status_code)}>{selectedEvent.status_code}</Tag>
                ) : (
                  '-'
                )}
              </div>
              <div>
                <strong>Duration:</strong>{' '}
                {selectedEvent.duration_ms !== null
                  ? `${selectedEvent.duration_ms} ms`
                  : '-'}
              </div>
              <div>
                <strong>Tenant ID:</strong> {selectedEvent.tenant_id || '-'}
              </div>
              <div>
                <strong>User ID:</strong> {selectedEvent.user_id || '-'}
              </div>
              <div>
                <strong>Request ID:</strong> {selectedEvent.request_id || '-'}
              </div>
              <div>
                <strong>Message:</strong>
                <div style={{ marginTop: '8px', padding: '8px', background: '#f5f5f5', borderRadius: '4px' }}>
                  {selectedEvent.message}
                </div>
              </div>
              {selectedEvent.meta && (
                <div>
                  <strong>Metadata:</strong>
                  <pre
                    style={{
                      marginTop: '8px',
                      padding: '8px',
                      background: '#f5f5f5',
                      borderRadius: '4px',
                      maxHeight: '400px',
                      overflow: 'auto',
                    }}
                  >
                    {JSON.stringify(selectedEvent.meta, null, 2)}
                  </pre>
                </div>
              )}
            </Space>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AdminMonitoringPage;

