import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Alert, Button, Spin } from 'antd';
import { BarChartOutlined, UserOutlined, DatabaseOutlined, ApiOutlined, ReloadOutlined } from '@ant-design/icons';
import { adminService } from '@/services/api';

const AdminUsagePage: React.FC = () => {
  const [stats, setStats] = useState({
    total_users: 0,
    active_users: 0,
    total_records: 0,
    api_calls_30d: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await adminService.getStatsSummary();
      setStats(data);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || 'Failed to load usage statistics';
      
      if (status === 401 || status === 403) {
        setError(status === 401
          ? 'Not authenticated. Please log in again.'
          : 'Access denied. Admin privileges required.');
      } else {
        setError(detail);
      }
      console.error('Error fetching usage stats:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  return (
    <div>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
          <BarChartOutlined style={{ marginRight: '8px' }} />
          Usage Analytics
        </h1>
        <Button icon={<ReloadOutlined />} onClick={fetchStats} loading={loading}>
          Refresh
        </Button>
      </div>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      <Spin spinning={loading}>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Total Users"
                value={stats.total_users}
                prefix={<UserOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Active Users"
                value={stats.active_users}
                prefix={<UserOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="Total Records"
                value={stats.total_records}
                prefix={<DatabaseOutlined />}
                valueStyle={{ color: '#722ed1' }}
                formatter={(value) => (value as number).toLocaleString()}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <Card>
              <Statistic
                title="API Calls (30d)"
                value={stats.api_calls_30d}
                prefix={<ApiOutlined />}
                valueStyle={{ color: '#eb2f96' }}
                formatter={(value) => (value as number).toLocaleString()}
              />
            </Card>
          </Col>
        </Row>
      </Spin>
    </div>
  );
};

export default AdminUsagePage;

