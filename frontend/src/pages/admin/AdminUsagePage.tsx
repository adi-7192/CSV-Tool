import React from 'react';
import { Card, Row, Col, Statistic } from 'antd';
import { BarChartOutlined, UserOutlined, DatabaseOutlined, ApiOutlined } from '@ant-design/icons';

const AdminUsagePage: React.FC = () => {
  // Mock data - replace with actual API call
  const stats = {
    totalUsers: 150,
    activeUsers: 120,
    totalRecords: 50000,
    apiCalls: 12500,
  };

  return (
    <div>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
          <BarChartOutlined style={{ marginRight: '8px' }} />
          Usage Analytics
        </h1>
      </div>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Users"
              value={stats.totalUsers}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Active Users"
              value={stats.activeUsers}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Records"
              value={stats.totalRecords}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="API Calls (30d)"
              value={stats.apiCalls}
              prefix={<ApiOutlined />}
              valueStyle={{ color: '#eb2f96' }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default AdminUsagePage;

