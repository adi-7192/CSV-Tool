import React from 'react';
import { Card, Table, Tag, Button, Space } from 'antd';
import { DatabaseOutlined } from '@ant-design/icons';

const AdminTenantsPage: React.FC = () => {
  // Mock data - replace with actual API call
  const mockTenants = [
    { id: '1', name: 'Tenant A', records: 15000, status: 'active', createdAt: '2025-01-01' },
    { id: '2', name: 'Tenant B', records: 8500, status: 'active', createdAt: '2025-01-02' },
    { id: '3', name: 'Tenant C', records: 25000, status: 'inactive', createdAt: '2025-01-01' },
  ];

  const columns = [
    {
      title: 'Tenant Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Records',
      dataIndex: 'records',
      key: 'records',
      render: (records: number) => records.toLocaleString(),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'default'}>{status}</Tag>
      ),
    },
    {
      title: 'Created At',
      dataIndex: 'createdAt',
      key: 'createdAt',
    },
    {
      title: 'Actions',
      key: 'actions',
      render: () => (
        <Space>
          <Button size="small">View Data</Button>
          <Button size="small" danger>Delete</Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
          <DatabaseOutlined style={{ marginRight: '8px' }} />
          Tenants & Data Management
        </h1>
      </div>
      <Card>
        <Table
          dataSource={mockTenants}
          columns={columns}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    </div>
  );
};

export default AdminTenantsPage;

