import React from 'react';
import { Card, Descriptions, Tag, Button } from 'antd';
import { SettingOutlined, ReloadOutlined } from '@ant-design/icons';

const AdminSystemPage: React.FC = () => {
  // Mock system info - replace with actual API call
  const systemInfo = {
    version: '1.0.0',
    environment: 'production',
    database: 'DuckDB',
    vectorDb: 'ChromaDB',
    status: 'healthy',
    uptime: '15 days',
    lastBackup: '2025-12-03 10:00:00',
  };

  return (
    <div>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
          <SettingOutlined style={{ marginRight: '8px' }} />
          System Settings
        </h1>
        <Button icon={<ReloadOutlined />}>Refresh</Button>
      </div>
      <Card>
        <Descriptions title="System Information" bordered column={2}>
          <Descriptions.Item label="Version">{systemInfo.version}</Descriptions.Item>
          <Descriptions.Item label="Environment">
            <Tag color={systemInfo.environment === 'production' ? 'green' : 'orange'}>
              {systemInfo.environment}
            </Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Database">{systemInfo.database}</Descriptions.Item>
          <Descriptions.Item label="Vector Database">{systemInfo.vectorDb}</Descriptions.Item>
          <Descriptions.Item label="Status">
            <Tag color="green">{systemInfo.status}</Tag>
          </Descriptions.Item>
          <Descriptions.Item label="Uptime">{systemInfo.uptime}</Descriptions.Item>
          <Descriptions.Item label="Last Backup" span={2}>
            {systemInfo.lastBackup}
          </Descriptions.Item>
        </Descriptions>
      </Card>
    </div>
  );
};

export default AdminSystemPage;

