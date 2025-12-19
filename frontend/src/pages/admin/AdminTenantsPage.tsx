import React, { useState, useEffect } from 'react';
import { Card, Table, Tag, Button, Space, Alert, Spin, Modal, message } from 'antd';
import { DatabaseOutlined, ReloadOutlined, DeleteOutlined } from '@ant-design/icons';
import { adminService, TenantUsage } from '@/services/api';
import dayjs from 'dayjs';

const AdminTenantsPage: React.FC = () => {
  const [tenants, setTenants] = useState<TenantUsage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [selectedTenant, setSelectedTenant] = useState<TenantUsage | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleting, setDeleting] = useState(false);

  const fetchTenants = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await adminService.getTenantsUsage();
      setTenants(data || []);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || 'Failed to load tenants';
      
      if (status === 401 || status === 403) {
        setError(status === 401
          ? 'Not authenticated. Please log in again.'
          : 'Access denied. Admin privileges required.');
      } else {
        setError(detail);
      }
      console.error('Error fetching tenants:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTenants();
  }, []);

  const handleDelete = (tenant: TenantUsage) => {
    setSelectedTenant(tenant);
    setDeleteModalVisible(true);
    setDeleteConfirmText('');
  };

  const confirmDelete = async () => {
    if (deleteConfirmText !== 'DELETE') {
      message.error('Please type "DELETE" to confirm');
      return;
    }

    if (!selectedTenant) return;

    setDeleting(true);
    try {
      const result = await adminService.deleteTenant(selectedTenant.tenant_id, false);
      message.success(`Deleted ${result.deleted_rows} rows for tenant ${selectedTenant.user_email}`);
      setDeleteModalVisible(false);
      setDeleteConfirmText('');
      setSelectedTenant(null);
      await fetchTenants();
    } catch (err: any) {
      message.error(err?.response?.data?.detail || 'Failed to delete tenant data');
    } finally {
      setDeleting(false);
    }
  };

  const columns = [
    {
      title: 'User Email',
      dataIndex: 'user_email',
      key: 'user_email',
      sorter: (a: TenantUsage, b: TenantUsage) => a.user_email.localeCompare(b.user_email),
    },
    {
      title: 'Tenant ID',
      dataIndex: 'tenant_id',
      key: 'tenant_id',
      render: (tenantId: string) => <code style={{ fontSize: '12px' }}>{tenantId}</code>,
    },
    {
      title: 'Files',
      dataIndex: 'file_count',
      key: 'file_count',
      render: (count: number) => count.toLocaleString(),
      sorter: (a: TenantUsage, b: TenantUsage) => a.file_count - b.file_count,
    },
    {
      title: 'Records',
      dataIndex: 'row_count',
      key: 'row_count',
      render: (count: number) => count.toLocaleString(),
      sorter: (a: TenantUsage, b: TenantUsage) => a.row_count - b.row_count,
    },
    {
      title: 'Last Upload',
      dataIndex: 'last_upload_at',
      key: 'last_upload_at',
      render: (date: string | null) => date ? dayjs(date).format('YYYY-MM-DD HH:mm') : 'Never',
      sorter: (a: TenantUsage, b: TenantUsage) => {
        if (!a.last_upload_at) return 1;
        if (!b.last_upload_at) return -1;
        return dayjs(a.last_upload_at).unix() - dayjs(b.last_upload_at).unix();
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: TenantUsage) => (
        <Space>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDelete(record)}
          >
            Delete Data
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
          <DatabaseOutlined style={{ marginRight: '8px' }} />
          Tenants & Data Management
        </h1>
        <Button icon={<ReloadOutlined />} onClick={fetchTenants} loading={loading}>
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

      <Card>
        <Table
          dataSource={tenants}
          columns={columns}
          rowKey="tenant_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} tenants`,
          }}
        />
      </Card>

      <Modal
        title="Delete Tenant Data"
        open={deleteModalVisible}
        onCancel={() => {
          setDeleteModalVisible(false);
          setDeleteConfirmText('');
          setSelectedTenant(null);
        }}
        onOk={confirmDelete}
        confirmLoading={deleting}
        okText="Delete"
        okButtonProps={{ danger: true }}
      >
        <div>
          <p>
            <strong>Warning:</strong> This will permanently delete all data for tenant{' '}
            <code>{selectedTenant?.tenant_id}</code> (user: {selectedTenant?.user_email}).
          </p>
          <p>This action cannot be undone.</p>
          <p>
            <strong>Files:</strong> {selectedTenant?.file_count || 0}
            <br />
            <strong>Records:</strong> {selectedTenant?.row_count?.toLocaleString() || 0}
          </p>
          <p style={{ marginTop: '16px' }}>
            Type <strong>DELETE</strong> to confirm:
          </p>
          <input
            type="text"
            value={deleteConfirmText}
            onChange={(e) => setDeleteConfirmText(e.target.value)}
            placeholder="Type DELETE to confirm"
            style={{
              width: '100%',
              padding: '8px',
              marginTop: '8px',
              border: '1px solid #d9d9d9',
              borderRadius: '4px',
            }}
          />
        </div>
      </Modal>
    </div>
  );
};

export default AdminTenantsPage;

