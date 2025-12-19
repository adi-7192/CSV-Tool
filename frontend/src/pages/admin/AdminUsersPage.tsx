import React, { useState, useEffect } from 'react';
import { Card, Table, Tag, Button, Input, Space, Alert, Spin, Modal, message, Checkbox } from 'antd';
import { SearchOutlined, UserOutlined, ReloadOutlined, DeleteOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { adminService, User, TenantUsage } from '@/services/api';

const AdminUsersPage: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [usageStats, setUsageStats] = useState<Record<string, TenantUsage>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isAuthError, setIsAuthError] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState('');
  const [deleteUserCheckbox, setDeleteUserCheckbox] = useState(false);
  const [selectedTenantId, setSelectedTenantId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [activating, setActivating] = useState<number | null>(null);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      setError(null);
      setIsAuthError(false);
      const [usersData, usageData] = await Promise.all([
        adminService.getUsers(),
        adminService.getTenantsUsage().catch(() => []) // Don't fail if usage endpoint fails
      ]);
      setUsers(usersData || []);
      
      // Create a map of tenant_id -> usage stats
      const usageMap: Record<string, TenantUsage> = {};
      usageData.forEach((usage) => {
        usageMap[usage.tenant_id] = usage;
      });
      setUsageStats(usageMap);
    } catch (err: any) {
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || 'Failed to load users';
      
      // Check if it's an auth error (401 or 403)
      if (status === 401 || status === 403) {
        setIsAuthError(true);
        setError(status === 401 
          ? 'Not authenticated. Please log in again.' 
          : 'Access denied. Admin privileges required.');
      } else {
        setError(detail);
      }
      console.error('Error fetching users:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const filteredUsers = users.filter(user =>
    user.email.toLowerCase().includes(searchText.toLowerCase())
  );

  const handleActivate = async (userId: number) => {
    setActivating(userId);
    try {
      const result = await adminService.activateUser(userId);
      message.success(result.message);
      await fetchUsers();
    } catch (err: any) {
      message.error(err?.response?.data?.detail || 'Failed to activate user');
    } finally {
      setActivating(null);
    }
  };

  const handleDeactivate = async (userId: number) => {
    setActivating(userId);
    try {
      const result = await adminService.deactivateUser(userId);
      message.success(result.message);
      await fetchUsers();
    } catch (err: any) {
      message.error(err?.response?.data?.detail || 'Failed to deactivate user');
    } finally {
      setActivating(null);
    }
  };

  const handleDeleteTenant = (tenantId: string) => {
    setSelectedTenantId(tenantId);
    setDeleteModalVisible(true);
    setDeleteConfirmText('');
    setDeleteUserCheckbox(false);
  };

  const confirmDeleteTenant = async () => {
    if (deleteConfirmText !== 'DELETE') {
      message.error('Please type "DELETE" to confirm');
      return;
    }

    if (!selectedTenantId) return;

    setDeleting(true);
    try {
      const result = await adminService.deleteTenant(selectedTenantId, deleteUserCheckbox);
      message.success(`Deleted ${result.deleted_rows} rows and ${result.users_deleted} user(s)`);
      setDeleteModalVisible(false);
      setDeleteConfirmText('');
      setSelectedTenantId(null);
      await fetchUsers();
    } catch (err: any) {
      message.error(err?.response?.data?.detail || 'Failed to delete tenant data');
    } finally {
      setDeleting(false);
    }
  };

  const columns = [
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
      sorter: (a: User, b: User) => a.email.localeCompare(b.email),
    },
    {
      title: 'Status',
      key: 'status',
      render: (_: any, record: User) => (
        <Tag color={record.is_active !== false ? 'green' : 'red'}>
          {record.is_active !== false ? 'Active' : 'Deactivated'}
        </Tag>
      ),
    },
    {
      title: 'Role',
      dataIndex: 'role',
      key: 'role',
      render: (role: string) => (
        <Tag color={role === 'admin' ? 'red' : 'blue'}>{role}</Tag>
      ),
    },
    {
      title: 'Plan',
      dataIndex: 'plan',
      key: 'plan',
      render: (plan: string) => (
        <Tag color={plan === 'enterprise' ? 'purple' : plan === 'pro' ? 'orange' : 'default'}>
          {plan}
        </Tag>
      ),
    },
    {
      title: 'Usage',
      key: 'usage',
      render: (_: any, record: User) => {
        const tenantId = record.tenant_id || String(record.id);
        const usage = usageStats[tenantId];
        if (!usage) return '-';
        return (
          <Space direction="vertical" size="small" style={{ fontSize: '12px' }}>
            <span>Files: {usage.file_count}</span>
            <span>Rows: {usage.row_count.toLocaleString()}</span>
            {usage.last_upload_at && (
              <span style={{ color: '#64748B' }}>
                Last: {new Date(usage.last_upload_at).toLocaleDateString()}
              </span>
            )}
          </Space>
        );
      },
    },
    {
      title: 'Last Login',
      dataIndex: 'last_login_at',
      key: 'last_login_at',
      render: (date: string | null | undefined) => 
        date ? new Date(date).toLocaleString() : 'Never',
      sorter: (a: User, b: User) => {
        const aDate = a.last_login_at ? new Date(a.last_login_at).getTime() : 0;
        const bDate = b.last_login_at ? new Date(b.last_login_at).getTime() : 0;
        return aDate - bDate;
      },
    },
    {
      title: 'Created At',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleDateString(),
      sorter: (a: User, b: User) => 
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: User) => {
        const tenantId = record.tenant_id || String(record.id);
        const usage = usageStats[tenantId];
        const hasData = usage && (usage.file_count > 0 || usage.row_count > 0);
        
        return (
          <Space>
            {record.is_active !== false ? (
              <Button
                size="small"
                danger
                icon={<CloseCircleOutlined />}
                loading={activating === record.id}
                onClick={() => handleDeactivate(record.id)}
              >
                Deactivate
              </Button>
            ) : (
              <Button
                size="small"
                type="primary"
                icon={<CheckCircleOutlined />}
                loading={activating === record.id}
                onClick={() => handleActivate(record.id)}
              >
                Activate
              </Button>
            )}
            {hasData && (
              <Button
                size="small"
                danger
                icon={<DeleteOutlined />}
                onClick={() => handleDeleteTenant(tenantId)}
              >
                Delete Data
              </Button>
            )}
          </Space>
        );
      },
    },
  ];

  if (loading && users.length === 0) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', margin: 0 }}>
          <UserOutlined style={{ marginRight: '8px' }} />
          User Management
        </h1>
        <Space>
          <Input
            placeholder="Search users..."
            prefix={<SearchOutlined />}
            style={{ width: '300px' }}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
          <Button icon={<ReloadOutlined />} onClick={fetchUsers} loading={loading}>
            Refresh
          </Button>
        </Space>
      </div>

      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '24px' }}
        />
      )}

      <Card>
        {/* Only show "No users found" when we have a successful load with no users, not on auth errors */}
        {filteredUsers.length === 0 && !loading && !isAuthError ? (
          <div style={{ textAlign: 'center', padding: '48px', color: '#64748B' }}>
            {searchText ? 'No users found matching your search' : 'No users found'}
          </div>
        ) : isAuthError && !loading ? (
          <div style={{ textAlign: 'center', padding: '48px', color: '#64748B' }}>
            Unable to load users due to authentication issue. Please try logging in again.
          </div>
        ) : (
          <Table
            dataSource={filteredUsers}
            columns={columns}
            rowKey="id"
            pagination={{ pageSize: 10 }}
            loading={loading}
          />
        )}
      </Card>

      {/* Delete Tenant Data Modal */}
      <Modal
        title={
          <span>
            <DeleteOutlined style={{ color: '#ff4d4f', marginRight: '8px' }} />
            Delete Tenant Data?
          </span>
        }
        open={deleteModalVisible}
        onCancel={() => {
          setDeleteModalVisible(false);
          setDeleteConfirmText('');
          setDeleteUserCheckbox(false);
          setSelectedTenantId(null);
        }}
        footer={[
          <Button key="cancel" onClick={() => {
            setDeleteModalVisible(false);
            setDeleteConfirmText('');
            setDeleteUserCheckbox(false);
            setSelectedTenantId(null);
          }}>
            Cancel
          </Button>,
          <Button
            key="delete"
            type="primary"
            danger
            loading={deleting}
            disabled={deleteConfirmText !== 'DELETE'}
            onClick={confirmDeleteTenant}
          >
            Delete All Data
          </Button>,
        ]}
      >
        <Alert
          message="Warning"
          description="This will permanently delete all data for this tenant, including all uploaded files, sales data, and analytics. This action cannot be undone."
          type="warning"
          showIcon
          style={{ marginBottom: '16px' }}
        />
        
        <p style={{ marginBottom: '12px' }}>
          To confirm, please type <strong style={{ color: '#ff4d4f' }}>DELETE</strong> in the box below:
        </p>
        
        <Input
          placeholder="Type DELETE to confirm"
          value={deleteConfirmText}
          onChange={(e) => setDeleteConfirmText(e.target.value)}
          style={{ marginBottom: '16px' }}
        />
        
        <Checkbox
          checked={deleteUserCheckbox}
          onChange={(e) => setDeleteUserCheckbox(e.target.checked)}
        >
          Also delete user account(s) for this tenant
        </Checkbox>
      </Modal>
    </div>
  );
};

export default AdminUsersPage;
