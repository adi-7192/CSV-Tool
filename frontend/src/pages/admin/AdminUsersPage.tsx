import React, { useState, useEffect } from 'react';
import { Card, Table, Tag, Button, Input, Space, Alert, Spin } from 'antd';
import { SearchOutlined, UserOutlined, ReloadOutlined } from '@ant-design/icons';
import { adminService, User } from '@/services/api';

const AdminUsersPage: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isAuthError, setIsAuthError] = useState(false);
  const [searchText, setSearchText] = useState('');

  const fetchUsers = async () => {
    try {
      setLoading(true);
      setError(null);
      setIsAuthError(false);
      const data = await adminService.getUsers();
      setUsers(data || []);
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

  const columns = [
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
      sorter: (a: User, b: User) => a.email.localeCompare(b.email),
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
      title: 'Onboarded',
      dataIndex: 'onboarded',
      key: 'onboarded',
      render: (onboarded: boolean) => (
        <Tag color={onboarded ? 'green' : 'default'}>
          {onboarded ? 'Yes' : 'No'}
        </Tag>
      ),
    },
    {
      title: 'Created At',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleDateString(),
      sorter: (a: User, b: User) => 
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
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
    </div>
  );
};

export default AdminUsersPage;

