import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Avatar, Dropdown, Button, Badge } from 'antd';
import {
  UserOutlined,
  TeamOutlined,
  DatabaseOutlined,
  BarChartOutlined,
  SettingOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  WarningOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import { useAuthStore } from '@/store/authStore';

const { Header, Sider, Content } = Layout;

interface AdminLayoutProps {
  children: React.ReactNode;
}

const AdminLayout: React.FC<AdminLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const profileMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
      onClick: () => {
        navigate('/app/profile');
      },
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
      onClick: () => {
        navigate('/app/settings');
      },
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      danger: true,
      onClick: handleLogout,
    },
  ];

  const menuItems: MenuProps['items'] = [
    {
      key: '/admin/users',
      icon: <TeamOutlined />,
      label: <Link to="/admin/users">Users</Link>,
    },
    {
      key: '/admin/tenants',
      icon: <DatabaseOutlined />,
      label: <Link to="/admin/tenants">Tenants/Data</Link>,
    },
    {
      key: '/admin/usage',
      icon: <BarChartOutlined />,
      label: <Link to="/admin/usage">Usage</Link>,
    },
    {
      key: '/admin/system',
      icon: <SettingOutlined />,
      label: <Link to="/admin/system">System</Link>,
    },
    {
      key: '/admin/monitoring',
      icon: <WarningOutlined />,
      label: <Link to="/admin/monitoring">Monitoring</Link>,
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
        }}
      >
        <div
          style={{
            height: '64px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #E2E8F0',
            marginBottom: '16px',
          }}
        >
          <div
            style={{
              fontSize: collapsed ? '16px' : '18px',
              fontWeight: '700',
              color: '#6366F1',
            }}
          >
            {collapsed ? 'A' : 'Admin'}
          </div>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
        />
      </Sider>
      <Layout style={{ marginLeft: collapsed ? 80 : 200, transition: 'margin-left 0.2s' }}>
        <Header
          style={{
            padding: '0 24px',
            background: '#FFFFFF',
            borderBottom: '1px solid #E2E8F0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'sticky',
            top: 0,
            zIndex: 100,
          }}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ fontSize: '16px', width: 64, height: 64 }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <Badge
              count="Admin"
              style={{
                backgroundColor: '#6366F1',
                fontSize: '10px',
                padding: '2px 8px',
                borderRadius: '4px',
              }}
            />
            <Dropdown menu={{ items: profileMenuItems }} placement="bottomRight">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                <span style={{ fontSize: '14px', fontWeight: '500' }}>
                  {user?.name || (user?.email ? user.email.split('@')[0] : 'Admin')}
                </span>
                <Avatar
                  size={32}
                  icon={<UserOutlined />}
                  style={{ backgroundColor: '#6366F1' }}
                />
              </div>
            </Dropdown>
          </div>
        </Header>
        <Content
          style={{
            margin: '24px',
            padding: '24px',
            background: '#FFFFFF',
            borderRadius: '8px',
            minHeight: 'calc(100vh - 112px)',
            overflowY: 'auto',
            overflowX: 'hidden',
          }}
        >
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default AdminLayout;

