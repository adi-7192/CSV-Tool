/**
 * AppLayout Component
 * 
 * Modern, Basedash-inspired application layout with sidebar navigation and header.
 * Features collapsible sidebar, responsive design, and smooth transitions.
 */

import React, { useState, useEffect } from 'react';
import { Layout, Menu, Avatar, Dropdown, Button, Badge, Drawer, Breadcrumb, DatePicker } from 'antd';
import type { MenuProps } from 'antd';
import {
  DashboardOutlined,
  DatabaseOutlined,
  FolderOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
  BellOutlined,
  HomeOutlined,
  CrownOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import dayjs, { Dayjs } from 'dayjs';
import { useDataStore } from '@/store/dataStore';
import { useAuthStore } from '@/store/authStore';
import { getDataDateRange } from '@/services/dataService';
import { COLORS, SPACING } from '@/styles/design-tokens';
import './AppLayout.css';

const { RangePicker } = DatePicker;

const { Sider, Header, Content } = Layout;

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { dateRange, setDateRange } = useDataStore();
  const { logout, user } = useAuthStore();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  // Load date range from API
  useEffect(() => {
    const loadDateRange = async () => {
      try {
        const response = await getDataDateRange();
        if (response && response.has_data && response.start_date && response.end_date) {
          setDateRange(response.start_date, response.end_date);
          localStorage.setItem('dateRange', JSON.stringify({
            start: response.start_date,
            end: response.end_date,
          }));
        } else {
          setDateRange('', '');
          localStorage.removeItem('dateRange');
        }
      } catch (error) {
        console.error('Failed to load date range:', error);
        const savedRange = localStorage.getItem('dateRange');
        if (savedRange) {
          try {
            const parsed = JSON.parse(savedRange);
            if (parsed.start && parsed.end) {
              setDateRange(parsed.start, parsed.end);
            }
          } catch (e) {
            localStorage.removeItem('dateRange');
          }
        }
      }
    };

    const savedRange = localStorage.getItem('dateRange');
    if (savedRange) {
      try {
        const parsed = JSON.parse(savedRange);
        if (parsed.start && parsed.end) {
          setDateRange(parsed.start, parsed.end);
        }
      } catch (e) {
        localStorage.removeItem('dateRange');
      }
    }

    loadDateRange();

    const handleDataUpload = () => {
      loadDateRange();
    };

    window.addEventListener('dataUploaded', handleDataUpload);
    return () => {
      window.removeEventListener('dataUploaded', handleDataUpload);
    };
  }, [setDateRange]);

  // Handle date range change
  const handleDateRangeChange = async (dates: [Dayjs | null, Dayjs | null] | null) => {
    if (dates && dates[0] && dates[1]) {
      const startDate = dates[0].format('YYYY-MM-DD');
      const endDate = dates[1].format('YYYY-MM-DD');
      setDateRange(startDate, endDate);
      localStorage.setItem('dateRange', JSON.stringify({
        start: startDate,
        end: endDate,
      }));
    } else {
      try {
        const response = await getDataDateRange();
        if (response && response.has_data && response.start_date && response.end_date) {
          setDateRange(response.start_date, response.end_date);
          localStorage.setItem('dateRange', JSON.stringify({
            start: response.start_date,
            end: response.end_date,
          }));
        } else {
          setDateRange('', '');
          localStorage.removeItem('dateRange');
        }
      } catch (error) {
        console.error('Failed to reload date range:', error);
        setDateRange('', '');
        localStorage.removeItem('dateRange');
      }
    }
  };

  const dateRangeValue: [Dayjs, Dayjs] | null =
    dateRange.start && dateRange.end
      ? [dayjs(dateRange.start), dayjs(dateRange.end)]
      : null;

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) {
        setMobileDrawerOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Navigation menu items
  const menuItems: MenuProps['items'] = [
    {
      key: '/app/dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
      onClick: () => {
        navigate('/app/dashboard');
        if (isMobile) setMobileDrawerOpen(false);
      },
    },
    {
      key: '/app/workspace',
      icon: <DatabaseOutlined />,
      label: 'Data Workspace',
      onClick: () => {
        navigate('/app/workspace');
        if (isMobile) setMobileDrawerOpen(false);
      },
    },
    {
      key: '/app/data-management',
      icon: <FolderOutlined />,
      label: 'Data Management',
      onClick: () => {
        navigate('/app/data-management');
        if (isMobile) setMobileDrawerOpen(false);
      },
    },
    {
      key: '/app/analyst',
      icon: <DashboardOutlined />,
      label: 'AI Analyst',
      onClick: () => {
        navigate('/app/analyst');
        if (isMobile) setMobileDrawerOpen(false);
      },
    },
    // Admin link - only visible to admin users
    ...(user?.role === 'admin'
      ? [
          {
            type: 'divider' as const,
          },
          {
            key: '/admin/users',
            icon: <CrownOutlined />,
            label: 'Admin Panel',
            onClick: () => {
              navigate('/admin/users');
              if (isMobile) setMobileDrawerOpen(false);
            },
          },
        ]
      : []),
  ];

  // User profile menu items
  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
      onClick: () => {
        // TODO: Navigate to profile page
        console.log('Profile clicked');
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
      onClick: () => {
        logout();
        navigate('/');
      },
    },
  ];

  // Get current page title for breadcrumb
  const getPageTitle = () => {
    const path = location.pathname;
    if (path === '/dashboard') return 'Dashboard';
    if (path === '/workspace') return 'Data Workspace';
    if (path === '/data-management') return 'Data Management';
    if (path === '/analyst') return 'AI Analyst';
    if (path === '/settings') return 'Settings';
    return 'Page';
  };

  // Toggle sidebar
  const toggleSidebar = () => {
    if (isMobile) {
      setMobileDrawerOpen(!mobileDrawerOpen);
    } else {
      setCollapsed(!collapsed);
    }
  };

  // Sidebar content
  const sidebarContent = (
    <div className="app-sidebar">
      {/* Logo Section */}
      <div className="sidebar-logo">
        <div className="logo-container">
          <div className="logo-icon">D</div>
          {!collapsed && <span className="logo-text">Datadost</span>}
        </div>
        {!isMobile && (
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={toggleSidebar}
            className="sidebar-toggle"
          />
        )}
      </div>

      {/* Navigation Menu */}
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        className="sidebar-menu"
      />

      {/* User Profile Section */}
      <div className="sidebar-user">
        <div className="user-info">
          <Avatar
            size={collapsed ? 32 : 40}
            icon={<UserOutlined />}
            style={{
              backgroundColor: COLORS.primary,
              flexShrink: 0,
            }}
          />
          {!collapsed && (
            <div className="user-details">
              <div className="user-name">
                {user?.name || (user?.email ? user.email.split('@')[0] : 'Not signed in')}
              </div>
              <div className="user-email">{user?.email || 'Not signed in'}</div>
            </div>
          )}
        </div>
        {!collapsed && (
          <div className="user-actions">
            <Button
              type="text"
              icon={<SettingOutlined />}
              onClick={() => navigate('/settings')}
              className="user-action-btn"
            />
            <Button
              type="text"
              icon={<LogoutOutlined />}
              danger
              onClick={() => {
                logout();
                navigate('/');
              }}
              className="user-action-btn"
            />
          </div>
        )}
      </div>
    </div>
  );

  return (
    <Layout className="app-layout">
      {/* Desktop Sidebar */}
      {!isMobile && (
        <Sider
          width={240}
          collapsedWidth={80}
          collapsed={collapsed}
          className="app-sider"
          theme="light"
        >
          {sidebarContent}
        </Sider>
      )}

      {/* Mobile Drawer */}
      {isMobile && (
        <Drawer
          placement="left"
          closable={false}
          onClose={() => setMobileDrawerOpen(false)}
          open={mobileDrawerOpen}
          bodyStyle={{ padding: 0 }}
          width={240}
        >
          {sidebarContent}
        </Drawer>
      )}

      <Layout className="app-layout-main">
        {/* Header */}
        <Header className="app-header">
          <div className="header-left">
            {isMobile && (
              <Button
                type="text"
                icon={<MenuUnfoldOutlined />}
                onClick={toggleSidebar}
                className="mobile-menu-btn"
              />
            )}
            <Breadcrumb
              items={[
                {
                  href: '/dashboard',
                  title: <HomeOutlined />,
                },
                {
                  title: getPageTitle(),
                },
              ]}
              className="header-breadcrumb"
            />
          </div>

          <div className="header-right">
            {(location.pathname === '/app/dashboard' || location.pathname === '/dashboard') && (
              <RangePicker
                value={dateRangeValue}
                onChange={handleDateRangeChange}
                format="MMM DD, YYYY"
                placeholder={['Start Date', 'End Date']}
                className="header-date-picker"
                allowClear={true}
                inputReadOnly={true}
              />
            )}
            <Badge count={0} showZero={false}>
              <Button
                type="text"
                icon={<BellOutlined />}
                className="header-action-btn"
              />
            </Badge>
            <Dropdown
              menu={{ items: userMenuItems }}
              placement="bottomRight"
              trigger={['click']}
            >
              <Button
                type="text"
                className="header-user-btn"
              >
                <Avatar
                  size={32}
                  icon={<UserOutlined />}
                  style={{
                    backgroundColor: COLORS.primary,
                    marginRight: SPACING.xs,
                  }}
                />
                <span className="header-user-name">
                  {user?.name || (user?.email ? user.email.split('@')[0] : 'Welcome')}
                </span>
              </Button>
            </Dropdown>
          </div>
        </Header>

        {/* Content */}
        <Content className="app-content">
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default AppLayout;

