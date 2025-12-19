import React from 'react';
import { DatePicker, Dropdown, Avatar, Button } from 'antd';
import type { MenuProps } from 'antd';
import { UserOutlined, SettingOutlined, QuestionCircleOutlined, LogoutOutlined } from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import dayjs, { Dayjs } from 'dayjs';
import { useDataStore } from '@/store/dataStore';
import { useAuthStore } from '@/store/authStore';

const { RangePicker } = DatePicker;

const TopBar: React.FC = () => {
  const { dateRange, setDateRange } = useDataStore();
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();

  // Handle date range change
  const handleDateRangeChange = (dates: [Dayjs | null, Dayjs | null] | null) => {
    if (dates && dates[0] && dates[1]) {
      const startDate = dates[0].format('YYYY-MM-DD');
      const endDate = dates[1].format('YYYY-MM-DD');
      setDateRange(startDate, endDate);
      // Dashboard will re-fetch via useEffect when dateRange changes
    }
  };

  // Convert store dateRange to dayjs format for RangePicker
  const dateRangeValue: [Dayjs, Dayjs] | null = 
    dateRange.start && dateRange.end
      ? [dayjs(dateRange.start), dayjs(dateRange.end)]
      : null;

  // Profile menu items
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
      key: 'help',
      icon: <QuestionCircleOutlined />,
      label: 'Help',
      onClick: () => {
        console.log('Help clicked');
        // TODO: Show help modal or navigate to help page
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

  return (
    <div
      style={{
        height: '64px',
        backgroundColor: '#F8FAFC',
        borderBottom: '1px solid #E2E8F0',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      {/* Logo/Brand Section */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '24px',
        }}
      >
        <div
          style={{
            fontSize: '20px',
            fontWeight: '700',
            color: '#6366F1',
            letterSpacing: '-0.5px',
            cursor: 'pointer',
          }}
          onClick={() => navigate('/app/dashboard')}
        >
          Analytics Dashboard
        </div>
        
        {/* Navigation Links */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button
            type={location.pathname === '/app/dashboard' ? 'primary' : 'text'}
            onClick={() => navigate('/app/dashboard')}
            style={{
              fontWeight: location.pathname === '/app/dashboard' ? '600' : '400',
            }}
          >
            Dashboard
          </Button>
          <Button
            type={location.pathname === '/app/workspace' ? 'primary' : 'text'}
            onClick={() => navigate('/app/workspace')}
            style={{
              fontWeight: location.pathname === '/app/workspace' ? '600' : '400',
            }}
          >
            Data Workspace
          </Button>
          <Button
            type={location.pathname === '/app/data-management' ? 'primary' : 'text'}
            onClick={() => navigate('/app/data-management')}
            style={{
              fontWeight: location.pathname === '/app/data-management' ? '600' : '400',
            }}
          >
            Data Management
          </Button>
          <Button
            type={location.pathname === '/app/analyst' ? 'primary' : 'text'}
            onClick={() => navigate('/app/analyst')}
            style={{
              fontWeight: location.pathname === '/app/analyst' ? '600' : '400',
            }}
          >
            AI Analyst
          </Button>
          {user?.role === 'admin' && (
            <Button
              type={location.pathname.startsWith('/admin') ? 'primary' : 'text'}
              onClick={() => navigate('/admin')}
              style={{
                fontWeight: location.pathname.startsWith('/admin') ? '600' : '400',
                color: location.pathname.startsWith('/admin') ? undefined : '#6366F1',
              }}
            >
              Admin
            </Button>
          )}
        </div>
      </div>

      {/* Date Range Picker (Center) */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
        }}
      >
        <RangePicker
          value={dateRangeValue}
          onChange={handleDateRangeChange}
          format="MMM D, YYYY"
          style={{
            borderRadius: '6px',
            border: '1px solid #E2E8F0',
            padding: '8px 12px',
            backgroundColor: '#FFFFFF',
          }}
          allowClear={false}
        />
      </div>

      {/* Profile Menu (Right) */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
        }}
      >
        <Dropdown
          menu={{ items: profileMenuItems }}
          placement="bottomRight"
          trigger={['click']}
        >
          <Button
            type="text"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '4px 12px',
              height: 'auto',
              color: '#030712',
              borderRadius: '6px',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#F1F5F9';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <span style={{ fontSize: '14px', fontWeight: '500' }}>
              {user?.email || 'Welcome'}
            </span>
            <Avatar
              size={32}
              icon={<UserOutlined />}
              style={{
                backgroundColor: '#6366F1',
                border: '2px solid #E2E8F0',
              }}
            />
          </Button>
        </Dropdown>
      </div>
    </div>
  );
};

export default TopBar;
