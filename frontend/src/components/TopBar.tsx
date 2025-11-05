import React from 'react';
import { Layout, DatePicker, Dropdown, Space, Avatar } from 'antd';
import type { MenuProps } from 'antd';
import { UserOutlined, SettingOutlined, QuestionCircleOutlined, LogoutOutlined } from '@ant-design/icons';
import { useDataStore } from '@/store';
import type { Dayjs } from 'dayjs';
import dayjs from 'dayjs';

const { Header } = Layout;
const { RangePicker } = DatePicker;

const TopBar: React.FC = () => {
  const { dateRange, setDateRange } = useDataStore();

  // Convert date strings to Dayjs objects for DatePicker
  const dateRangeValue: [Dayjs, Dayjs] = [
    dayjs(dateRange.start),
    dayjs(dateRange.end),
  ];

  // Handle date range change
  const handleDateRangeChange = (dates: [Dayjs | null, Dayjs | null] | null) => {
    if (dates && dates[0] && dates[1]) {
      setDateRange(dates[0].format('YYYY-MM-DD'), dates[1].format('YYYY-MM-DD'));
    }
  };

  // Format date range display
  const formatDateRange = (start: string, end: string): string => {
    const startDate = dayjs(start);
    const endDate = dayjs(end);
    return `${startDate.format('MMM D')} - ${endDate.format('MMM D, YYYY')}`;
  };

  // Profile menu items
  const profileMenuItems: MenuProps['items'] = [
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
      onClick: () => {
        console.log('Settings clicked');
        // TODO: Implement settings
      },
    },
    {
      key: 'help',
      icon: <QuestionCircleOutlined />,
      label: 'Help',
      onClick: () => {
        console.log('Help clicked');
        // TODO: Implement help
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
        console.log('Logout clicked');
        // TODO: Implement logout
      },
    },
  ];

  return (
    <Header
      style={{
        height: '64px',
        padding: '0 24px',
        backgroundColor: '#F8FAFC',
        borderBottom: '1px solid #E2E8F0',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 1000,
      }}
    >
      {/* Logo/Brand Section */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          flex: '0 0 auto',
        }}
      >
        <div
          style={{
            width: '32px',
            height: '32px',
            backgroundColor: '#6366F1',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#FFFFFF',
            fontWeight: 'bold',
            fontSize: '16px',
          }}
        >
          A
        </div>
        <span
          style={{
            fontSize: '18px',
            fontWeight: '600',
            color: '#030712',
            letterSpacing: '-0.02em',
          }}
        >
          Analytics Dashboard
        </span>
      </div>

      {/* Date Range Picker Section */}
      <div
        style={{
          flex: '1 1 auto',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '12px',
        }}
      >
        <RangePicker
          value={dateRangeValue}
          onChange={handleDateRangeChange}
          format="MMM D, YYYY"
          placeholder={['Start Date', 'End Date']}
          style={{
            width: '300px',
            maxWidth: '100%',
          }}
          allowClear={false}
        />
        <span
          style={{
            fontSize: '14px',
            color: '#64748B',
            display: 'none', // Hide on mobile, show on desktop via media query
          }}
          className="date-range-text"
        >
          {formatDateRange(dateRange.start, dateRange.end)}
        </span>
      </div>

      {/* Profile Menu Section */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          flex: '0 0 auto',
        }}
      >
        <Dropdown
          menu={{ items: profileMenuItems }}
          placement="bottomRight"
          trigger={['click']}
        >
          <Space
            style={{
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: '6px',
              transition: 'background-color 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#F1F5F9';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
            }}
          >
            <span
              style={{
                fontSize: '14px',
                color: '#64748B',
                display: 'none', // Hide on mobile
              }}
              className="welcome-text"
            >
              Welcome
            </span>
            <Avatar
              icon={<UserOutlined />}
              style={{
                backgroundColor: '#6366F1',
                cursor: 'pointer',
              }}
            />
          </Space>
        </Dropdown>
      </div>

      <style>{`
        @media (min-width: 768px) {
          .date-range-text,
          .welcome-text {
            display: inline !important;
          }
        }
      `}</style>
    </Header>
  );
};

export default TopBar;

