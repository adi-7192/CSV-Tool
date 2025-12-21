/**
 * Profile Page
 * 
 * Displays user profile information including email, role, plan, tenant ID, etc.
 */

import React from 'react';
import {
  Card,
  Typography,
  Space,
  Descriptions,
  Tag,
  Button,
  message,
} from 'antd';
import {
  UserOutlined,
  MailOutlined,
  CrownOutlined,
  CheckCircleOutlined,
  CopyOutlined,
  IdcardOutlined,
  CalendarOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '@/store/authStore';
import { COLORS, SPACING, BORDER_RADIUS, SHADOWS } from '@/styles/designTokens';
import { useNavigate } from 'react-router-dom';

const { Title, Text } = Typography;

const Profile: React.FC = () => {
  const { user } = useAuthStore();
  const navigate = useNavigate();

  const handleCopyTenantId = () => {
    if (user?.tenant_id) {
      navigator.clipboard.writeText(user.tenant_id);
      message.success('Tenant ID copied to clipboard');
    }
  };

  if (!user) {
    return (
      <div
        style={{
          maxWidth: '800px',
          margin: '0 auto',
          padding: SPACING.lg,
        }}
      >
        <Card>
          <Text>Please log in to view your profile.</Text>
        </Card>
      </div>
    );
  }

  return (
    <div
      style={{
        maxWidth: '800px',
        margin: '0 auto',
        padding: SPACING.lg,
      }}
    >
      <Title level={2} style={{ marginBottom: SPACING.lg }}>
        <UserOutlined style={{ marginRight: SPACING.sm, color: COLORS.primary }} />
        Profile
      </Title>

      <Card
        style={{
          marginBottom: SPACING.lg,
          borderRadius: BORDER_RADIUS.md,
          boxShadow: SHADOWS.card,
        }}
      >
        <Descriptions
          title={
            <Space>
              <UserOutlined />
              <span>Account Information</span>
            </Space>
          }
          bordered
          column={1}
          labelStyle={{ fontWeight: 600, width: '200px' }}
        >
          <Descriptions.Item
            label={
              <Space>
                <MailOutlined />
                <span>Email</span>
              </Space>
            }
          >
            {user.email}
          </Descriptions.Item>

          <Descriptions.Item
            label={
              <Space>
                <IdcardOutlined />
                <span>Role</span>
              </Space>
            }
          >
            <Tag color={user.role === 'admin' ? 'red' : 'blue'}>
              {user.role === 'admin' ? 'Administrator' : 'User'}
            </Tag>
          </Descriptions.Item>

          <Descriptions.Item
            label={
              <Space>
                <CrownOutlined />
                <span>Plan</span>
              </Space>
            }
          >
            <Tag color={user.plan === 'enterprise' ? 'gold' : user.plan === 'pro' ? 'purple' : 'default'}>
              {user.plan.charAt(0).toUpperCase() + user.plan.slice(1)}
            </Tag>
          </Descriptions.Item>

          <Descriptions.Item
            label={
              <Space>
                <CheckCircleOutlined />
                <span>Onboarded</span>
              </Space>
            }
          >
            <Tag color={user.onboarded ? 'success' : 'warning'}>
              {user.onboarded ? 'Yes' : 'No'}
            </Tag>
          </Descriptions.Item>

          <Descriptions.Item
            label={
              <Space>
                <IdcardOutlined />
                <span>Tenant ID</span>
              </Space>
            }
          >
            <Space>
              <Text code style={{ fontSize: '12px' }}>
                {user.tenant_id || 'N/A'}
              </Text>
              {user.tenant_id && (
                <Button
                  type="text"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={handleCopyTenantId}
                  style={{ padding: 0 }}
                >
                  Copy
                </Button>
              )}
            </Space>
          </Descriptions.Item>

          <Descriptions.Item
            label={
              <Space>
                <CalendarOutlined />
                <span>Account Created</span>
              </Space>
            }
          >
            {user.created_at
              ? new Date(user.created_at).toLocaleString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })
              : 'N/A'}
          </Descriptions.Item>

          {user.last_login_at && (
            <Descriptions.Item
              label={
                <Space>
                  <CalendarOutlined />
                  <span>Last Login</span>
                </Space>
              }
            >
              {new Date(user.last_login_at).toLocaleString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </Descriptions.Item>
          )}

          <Descriptions.Item
            label={
              <Space>
                <CheckCircleOutlined />
                <span>Account Status</span>
              </Space>
            }
          >
            <Tag color={user.is_active ? 'success' : 'error'}>
              {user.is_active ? 'Active' : 'Inactive'}
            </Tag>
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card
        style={{
          borderRadius: BORDER_RADIUS.md,
          boxShadow: SHADOWS.card,
        }}
      >
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Title level={4} style={{ margin: 0 }}>
            Quick Actions
          </Title>
          <Space>
            <Button
              type="primary"
              onClick={() => navigate('/app/settings')}
            >
              Go to Settings
            </Button>
            {!user.onboarded && (
              <Button onClick={() => navigate('/app/onboarding')}>
                Complete Onboarding
              </Button>
            )}
          </Space>
        </Space>
      </Card>
    </div>
  );
};

export default Profile;

