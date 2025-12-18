import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Button, Typography, Space, message } from 'antd';
import { UploadOutlined, DatabaseOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { useAuthStore } from '@/store/authStore';

const { Title, Text, Paragraph } = Typography;

const Onboarding: React.FC = () => {
  const navigate = useNavigate();
  const { markOnboarded } = useAuthStore();

  const handleUploadCSV = async () => {
    try {
      await markOnboarded();
      navigate('/app/data-management');
    } catch (error) {
      console.error('Failed to mark as onboarded:', error);
      // Still navigate even if API call fails
      navigate('/app/data-management');
    }
  };

  const handleExploreSample = async () => {
    // Sample data feature is disabled for now
    // Each user must upload their own data for tenant isolation
    message.info('Sample data feature coming soon! Please upload your own CSV file to get started.');
    return;
  };

  const handleSkip = async () => {
    try {
      await markOnboarded();
      navigate('/app/dashboard');
    } catch (error) {
      console.error('Failed to mark as onboarded:', error);
      // Still navigate even if API call fails
      navigate('/app/dashboard');
    }
  };

  return (
    <div
      style={{
        maxWidth: '1000px',
        margin: '0 auto',
        padding: '40px 24px',
      }}
    >
      {/* Welcome Section */}
      <div style={{ textAlign: 'center', marginBottom: '48px' }}>
        <Title level={1} style={{ marginBottom: '16px' }}>
          Welcome to Datadost Analytics! 🎉
        </Title>
        <Paragraph style={{ fontSize: '18px', color: '#64748B', marginBottom: '8px' }}>
          You're on the <strong>Free plan</strong>.
        </Paragraph>
        <Text type="secondary" style={{ fontSize: '14px' }}>
          Get started by uploading your data or exploring with sample data.
        </Text>
      </div>

      {/* Next Steps */}
      <Card
        style={{
          marginBottom: '32px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        }}
      >
        <Title level={3} style={{ marginBottom: '24px' }}>
          Next Steps
        </Title>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Button
            type="primary"
            size="large"
            icon={<UploadOutlined />}
            onClick={handleUploadCSV}
            block
            style={{ height: '56px', fontSize: '16px' }}
          >
            Upload Your First CSV
          </Button>
          <Button
            size="large"
            icon={<DatabaseOutlined />}
            onClick={handleExploreSample}
            block
            disabled
            style={{ height: '56px', fontSize: '16px' }}
          >
            Explore with Sample Data (Coming Soon)
          </Button>
          <div style={{ textAlign: 'center' }}>
            <Button type="link" onClick={handleSkip} style={{ fontSize: '14px' }}>
              Skip for now
            </Button>
          </div>
        </Space>
      </Card>

      {/* Plans Section */}
      <Card
        title={
          <Title level={3} style={{ margin: 0 }}>
            Plans & Pricing
          </Title>
        }
        style={{
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '24px',
          }}
        >
          {/* Free Plan */}
          <Card
            style={{
              border: '2px solid #6366F1',
              borderRadius: '8px',
              position: 'relative',
            }}
            bodyStyle={{ padding: '24px' }}
          >
            <div style={{ marginBottom: '16px' }}>
              <div
                style={{
                  position: 'absolute',
                  top: '-12px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  backgroundColor: '#6366F1',
                  color: '#FFFFFF',
                  padding: '4px 16px',
                  borderRadius: '12px',
                  fontSize: '12px',
                  fontWeight: '600',
                }}
              >
                Current Plan
              </div>
            </div>
            <Title level={4} style={{ marginTop: '16px', marginBottom: '8px' }}>
              Free
            </Title>
            <div style={{ marginBottom: '24px' }}>
              <Text style={{ fontSize: '32px', fontWeight: '700' }}>$0</Text>
              <Text type="secondary">/forever</Text>
            </div>
            <Space direction="vertical" size="small" style={{ width: '100%', marginBottom: '24px' }}>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Up to 10,000 records</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Basic analytics</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>AI chat support</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Email support</Text>
              </div>
            </Space>
            <Button type="primary" block disabled>
              Current Plan
            </Button>
          </Card>

          {/* Pro Plan */}
          <Card
            style={{
              border: '1px solid #E2E8F0',
              borderRadius: '8px',
            }}
            bodyStyle={{ padding: '24px' }}
          >
            <Title level={4} style={{ marginBottom: '8px' }}>
              Pro
            </Title>
            <div style={{ marginBottom: '24px' }}>
              <Text style={{ fontSize: '32px', fontWeight: '700' }}>$29</Text>
              <Text type="secondary">/month</Text>
            </div>
            <Space direction="vertical" size="small" style={{ width: '100%', marginBottom: '24px' }}>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Unlimited records</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Advanced analytics</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Priority AI support</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Custom reports</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>API access</Text>
              </div>
            </Space>
            <Button block disabled>
              Coming Soon
            </Button>
          </Card>

          {/* Enterprise Plan */}
          <Card
            style={{
              border: '1px solid #E2E8F0',
              borderRadius: '8px',
            }}
            bodyStyle={{ padding: '24px' }}
          >
            <Title level={4} style={{ marginBottom: '8px' }}>
              Enterprise
            </Title>
            <div style={{ marginBottom: '24px' }}>
              <Text style={{ fontSize: '32px', fontWeight: '700' }}>Custom</Text>
            </div>
            <Space direction="vertical" size="small" style={{ width: '100%', marginBottom: '24px' }}>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Everything in Pro</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Dedicated support</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>Custom integrations</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>SLA guarantee</Text>
              </div>
              <div>
                <CheckCircleOutlined style={{ color: '#10B981', marginRight: '8px' }} />
                <Text>On-premise deployment</Text>
              </div>
            </Space>
            <Button block disabled>
              Coming Soon
            </Button>
          </Card>
        </div>
      </Card>
    </div>
  );
};

export default Onboarding;

