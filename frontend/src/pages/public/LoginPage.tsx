import React, { useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Form, Input, Button, Card, Alert } from 'antd';
import { MailOutlined, LockOutlined } from '@ant-design/icons';
import { useAuthStore } from '@/store/authStore';
import { dataService } from '@/services/api';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { login, loading, error, clearError, user } = useAuthStore();
  const [form] = Form.useForm();

  // Redirect if already logged in
  useEffect(() => {
    const checkDataAndRedirect = async () => {
      if (user && !loading) {
        try {
          // Check if user has data in the database
          const dataSummary = await dataService.getSummary();

          if (dataSummary.has_data && dataSummary.row_count > 0) {
            // User has data → go directly to dashboard
            navigate('/app/dashboard');
          } else {
            // User has no data → show onboarding
            navigate('/app/onboarding');
          }
        } catch (err) {
          // If check fails, default to onboarding (safer)
          console.error('Failed to check user data:', err);
          navigate('/app/onboarding');
        }
      }
    };

    checkDataAndRedirect();
  }, [user, loading, navigate]);

  const handleSubmit = async (values: { email: string; password: string }) => {
    try {
      clearError();
      await login(values.email, values.password);
      // Navigation will happen via useEffect when user state updates
    } catch (err) {
      // Error is handled by authStore
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 'calc(100vh - 200px)',
        padding: '24px',
      }}
    >
      <Card
        style={{
          width: '100%',
          maxWidth: '400px',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px', color: '#030712' }}>
            Welcome Back
          </h1>
          <p style={{ color: '#64748B', fontSize: '14px' }}>
            Sign in to your account to continue
          </p>
        </div>

        {error && (
          <Alert
            message={error}
            type="error"
            showIcon
            closable
            onClose={clearError}
            style={{ marginBottom: '24px' }}
          />
        )}

        <Form
          form={form}
          name="login"
          onFinish={handleSubmit}
          layout="vertical"
          size="large"
        >
          <Form.Item
            name="email"
            rules={[
              { required: true, message: 'Please input your email!' },
              { type: 'email', message: 'Please enter a valid email!' },
            ]}
          >
            <Input
              prefix={<MailOutlined />}
              placeholder="Email"
              autoComplete="email"
            />
          </Form.Item>

          <Form.Item
            name="password"
            rules={[{ required: true, message: 'Please input your password!' }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Password"
              autoComplete="current-password"
            />
          </Form.Item>

          <div style={{ textAlign: 'right', marginTop: '-16px', marginBottom: '16px' }}>
            <Link
              to="/forgot-password"
              style={{
                color: '#6366F1',
                fontSize: '14px',
                fontWeight: '500',
              }}
            >
              Forgot password?
            </Link>
          </div>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              block
              loading={loading}
              style={{ height: '44px', fontSize: '16px', fontWeight: '600' }}
            >
              Sign In
            </Button>
          </Form.Item>
        </Form>

        <div style={{ textAlign: 'center', marginTop: '24px' }}>
          <span style={{ color: '#64748B', fontSize: '14px' }}>
            Don't have an account?{' '}
            <Link to="/signup" style={{ color: '#6366F1', fontWeight: '600' }}>
              Sign up
            </Link>
          </span>
        </div>
      </Card>
    </div>
  );
};

export default LoginPage;

