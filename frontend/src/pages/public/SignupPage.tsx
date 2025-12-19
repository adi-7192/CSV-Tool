import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Form, Input, Button, Card, Alert } from 'antd';
import { MailOutlined, LockOutlined, UserOutlined } from '@ant-design/icons';
import { useAuthStore } from '@/store/authStore';
import { dataService } from '@/services/api';

const SignupPage: React.FC = () => {
  const navigate = useNavigate();
  const { register, loading, error, clearError, user } = useAuthStore();
  const [form] = Form.useForm();
  const [emailError, setEmailError] = useState<string>('');

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
          // If check fails, default to onboarding (new users won't have data)
          console.error('Failed to check user data:', err);
          navigate('/app/onboarding');
        }
      }
    };

    checkDataAndRedirect();
  }, [user, loading, navigate]);

  const handleSubmit = async (values: { email: string; password: string; name?: string }) => {
    try {
      clearError();
      setEmailError(''); // Clear email-specific error
      await register(values.email, values.password, values.name);
      // Navigation will happen via useEffect when user state updates
    } catch (err: any) {
      // Check if it's an email already exists error
      const errorMessage = err?.message || err?.response?.data?.detail || '';
      if (errorMessage.includes('EMAIL_ALREADY_EXISTS') || errorMessage.includes('already exists')) {
        setEmailError('An account with this email already exists. Please log in instead.');
        form.setFields([
          {
            name: 'email',
            errors: ['Email already registered'],
          },
        ]);
        // Clear the generic error since we're showing a specific one
        clearError();
      }
      // Other errors are handled by authStore and shown via the error Alert
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
            Create Account
          </h1>
          <p style={{ color: '#64748B', fontSize: '14px' }}>
            Get started with your free account today
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
          name="signup"
          onFinish={handleSubmit}
          layout="vertical"
          size="large"
        >
          <Form.Item
            name="name"
            rules={[{ required: false }]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="Full Name (optional)"
              autoComplete="name"
            />
          </Form.Item>

          <Form.Item
            name="email"
            validateStatus={emailError ? 'error' : ''}
            help={emailError || undefined}
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

          {/* Add link to login if email error */}
          {emailError && (
            <div style={{ marginBottom: '16px', textAlign: 'center' }}>
              <Link to="/login" style={{ color: '#6366F1', fontWeight: '600' }}>
                Go to Login →
              </Link>
            </div>
          )}

          <Form.Item
            name="password"
            rules={[
              { required: true, message: 'Please input your password!' },
              { min: 6, message: 'Password must be at least 6 characters!' },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Password"
              autoComplete="new-password"
            />
          </Form.Item>

          <Form.Item
            name="confirmPassword"
            dependencies={['password']}
            rules={[
              { required: true, message: 'Please confirm your password!' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('password') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('The two passwords do not match!'));
                },
              }),
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Confirm Password"
              autoComplete="new-password"
            />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              block
              loading={loading}
              style={{ height: '44px', fontSize: '16px', fontWeight: '600' }}
            >
              Create Account
            </Button>
          </Form.Item>
        </Form>

        <div style={{ textAlign: 'center', marginTop: '24px' }}>
          <span style={{ color: '#64748B', fontSize: '14px' }}>
            Already have an account?{' '}
            <Link to="/login" style={{ color: '#6366F1', fontWeight: '600' }}>
              Sign in
            </Link>
          </span>
        </div>
      </Card>
    </div>
  );
};

export default SignupPage;

