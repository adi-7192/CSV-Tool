import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Form, Input, Button, Card, Alert, Result } from 'antd';
import { MailOutlined, ArrowLeftOutlined } from '@ant-design/icons';
import { apiClient } from '@/services/api';

const ForgotPasswordPage: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (values: { email: string }) => {
    setLoading(true);
    setError(null);

    try {
      await apiClient.post('/api/auth/forgot-password', {
        email: values.email,
      });
      
      // Always show success (backend returns 200 regardless of email existence)
      setSubmitted(true);
    } catch (err: any) {
      // Even on error, show success for security (prevent email enumeration)
      // Only show error for network issues
      if (err.code === 'ERR_NETWORK' || !err.response) {
        setError('Unable to connect to server. Please check your internet connection.');
      } else {
        // For any other error, still show success state
        setSubmitted(true);
      }
    } finally {
      setLoading(false);
    }
  };

  // Success state after submission
  if (submitted) {
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
            maxWidth: '450px',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
          }}
        >
          <Result
            status="success"
            title="Check Your Email"
            subTitle={
              <div style={{ color: '#64748B', lineHeight: 1.6 }}>
                <p>
                  If an account with that email exists, we've sent a password reset link.
                </p>
                <p style={{ marginTop: '8px' }}>
                  The link will expire in <strong>15 minutes</strong>.
                </p>
                <p style={{ marginTop: '8px', fontSize: '13px' }}>
                  Don't see the email? Check your spam folder.
                </p>
              </div>
            }
            extra={[
              <Link to="/login" key="login">
                <Button type="primary" size="large">
                  Back to Login
                </Button>
              </Link>,
              <Button
                key="resend"
                size="large"
                onClick={() => {
                  setSubmitted(false);
                  form.resetFields();
                }}
                style={{ marginTop: '8px' }}
              >
                Try Different Email
              </Button>,
            ]}
          />
        </Card>
      </div>
    );
  }

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
        <div style={{ marginBottom: '8px' }}>
          <Link
            to="/login"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              color: '#64748B',
              fontSize: '14px',
              textDecoration: 'none',
            }}
          >
            <ArrowLeftOutlined /> Back to login
          </Link>
        </div>

        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px', color: '#030712' }}>
            Forgot Password?
          </h1>
          <p style={{ color: '#64748B', fontSize: '14px' }}>
            No worries! Enter your email and we'll send you a reset link.
          </p>
        </div>

        {error && (
          <Alert
            message={error}
            type="error"
            showIcon
            closable
            onClose={() => setError(null)}
            style={{ marginBottom: '24px' }}
          />
        )}

        <Form
          form={form}
          name="forgot-password"
          onFinish={handleSubmit}
          layout="vertical"
          size="large"
        >
          <Form.Item
            name="email"
            rules={[
              { required: true, message: 'Please enter your email!' },
              { type: 'email', message: 'Please enter a valid email!' },
            ]}
          >
            <Input
              prefix={<MailOutlined />}
              placeholder="Enter your email"
              autoComplete="email"
              autoFocus
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
              Send Reset Link
            </Button>
          </Form.Item>
        </Form>

        <div style={{ textAlign: 'center', marginTop: '24px' }}>
          <span style={{ color: '#64748B', fontSize: '14px' }}>
            Remember your password?{' '}
            <Link to="/login" style={{ color: '#6366F1', fontWeight: '600' }}>
              Sign in
            </Link>
          </span>
        </div>
      </Card>
    </div>
  );
};

export default ForgotPasswordPage;

