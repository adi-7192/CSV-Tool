import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { Form, Input, Button, Card, Alert, Result, Progress } from 'antd';
import { LockOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { apiClient } from '@/services/api';

const MIN_PASSWORD_LENGTH = 8;

/**
 * Password strength indicator component
 */
const PasswordStrength: React.FC<{ password: string }> = ({ password }) => {
  const getStrength = (pwd: string): { percent: number; status: 'exception' | 'active' | 'success'; text: string } => {
    if (!pwd) return { percent: 0, status: 'exception', text: '' };
    
    let score = 0;
    
    // Length checks
    if (pwd.length >= 8) score += 25;
    if (pwd.length >= 12) score += 15;
    
    // Character variety
    if (/[a-z]/.test(pwd)) score += 15;
    if (/[A-Z]/.test(pwd)) score += 15;
    if (/[0-9]/.test(pwd)) score += 15;
    if (/[^a-zA-Z0-9]/.test(pwd)) score += 15;
    
    if (score < 40) return { percent: score, status: 'exception', text: 'Weak' };
    if (score < 70) return { percent: score, status: 'active', text: 'Medium' };
    return { percent: Math.min(score, 100), status: 'success', text: 'Strong' };
  };

  const strength = getStrength(password);

  if (!password) return null;

  return (
    <div style={{ marginTop: '-16px', marginBottom: '16px' }}>
      <Progress
        percent={strength.percent}
        status={strength.status}
        showInfo={false}
        size="small"
      />
      <span style={{ 
        fontSize: '12px', 
        color: strength.status === 'exception' ? '#ff4d4f' : strength.status === 'active' ? '#faad14' : '#52c41a' 
      }}>
        Password strength: {strength.text}
      </span>
    </div>
  );
};

/**
 * Password validation requirements display
 */
const PasswordRequirements: React.FC<{ password: string }> = ({ password }) => {
  const requirements = [
    { label: 'At least 8 characters', met: password.length >= MIN_PASSWORD_LENGTH },
    { label: 'Contains a letter', met: /[a-zA-Z]/.test(password) },
    { label: 'Contains a number', met: /[0-9]/.test(password) },
  ];

  return (
    <div style={{ marginBottom: '16px', padding: '12px', backgroundColor: '#f8fafc', borderRadius: '8px' }}>
      <div style={{ fontSize: '13px', fontWeight: '500', marginBottom: '8px', color: '#475569' }}>
        Password requirements:
      </div>
      {requirements.map((req, index) => (
        <div
          key={index}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '13px',
            color: req.met ? '#10b981' : '#94a3b8',
            marginBottom: '4px',
          }}
        >
          {req.met ? (
            <CheckCircleOutlined style={{ color: '#10b981' }} />
          ) : (
            <CloseCircleOutlined style={{ color: '#cbd5e1' }} />
          )}
          {req.label}
        </div>
      ))}
    </div>
  );
};

const ResetPasswordPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [password, setPassword] = useState('');
  
  // Get token from URL query params
  const token = searchParams.get('token');

  // Check if token exists
  useEffect(() => {
    if (!token) {
      setError('Invalid password reset link. Please request a new one.');
    }
  }, [token]);

  /**
   * Validate password meets requirements
   */
  const validatePassword = (pwd: string): string | null => {
    if (pwd.length < MIN_PASSWORD_LENGTH) {
      return `Password must be at least ${MIN_PASSWORD_LENGTH} characters`;
    }
    if (!/[a-zA-Z]/.test(pwd)) {
      return 'Password must contain at least one letter';
    }
    if (!/[0-9]/.test(pwd)) {
      return 'Password must contain at least one number';
    }
    return null;
  };

  const handleSubmit = async (values: { password: string; confirmPassword: string }) => {
    // Client-side validation
    const passwordError = validatePassword(values.password);
    if (passwordError) {
      setError(passwordError);
      return;
    }

    if (values.password !== values.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!token) {
      setError('Invalid reset token. Please request a new password reset link.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await apiClient.post('/api/auth/reset-password', {
        token: token,
        new_password: values.password,
      });
      
      setSuccess(true);
      
      // Redirect to login after 3 seconds
      setTimeout(() => {
        navigate('/login');
      }, 3000);
    } catch (err: any) {
      console.error('Password reset error:', err);
      
      if (err.response?.status === 400) {
        const detail = err.response?.data?.detail;
        if (detail === 'INVALID_OR_EXPIRED_TOKEN') {
          setError('This password reset link is invalid or has expired. Please request a new one.');
        } else if (detail === 'TOKEN_ALREADY_USED') {
          setError('This password reset link has already been used. Please request a new one if needed.');
        } else if (typeof detail === 'string' && detail.includes('password')) {
          setError(detail);
        } else {
          setError('Invalid password reset link. Please request a new one.');
        }
      } else if (err.code === 'ERR_NETWORK' || !err.response) {
        setError('Unable to connect to server. Please check your internet connection.');
      } else {
        setError('Something went wrong. Please try again or request a new reset link.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Success state
  if (success) {
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
            title="Password Reset Successful!"
            subTitle={
              <div style={{ color: '#64748B', lineHeight: 1.6 }}>
                <p>Your password has been changed successfully.</p>
                <p style={{ marginTop: '8px' }}>
                  Redirecting you to login in a few seconds...
                </p>
              </div>
            }
            extra={[
              <Link to="/login" key="login">
                <Button type="primary" size="large">
                  Go to Login
                </Button>
              </Link>,
            ]}
          />
        </Card>
      </div>
    );
  }

  // No token error state
  if (!token) {
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
            status="error"
            title="Invalid Reset Link"
            subTitle="This password reset link is invalid or has expired."
            extra={[
              <Link to="/forgot-password" key="forgot">
                <Button type="primary" size="large">
                  Request New Link
                </Button>
              </Link>,
              <Link to="/login" key="login">
                <Button size="large" style={{ marginTop: '8px' }}>
                  Back to Login
                </Button>
              </Link>,
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
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ fontSize: '28px', fontWeight: '700', marginBottom: '8px', color: '#030712' }}>
            Reset Your Password
          </h1>
          <p style={{ color: '#64748B', fontSize: '14px' }}>
            Enter your new password below
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

        <PasswordRequirements password={password} />

        <Form
          form={form}
          name="reset-password"
          onFinish={handleSubmit}
          layout="vertical"
          size="large"
        >
          <Form.Item
            name="password"
            rules={[
              { required: true, message: 'Please enter your new password!' },
              { min: MIN_PASSWORD_LENGTH, message: `Password must be at least ${MIN_PASSWORD_LENGTH} characters` },
              {
                validator: (_, value) => {
                  if (value && !/[a-zA-Z]/.test(value)) {
                    return Promise.reject('Password must contain at least one letter');
                  }
                  return Promise.resolve();
                },
              },
              {
                validator: (_, value) => {
                  if (value && !/[0-9]/.test(value)) {
                    return Promise.reject('Password must contain at least one number');
                  }
                  return Promise.resolve();
                },
              },
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="New password"
              autoComplete="new-password"
              onChange={(e) => setPassword(e.target.value)}
            />
          </Form.Item>

          <PasswordStrength password={password} />

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
                  return Promise.reject(new Error('Passwords do not match'));
                },
              }),
            ]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Confirm new password"
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
              Reset Password
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

export default ResetPasswordPage;

