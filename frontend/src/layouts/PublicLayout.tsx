import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from 'antd';
import { useAuthStore } from '@/store/authStore';

interface PublicLayoutProps {
  children: React.ReactNode;
}

const PublicLayout: React.FC<PublicLayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const { user } = useAuthStore();

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation */}
      <header
        style={{
          backgroundColor: '#FFFFFF',
          borderBottom: '1px solid #E2E8F0',
          padding: '16px 24px',
          position: 'sticky',
          top: 0,
          zIndex: 100,
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div
          style={{
            maxWidth: '1200px',
            margin: '0 auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          {/* Logo */}
          <Link
            to="/"
            style={{
              fontSize: '20px',
              fontWeight: '700',
              color: '#6366F1',
              textDecoration: 'none',
              letterSpacing: '-0.5px',
            }}
          >
            Analytics Dashboard
          </Link>

          {/* Navigation Links */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
            <Link
              to="/"
              style={{
                color: '#64748B',
                textDecoration: 'none',
                fontSize: '14px',
                fontWeight: '500',
              }}
            >
              Product
            </Link>
            <Link
              to="/pricing"
              style={{
                color: '#64748B',
                textDecoration: 'none',
                fontSize: '14px',
                fontWeight: '500',
              }}
            >
              Pricing
            </Link>

            {/* Auth Buttons */}
            {user ? (
              <Button
                type="primary"
                onClick={() => {
                  if (user.role === 'admin') {
                    navigate('/admin');
                  } else {
                    navigate('/app/dashboard');
                  }
                }}
              >
                Go to Dashboard
              </Button>
            ) : (
              <>
                <Button
                  type="text"
                  onClick={() => navigate('/login')}
                  style={{ color: '#64748B' }}
                >
                  Login
                </Button>
                <Button
                  type="primary"
                  onClick={() => navigate('/signup')}
                >
                  Get Started
                </Button>
              </>
            )}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>{children}</main>

      {/* Footer */}
      <footer
        style={{
          backgroundColor: '#F8FAFC',
          borderTop: '1px solid #E2E8F0',
          padding: '48px 24px',
          marginTop: 'auto',
        }}
      >
        <div
          style={{
            maxWidth: '1200px',
            margin: '0 auto',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '32px',
          }}
        >
          <div>
            <h4 style={{ marginBottom: '16px', color: '#030712', fontSize: '14px', fontWeight: '600' }}>
              Product
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ marginBottom: '8px' }}>
                <Link to="/" style={{ color: '#64748B', textDecoration: 'none', fontSize: '14px' }}>
                  Features
                </Link>
              </li>
              <li style={{ marginBottom: '8px' }}>
                <Link to="/pricing" style={{ color: '#64748B', textDecoration: 'none', fontSize: '14px' }}>
                  Pricing
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h4 style={{ marginBottom: '16px', color: '#030712', fontSize: '14px', fontWeight: '600' }}>
              Legal
            </h4>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ marginBottom: '8px' }}>
                <a href="#" style={{ color: '#64748B', textDecoration: 'none', fontSize: '14px' }}>
                  Privacy Policy
                </a>
              </li>
              <li style={{ marginBottom: '8px' }}>
                <a href="#" style={{ color: '#64748B', textDecoration: 'none', fontSize: '14px' }}>
                  Terms of Service
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div
          style={{
            marginTop: '32px',
            paddingTop: '24px',
            borderTop: '1px solid #E2E8F0',
            textAlign: 'center',
            color: '#94A3B8',
            fontSize: '12px',
          }}
        >
          © 2025 Analytics Dashboard. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default PublicLayout;

