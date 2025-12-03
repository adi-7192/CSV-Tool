import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from 'antd';
import { ArrowRightOutlined, BarChartOutlined, RobotOutlined, CloudUploadOutlined } from '@ant-design/icons';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '80px 24px' }}>
      {/* Hero Section */}
      <div style={{ textAlign: 'center', marginBottom: '80px' }}>
        <h1
          style={{
            fontSize: '56px',
            fontWeight: '800',
            color: '#030712',
            marginBottom: '24px',
            lineHeight: '1.1',
          }}
        >
          AI-Powered Sales Analytics
          <br />
          <span style={{ color: '#6366F1' }}>Made Simple</span>
        </h1>
        <p
          style={{
            fontSize: '20px',
            color: '#64748B',
            marginBottom: '40px',
            maxWidth: '600px',
            margin: '0 auto 40px',
            lineHeight: '1.6',
          }}
        >
          Transform your sales data into actionable insights with our intelligent analytics platform.
          Upload your data, ask questions in plain English, and get instant answers.
        </p>
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Button
            type="primary"
            size="large"
            onClick={() => navigate('/signup')}
            style={{ height: '48px', paddingLeft: '32px', paddingRight: '32px', fontSize: '16px' }}
          >
            Get Started Free
            <ArrowRightOutlined style={{ marginLeft: '8px' }} />
          </Button>
          <Button
            size="large"
            onClick={() => navigate('/login')}
            style={{ height: '48px', paddingLeft: '32px', paddingRight: '32px', fontSize: '16px' }}
          >
            Sign In
          </Button>
        </div>
      </div>

      {/* Features Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '32px',
          marginTop: '80px',
        }}
      >
        <div
          style={{
            padding: '32px',
            backgroundColor: '#FFFFFF',
            borderRadius: '12px',
            border: '1px solid #E2E8F0',
            textAlign: 'center',
          }}
        >
          <BarChartOutlined
            style={{ fontSize: '48px', color: '#6366F1', marginBottom: '16px' }}
          />
          <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#030712' }}>
            Real-Time Analytics
          </h3>
          <p style={{ color: '#64748B', fontSize: '14px', lineHeight: '1.6' }}>
            Get instant insights into your sales performance with live dashboards and metrics.
          </p>
        </div>

        <div
          style={{
            padding: '32px',
            backgroundColor: '#FFFFFF',
            borderRadius: '12px',
            border: '1px solid #E2E8F0',
            textAlign: 'center',
          }}
        >
          <RobotOutlined
            style={{ fontSize: '48px', color: '#6366F1', marginBottom: '16px' }}
          />
          <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#030712' }}>
            AI-Powered Insights
          </h3>
          <p style={{ color: '#64748B', fontSize: '14px', lineHeight: '1.6' }}>
            Ask questions in plain English and get intelligent answers powered by advanced AI.
          </p>
        </div>

        <div
          style={{
            padding: '32px',
            backgroundColor: '#FFFFFF',
            borderRadius: '12px',
            border: '1px solid #E2E8F0',
            textAlign: 'center',
          }}
        >
          <CloudUploadOutlined
            style={{ fontSize: '48px', color: '#6366F1', marginBottom: '16px' }}
          />
          <h3 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '12px', color: '#030712' }}>
            Easy Data Import
          </h3>
          <p style={{ color: '#64748B', fontSize: '14px', lineHeight: '1.6' }}>
            Upload CSV or Excel files and start analyzing your data in minutes.
          </p>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;

