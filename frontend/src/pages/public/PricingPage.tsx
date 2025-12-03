import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Card } from 'antd';
import { CheckOutlined } from '@ant-design/icons';

const PricingPage: React.FC = () => {
  const navigate = useNavigate();

  const plans = [
    {
      name: 'Free',
      price: '$0',
      period: 'forever',
      description: 'Perfect for getting started',
      features: [
        'Up to 10,000 records',
        'Basic analytics',
        'AI chat support',
        'Email support',
      ],
      cta: 'Get Started',
      popular: false,
    },
    {
      name: 'Pro',
      price: '$29',
      period: 'month',
      description: 'For growing businesses',
      features: [
        'Unlimited records',
        'Advanced analytics',
        'Priority AI support',
        'Custom reports',
        'API access',
      ],
      cta: 'Start Free Trial',
      popular: true,
    },
    {
      name: 'Enterprise',
      price: 'Custom',
      period: '',
      description: 'For large organizations',
      features: [
        'Everything in Pro',
        'Dedicated support',
        'Custom integrations',
        'SLA guarantee',
        'On-premise deployment',
      ],
      cta: 'Contact Sales',
      popular: false,
    },
  ];

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '80px 24px' }}>
      <div style={{ textAlign: 'center', marginBottom: '64px' }}>
        <h1
          style={{
            fontSize: '48px',
            fontWeight: '800',
            color: '#030712',
            marginBottom: '16px',
          }}
        >
          Simple, Transparent Pricing
        </h1>
        <p style={{ fontSize: '18px', color: '#64748B', maxWidth: '600px', margin: '0 auto' }}>
          Choose the plan that's right for your business. All plans include our core analytics features.
        </p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '32px',
        }}
      >
        {plans.map((plan) => (
          <Card
            key={plan.name}
            style={{
              border: plan.popular ? '2px solid #6366F1' : '1px solid #E2E8F0',
              borderRadius: '12px',
              position: 'relative',
            }}
            bodyStyle={{ padding: '32px' }}
          >
            {plan.popular && (
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
                Most Popular
              </div>
            )}
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '8px', color: '#030712' }}>
                {plan.name}
              </h3>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '4px', marginBottom: '8px' }}>
                <span style={{ fontSize: '40px', fontWeight: '800', color: '#030712' }}>
                  {plan.price}
                </span>
                {plan.period && (
                  <span style={{ fontSize: '16px', color: '#64748B' }}>/{plan.period}</span>
                )}
              </div>
              <p style={{ fontSize: '14px', color: '#64748B' }}>{plan.description}</p>
            </div>

            <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 32px 0' }}>
              {plan.features.map((feature, index) => (
                <li
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    marginBottom: '12px',
                    fontSize: '14px',
                    color: '#030712',
                  }}
                >
                  <CheckOutlined style={{ color: '#10B981', fontSize: '16px' }} />
                  {feature}
                </li>
              ))}
            </ul>

            <Button
              type={plan.popular ? 'primary' : 'default'}
              block
              size="large"
              onClick={() => {
                if (plan.name === 'Enterprise') {
                  // Contact sales action
                  window.location.href = 'mailto:sales@example.com';
                } else {
                  navigate('/signup');
                }
              }}
            >
              {plan.cta}
            </Button>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default PricingPage;

