/**
 * ScrollToTop Component
 * 
 * Button that appears after scrolling down to quickly return to top.
 * Includes smooth animations and accessibility features.
 */

import React, { useState, useEffect } from 'react';
import { Button } from 'antd';
import { UpOutlined } from '@ant-design/icons';
import { COLORS, BORDER_RADIUS, SHADOWS } from '@/styles/design-tokens';
import '../ScrollToTop.css';

const ScrollToTop: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const toggleVisibility = () => {
      if (window.pageYOffset > 300) {
        setIsVisible(true);
      } else {
        setIsVisible(false);
      }
    };

    window.addEventListener('scroll', toggleVisibility);
    return () => window.removeEventListener('scroll', toggleVisibility);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };

  return (
    <Button
      type="primary"
      icon={<UpOutlined />}
      onClick={scrollToTop}
      className={`scroll-to-top ${isVisible ? 'visible' : ''}`}
      aria-label="Scroll to top"
      style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        width: '48px',
        height: '48px',
        borderRadius: BORDER_RADIUS.full,
        backgroundColor: COLORS.primary,
        borderColor: COLORS.primary,
        boxShadow: SHADOWS.md,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
    />
  );
};

export default ScrollToTop;

