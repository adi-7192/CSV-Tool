import React from 'react';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#FFFFFF' }}>
      {children}
    </div>
  );
};

export default Layout;

