/**
 * DataAwareRedirect Component
 * 
 * Redirects users based on whether they have data:
 * - Has data → Dashboard
 * - No data → Onboarding
 */
import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { dataService } from '@/services/api';

const DataAwareRedirect: React.FC = () => {
  const [hasData, setHasData] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkData = async () => {
      try {
        const summary = await dataService.getSummary();
        setHasData(summary.has_data && summary.row_count > 0);
      } catch (error) {
        console.error('Failed to check data:', error);
        // Default to onboarding if check fails (safer for new users)
        setHasData(false);
      } finally {
        setLoading(false);
      }
    };

    checkData();
  }, []);

  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
        }}
      >
        <div>Loading...</div>
      </div>
    );
  }

  // Redirect based on data availability
  if (hasData) {
    return <Navigate to="/app/dashboard" replace />;
  } else {
    return <Navigate to="/app/onboarding" replace />;
  }
};

export default DataAwareRedirect;

