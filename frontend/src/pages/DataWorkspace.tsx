import React from 'react';
import PerformanceTable, { PerformanceTableRow } from '@/components/PerformanceTable';

const DataWorkspace: React.FC = () => {
  // Sample SKU performance data
  const sampleData: PerformanceTableRow[] = [
    {
      id: '1',
      sku: 'SKU-001',
      asin: 'B001234567',
      unitsSold: 1200,
      revenue: 234567.89,
      refundRatio: 2.3,
      rating: 4.5,
      trend: 12.3,
    },
    {
      id: '2',
      sku: 'SKU-002',
      asin: 'B002345678',
      unitsSold: 890,
      revenue: 156234.56,
      refundRatio: 5.1,
      rating: 4.2,
      trend: -3.2,
    },
    {
      id: '3',
      sku: 'SKU-003',
      asin: 'B003456789',
      unitsSold: 2340,
      revenue: 489012.34,
      refundRatio: 1.8,
      rating: 4.8,
      trend: 8.7,
    },
    {
      id: '4',
      sku: 'SKU-004',
      asin: 'B004567890',
      unitsSold: 560,
      revenue: 89012.45,
      refundRatio: 6.5,
      rating: 3.9,
      trend: -5.4,
    },
    {
      id: '5',
      sku: 'SKU-005',
      asin: 'B005678901',
      unitsSold: 1780,
      revenue: 345678.90,
      refundRatio: 3.8,
      rating: 4.6,
      trend: 15.2,
    },
    {
      id: '6',
      sku: 'SKU-006',
      asin: 'B006789012',
      unitsSold: 945,
      revenue: 189234.67,
      refundRatio: 2.9,
      rating: 4.4,
      trend: 7.1,
    },
    {
      id: '7',
      sku: 'SKU-007',
      asin: 'B007890123',
      unitsSold: 1234,
      revenue: 267890.12,
      refundRatio: 4.2,
      rating: 4.3,
      trend: -1.8,
    },
    {
      id: '8',
      sku: 'SKU-008',
      asin: 'B008901234',
      unitsSold: 2100,
      revenue: 456789.01,
      refundRatio: 1.5,
      rating: 4.9,
      trend: 22.5,
    },
    {
      id: '9',
      sku: 'SKU-009',
      asin: 'B009012345',
      unitsSold: 678,
      revenue: 123456.78,
      refundRatio: 7.2,
      rating: 3.7,
      trend: -8.3,
    },
    {
      id: '10',
      sku: 'SKU-010',
      asin: 'B010123456',
      unitsSold: 1567,
      revenue: 298765.43,
      refundRatio: 2.1,
      rating: 4.7,
      trend: 9.4,
    },
    {
      id: '11',
      sku: 'SKU-011',
      asin: 'B011234567',
      unitsSold: 789,
      revenue: 145678.90,
      refundRatio: 3.5,
      rating: 4.1,
      trend: 4.2,
    },
    {
      id: '12',
      sku: 'SKU-012',
      asin: 'B012345678',
      unitsSold: 2345,
      revenue: 512345.67,
      refundRatio: 1.2,
      rating: 4.8,
      trend: 18.9,
    },
  ];

  const handleRowClick = (row: PerformanceTableRow) => {
    console.log('Row clicked:', row);
    // TODO: Navigate to SKU detail page or show modal
  };

  return (
    <div style={{ padding: '24px' }}>
      <h1 style={{ marginBottom: '24px', color: '#030712' }}>Data Workspace</h1>
      
      <PerformanceTable
        data={sampleData}
        onRowClick={handleRowClick}
      />
    </div>
  );
};

export default DataWorkspace;
