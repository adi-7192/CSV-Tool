/**
 * API Integration Test Script
 * 
 * This script tests the API integration and verifies data flow.
 * Run with: npm run test:api (or manually)
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface TestResult {
  name: string;
  passed: boolean;
  message: string;
  data?: any;
}

const testResults: TestResult[] = [];

// Helper function to make API calls
async function testAPI(
  name: string,
  endpoint: string,
  params?: Record<string, string>
): Promise<TestResult> {
  try {
    console.log(`\n🧪 Testing: ${name}`);
    console.log(`   Endpoint: ${endpoint}`);
    console.log(`   Params:`, params || {});

    const startTime = Date.now();
    const response = await axios.get(`${API_BASE_URL}${endpoint}`, {
      params,
      timeout: 10000,
    });
    const duration = Date.now() - startTime;

    console.log(`   ✅ Status: ${response.status}`);
    console.log(`   ⏱️  Duration: ${duration}ms`);

    return {
      name,
      passed: response.status === 200 && response.data !== null,
      message: `Status: ${response.status}, Duration: ${duration}ms`,
      data: response.data,
    };
  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`);
    if (error.response) {
      console.log(`   Status: ${error.response.status}`);
      console.log(`   Data:`, error.response.data);
    }
    return {
      name,
      passed: false,
      message: error.response
        ? `Status: ${error.response.status}, ${error.response.data?.detail || error.message}`
        : error.message,
    };
  }
}

// Test 1: Backend Health Check
async function testBackendHealth(): Promise<TestResult> {
  return await testAPI('Backend Health Check', '/api/health');
}

// Test 2: Metrics Endpoint
async function testMetrics(): Promise<TestResult> {
  return await testAPI('Metrics API', '/api/metrics', {
    start_date: '2025-07-01',
    end_date: '2025-09-30',
  });
}

// Test 3: Metrics Trend Endpoint
async function testMetricsTrend(): Promise<TestResult> {
  return await testAPI('Metrics Trend API', '/api/metrics/trend', {
    start_date: '2025-07-01',
    end_date: '2025-09-30',
    group_by: 'day',
  });
}

// Test 4: Top Products Endpoint
async function testTopProducts(): Promise<TestResult> {
  return await testAPI('Top Products API', '/api/metrics/top-products', {
    start_date: '2025-07-01',
    end_date: '2025-09-30',
    limit: '50',
  });
}

// Test 5: Data Status Endpoint
async function testDataStatus(): Promise<TestResult> {
  return await testAPI('Data Status API', '/api/data/status');
}

// Validate Metrics Response
function validateMetrics(data: any): TestResult {
  const checks = {
    hasData: !!data?.data,
    hasGrossRevenue: typeof data?.data?.gross_revenue === 'number',
    hasNetRevenue: typeof data?.data?.net_revenue === 'number',
    hasOrders: typeof data?.data?.orders === 'number',
    hasPeriod: !!data?.period,
  };

  const allPassed = Object.values(checks).every((v) => v === true);
  const failures = Object.entries(checks)
    .filter(([_, v]) => !v)
    .map(([key]) => key);

  return {
    name: 'Metrics Data Validation',
    passed: allPassed,
    message: allPassed
      ? 'All metrics fields present and valid'
      : `Missing fields: ${failures.join(', ')}`,
    data: checks,
  };
}

// Main test function
async function runTests() {
  console.log('🚀 Starting API Integration Tests');
  console.log(`📍 API Base URL: ${API_BASE_URL}`);
  console.log('='.repeat(60));

  // Test 1: Backend Health
  const healthResult = await testBackendHealth();
  testResults.push(healthResult);

  if (!healthResult.passed) {
    console.log('\n❌ Backend is not available. Please start the backend server.');
    console.log('   Run: cd backend && uvicorn main:app --reload');
    return;
  }

  // Test 2: Metrics
  const metricsResult = await testMetrics();
  testResults.push(metricsResult);

  if (metricsResult.passed && metricsResult.data) {
    const validationResult = validateMetrics(metricsResult.data);
    testResults.push(validationResult);
  }

  // Test 3: Metrics Trend
  const trendResult = await testMetricsTrend();
  testResults.push(trendResult);

  // Test 4: Top Products
  const productsResult = await testTopProducts();
  testResults.push(productsResult);

  // Test 5: Data Status
  const statusResult = await testDataStatus();
  testResults.push(statusResult);

  // Print Summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 TEST SUMMARY');
  console.log('='.repeat(60));

  const passed = testResults.filter((r) => r.passed).length;
  const failed = testResults.filter((r) => !r.passed).length;

  testResults.forEach((result) => {
    const icon = result.passed ? '✅' : '❌';
    console.log(`${icon} ${result.name}: ${result.message}`);
  });

  console.log('\n' + '='.repeat(60));
  console.log(`Total: ${testResults.length} | Passed: ${passed} | Failed: ${failed}`);
  console.log('='.repeat(60));

  if (failed === 0) {
    console.log('\n🎉 All tests passed! API integration is working correctly.');
  } else {
    console.log('\n⚠️  Some tests failed. Please check the errors above.');
  }
}

// Run tests if executed directly
// Note: This is a utility file, tests should be run manually or via test runner
export { runTests, testResults };

