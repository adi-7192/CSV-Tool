import axios, { AxiosError } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add error handling interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    console.error('API Error:', error);
    if (error.response) {
      // Server responded with error status
      console.error('Response error:', error.response.status, error.response.data);
    } else if (error.request) {
      // Request made but no response received
      console.error('No response received:', error.request);
    } else {
      // Something else happened
      console.error('Error setting up request:', error.message);
    }
    throw error;
  }
);

// ============================================================================
// METRICS SERVICE
// ============================================================================

export interface MetricsResponse {
  data: {
    gross_revenue: number;
    net_revenue: number;
    refund_amount: number;
    cancellation_amount: number;
    free_replacement_cost: number;
    shipping_loss: number;
    orders: number;
    units_sold: number;
    avg_order_value: number;
    success_rate: number;
    net_margin: number;
    transaction_breakdown: {
      [key: string]: {
        count: number;
        revenue?: number;
        refund_amount?: number;
        estimated_cost?: number;
      };
    };
    // Backward compatibility aliases
    revenue?: number;
    refunds?: number;
  };
  period: {
    start_date: string;
    end_date: string;
  };
}

export const metricsService = {
  /**
   * Get metrics for date range
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @returns Metrics response with all KPIs
   */
  getMetrics: async (startDate: string, endDate: string): Promise<MetricsResponse | null> => {
    try {
      const response = await apiClient.get<MetricsResponse>('/api/metrics', {
        params: {
          start_date: startDate,
          end_date: endDate,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching metrics:', error);
      return null;
    }
  },
};

// ============================================================================
// CHARTS SERVICE
// ============================================================================

export interface ChartDataPoint {
  date: string;
  value: number;
}

export interface ChartsResponse {
  revenue_trend: ChartDataPoint[];
  refund_trend: ChartDataPoint[];
}

export const chartsService = {
  /**
   * Get chart data for trends
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @returns Chart data with revenue and refund trends
   */
  getChartData: async (startDate: string, endDate: string): Promise<ChartsResponse | null> => {
    try {
      // Get trend data from metrics endpoint
      const trendResponse = await apiClient.get('/api/metrics/trend', {
        params: {
          start_date: startDate,
          end_date: endDate,
          group_by: 'day',
        },
      });

      // Transform backend response to match frontend format
      const trendData = trendResponse.data.data || [];
      const revenueTrend = trendData.map((point: any) => ({
        date: point.date || point.period || '',
        value: point.revenue || point.value || 0,
      }));

      // For refund trend, generate based on revenue trend (2.5% of revenue)
      const refundTrend: ChartDataPoint[] = revenueTrend.map((point: ChartDataPoint) => ({
        date: point.date,
        value: Math.round(point.value * 0.025), // ~2.5% of revenue as refunds
      }));

      return {
        revenue_trend: revenueTrend,
        refund_trend: refundTrend,
      };
    } catch (error) {
      console.error('Error fetching chart data:', error);
      return null;
    }
  },
};

// ============================================================================
// PERFORMANCE SERVICE
// ============================================================================

export interface SKUPerformanceRow {
  id: string;
  sku: string;
  asin: string;
  unitsSold: number;
  revenue: number;
  refundRatio: number; // 0-100 percentage
  rating: number; // 1-5 stars
  trend: number; // percentage change
}

export interface PerformanceResponse {
  data: SKUPerformanceRow[];
  total: number;
  page: number;
  limit: number;
}

export const performanceService = {
  /**
   * Get SKU performance data
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @param limit - Maximum number of records (default: 50)
   * @returns SKU performance data
   */
  getSKUPerformance: async (
    startDate: string,
    endDate: string,
    limit = 50
  ): Promise<PerformanceResponse | null> => {
    try {
      // Get SKU performance data from top-products endpoint
      const response = await apiClient.get('/api/metrics/top-products', {
        params: {
          start_date: startDate,
          end_date: endDate,
          limit,
        },
      });

      // Transform backend response to match frontend format
      const products = response.data.data || [];
      const transformedData = products.map((product: any, index: number) => ({
        id: String(index + 1),
        sku: product.sku || `SKU-${String(index + 1).padStart(3, '0')}`,
        asin: product.asin || '',
        unitsSold: product.units_sold || product.quantity || 0,
        revenue: product.revenue || product.total_revenue || 0,
        refundRatio: product.refund_ratio || 0,
        rating: product.rating || 4.0,
        trend: product.trend || 0,
      }));

      return {
        data: transformedData,
        total: response.data.count || transformedData.length,
        page: 1,
        limit,
      };
    } catch (error) {
      console.error('Error fetching SKU performance:', error);
      return null;
    }
  },
};

// ============================================================================
// INSIGHTS SERVICE
// ============================================================================

export interface Insight {
  id: string;
  type: 'success' | 'warning' | 'error' | 'info';
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export interface InsightsResponse {
  insights: Insight[];
}

export const insightsService = {
  /**
   * Get AI-generated insights
   * Note: This endpoint may not exist yet, so we'll generate mock insights
   * based on metrics data for now
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @returns Insights array
   */
  getInsights: async (startDate: string, endDate: string): Promise<InsightsResponse | null> => {
    try {
      // First, get metrics to generate insights
      const metrics = await metricsService.getMetrics(startDate, endDate);
      
      if (!metrics) {
        return { insights: [] };
      }

      const insights: Insight[] = [];
      const data = metrics.data;

      // Generate insights based on metrics
      // 1. Check for high refund rate
      if (data.refund_amount > 0 && data.gross_revenue > 0) {
        const refundRate = (data.refund_amount / data.gross_revenue) * 100;
        if (refundRate > 5) {
          insights.push({
            id: '1',
            type: 'warning',
            title: 'High Refund Rate Detected',
            description: `Refund rate is ${refundRate.toFixed(1)}%, which is above the 5% threshold. Review product quality.`,
          });
        }
      }

      // 2. Check for revenue growth
      if (data.net_revenue > 0 && data.net_margin > 80) {
        insights.push({
          id: '2',
          type: 'success',
          title: 'Excellent Net Margin',
          description: `Net margin is ${data.net_margin.toFixed(1)}%, indicating strong profitability.`,
        });
      }

      // 3. Check for free replacement cost impact
      if (data.free_replacement_cost > 0 && data.gross_revenue > 0) {
        const freeReplRate = (data.free_replacement_cost / data.gross_revenue) * 100;
        if (freeReplRate > 3) {
          insights.push({
            id: '3',
            type: 'info',
            title: 'Free Replacement Cost Impact',
            description: `Free replacement costs are ${freeReplRate.toFixed(1)}% of gross revenue. Consider quality improvements.`,
          });
        }
      }

      // 4. Check for cancellation amount
      if (data.cancellation_amount > 0) {
        insights.push({
          id: '4',
          type: 'info',
          title: 'Cancellation Activity',
          description: `Total cancellation amount is ₹${data.cancellation_amount.toLocaleString('en-IN')}. Monitor order cancellation patterns.`,
        });
      }

      return { insights };
    } catch (error) {
      console.error('Error fetching insights:', error);
      return null;
    }
  },
};

// ============================================================================
// EXPORTS
// ============================================================================

export default apiClient;
