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
// REGION SERVICE
// ============================================================================

export interface RegionRevenue {
  city: string;  // Changed from 'region' to 'city'
  revenue: number;
}

export interface RegionRevenueResponse {
  data: RegionRevenue[];
  count: number;
}

export interface RegionSKU {
  sku: string;
  asin: string;
  units: number;
  revenue: number;
}

export interface RegionSKUsResponse {
  data: RegionSKU[];
  count: number;
}

export const regionService = {
  /**
   * Get revenue by city for a specific date range
   * @param startDate - Start date (YYYY-MM-DD) - optional, if not provided uses all-time data
   * @param endDate - End date (YYYY-MM-DD) - optional, if not provided uses all-time data
   * @param limit - Number of cities to return (default: 10)
   * @returns City revenue data
   */
  getRevenueByCity: async (
    startDate?: string,
    endDate?: string,
    limit = 10
  ): Promise<RegionRevenueResponse | null> => {
    try {
      const url = '/api/metrics/revenue-by-city';
      const params: any = { limit };
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
      console.log(`[API] Fetching revenue by city: ${url}`, params);
      const response = await apiClient.get<RegionRevenueResponse>(url, { params });
      console.log('[API] City revenue response:', response.data);
      console.log(`[API] City revenue count: ${response.data?.count || 0}`);
      console.log('[API] City revenue data:', response.data?.data);
      return response.data;
    } catch (error) {
      console.error('[API] Error fetching revenue by city:', error);
      if (error instanceof Error) {
        console.error('[API] Error details:', error.message);
      }
      return null;
    }
  },

  /**
   * Get top SKUs for a specific city
   * @param city - City name
   * @param limit - Number of SKUs to return (default: 10)
   * @returns City SKUs data
   */
  getSKUsByCity: async (
    city: string,
    limit = 10
  ): Promise<RegionSKUsResponse | null> => {
    try {
      const url = `/api/metrics/revenue-by-city/skus/${encodeURIComponent(city)}`;
      const params = { limit };
      console.log(`[API] Fetching SKUs for city: ${url}`, params);
      const response = await apiClient.get<RegionSKUsResponse>(url, { params });
      console.log('[API] City SKUs response:', response.data);
      return response.data;
    } catch (error) {
      console.error('[API] Error fetching SKUs by city:', error);
      if (error instanceof Error) {
        console.error('[API] Error details:', error.message);
      }
      return null;
    }
  },
};

// ============================================================================
// MOVERS & DECLINERS SERVICE
// ============================================================================

export interface MoverDeclinerItem {
  sku: string;
  revenue: number;
  wow_change: number; // Week-over-week change percentage
}

export interface MoversDeclinersResponse {
  movers: MoverDeclinerItem[];
  decliners: MoverDeclinerItem[];
  label?: string; // Comparison label (e.g., "Last Week vs Weekly Avg")
  granularity?: string; // Period granularity ("daily", "weekly", or "monthly")
}

export const moversDeclinersService = {
  /**
   * Get movers and decliners SKUs
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @param limit - Number of SKUs per category (default: 10)
   * @returns Movers and decliners data
   */
  getMoversDecliners: async (
    startDate: string,
    endDate: string,
    limit = 10
  ): Promise<MoversDeclinersResponse | null> => {
    try {
      const url = '/api/metrics/movers-decliners';
      const params = {
        start_date: startDate,
        end_date: endDate,
        limit,
      };
      
      console.log('\n' + '='.repeat(80));
      console.log('🔍 API SERVICE DEBUG: MOVERS & DECLINERS');
      console.log('='.repeat(80));
      console.log(`[API] Fetching movers & decliners: ${url}`);
      console.log('[API] Request params:', params);
      
      const response = await apiClient.get<MoversDeclinersResponse>(url, { params });
      
      console.log('[API] Response status:', response.status);
      console.log('[API] Full response data:', JSON.stringify(response.data, null, 2));
      console.log('[API] Response structure:', {
        hasMovers: !!response.data?.movers,
        hasDecliners: !!response.data?.decliners,
        moversCount: response.data?.movers?.length || 0,
        declinersCount: response.data?.decliners?.length || 0,
      });
      
      if (response.data?.movers && response.data.movers.length > 0) {
        console.log('[API] Sample movers from response:', response.data.movers.slice(0, 3));
      }
      if (response.data?.decliners && response.data.decliners.length > 0) {
        console.log('[API] Sample decliners from response:', response.data.decliners.slice(0, 3));
      }
      
      console.log('='.repeat(80) + '\n');
      
      return response.data;
    } catch (error) {
      console.error('[API] ❌ Error fetching movers & decliners:', error);
      if (error instanceof Error) {
        console.error('[API] Error details:', error.message);
      }
      return null;
    }
  },
};

// ============================================================================
// TOP PRODUCTS PERFORMANCE SERVICE
// ============================================================================

export interface TopProductPerformance {
  sku: string;
  periods: number[];
  growth_rates: (number | null)[];
  total_volume: number;
}

export interface TopProductsPerformanceResponse {
  products: TopProductPerformance[];
  period_labels: string[];
  view_type: string;
}

export const topProductsPerformanceService = {
  /**
   * Get top products performance tracker
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @param viewType - 'monthly' or 'quarterly' (default: 'monthly')
   * @param limit - Number of products to return (default: 10)
   * @returns Top products performance data
   */
  getTopProductsPerformance: async (
    startDate: string,
    endDate: string,
    viewType: 'monthly' | 'quarterly' = 'monthly',
    limit = 10
  ): Promise<TopProductsPerformanceResponse | null> => {
    try {
      const url = '/api/metrics/top-products-performance';
      const params = {
        start_date: startDate,
        end_date: endDate,
        view_type: viewType,
        limit,
      };
      console.log(`[API] Fetching top products performance: ${url}`, params);
      const response = await apiClient.get<TopProductsPerformanceResponse>(url, { params });
      console.log('[API] Top products performance response:', response.data);
      return response.data;
    } catch (error) {
      console.error('[API] Error fetching top products performance:', error);
      if (error instanceof Error) {
        console.error('[API] Error details:', error.message);
      }
      return null;
    }
  },
};

// ============================================================================
// PRODUCT QUALITY ISSUES SERVICE
// ============================================================================

export interface RefundData {
  sku: string;
  units_sold: number;
  refunds: number;
  refund_percentage: number;
  lost_revenue: number;
}

export interface CancellationData {
  sku: string;
  units_ordered: number;
  cancelled: number;
  cancel_percentage: number;
}

export interface ReplacementData {
  sku: string;
  replacements: number;
  total_loss: number;
}

export interface QualityIssuesResponse<T> {
  data: T[];
}

export const qualityIssuesService = {
  /**
   * Get refunds data for Product Quality Issues dashboard
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @param limit - Number of products to return (default: 10)
   * @returns Refunds data
   */
  getRefundsData: async (
    startDate: string,
    endDate: string,
    limit = 10
  ): Promise<QualityIssuesResponse<RefundData> | null> => {
    try {
      const url = '/api/metrics/quality-issues/refunds';
      const params = {
        start_date: startDate,
        end_date: endDate,
        limit,
      };
      console.log(`[API] Fetching refunds data: ${url}`, params);
      const response = await apiClient.get<QualityIssuesResponse<RefundData>>(url, { params });
      console.log('[API] Refunds data response:', response.data);
      return response.data;
    } catch (error) {
      console.error('[API] Error fetching refunds data:', error);
      if (error instanceof Error) {
        console.error('[API] Error details:', error.message);
      }
      return null;
    }
  },

  /**
   * Get cancellations data for Product Quality Issues dashboard
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @param limit - Number of products to return (default: 10)
   * @returns Cancellations data
   */
  getCancellationsData: async (
    startDate: string,
    endDate: string,
    limit = 10
  ): Promise<QualityIssuesResponse<CancellationData> | null> => {
    try {
      const url = '/api/metrics/quality-issues/cancellations';
      const params = {
        start_date: startDate,
        end_date: endDate,
        limit,
      };
      console.log(`[API] Fetching cancellations data: ${url}`, params);
      const response = await apiClient.get<QualityIssuesResponse<CancellationData>>(url, { params });
      console.log('[API] Cancellations data response:', response.data);
      return response.data;
    } catch (error) {
      console.error('[API] Error fetching cancellations data:', error);
      if (error instanceof Error) {
        console.error('[API] Error details:', error.message);
      }
      return null;
    }
  },

  /**
   * Get free replacements data for Product Quality Issues dashboard
   * @param startDate - Start date (YYYY-MM-DD)
   * @param endDate - End date (YYYY-MM-DD)
   * @param limit - Number of products to return (default: 10)
   * @returns Free replacements data
   */
  getReplacementsData: async (
    startDate: string,
    endDate: string,
    limit = 10
  ): Promise<QualityIssuesResponse<ReplacementData> | null> => {
    try {
      const url = '/api/metrics/quality-issues/replacements';
      const params = {
        start_date: startDate,
        end_date: endDate,
        limit,
      };
      console.log(`[API] Fetching replacements data: ${url}`, params);
      const response = await apiClient.get<QualityIssuesResponse<ReplacementData>>(url, { params });
      console.log('[API] Replacements data response:', response.data);
      return response.data;
    } catch (error) {
      console.error('[API] Error fetching replacements data:', error);
      if (error instanceof Error) {
        console.error('[API] Error details:', error.message);
      }
      return null;
    }
  },
};

// ============================================================================
// EXPORTS
// ============================================================================

// ============================================================================
// CHAT SERVICE
// ============================================================================

export interface ChatResponse {
  answer: string;
  sql: string | null;
  data: any;
  execution_time: number | null;
  error: string | null;
  provider?: string | null;
  confidence?: number | null;
  suggestion?: string | null;
}

export interface ChatRequest {
  question: string;
  context?: {
    start_date?: string;
    end_date?: string;
  };
}

export const chatService = {
  /**
   * Ask a natural language question about the data
   * @param question - Natural language question
   * @param context - Optional context (date filters, etc.)
   * @returns Chat response with answer, SQL, and data
   */
  askQuestion: async (
    question: string,
    context?: { start_date?: string; end_date?: string }
  ): Promise<ChatResponse | null> => {
    try {
      const requestBody: ChatRequest = { question };
      if (context) {
        requestBody.context = context;
      }
      
      // Get user ID from localStorage (same as API key service)
      const userId = localStorage.getItem('userId') || 'user-123';
      
      const response = await apiClient.post<ChatResponse>('/api/chat/ask', requestBody, {
        headers: {
          'X-User-ID': userId,
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error asking question:', error);
      if (error instanceof AxiosError) {
        console.error('Response:', error.response?.data);
      }
      return null;
    }
  },
};

export default apiClient;
