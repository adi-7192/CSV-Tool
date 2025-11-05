import { create } from 'zustand';
import {
  metricsService,
  chartsService,
  performanceService,
  insightsService,
} from '@/services/api';

export interface DateRange {
  start: string;
  end: string;
}

export interface MetricsData {
  gross_revenue: number;
  net_revenue: number;
  refund_amount: number;
  cancellation_amount: number;
  free_replacement_cost: number;
  shipping_loss: number;
  orders: number;
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
  revenue?: number;
  refunds?: number;
}

export interface InsightType {
  id: string;
  type: 'success' | 'warning' | 'error' | 'info';
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface DataStore {
  // Data
  metrics: MetricsData | null;
  chartData: { revenue_trend: any[]; refund_trend: any[] } | null;
  skuPerformance: any[] | null;
  insights: InsightType[] | null;

  // Date range
  dateRange: DateRange;

  // Loading/Error states
  metricsLoading: boolean;
  chartsLoading: boolean;
  skuLoading: boolean;
  insightsLoading: boolean;
  error: string | null;

  // Methods
  fetchMetrics: (start: string, end: string) => Promise<void>;
  fetchChartData: (start: string, end: string) => Promise<void>;
  fetchSKUPerformance: (start: string, end: string) => Promise<void>;
  fetchInsights: (start: string, end: string) => Promise<void>;
  setDateRange: (start: string, end: string) => void;
  reset: () => void;
}

export const useDataStore = create<DataStore>((set) => ({
  // Initial state
  metrics: null,
  chartData: null,
  skuPerformance: null,
  insights: null,
  dateRange: { start: '2025-07-01', end: '2025-09-30' },
  metricsLoading: false,
  chartsLoading: false,
  skuLoading: false,
  insightsLoading: false,
  error: null,

  // Fetch metrics
  fetchMetrics: async (start: string, end: string) => {
    set({ metricsLoading: true, error: null });
    try {
      const response = await metricsService.getMetrics(start, end);
      if (response) {
        set({
          metrics: response.data,
          metricsLoading: false,
        });
      } else {
        set({
          metrics: null,
          metricsLoading: false,
          error: 'Failed to fetch metrics',
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      set({
        error: errorMessage,
        metricsLoading: false,
      });
    }
  },

  // Fetch chart data
  fetchChartData: async (start: string, end: string) => {
    set({ chartsLoading: true, error: null });
    try {
      const response = await chartsService.getChartData(start, end);
      if (response) {
        set({
          chartData: {
            revenue_trend: response.revenue_trend || [],
            refund_trend: response.refund_trend || [],
          },
          chartsLoading: false,
        });
      } else {
        set({
          chartData: null,
          chartsLoading: false,
          error: 'Failed to fetch chart data',
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      set({
        error: errorMessage,
        chartsLoading: false,
      });
    }
  },

  // Fetch SKU performance
  fetchSKUPerformance: async (start: string, end: string) => {
    set({ skuLoading: true, error: null });
    try {
      const response = await performanceService.getSKUPerformance(start, end, 50);
      if (response) {
        set({
          skuPerformance: response.data || [],
          skuLoading: false,
        });
      } else {
        set({
          skuPerformance: null,
          skuLoading: false,
          error: 'Failed to fetch SKU performance',
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      set({
        error: errorMessage,
        skuLoading: false,
      });
    }
  },

  // Fetch insights
  fetchInsights: async (start: string, end: string) => {
    set({ insightsLoading: true, error: null });
    try {
      const response = await insightsService.getInsights(start, end);
      if (response) {
        set({
          insights: response.insights || [],
          insightsLoading: false,
        });
      } else {
        set({
          insights: null,
          insightsLoading: false,
          error: 'Failed to fetch insights',
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      set({
        error: errorMessage,
        insightsLoading: false,
      });
    }
  },

  // Set date range
  setDateRange: (start: string, end: string) => {
    set({ dateRange: { start, end } });
    // Note: Fetching is handled in useEffect in Dashboard.tsx
    // This prevents unnecessary API calls when just updating the range
  },

  // Reset store
  reset: () => {
    set({
      metrics: null,
      chartData: null,
      skuPerformance: null,
      insights: null,
      metricsLoading: false,
      chartsLoading: false,
      skuLoading: false,
      insightsLoading: false,
      error: null,
    });
  },
}));
