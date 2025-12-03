import { create } from 'zustand';
import {
  metricsService,
  chartsService,
  performanceService,
  insightsService,
  regionService,
  moversDeclinersService,
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

export interface RegionRevenue {
  region: string;
  revenue: number;
}

export interface MoverDeclinerItem {
  sku: string;
  revenue: number;
  wow_change: number;
}

interface DataStore {
  // Data
  metrics: MetricsData | null;
  chartData: { revenue_trend: any[]; refund_trend: any[] } | null;
  skuPerformance: any[] | null;
  insights: InsightType[] | null;
  regionRevenue: RegionRevenue[] | null;
  moversDecliners: { 
    movers: MoverDeclinerItem[]; 
    decliners: MoverDeclinerItem[];
    label?: string;
    granularity?: string;
  } | null;

  // Date range
  dateRange: DateRange;

  // Loading/Error states
  metricsLoading: boolean;
  chartsLoading: boolean;
  skuLoading: boolean;
  insightsLoading: boolean;
  regionLoading: boolean;
  moversDeclinersLoading: boolean;
  error: string | null;

  // Methods
  fetchMetrics: (start: string, end: string) => Promise<boolean>;
  fetchChartData: (start: string, end: string) => Promise<void>;
  fetchSKUPerformance: (start: string, end: string) => Promise<void>;
  fetchInsights: (start: string, end: string) => Promise<void>;
  fetchRegionRevenue: (start: string, end: string) => Promise<void>;
  fetchMoversDecliners: (start: string, end: string) => Promise<void>;
  setDateRange: (start: string, end: string) => void;
  reset: () => void;
}

export const useDataStore = create<DataStore>((set) => ({
  // Initial state
  metrics: null,
  chartData: null,
  skuPerformance: null,
  insights: null,
  regionRevenue: null,
  moversDecliners: null,
  dateRange: { start: '2025-07-01', end: '2025-09-30' },
  metricsLoading: false,
  chartsLoading: false,
  skuLoading: false,
  insightsLoading: false,
  regionLoading: false,
  moversDeclinersLoading: false,
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
        // Store has_data flag for empty state check
        return response.has_data ?? true;
      } else {
        set({
          metrics: null,
          metricsLoading: false,
          error: 'Failed to fetch metrics',
        });
        return false;
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      set({
        error: errorMessage,
        metricsLoading: false,
      });
      return false;
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

  // Fetch region revenue (now city revenue - filtered by date range)
  fetchRegionRevenue: async (start: string, end: string) => {
    set({ regionLoading: true, error: null });
    try {
      console.log(`[DataStore] Fetching revenue by city for date range: ${start} to ${end}`);
      const response = await regionService.getRevenueByCity(start, end, 10);
      console.log('[DataStore] City revenue response:', response);
      if (response) {
        const data = response.data || [];
        console.log(`[DataStore] City revenue data count: ${data.length}`);
        if (data.length > 0) {
          console.log('[DataStore] Sample city data:', data.slice(0, 3));
          // Map 'city' field to 'region' for backward compatibility with Dashboard component
          const mappedData = data.map(item => ({
            region: item.city,  // Map city -> region for Dashboard
            revenue: item.revenue
          }));
          set({
            regionRevenue: mappedData,
            regionLoading: false,
          });
        } else {
          set({
            regionRevenue: [],
            regionLoading: false,
          });
        }
      } else {
        console.warn('[DataStore] City revenue response is null');
        set({
          regionRevenue: null,
          regionLoading: false,
          error: 'Failed to fetch city revenue',
        });
      }
    } catch (error) {
      console.error('[DataStore] Error fetching city revenue:', error);
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      set({
        error: errorMessage,
        regionLoading: false,
      });
    }
  },

  // Fetch movers and decliners
  fetchMoversDecliners: async (start: string, end: string) => {
    set({ moversDeclinersLoading: true, error: null });
    try {
      console.log('\n' + '='.repeat(80));
      console.log('🔍 DATASTORE DEBUG: FETCHING MOVERS & DECLINERS');
      console.log('='.repeat(80));
      console.log(`[DataStore] Fetching movers & decliners from ${start} to ${end}`);
      
      const response = await moversDeclinersService.getMoversDecliners(start, end, 10);
      
      console.log('[DataStore] Raw API Response:', response);
      console.log('[DataStore] Response Type:', typeof response);
      console.log('[DataStore] Has movers:', !!response?.movers);
      console.log('[DataStore] Has decliners:', !!response?.decliners);
      
      if (response) {
        console.log('[DataStore] Movers count:', response.movers?.length || 0);
        console.log('[DataStore] Decliners count:', response.decliners?.length || 0);
        
        if (response.movers && response.movers.length > 0) {
          console.log('[DataStore] Sample movers:', response.movers.slice(0, 3));
        }
        if (response.decliners && response.decliners.length > 0) {
          console.log('[DataStore] Sample decliners:', response.decliners.slice(0, 3));
        }
        
        set({
          moversDecliners: {
            movers: response.movers || [],
            decliners: response.decliners || [],
            label: response.label,
            granularity: response.granularity,
          },
          moversDeclinersLoading: false,
        });
        
        console.log('[DataStore] ✅ Successfully stored movers & decliners');
      } else {
        console.warn('[DataStore] ⚠️  API Response is null or undefined');
        set({
          moversDecliners: null,
          moversDeclinersLoading: false,
          error: 'Failed to fetch movers and decliners',
        });
      }
      console.log('='.repeat(80) + '\n');
    } catch (error) {
      console.error('[DataStore] ❌ Error fetching movers & decliners:', error);
      const errorMessage =
        error instanceof Error ? error.message : 'Unknown error occurred';
      set({
        error: errorMessage,
        moversDeclinersLoading: false,
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
      regionRevenue: null,
      moversDecliners: null,
      metricsLoading: false,
      chartsLoading: false,
      skuLoading: false,
      insightsLoading: false,
      regionLoading: false,
      moversDeclinersLoading: false,
      error: null,
    });
  },
}));
