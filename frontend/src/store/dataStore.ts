import { create } from 'zustand';

export interface DateRange {
  start: string;
  end: string;
}

interface DataStore {
  // Metrics data
  metrics: any;
  chartData: any;
  skuPerformance: any;
  insights: any[];
  // Date range
  dateRange: DateRange;
  // Loading & errors
  loading: boolean;
  error: string | null;
  // Methods
  setMetrics: (metrics: any) => void;
  setChartData: (data: any) => void;
  setSKUPerformance: (data: any) => void;
  setInsights: (insights: any[]) => void;
  setDateRange: (start: string, end: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useDataStore = create<DataStore>((set) => ({
  metrics: null,
  chartData: null,
  skuPerformance: null,
  insights: [],
  dateRange: { start: '2025-07-01', end: '2025-09-30' },
  loading: false,
  error: null,
  setMetrics: (metrics) => set({ metrics }),
  setChartData: (data) => set({ chartData: data }),
  setSKUPerformance: (data) => set({ skuPerformance: data }),
  setInsights: (insights) => set({ insights }),
  setDateRange: (start, end) => set({ dateRange: { start, end } }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}));
