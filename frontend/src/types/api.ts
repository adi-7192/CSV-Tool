export interface ApiResponse<T> {
  data: T;
  error?: string;
}

export interface MetricsResponse {
  data: {
    gross_revenue: number;
    refund_amount: number;
    cancellation_amount: number;
    free_replacement_cost: number;
    net_revenue: number;
    net_margin: number;
    orders: number;
    avg_order_value: number;
    success_rate: number;
    refund_rate: number;
    transaction_breakdown: Record<string, any>;
    revenue?: number; // backward compatibility
    refunds?: number; // backward compatibility
  };
  period: {
    start_date: string;
    end_date: string;
  };
}

export interface ChatRequest {
  question: string;
}

export interface ChatResponse {
  answer: string;
  sql: string | null;
  data: any;
  execution_time: number | null;
  error: string | null;
}

