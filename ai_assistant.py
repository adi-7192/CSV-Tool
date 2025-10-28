"""
AI Assistant Module for CSV Analytics Dashboard

This module provides natural language query capabilities for business data
using local Ollama with llama3.1:8b model. Converts natural language questions
into SQL queries and executes them against DuckDB database.

Features:
- Local Ollama integration (no internet required)
- Natural language to SQL conversion
- SQL safety validation
- Indian currency formatting (Lakhs/Crores)
- Business-friendly response formatting
- Error handling and graceful failures

Usage:
    python ai_assistant.py  # Test the module independently
"""

import requests
import json
import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import numpy as np
from collections import defaultdict
import math
import logging
import traceback
from functools import wraps

# Import our existing database manager
try:
    from db_manager import query_data, get_row_count, table_exists
except ImportError:
    print("❌ Error: db_manager.py not found. Make sure it's in the same directory.")
    sys.exit(1)

# Configure logging for error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_assistant_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Error handling decorators
def handle_ollama_errors(func):
    """Decorator to handle Ollama connection and API errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama connection error in {func.__name__}: {e}")
            return {
                'success': False,
                'error': 'AI assistant is currently unavailable. Please ensure Ollama is running.',
                'error_type': 'connection_error'
            }
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama timeout error in {func.__name__}: {e}")
            return {
                'success': False,
                'error': 'This query is taking too long. Try a simpler question.',
                'error_type': 'timeout_error'
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request error in {func.__name__}: {e}")
            return {
                'success': False,
                'error': 'AI assistant is experiencing issues. Please try again.',
                'error_type': 'request_error'
            }
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': 'I encountered an unexpected error. Please try rephrasing your question.',
                'error_type': 'unexpected_error'
            }
    return wrapper

def handle_database_errors(func):
    """Decorator to handle database query errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': 'Unable to retrieve data. Please try again.',
                'error_type': 'database_error'
            }
    return wrapper

def handle_sql_errors(func):
    """Decorator to handle SQL generation and validation errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"SQL error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': 'I couldn\'t understand that question. Could you rephrase it?',
                'error_type': 'sql_error'
            }
    return wrapper

def handle_llm_response_errors(func):
    """Decorator to handle LLM response parsing errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"LLM response error in {func.__name__}: {e}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': 'I\'m having trouble formulating a response. Please try a different question.',
                'error_type': 'llm_response_error'
            }
    return wrapper

def timeout_handler(seconds=30):
    """Decorator to add timeout handling to functions using threading"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import threading
            import time
            
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                logger.error(f"Timeout error in {func.__name__}: Operation timed out after {seconds} seconds")
                return {
                    'success': False,
                    'error': 'This query is taking too long. Try a simpler question.',
                    'error_type': 'timeout_error'
                }
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator

class PredictiveAnalytics:
    """Predictive analytics and business intelligence for the AI assistant"""
    
    def __init__(self):
        """Initialize predictive analytics"""
        self.confidence_threshold = 0.7  # Minimum confidence for predictions
        self.min_data_points = 3  # Minimum data points needed for analysis
    
    def is_predictive_question(self, question: str) -> bool:
        """Detect if question requires predictive analytics"""
        question_lower = question.lower()
        
        predictive_keywords = [
            'predict', 'forecast', 'will be', 'next month', 'next quarter', 'next year',
            'future', 'trend', 'growing', 'declining', 'should i', 'recommend',
            'restock', 'run out', 'deplete', 'inventory', 'demand', 'seasonal',
            'pattern', 'anomaly', 'unusual', 'spike', 'drop', 'peak', 'slow',
            'optimize', 'best time', 'worst time', 'when to', 'how much'
        ]
        
        return any(keyword in question_lower for keyword in predictive_keywords)
    
    def analyze_trends(self, data: pd.DataFrame, metric: str, group_by: str = None) -> Dict[str, Any]:
        """Analyze trends in data"""
        if data.empty or len(data) < self.min_data_points:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        try:
            # Convert date column if needed
            if 'Invoice Date' in data.columns:
                data['date'] = pd.to_datetime(data['Invoice Date'])
                data = data.sort_values('date')
            
            if group_by and group_by in data.columns:
                # Group analysis
                trends = {}
                for group_value in data[group_by].unique():
                    group_data = data[data[group_by] == group_value]
                    if len(group_data) >= self.min_data_points:
                        trends[str(group_value)] = self._calculate_trend(group_data[metric])
                return {'trends': trends, 'overall_trend': self._calculate_trend(data[metric])}
            else:
                # Overall trend
                return self._calculate_trend(data[metric])
                
        except Exception as e:
            print(f"Error in trend analysis: {e}")
            return {'trend': 'error', 'confidence': 0.0}
    
    def _calculate_trend(self, values: pd.Series) -> Dict[str, Any]:
        """Calculate trend direction and strength"""
        if len(values) < self.min_data_points:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        try:
            # Simple linear trend calculation
            x = np.arange(len(values))
            y = values.values
            
            # Calculate slope and correlation
            slope = np.polyfit(x, y, 1)[0]
            correlation = np.corrcoef(x, y)[0, 1]
            
            # Determine trend direction
            if abs(slope) < 0.01:  # Very small change
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'growing'
            else:
                trend_direction = 'declining'
            
            # Calculate confidence based on correlation
            confidence = abs(correlation) if not np.isnan(correlation) else 0.0
            
            # Calculate percentage change
            if len(values) > 1:
                first_value = values.iloc[0]
                last_value = values.iloc[-1]
                if first_value != 0:
                    pct_change = ((last_value - first_value) / abs(first_value)) * 100
                else:
                    pct_change = 0
            else:
                pct_change = 0
            
            return {
                'trend': trend_direction,
                'confidence': confidence,
                'slope': slope,
                'pct_change': pct_change,
                'first_value': first_value if len(values) > 0 else 0,
                'last_value': last_value if len(values) > 0 else 0
            }
            
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return {'trend': 'error', 'confidence': 0.0}
    
    def forecast_revenue(self, historical_data: pd.DataFrame, periods: int = 1) -> Dict[str, Any]:
        """Forecast future revenue based on historical data"""
        if historical_data.empty or len(historical_data) < self.min_data_points:
            return {'forecast': 0, 'confidence': 0.0, 'error': 'Insufficient historical data'}
        
        try:
            # Convert dates and sort
            historical_data['date'] = pd.to_datetime(historical_data['Invoice Date'])
            historical_data = historical_data.sort_values('date')
            
            # Group by month for monthly revenue
            historical_data['year_month'] = historical_data['date'].dt.to_period('M')
            monthly_revenue = historical_data.groupby('year_month')['revenue_calc'].sum()
            
            if len(monthly_revenue) < self.min_data_points:
                return {'forecast': 0, 'confidence': 0.0, 'error': 'Insufficient monthly data'}
            
            # Calculate trend
            trend_analysis = self._calculate_trend(monthly_revenue)
            
            # Simple linear forecast
            last_revenue = monthly_revenue.iloc[-1]
            avg_monthly_change = trend_analysis.get('slope', 0)
            
            # Forecast next period(s)
            forecast_revenue = last_revenue + (avg_monthly_change * periods)
            confidence = trend_analysis.get('confidence', 0.0)
            
            # Add some uncertainty
            uncertainty = max(0.1, 1 - confidence)  # At least 10% uncertainty
            lower_bound = forecast_revenue * (1 - uncertainty)
            upper_bound = forecast_revenue * (1 + uncertainty)
            
            return {
                'forecast': max(0, forecast_revenue),  # Don't forecast negative revenue
                'confidence': confidence,
                'lower_bound': max(0, lower_bound),
                'upper_bound': upper_bound,
                'trend': trend_analysis.get('trend', 'unknown'),
                'pct_change': trend_analysis.get('pct_change', 0),
                'last_month_revenue': last_revenue
            }
            
        except Exception as e:
            print(f"Error in revenue forecasting: {e}")
            return {'forecast': 0, 'confidence': 0.0, 'error': str(e)}
    
    def analyze_product_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze product performance and provide recommendations"""
        if data.empty:
            return {'products': {}, 'recommendations': []}
        
        try:
            # Group by SKU and analyze
            product_analysis = {}
            recommendations = []
            
            for sku in data['Sku'].unique():
                if pd.isna(sku):
                    continue
                    
                sku_data = data[data['Sku'] == sku]
                
                # Calculate metrics
                total_revenue = sku_data['revenue_calc'].sum()
                total_orders = sku_data['Invoice Number'].nunique()
                avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
                
                # Analyze trend
                sku_data_sorted = sku_data.sort_values('Invoice Date')
                trend = self._calculate_trend(sku_data_sorted['revenue_calc'])
                
                product_analysis[str(sku)] = {
                    'total_revenue': total_revenue,
                    'total_orders': total_orders,
                    'avg_order_value': avg_order_value,
                    'trend': trend.get('trend', 'unknown'),
                    'confidence': trend.get('confidence', 0.0),
                    'pct_change': trend.get('pct_change', 0)
                }
                
                # Generate recommendations
                if trend.get('trend') == 'declining' and trend.get('confidence', 0) > 0.6:
                    recommendations.append({
                        'sku': str(sku),
                        'action': 'review',
                        'reason': f"Declining sales trend ({trend.get('pct_change', 0):.1f}% change)",
                        'priority': 'high' if trend.get('pct_change', 0) < -20 else 'medium'
                    })
                elif trend.get('trend') == 'growing' and trend.get('confidence', 0) > 0.6:
                    recommendations.append({
                        'sku': str(sku),
                        'action': 'restock',
                        'reason': f"Growing sales trend ({trend.get('pct_change', 0):.1f}% change)",
                        'priority': 'high' if trend.get('pct_change', 0) > 20 else 'medium'
                    })
            
            return {
                'products': product_analysis,
                'recommendations': recommendations,
                'top_performers': sorted(product_analysis.items(), 
                                       key=lambda x: x[1]['total_revenue'], reverse=True)[:5],
                'declining_products': [p for p in product_analysis.items() 
                                     if p[1]['trend'] == 'declining' and p[1]['confidence'] > 0.6]
            }
            
        except Exception as e:
            print(f"Error in product analysis: {e}")
            return {'products': {}, 'recommendations': [], 'error': str(e)}
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual patterns in data"""
        anomalies = []
        
        try:
            if data.empty or len(data) < self.min_data_points:
                return anomalies
            
            # Convert dates
            data['date'] = pd.to_datetime(data['Invoice Date'])
            data = data.sort_values('date')
            
            # Daily revenue analysis
            daily_revenue = data.groupby(data['date'].dt.date)['revenue_calc'].sum()
            
            if len(daily_revenue) >= 7:  # Need at least a week of data
                # Calculate rolling average and standard deviation
                rolling_mean = daily_revenue.rolling(window=7, min_periods=3).mean()
                rolling_std = daily_revenue.rolling(window=7, min_periods=3).std()
                
                # Detect outliers (more than 2 standard deviations from mean)
                for date, revenue in daily_revenue.items():
                    if pd.notna(rolling_mean[date]) and pd.notna(rolling_std[date]):
                        if rolling_std[date] > 0:
                            z_score = abs(revenue - rolling_mean[date]) / rolling_std[date]
                            if z_score > 2:
                                anomalies.append({
                                    'date': date,
                                    'type': 'revenue_spike' if revenue > rolling_mean[date] else 'revenue_drop',
                                    'value': revenue,
                                    'expected': rolling_mean[date],
                                    'severity': 'high' if z_score > 3 else 'medium',
                                    'z_score': z_score
                                })
            
            return anomalies
            
        except Exception as e:
            print(f"Error detecting anomalies: {e}")
            return []
    
    def get_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns in data"""
        try:
            if data.empty or len(data) < 30:  # Need at least a month of data
                return {'patterns': {}, 'insights': []}
            
            data['date'] = pd.to_datetime(data['Invoice Date'])
            data['month'] = data['date'].dt.month
            data['day_of_week'] = data['date'].dt.dayofweek
            data['day_of_month'] = data['date'].dt.day
            
            patterns = {}
            insights = []
            
            # Monthly patterns
            monthly_revenue = data.groupby('month')['revenue_calc'].sum()
            if len(monthly_revenue) > 1:
                best_month = monthly_revenue.idxmax()
                worst_month = monthly_revenue.idxmin()
                patterns['monthly'] = {
                    'best_month': best_month,
                    'worst_month': worst_month,
                    'monthly_revenue': monthly_revenue.to_dict()
                }
                insights.append(f"Best performing month: {best_month}, Worst: {worst_month}")
            
            # Day of week patterns
            daily_revenue = data.groupby('day_of_week')['revenue_calc'].sum()
            if len(daily_revenue) > 1:
                best_day = daily_revenue.idxmax()
                worst_day = daily_revenue.idxmin()
                patterns['daily'] = {
                    'best_day': best_day,
                    'worst_day': worst_day,
                    'daily_revenue': daily_revenue.to_dict()
                }
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                insights.append(f"Best performing day: {day_names[best_day]}, Worst: {day_names[worst_day]}")
            
            return {'patterns': patterns, 'insights': insights}
            
        except Exception as e:
            print(f"Error analyzing seasonal patterns: {e}")
            return {'patterns': {}, 'insights': [], 'error': str(e)}


class AIAssistant:
    """AI Assistant for natural language business queries"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize AI Assistant with Ollama connection
        
        Args:
            ollama_url: URL of local Ollama service (default: http://localhost:11434)
        """
        self.ollama_url = ollama_url
        self.model = "llama3.1:8b"
        self.is_ollama_available = False
        self.database_schema = {}
        self.query_cache = {}  # Simple cache for repeated questions
        
        try:
            # Test Ollama connection
            if self._test_ollama_connection():
                self.is_ollama_available = True
                print("✅ AI Assistant initialized successfully")
                print(f"🔗 Connected to Ollama at {ollama_url}")
                print(f"🤖 Using model: {self.model}")
            else:
                print("⚠️ AI Assistant initialized with limited functionality")
                print("💡 Ollama is not available - some features will be disabled")
            
            # Load database schema safely
            self.database_schema = self._get_database_schema()
            
            # Initialize predictive analytics
            self.predictive_analytics = PredictiveAnalytics()
            
        except Exception as e:
            logger.error(f"Error initializing AI Assistant: {e}\n{traceback.format_exc()}")
            print("⚠️ AI Assistant initialized with limited functionality due to initialization error")
            self.is_ollama_available = False
    
    def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama service"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model in model_names:
                    print(f"✅ Ollama connection successful. Model {self.model} available.")
                    return True
                else:
                    print(f"❌ Model {self.model} not found. Available models: {model_names}")
                    return False
            else:
                print(f"❌ Ollama connection failed. Status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Cannot connect to Ollama: {e}")
            print("❌ Cannot connect to Ollama service")
            print("💡 Make sure Ollama is running: ollama serve")
            return False
        except requests.exceptions.Timeout as e:
            logger.warning(f"Ollama connection timeout: {e}")
            print("❌ Ollama connection timed out")
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama request error: {e}")
            print(f"❌ Ollama request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error testing Ollama connection: {e}")
            print(f"❌ Unexpected error connecting to Ollama: {e}")
            return False
    
    def _get_database_schema(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            if not table_exists('sales'):
                logger.warning("Sales table does not exist")
                return {}
            
            # Get table schema
            schema_query = "DESCRIBE sales"
            schema_result = query_data(schema_query)
            
            if schema_result.empty:
                logger.warning("No schema information available")
                return {}
            
            # Get sample data for context
            sample_query = "SELECT * FROM sales LIMIT 5"
            sample_result = query_data(sample_query)
            
            # Get column statistics
            stats_query = """
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT "Invoice Number") as unique_orders,
                    COUNT(DISTINCT "Sku") as unique_skus,
                    COUNT(DISTINCT "Ship To City") as unique_cities,
                    MIN("Invoice Date") as earliest_date,
                    MAX("Invoice Date") as latest_date
                FROM sales
            """
            stats_result = query_data(stats_query)
            
            schema_info = {
                'columns': schema_result['column_name'].tolist(),
                'column_types': dict(zip(schema_result['column_name'], schema_result['column_type'])),
                'sample_data': sample_result.to_dict('records') if not sample_result.empty else [],
                'statistics': stats_result.iloc[0].to_dict() if not stats_result.empty else {}
            }
            
            print(f"📊 Database schema loaded: {len(schema_info['columns'])} columns")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error loading database schema: {e}\n{traceback.format_exc()}")
            print(f"⚠️ Could not load database schema: {e}")
            return {}
    
    def _generate_sql_prompt(self, question: str) -> str:
        """Generate prompt for SQL generation"""
        
        schema_info = self.database_schema
        if not schema_info:
            return f"""
            Generate a SQL query for this question: "{question}"
            
            Database: DuckDB
            Table: sales
            
            Important:
            - Use double quotes around column names
            - Format dates as 'YYYY-MM-DD'
            - Only use SELECT statements (no INSERT, UPDATE, DELETE, DROP)
            - Return only the SQL query, no explanations
            """
        
        columns = schema_info['columns']
        sample_data = schema_info['sample_data']
        stats = schema_info['statistics']
        
        # Special handling for declining/trend analysis questions
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ['declining', 'decreasing', 'trend', 'compare', 'vs', 'versus']):
            return self._generate_trend_analysis_prompt(question, columns, sample_data, stats)
        
        prompt = f"""
        You are a SQL expert. Generate a SQL query for this business question: "{question}"
        
        Database: DuckDB
        Table: sales
        
        Available columns: {', '.join(columns)}
        
        Sample data structure:
        {json.dumps(sample_data[:2], indent=2) if sample_data else 'No sample data'}
        
        Database statistics:
        - Total rows: {stats.get('total_rows', 'Unknown')}
        - Unique orders: {stats.get('unique_orders', 'Unknown')}
        - Unique SKUs: {stats.get('unique_skus', 'Unknown')}
        - Unique cities: {stats.get('unique_cities', 'Unknown')}
        - Date range: {stats.get('earliest_date', 'Unknown')} to {stats.get('latest_date', 'Unknown')}
        
        Important rules:
        1. Use double quotes around column names (e.g., "Invoice Date", "Invoice Amount")
        2. Format dates as 'YYYY-MM-DD' strings
        3. Only use SELECT statements (no INSERT, UPDATE, DELETE, DROP, ALTER)
        4. Use proper SQL syntax for DuckDB
        5. For revenue calculations, use "Invoice Amount" or "revenue_calc" column
        6. For order counts, use COUNT(DISTINCT "Invoice Number")
        7. For product analysis, use "Sku" and "Asin" columns
        8. For regional analysis, use "Ship To City" column
        9. For date filtering, use "Invoice Date" column
        10. For transaction analysis, use "Transaction Type" or "transaction_type" column
        11. NEVER use window functions (LAG, LEAD, ROW_NUMBER) in HAVING or WHERE clauses
        12. For trend analysis, use subqueries or CTEs instead of window functions in HAVING
        13. For declining analysis, compare current period vs previous period using separate queries
        14. For date extraction, use DATE_PART('month', CAST("Invoice Date" AS DATE)) instead of EXTRACT(MONTH FROM "Invoice Date")
        15. For year extraction, use DATE_PART('year', CAST("Invoice Date" AS DATE)) instead of EXTRACT(YEAR FROM "Invoice Date")
        16. For day extraction, use DATE_PART('day', CAST("Invoice Date" AS DATE)) instead of EXTRACT(DAY FROM "Invoice Date")
        17. Return ONLY the SQL query, no explanations or markdown
        
        Generate the SQL query:
        """
        
        return prompt
    
    def _generate_trend_analysis_prompt(self, question: str, columns: List[str], sample_data: List[Dict], stats: Dict) -> str:
        """Generate specialized prompt for trend analysis questions"""
        return f"""
        You are a SQL expert. Generate a SQL query for this trend analysis question: "{question}"
        
        Database: DuckDB
        Table: sales
        
        Available columns: {', '.join(columns)}
        
        Sample data structure:
        {json.dumps(sample_data[:2], indent=2) if sample_data else 'No sample data'}
        
        Database statistics:
        - Total rows: {stats.get('total_rows', 'Unknown')}
        - Unique orders: {stats.get('unique_orders', 'Unknown')}
        - Unique SKUs: {stats.get('unique_skus', 'Unknown')}
        - Unique cities: {stats.get('unique_cities', 'Unknown')}
        - Date range: {stats.get('earliest_date', 'Unknown')} to {stats.get('latest_date', 'Unknown')}
        
        CRITICAL RULES FOR TREND ANALYSIS:
        1. NEVER use window functions (LAG, LEAD, ROW_NUMBER) in HAVING or WHERE clauses
        2. For declining analysis, use subqueries to compare periods
        3. Use CTEs (WITH clauses) for complex trend analysis
        4. For "declining SKUs", compare current month vs previous month using separate subqueries
        5. Use proper date filtering with "Invoice Date" column
        6. Group by "Sku" for product analysis
        7. Use COUNT(DISTINCT "Invoice Number") for order counts
        8. Use SUM("revenue_calc") for revenue calculations
        9. NEVER use LAG() or LEAD() functions - use JOINs between period subqueries instead
        10. For declining products, compare current period orders/revenue vs previous period
        11. For date extraction, use DATE_PART('month', CAST("Invoice Date" AS DATE)) instead of EXTRACT(MONTH FROM "Invoice Date")
        12. For year extraction, use DATE_PART('year', CAST("Invoice Date" AS DATE)) instead of EXTRACT(YEAR FROM "Invoice Date")
        13. For day extraction, use DATE_PART('day', CAST("Invoice Date" AS DATE)) instead of EXTRACT(DAY FROM "Invoice Date")
        14. NEVER use DATE_SUB() - use CURRENT_DATE - INTERVAL '6 MONTH' for date filtering
        15. NEVER use NOW() - use CURRENT_DATE instead
        16. ALWAYS use INTERVAL '6 MONTH' (with quotes) not INTERVAL 6 MONTH
        
        Example pattern for declining SKUs (DuckDB-compatible):
        WITH current_period AS (
            SELECT "Sku", COUNT(DISTINCT "Invoice Number") as current_orders,
                   SUM("revenue_calc") as current_revenue
            FROM sales 
            WHERE CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '1 MONTH'
            GROUP BY "Sku"
        ),
        previous_period AS (
            SELECT "Sku", COUNT(DISTINCT "Invoice Number") as previous_orders,
                   SUM("revenue_calc") as previous_revenue
            FROM sales 
            WHERE CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '2 MONTH' 
            AND CAST("Invoice Date" AS DATE) < CURRENT_DATE - INTERVAL '1 MONTH'
            GROUP BY "Sku"
        )
        SELECT c."Sku", c.current_orders, p.previous_orders,
               (c.current_orders - COALESCE(p.previous_orders, 0)) as order_change,
               c.current_revenue, p.previous_revenue,
               (c.current_revenue - COALESCE(p.previous_revenue, 0)) as revenue_change
        FROM current_period c
        LEFT JOIN previous_period p ON c."Sku" = p."Sku"
        WHERE c.current_orders < COALESCE(p.previous_orders, 0)
           OR c.current_revenue < COALESCE(p.previous_revenue, 0)
        ORDER BY order_change ASC, revenue_change ASC
        
        Generate the SQL query:
        """
    
    def _validate_sql_safety(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        sql_upper = sql.upper().strip()
        
        # Dangerous operations to block
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE',
            'EXEC', 'EXECUTE', 'SP_', 'XP_', 'UNION',
            'INFORMATION_SCHEMA', 'SYS.', 'PG_', 'MYSQL.'
        ]
        
        # Remove SQL comments to check for other dangerous patterns
        # Single-line comments (--) are generally safe if they're just annotations
        sql_no_comments = re.sub(r'--.*?(\n|$)', '', sql)
        
        # Check for multi-line comments which could hide dangerous SQL
        if '/*' in sql or '*/' in sql:
            return False, "Multi-line comments (/* */) are not allowed for security reasons"
        
        # Check for invalid window function usage
        if 'HAVING' in sql_upper and any(func in sql_upper for func in ['LAG(', 'LEAD(', 'ROW_NUMBER(', 'RANK(', 'DENSE_RANK(']):
            return False, "Window functions cannot be used in HAVING clause"
        
        # Check for window functions in WHERE clause (also problematic)
        if 'WHERE' in sql_upper and any(func in sql_upper for func in ['LAG(', 'LEAD(', 'ROW_NUMBER(', 'RANK(', 'DENSE_RANK(']):
            return False, "Window functions cannot be used in WHERE clause - use subqueries or CTEs instead"
        
        # Check for EXTRACT functions (not supported in DuckDB)
        if 'EXTRACT(' in sql_upper:
            return False, "EXTRACT function not supported in DuckDB - use DATE_PART('month', CAST(column AS DATE)) instead"
        
        # Check for DATE_SUB function (not supported in DuckDB)
        if 'DATE_SUB(' in sql_upper:
            return False, "DATE_SUB function not supported in DuckDB - use CURRENT_DATE - INTERVAL '6 MONTH' instead"
        
        # Check for NOW() function (not supported in DuckDB)
        if 'NOW()' in sql_upper:
            return False, "NOW() function not supported in DuckDB - use CURRENT_DATE instead"
        
        # Check for INTERVAL without quotes (common error)
        if 'INTERVAL ' in sql_upper and 'INTERVAL \'' not in sql_upper:
            return False, "INTERVAL must use quotes in DuckDB - use INTERVAL '6 MONTH' not INTERVAL 6 MONTH"
        
        # Check for DATE_PART without proper casting (common error)
        if 'DATE_PART(' in sql_upper and 'CAST(' not in sql_upper:
            return False, "DATE_PART requires CAST to DATE type - use DATE_PART('month', CAST(column AS DATE))"
        
        # Check for VARCHAR date comparisons without casting (critical error)
        if '"Invoice Date"' in sql_upper and 'CAST(' not in sql_upper and ('>=' in sql_upper or '<=' in sql_upper or '>' in sql_upper or '<' in sql_upper):
            return False, "Invoice Date is VARCHAR - must cast to DATE for comparisons: CAST(\"Invoice Date\" AS DATE) >= CURRENT_DATE"
        
        # Check for incorrect column names (common error)
        if any(col in sql_upper for col in ['"DATE"', '"ASSET_NAME"', '"PRODUCT_NAME"', '"CITY"', '"REVENUE"', '"AMOUNT"', '"SALES"']):
            return False, "Use correct column names: \"Invoice Date\", \"Sku\", \"Ship To City\", \"revenue_calc\", \"transaction_type\", \"Invoice Number\""
        
        # Check for dangerous keywords in the comment-cleaned SQL
        sql_no_comments_upper = sql_no_comments.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_no_comments_upper:
                return False, f"Dangerous keyword '{keyword}' detected"
        
        # Must start with SELECT or WITH (for CTEs)
        if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
            return False, "Query must start with SELECT or WITH"
        
        # Check for proper column name quoting
        if '"' not in sql and any(col in sql_upper for col in ['INVOICE DATE', 'INVOICE AMOUNT', 'INVOICE NUMBER', 'SHIP TO CITY']):
            return False, "Column names must be quoted with double quotes"
        
        return True, ""
    
    @timeout_handler(seconds=60)
    def _call_ollama(self, prompt: str, max_retries: int = 3) -> str:
        """Call Ollama API to generate SQL with retry logic"""
        if not self.is_ollama_available:
            raise ConnectionError("Ollama is not available")
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent SQL generation
                        "top_p": 0.9,
                        "num_predict": 200,  # Further reduced for faster responses
                        "num_ctx": 2048,     # Smaller context window for speed
                        "repeat_penalty": 1.1
                    }
                }
                
                print(f"🔄 Ollama API call (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=60  # Increased timeout for complex queries
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    logger.error(f"Ollama API returned status code {response.status_code}")
                    raise Exception(f"Ollama API returned status code {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    print(f"⏱️ Ollama timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(2)  # Brief wait before retry
                    continue
                else:
                    raise Exception(f"Ollama API timeout after {max_retries} attempts")
            except requests.exceptions.ConnectionError:
                logger.error("Ollama connection lost")
                raise ConnectionError("Lost connection to Ollama service")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama API error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"⚠️ Ollama API error: {str(e)}, retrying...")
                    import time
                    time.sleep(2)
                    continue
                else:
                    raise Exception(f"Failed to call Ollama after {max_retries} attempts: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in Ollama call: {e}")
                if attempt < max_retries - 1:
                    print(f"⚠️ Unexpected error: {str(e)}, retrying...")
                    import time
                    time.sleep(2)
                    continue
                else:
                    raise Exception(f"Unexpected error after {max_retries} attempts: {e}")
    
    def _format_currency(self, amount: float) -> str:
        """Format currency in Indian format (Lakhs/Crores)"""
        if amount == 0:
            return "₹0"
        
        # Handle negative amounts
        is_negative = amount < 0
        amount = abs(amount)
        
        # Indian number formatting
        if amount >= 10000000:  # 1 crore = 10 million
            crores = amount / 10000000
            if crores >= 100:
                return f"₹{crores:.1f} Cr" if not is_negative else f"-₹{crores:.1f} Cr"
            else:
                return f"₹{crores:.2f} Cr" if not is_negative else f"-₹{crores:.2f} Cr"
        
        elif amount >= 100000:  # 1 lakh = 100 thousand
            lakhs = amount / 100000
            return f"₹{lakhs:.2f} L" if not is_negative else f"-₹{lakhs:.2f} L"
        
        elif amount >= 1000:  # Thousands
            thousands = amount / 1000
            return f"₹{thousands:.1f}K" if not is_negative else f"-₹{thousands:.1f}K"
        
        else:  # Less than 1000
            return f"₹{amount:.0f}" if not is_negative else f"-₹{amount:.0f}"
    
    def _add_intelligent_context(self, question: str, sql: str, result: pd.DataFrame) -> str:
        """Add intelligent context, insights, and analysis to SQL results"""
        try:
            # Start with base SQL response
            base_response = self._format_business_response(question, sql, result)
            
            if result.empty:
                return base_response
            
            question_lower = question.lower()
            
            # Detect if we need to add contextual information
            needs_context = any(keyword in question_lower for keyword in [
                'revenue', 'sales', 'amount', 'total', 'how much', 'what is', 'show'
            ])
            
            if not needs_context:
                return base_response
            
            # Add contextual insights
            insights = []
            
            # For revenue/sales questions, add comparison with previous period
            if 'revenue' in question_lower or 'sales' in question_lower:
                comparison = self._compare_with_previous_period(result)
                if comparison:
                    insights.append(f"📈 {comparison}")
            
            # Add top/worst performers for aggregate queries
            if len(result) == 1 and len(result.columns) == 1:
                additional_insights = self._get_additional_insights(question)
                insights.extend(additional_insights)
            
            # Add trend indicators for time-based queries
            if 'month' in question_lower or 'date' in question_lower:
                trend_insight = self._analyze_trends_from_result(result)
                if trend_insight:
                    insights.append(trend_insight)
            
            # For total revenue questions, always add context
            if 'total' in question_lower and ('revenue' in question_lower or 'sales' in question_lower):
                comparison = self._compare_with_previous_period(result)
                if comparison:
                    insights.append(comparison)
                
                # Add top city insight
                top_insight = self._get_top_city_insight()
                if top_insight:
                    insights.append(top_insight)
            
            # Format response with insights
            if insights:
                enhanced_response = base_response.replace("📊 **Query Result**:", "📊 **Answer**:")
                enhanced_response += "\n\n💡 **Insights:**\n"
                for insight in insights:
                    enhanced_response += f"{insight}\n"
                return enhanced_response
            
            return base_response
            
        except Exception as e:
            print(f"⚠️ Error adding context: {e}")
            # Return base response if context enhancement fails
            return self._format_business_response(question, sql, result)
    
    def _compare_with_previous_period(self, result: pd.DataFrame) -> str:
        """Compare current result with previous period"""
        try:
            if result.empty:
                return None
            
            # Get historical data for comparison
            historical = query_data("SELECT DATE_PART('month', CAST(\"Invoice Date\" AS DATE)) as month, SUM(\"revenue_calc\") as revenue FROM sales GROUP BY month ORDER BY month")
            
            if historical.empty or len(historical) < 2:
                return None
            
            # Get current and previous period revenue
            last_month = historical.iloc[-1]['revenue']
            prev_month = historical.iloc[-2]['revenue'] if len(historical) >= 2 else 0
            
            # Check for valid numeric values and avoid NA issues
            if pd.notna(last_month) and pd.notna(prev_month) and prev_month > 0:
                change_pct = ((last_month - prev_month) / prev_month) * 100
                direction = "up" if change_pct > 0 else "down"
                return f"Revenue is {direction} {abs(change_pct):.1f}% compared to previous month"
            
            return None
            
        except Exception as e:
            print(f"Error in comparison: {e}")
            return None
    
    def _get_top_city_insight(self) -> str:
        """Get top performing city insight"""
        try:
            top_city = query_data("SELECT \"Ship To City\", SUM(\"revenue_calc\") as revenue FROM sales GROUP BY \"Ship To City\" ORDER BY revenue DESC LIMIT 1")
            if not top_city.empty:
                city = top_city.iloc[0]['Ship To City']
                revenue = self._format_currency(top_city.iloc[0]['revenue'])
                return f"🏆 Top performing city: {city} with {revenue}"
        except Exception as e:
            print(f"Error getting top city insight: {e}")
        return None
    
    def _get_additional_insights(self, question: str) -> List[str]:
        """Get additional insights based on question type"""
        insights = []
        
        try:
            question_lower = question.lower()
            
            # Top performers
            if 'top' in question_lower or 'best' in question_lower:
                top_city = query_data("SELECT \"Ship To City\", SUM(\"revenue_calc\") as revenue FROM sales GROUP BY \"Ship To City\" ORDER BY revenue DESC LIMIT 1")
                if not top_city.empty:
                    city = top_city.iloc[0]['Ship To City']
                    revenue = self._format_currency(top_city.iloc[0]['revenue'])
                    insights.append(f"🏆 Top performing city: {city} with {revenue}")
            
            # Declining products
            if 'declining' in question_lower or 'worst' in question_lower:
                declining = query_data("SELECT \"Sku\", SUM(\"revenue_calc\") as revenue FROM sales GROUP BY \"Sku\" ORDER BY revenue ASC LIMIT 1")
                if not declining.empty:
                    sku = declining.iloc[0]['Sku']
                    revenue = self._format_currency(declining.iloc[0]['revenue'])
                    insights.append(f"📉 Lowest performing product: {sku} with {revenue}")
            
        except Exception as e:
            print(f"Error getting additional insights: {e}")
        
        return insights
    
    def _analyze_trends_from_result(self, result: pd.DataFrame) -> str:
        """Analyze trends from query result"""
        try:
            if result.empty or len(result) < 2:
                return None
            
            # Check if result has date/time columns
            date_cols = [col for col in result.columns if 'date' in col.lower() or 'month' in col.lower()]
            value_cols = [col for col in result.columns if 'revenue' in col.lower() or 'amount' in col.lower()]
            
            if date_cols and value_cols:
                # Simple trend calculation
                if len(result) >= 2:
                    first_value = result[value_cols[0]].iloc[0]
                    last_value = result[value_cols[0]].iloc[-1]
                    
                    if first_value and first_value > 0:
                        trend_pct = ((last_value - first_value) / abs(first_value)) * 100
                        direction = "increasing" if trend_pct > 0 else "decreasing"
                        return f"📊 Trend: {direction} by {abs(trend_pct):.1f}% over this period"
            
            return None
            
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return None
    
    def _format_business_response(self, question: str, sql: str, result: pd.DataFrame) -> str:
        """Format query result into business-friendly response"""
        
        if result.empty:
            return f"📊 **Query Result**: No data found for your question.\n\n**Question**: {question}\n**SQL**: `{sql}`"
        
        # Get the first row for single-value results
        if len(result) == 1 and len(result.columns) == 1:
            value = result.iloc[0, 0]
            column_name = result.columns[0]
            
            # Format currency values
            if 'amount' in column_name.lower() or 'revenue' in column_name.lower():
                if isinstance(value, (int, float)):
                    formatted_value = self._format_currency(value)
                else:
                    formatted_value = str(value)
            else:
                formatted_value = f"{value:,}" if isinstance(value, (int, float)) else str(value)
            
            return f"📊 **Answer**: {formatted_value}\n\n**Question**: {question}\n**SQL**: `{sql}`"
        
        # Format table results
        response = f"📊 **Query Result**:\n\n"
        
        # Format each row
        for idx, row in result.iterrows():
            row_text = []
            for col, val in row.items():
                if 'amount' in col.lower() or 'revenue' in col.lower():
                    if isinstance(val, (int, float)):
                        formatted_val = self._format_currency(val)
                    else:
                        formatted_val = str(val)
                else:
                    formatted_val = f"{val:,}" if isinstance(val, (int, float)) else str(val)
                
                row_text.append(f"**{col}**: {formatted_val}")
            
            response += f"• {' | '.join(row_text)}\n"
        
        response += f"\n**Question**: {question}\n**SQL**: `{sql}`"
        return response
    
    @timeout_handler(seconds=45)
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return formatted response
        
        Args:
            question: Natural language business question
            
        Returns:
            Dictionary with response, SQL, and metadata
        """
        try:
            print(f"🤔 Processing question: {question}")
            
            # Check cache first for repeated questions
            question_key = question.lower().strip()
            if question_key in self.query_cache:
                print("📋 Using cached result")
                cached_result = self.query_cache[question_key]
                cached_result['cached'] = True
                return cached_result
            
            # Check if Ollama is available
            if not self.is_ollama_available:
                return self._handle_offline_mode(question)
            
            # Check if this is a predictive analytics question
            if self.predictive_analytics.is_predictive_question(question):
                print("🔮 Detected predictive analytics question")
                result = self._handle_predictive_question(question)
            else:
                # Use intelligent SQL generation for all other questions
                print("🧠 Using intelligent SQL generation...")
                result = self._handle_intelligent_sql_question(question)
            
            # Cache successful results (limit cache size)
            if result['success'] and len(self.query_cache) < 50:
                self.query_cache[question_key] = result.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': 'I encountered an unexpected error. Please try rephrasing your question.',
                'question': question,
                'sql': None,
                'error_type': 'unexpected_error'
            }
    
    def _handle_offline_mode(self, question: str) -> Dict[str, Any]:
        """Handle questions when Ollama is not available"""
        try:
            # Try to provide basic responses without LLM
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['total', 'revenue', 'sales']):
                # Try to get basic revenue data
                try:
                    result = query_data("SELECT SUM(\"revenue_calc\") as total_revenue FROM sales")
                    if not result.empty:
                        total_revenue = result.iloc[0]['total_revenue']
                        formatted_revenue = self._format_currency(total_revenue)
                        return {
                            'success': True,
                            'response': f"📊 **Total Revenue**: {formatted_revenue}\n\n*Note: AI features are limited when Ollama is not available.*",
                            'question': question,
                            'sql': 'SELECT SUM("revenue_calc") as total_revenue FROM sales',
                            'offline_mode': True
                        }
                except Exception as e:
                    logger.error(f"Error in offline mode revenue query: {e}")
            
            # Generic offline response
            return {
                'success': False,
                'error': 'AI assistant is currently unavailable. Please ensure Ollama is running for full functionality.',
                'question': question,
                'error_type': 'offline_mode',
                'offline_mode': True
            }
            
        except Exception as e:
            logger.error(f"Error in offline mode: {e}")
            return {
                'success': False,
                'error': 'AI assistant is currently unavailable. Please ensure Ollama is running.',
                'question': question,
                'error_type': 'offline_mode'
            }
    
    def _handle_intelligent_sql_question(self, question: str) -> Dict[str, Any]:
        """
        Handle any business question using intelligent SQL generation
        
        Args:
            question: Natural language business question
            
        Returns:
            Dictionary with response, SQL, and metadata
        """
        try:
            print(f"🧠 Analyzing question: {question}")
            
            # Generate intelligent SQL using LLM
            sql = self._generate_intelligent_sql(question)
            if not sql:
                logger.warning(f"Could not generate SQL for question: {question}")
                return {
                    'success': False,
                    'error': 'I couldn\'t understand that question. Could you rephrase it?',
                    'question': question,
                    'error_type': 'sql_generation_failed'
                }
            
            print(f"🔍 Generated SQL: {sql}")
            
            # Validate SQL safety
            is_safe, error_msg = self._validate_sql_safety(sql)
            if not is_safe:
                logger.warning(f"SQL safety validation failed: {error_msg}")
                return {
                    'success': False,
                    'error': 'I couldn\'t understand that question. Could you rephrase it?',
                    'question': question,
                    'sql': sql,
                    'error_type': 'sql_safety_failed'
                }
            
            # Execute SQL query
            print("📊 Executing SQL query...")
            try:
                result = query_data(sql)
                
                # Check result size for performance
                if len(result) > 1000:
                    logger.warning(f"Large result set ({len(result)} rows), limiting for performance")
                    result = result.head(1000)  # Limit to 1000 rows for performance
                    
            except Exception as e:
                logger.error(f"Database query execution failed: {e}")
                return {
                    'success': False,
                    'error': 'Unable to retrieve data. Please try again.',
                    'question': question,
                    'sql': sql,
                    'error_type': 'database_query_failed'
                }
            
            # Generate intelligent analysis of results using LLM
            print("🧠 Generating intelligent analysis...")
            try:
                intelligent_response = self._analyze_with_llm(question, result)
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                # Fallback to basic formatting
                intelligent_response = self._format_conversational_fallback(question, result)
            
            return {
                'success': True,
                'response': intelligent_response,
                'question': question,
                'sql': sql,
                'result_rows': len(result),
                'result_columns': len(result.columns) if not result.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing intelligent SQL question '{question}': {e}\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': 'I encountered an unexpected error. Please try rephrasing your question.',
                'question': question,
                'sql': None,
                'error_type': 'unexpected_error'
            }
    
    def _generate_intelligent_sql(self, question: str) -> Optional[str]:
        """
        Generate SQL query using LLM understanding of business questions
        
        Args:
            question: Natural language business question
            
        Returns:
            Generated SQL query or None if failed
        """
        try:
            # Create comprehensive prompt for LLM
            prompt = self._create_intelligent_sql_prompt(question)
            
            # Call Ollama to generate SQL
            print("🤖 Generating SQL with intelligent understanding...")
            try:
                sql_response = self._call_ollama(prompt)
            except ConnectionError as e:
                logger.error(f"Ollama connection error in SQL generation: {e}")
                return None
            except Exception as e:
                logger.error(f"Ollama error in SQL generation: {e}")
                return None
            
            # Extract SQL from response
            sql = self._extract_sql_from_response(sql_response)
            
            if not sql:
                logger.warning("No SQL found in LLM response")
                return None
            
            # Clean up SQL
            sql = self._clean_sql_query(sql)
            
            return sql
            
        except Exception as e:
            logger.error(f"Error generating intelligent SQL: {e}\n{traceback.format_exc()}")
            return None
    
    def _create_intelligent_sql_prompt(self, question: str) -> str:
        """
        Create comprehensive prompt for intelligent SQL generation
        
        Args:
            question: Business question to analyze
            
        Returns:
            Detailed prompt for LLM
        """
        schema_info = self.database_schema
        if not schema_info:
            return f"""
            Generate a SQL query for this business question: "{question}"
            
            Database: DuckDB
            Table: sales
            
            Important:
            - Use double quotes around column names
            - Format dates as 'YYYY-MM-DD'
            - Only use SELECT statements (no INSERT, UPDATE, DELETE, DROP)
            - Return only the SQL query, no explanations
            """
        
        columns = schema_info['columns']
        sample_data = schema_info['sample_data']
        stats = schema_info['statistics']
        
        prompt = f"""
        You are an expert business analyst and SQL developer. Your task is to understand business questions and generate appropriate SQL queries.

        BUSINESS QUESTION: "{question}"

        DATABASE SCHEMA:
        Table: sales
        Columns: {', '.join(columns)}
        
        SAMPLE DATA STRUCTURE:
        {json.dumps(sample_data[:3], indent=2) if sample_data else 'No sample data available'}
        
        DATABASE STATISTICS:
        - Total rows: {stats.get('total_rows', 'Unknown')}
        - Unique orders: {stats.get('unique_orders', 'Unknown')}
        - Unique SKUs: {stats.get('unique_skus', 'Unknown')}
        - Unique cities: {stats.get('unique_cities', 'Unknown')}
        - Date range: {stats.get('earliest_date', 'Unknown')} to {stats.get('latest_date', 'Unknown')}

        BUSINESS CONTEXT:
        This is an e-commerce sales database. Key business concepts:
        - Revenue: Use "revenue_calc" column for accurate revenue calculations
        - Orders: Count distinct "Invoice Number" for order counts
        - Products: Use "Sku" and "Asin" for product identification
        - Regions: Use "Ship To City" for geographic analysis
        - Time: Use "Invoice Date" for temporal analysis
        - Transaction Types: "transaction_type" includes Shipment, Refund, FreeReplacement, etc.

        BUSINESS QUESTION ANALYSIS:
        Understand what the user is asking for:
        - "Loss making products" = Products with high costs, refunds, or negative margins
        - "Best sellers" = Products with highest revenue or order volume
        - "Top performers" = Products with best overall performance metrics
        - "Declining sales" = Products with decreasing trends over time
        - "Regional comparison" = Analysis grouped by "Ship To City"
        - "Monthly trends" = Time-series analysis by month
        - "Average order value" = Revenue per order calculation
        - "High refund rates" = Products with many refunds relative to sales

        SQL GENERATION RULES:
        1. Use double quotes around all column names (e.g., "Invoice Date", "Invoice Amount")
        2. Use DATE_PART('month', CAST("Invoice Date" AS DATE)) for month extraction
        3. Use DATE_PART('year', CAST("Invoice Date" AS DATE)) for year extraction
        4. Only use SELECT statements (no INSERT, UPDATE, DELETE, DROP, ALTER)
        5. Use proper DuckDB SQL syntax
        6. For revenue calculations, prefer "revenue_calc" over "Invoice Amount"
        7. For order counts, use COUNT(DISTINCT "Invoice Number")
        8. For product analysis, group by "Sku" or "Asin"
        9. For regional analysis, group by "Ship To City"
        10. For time analysis, use "Invoice Date" with proper date functions
        11. For transaction filtering, use "transaction_type" column
        12. ALWAYS use LIMIT to control result size (max 20 rows for performance)
        13. Use ORDER BY to sort results meaningfully
        14. Use WHERE clauses to filter data appropriately
        15. Use CASE statements for conditional logic
        16. Use subqueries or CTEs for complex analysis
        17. NEVER use window functions in HAVING or WHERE clauses
        18. For trend analysis, compare periods using separate subqueries
        19. For performance: Filter by recent dates when possible (last 6 months)
        20. For performance: Use specific transaction types instead of all data
        21. For performance: Aggregate data at appropriate levels (monthly vs daily)
        22. For performance: Use indexes-friendly WHERE clauses
        23. CRITICAL: Use DuckDB date functions - DATE_SUB() is NOT supported
        24. For date filtering: Use "Invoice Date" >= CURRENT_DATE - INTERVAL 6 MONTH
        25. For date arithmetic: Use INTERVAL '6 MONTH' not INTERVAL 6 MONTH
        26. For current date: Use CURRENT_DATE (not NOW())

        EXAMPLES OF BUSINESS QUESTION → SQL MAPPING:
        
        Question: "What products are making me lose money?"
        SQL: SELECT "Sku", SUM("revenue_calc") as total_revenue, 
             COUNT(CASE WHEN "transaction_type" = 'Refund' THEN 1 END) as refund_count,
             COUNT(DISTINCT "Invoice Number") as order_count
             FROM sales WHERE "Sku" IS NOT NULL AND CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
             GROUP BY "Sku" 
             HAVING refund_count > 0 OR total_revenue < 0
             ORDER BY total_revenue ASC LIMIT 10

        Question: "Show me top 10 best sellers"
        SQL: SELECT "Sku", SUM("revenue_calc") as total_revenue,
             COUNT(DISTINCT "Invoice Number") as order_count
             FROM sales WHERE "transaction_type" IN ('Shipment', 'Fulfillment') 
             AND CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
             GROUP BY "Sku" 
             ORDER BY total_revenue DESC LIMIT 10

        Question: "Which region has declining sales?"
        SQL: WITH current_month AS (
             SELECT "Ship To City", SUM("revenue_calc") as revenue
             FROM sales WHERE DATE_PART('month', CAST("Invoice Date" AS DATE)) = DATE_PART('month', CURRENT_DATE)
             AND "transaction_type" IN ('Shipment', 'Fulfillment')
             GROUP BY "Ship To City"
             ),
             previous_month AS (
             SELECT "Ship To City", SUM("revenue_calc") as revenue
             FROM sales WHERE DATE_PART('month', CAST("Invoice Date" AS DATE)) = DATE_PART('month', CURRENT_DATE) - 1
             AND "transaction_type" IN ('Shipment', 'Fulfillment')
             GROUP BY "Ship To City"
             )
             SELECT c."Ship To City", c.revenue as current_revenue, p.revenue as previous_revenue,
             ((c.revenue - p.revenue) / NULLIF(p.revenue, 0)) * 100 as change_pct
             FROM current_month c LEFT JOIN previous_month p ON c."Ship To City" = p."Ship To City"
             WHERE c.revenue < p.revenue ORDER BY change_pct ASC LIMIT 10

        Question: "What's my average order value?"
        SQL: SELECT AVG("revenue_calc") as avg_order_value,
             COUNT(DISTINCT "Invoice Number") as total_orders,
             SUM("revenue_calc") as total_revenue
             FROM sales WHERE "transaction_type" IN ('Shipment', 'Fulfillment')
             AND CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'

        Question: "Products with high refund rates"
        SQL: SELECT "Sku", 
             COUNT(DISTINCT CASE WHEN "transaction_type" = 'Refund' THEN "Invoice Number" END) as refund_orders,
             COUNT(DISTINCT "Invoice Number") as total_orders,
             (COUNT(DISTINCT CASE WHEN "transaction_type" = 'Refund' THEN "Invoice Number" END) * 100.0 / 
              COUNT(DISTINCT "Invoice Number")) as refund_rate
             FROM sales WHERE "Sku" IS NOT NULL 
             AND CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
             GROUP BY "Sku"
             HAVING refund_rate > 20
             ORDER BY refund_rate DESC LIMIT 10

        CRITICAL DUCKDB COMPATIBILITY RULES:
        - NEVER use DATE_SUB() function - use CURRENT_DATE - INTERVAL '6 MONTH' instead
        - NEVER use EXTRACT() function - use DATE_PART('month', CAST(column AS DATE)) instead
        - NEVER use NOW() function - use CURRENT_DATE instead
        - ALWAYS use INTERVAL '6 MONTH' (with quotes) not INTERVAL 6 MONTH
        - ALWAYS use double quotes around column names
        - ALWAYS use CAST(column AS DATE) with DATE_PART functions
        - CRITICAL: "Invoice Date" is VARCHAR - must cast to DATE for comparisons
        - For date filtering: CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
        - For month extraction: DATE_PART('month', CAST("Invoice Date" AS DATE))
        - For year extraction: DATE_PART('year', CAST("Invoice Date" AS DATE))
        - For day extraction: DATE_PART('day', CAST("Invoice Date" AS DATE))
        - For date comparisons: ALWAYS cast VARCHAR dates to DATE type
        - For date sorting: ORDER BY CAST("Invoice Date" AS DATE)

        COMMON DUCKDB PATTERNS FOR ALL QUESTION TYPES:
        
        Revenue Questions:
        SELECT "Sku", SUM("revenue_calc") as total_revenue FROM sales 
        WHERE "transaction_type" IN ('Shipment', 'Fulfillment') 
        AND CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
        GROUP BY "Sku" ORDER BY total_revenue DESC LIMIT 10
        
        Order Count Questions:
        SELECT "Sku", COUNT(DISTINCT "Invoice Number") as order_count FROM sales 
        WHERE CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
        GROUP BY "Sku" ORDER BY order_count DESC LIMIT 10
        
        Date Range Questions:
        SELECT DATE_PART('month', CAST("Invoice Date" AS DATE)) as month, 
               SUM("revenue_calc") as revenue FROM sales 
        WHERE CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
        GROUP BY month ORDER BY month
        
        Regional Questions:
        SELECT "Ship To City", SUM("revenue_calc") as revenue FROM sales 
        WHERE "transaction_type" IN ('Shipment', 'Fulfillment')
        AND CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
        GROUP BY "Ship To City" ORDER BY revenue DESC LIMIT 10
        
        Product Performance Questions:
        SELECT "Sku", SUM("revenue_calc") as revenue, 
               COUNT(DISTINCT "Invoice Number") as orders,
               COUNT(CASE WHEN "transaction_type" = 'Refund' THEN 1 END) as refunds
        FROM sales WHERE CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
        GROUP BY "Sku" ORDER BY revenue ASC LIMIT 10

        MANDATORY REQUIREMENTS:
        1. ALWAYS use "Invoice Date" column (not "date" or other variations)
        2. ALWAYS cast "Invoice Date" to DATE: CAST("Invoice Date" AS DATE)
        3. ALWAYS use proper date filtering: CAST("Invoice Date" AS DATE) >= CURRENT_DATE - INTERVAL '6 MONTH'
        4. ALWAYS use "revenue_calc" for revenue calculations
        5. ALWAYS use "Sku" for product identification
        6. ALWAYS use "transaction_type" for filtering transactions
        7. ALWAYS use "Invoice Number" for order counting
        8. ALWAYS use "Ship To City" for regional analysis

        INSTRUCTIONS:
        1. Analyze the business question carefully
        2. Identify what data is needed
        3. Determine appropriate grouping and filtering
        4. Generate a DuckDB-compatible SQL query that answers the question
        5. Return ONLY the SQL query, no explanations or markdown
        6. Ensure the query follows ALL DuckDB compatibility rules above
        7. MANDATORY: Use CAST("Invoice Date" AS DATE) for all date operations

        Generate the DuckDB-compatible SQL query for: "{question}"
        """
        
        return prompt
    
    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """
        Extract SQL query from LLM response
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted SQL query or None
        """
        try:
            # Remove any markdown formatting
            sql = response.strip()
            
            # Handle markdown code blocks
            if sql.startswith('```sql'):
                sql = sql.replace('```sql', '').replace('```', '').strip()
            elif sql.startswith('```'):
                sql = sql.replace('```', '').strip()
            
            # Extract SQL from explanatory text (look for ```sql blocks)
            if '```sql' in sql:
                sql_start = sql.find('```sql') + 6
                sql_end = sql.find('```', sql_start)
                if sql_end > sql_start:
                    sql = sql[sql_start:sql_end].strip()
            
            # Clean up common LLM artifacts
            sql = re.sub(r'^Here\'s the SQL query:\s*', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'^The SQL query is:\s*', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'^SQL:\s*', '', sql, flags=re.IGNORECASE)
            
            # Ensure it starts with SELECT or WITH
            if not (sql.upper().startswith('SELECT') or sql.upper().startswith('WITH')):
                return None
            
            return sql
            
        except Exception as e:
            print(f"❌ Error extracting SQL: {e}")
            return None
    
    def _clean_sql_query(self, sql: str) -> str:
        """
        Clean and normalize SQL query
        
        Args:
            sql: Raw SQL query
            
        Returns:
            Cleaned SQL query
        """
        try:
            # Remove extra whitespace and normalize
            sql = re.sub(r'\s+', ' ', sql.strip())
            
            # Ensure proper spacing around keywords
            sql = re.sub(r'\bSELECT\b', 'SELECT ', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bFROM\b', ' FROM ', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bWHERE\b', ' WHERE ', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bGROUP BY\b', ' GROUP BY ', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bORDER BY\b', ' ORDER BY ', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bLIMIT\b', ' LIMIT ', sql, flags=re.IGNORECASE)
            sql = re.sub(r'\bHAVING\b', ' HAVING ', sql, flags=re.IGNORECASE)
            
            # Remove trailing semicolon if present
            sql = sql.rstrip(';')
            
            return sql
            
        except Exception as e:
            print(f"❌ Error cleaning SQL: {e}")
            return sql
    
    def _analyze_with_llm(self, question: str, result: pd.DataFrame) -> str:
        """
        Use LLM to analyze SQL results and generate business insights
        
        Args:
            question: Original business question
            result: SQL query result DataFrame
            
        Returns:
            Intelligent business analysis with insights and recommendations
        """
        try:
            if result.empty:
                return f"""📊 **No Data Found**

I couldn't find any data matching your question: "{question}"

This might mean:
• No records exist for the criteria you're looking for
• The data might be in a different format than expected
• Try rephrasing your question or being more specific"""
            
            # Convert DataFrame to a readable format for LLM
            data_summary = self._prepare_data_for_llm(result)
            
            # Generate LLM prompt for analysis
            prompt = self._create_analysis_prompt(question, data_summary, result)
            
            # Call Ollama for intelligent analysis
            print("🤖 Generating insights with LLM...")
            try:
                analysis = self._call_ollama(prompt)
                return analysis
            except ConnectionError as e:
                logger.error(f"Ollama connection error in analysis: {e}")
                return self._format_conversational_fallback(question, result)
            except Exception as e:
                logger.error(f"Ollama error in analysis: {e}")
                return self._format_conversational_fallback(question, result)
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}\n{traceback.format_exc()}")
            # Fallback to conversational formatting
            return self._format_conversational_fallback(question, result)
    
    def _prepare_data_for_llm(self, result: pd.DataFrame) -> str:
        """Convert DataFrame to readable format for LLM analysis with business context"""
        try:
            # Limit to first 8 rows for efficiency (reduced from 15)
            display_rows = min(8, len(result))
            summary = f"Total records: {len(result)}\n\n"
            summary += "Key business data:\n"
            
            # Show column headers with business context
            summary += f"Columns: {', '.join(result.columns)}\n\n"
            
            # Show sample data with business-friendly descriptions (more concise)
            for i in range(display_rows):
                row = result.iloc[i]
                row_data = []
                for col in result.columns:
                    value = row[col]
                    if isinstance(value, (int, float)):
                        if 'revenue' in col.lower() or 'amount' in col.lower():
                            row_data.append(f"{col}={self._format_currency(value)}")
                        elif 'rate' in col.lower() or 'percent' in col.lower():
                            row_data.append(f"{col}={value:.1f}%")
                        elif 'count' in col.lower() or 'orders' in col.lower():
                            row_data.append(f"{col}={value:,}")
                        else:
                            row_data.append(f"{col}={value:,}")
                    else:
                        row_data.append(f"{col}={str(value)[:20]}")  # Truncate long strings
                summary += f"{i+1}. " + " | ".join(row_data) + "\n"
            
            if len(result) > display_rows:
                summary += f"\n... and {len(result) - display_rows} more items"
            
            # Add concise business context
            summary += f"\n\nContext: E-commerce sales data with products, revenue, orders."
            
            return summary
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return str(result)
    
    def _create_analysis_prompt(self, question: str, data_summary: str, result: pd.DataFrame) -> str:
        """Create prompt for LLM to analyze results and generate conversational insights"""
        
        prompt = f"""
You are a business advisor analyzing data for: "{question}"

DATA: {data_summary}

Provide a concise, conversational business analysis:

1. Start with: "Here's what I found in your data..."
2. Highlight key insights naturally
3. Focus on actionable recommendations
4. Use simple language, avoid jargon
5. Keep response under 200 words
6. Be specific about what needs attention

Format:
- Key findings first
- Specific recommendations
- Next steps

Generate a brief, actionable business response:
"""
        
        return prompt
    
    def _format_conversational_fallback(self, question: str, result: pd.DataFrame) -> str:
        """Fallback conversational response when LLM analysis fails"""
        try:
            if result.empty:
                return f"""I couldn't find any data matching your question: "{question}"

This might mean:
• No records exist for what you're looking for
• The data might be in a different format than expected
• Try rephrasing your question or being more specific"""
            
            # Create a simple conversational response
            response = f"""Let me share what I found for your question: "{question}"

I've got {len(result)} results to show you:

"""
            
            # Show first few items conversationally
            display_rows = min(5, len(result))
            for i in range(display_rows):
                row = result.iloc[i]
                response += f"**Item {i+1}:** "
                row_text = []
                for col in result.columns:
                    value = row[col]
                    if isinstance(value, (int, float)):
                        if 'revenue' in col.lower() or 'amount' in col.lower():
                            row_text.append(f"{col}: {self._format_currency(value)}")
                        else:
                            row_text.append(f"{col}: {value:,}")
                    else:
                        row_text.append(f"{col}: {value}")
                response += " | ".join(row_text) + "\n"
            
            if len(result) > display_rows:
                response += f"\n... and {len(result) - display_rows} more items"
            
            response += f"""

This gives you a snapshot of what's happening. Would you like me to dive deeper into any specific aspect?"""
            
            return response
            
        except Exception as e:
            print(f"Error in conversational fallback: {e}")
            return f"I found some data for your question: '{question}' but had trouble formatting it properly. There are {len(result)} results available."
    
    def _format_intelligent_response(self, question: str, sql: str, result: pd.DataFrame) -> str:
        """
        Format SQL result into intelligent business response
        
        Args:
            question: Original business question
            sql: Generated SQL query
            result: Query result DataFrame
            
        Returns:
            Formatted business response
        """
        try:
            if result.empty:
                return f"""📊 **No Data Found**

I couldn't find any data matching your question: "{question}"

This might mean:
• No records exist for the criteria you're looking for
• The data might be in a different format than expected
• Try rephrasing your question or being more specific

**Generated Query**: `{sql}`"""
            
            # Analyze the question to determine response format
            question_lower = question.lower()
            
            # Determine response type based on question
            if any(word in question_lower for word in ['loss', 'losing', 'problem', 'issue', 'bad']):
                return self._format_problem_analysis_response(question, result)
            elif any(word in question_lower for word in ['best', 'top', 'highest', 'good', 'perform']):
                return self._format_performance_response(question, result)
            elif any(word in question_lower for word in ['average', 'mean', 'typical']):
                return self._format_average_response(question, result)
            elif any(word in question_lower for word in ['trend', 'declining', 'growing', 'change']):
                return self._format_trend_response(question, result)
            else:
                return self._format_general_response(question, result)
                
        except Exception as e:
            print(f"❌ Error formatting response: {e}")
            return f"📊 **Query Result**: {len(result)} rows returned\n\n**Question**: {question}\n**SQL**: `{sql}`"
    
    def _format_problem_analysis_response(self, question: str, result: pd.DataFrame) -> str:
        """Format response for problem analysis questions"""
        response = f"""⚠️ **Problem Analysis**

Based on your question: "{question}"

I found {len(result)} items that need attention:

"""
        
        for i, (_, row) in enumerate(result.iterrows(), 1):
            response += f"🔴 **{i}. {row.iloc[0] if len(row) > 0 else 'Item'}**\n"
            
            # Add key metrics
            for col in result.columns[1:]:
                value = row[col]
                if isinstance(value, (int, float)):
                    if 'revenue' in col.lower() or 'amount' in col.lower():
                        response += f"• {col}: {self._format_currency(value)}\n"
                    elif 'rate' in col.lower() or 'percent' in col.lower():
                        response += f"• {col}: {value:.1f}%\n"
                    else:
                        response += f"• {col}: {value:,}\n"
                else:
                    response += f"• {col}: {value}\n"
            response += "\n"
        
        response += """💡 **Recommendations:**
• Review these items for potential issues
• Consider corrective actions based on the metrics
• Monitor trends to prevent future problems"""
        
        return response
    
    def _format_performance_response(self, question: str, result: pd.DataFrame) -> str:
        """Format response for performance analysis questions"""
        response = f"""🚀 **Performance Analysis**

Based on your question: "{question}"

Here are your top {len(result)} performers:

"""
        
        for i, (_, row) in enumerate(result.iterrows(), 1):
            response += f"📈 **{i}. {row.iloc[0] if len(row) > 0 else 'Item'}**\n"
            
            # Add key metrics
            for col in result.columns[1:]:
                value = row[col]
                if isinstance(value, (int, float)):
                    if 'revenue' in col.lower() or 'amount' in col.lower():
                        response += f"• {col}: {self._format_currency(value)}\n"
                    elif 'rate' in col.lower() or 'percent' in col.lower():
                        response += f"• {col}: {value:.1f}%\n"
                    else:
                        response += f"• {col}: {value:,}\n"
                else:
                    response += f"• {col}: {value}\n"
            response += "\n"
        
        response += """💡 **Insights:**
• These are your best-performing items
• Consider increasing focus on top performers
• Use these as benchmarks for other items"""
        
        return response
    
    def _format_average_response(self, question: str, result: pd.DataFrame) -> str:
        """Format response for average/statistical questions"""
        if len(result) == 1 and len(result.columns) == 1:
            value = result.iloc[0, 0]
            if isinstance(value, (int, float)):
                if 'revenue' in question.lower() or 'amount' in question.lower():
                    formatted_value = self._format_currency(value)
                else:
                    formatted_value = f"{value:,.2f}"
            else:
                formatted_value = str(value)
            
            return f"""📊 **Answer**

{question}

**Result**: {formatted_value}

This represents the average value across all relevant data points."""
        
        return self._format_general_response(question, result)
    
    def _format_trend_response(self, question: str, result: pd.DataFrame) -> str:
        """Format response for trend analysis questions"""
        response = f"""📈 **Trend Analysis**

Based on your question: "{question}"

Trend analysis results:

"""
        
        for i, (_, row) in enumerate(result.iterrows(), 1):
            response += f"📊 **{i}. {row.iloc[0] if len(row) > 0 else 'Item'}**\n"
            
            # Add key metrics
            for col in result.columns[1:]:
                value = row[col]
                if isinstance(value, (int, float)):
                    if 'revenue' in col.lower() or 'amount' in col.lower():
                        response += f"• {col}: {self._format_currency(value)}\n"
                    elif 'change' in col.lower() or 'trend' in col.lower():
                        direction = "📈" if value > 0 else "📉" if value < 0 else "➡️"
                        response += f"• {col}: {direction} {value:.1f}%\n"
                    elif 'rate' in col.lower() or 'percent' in col.lower():
                        response += f"• {col}: {value:.1f}%\n"
                    else:
                        response += f"• {col}: {value:,}\n"
                else:
                    response += f"• {col}: {value}\n"
            response += "\n"
        
        response += """💡 **Trend Insights:**
• Monitor these trends regularly
• Take action on declining trends
• Capitalize on positive trends"""
        
        return response
    
    def _format_general_response(self, question: str, result: pd.DataFrame) -> str:
        """Format general response for any question"""
        response = f"""📊 **Query Results**

Question: "{question}"

Found {len(result)} results:

"""
        
        # Show first few rows
        display_rows = min(5, len(result))
        for i in range(display_rows):
            row = result.iloc[i]
            response += f"**{i+1}.** "
            row_text = []
            for col in result.columns:
                value = row[col]
                if isinstance(value, (int, float)):
                    if 'revenue' in col.lower() or 'amount' in col.lower():
                        row_text.append(f"{col}: {self._format_currency(value)}")
                    else:
                        row_text.append(f"{col}: {value:,}")
                else:
                    row_text.append(f"{col}: {value}")
            response += " | ".join(row_text) + "\n"
        
        if len(result) > display_rows:
            response += f"\n... and {len(result) - display_rows} more results"
        
        return response
    
    def _handle_predictive_question(self, question: str) -> Dict[str, Any]:
        """Handle predictive analytics questions"""
        try:
            print("🔮 Processing predictive analytics question...")
            
            # Get historical data for analysis
            historical_data = query_data("SELECT * FROM sales ORDER BY \"Invoice Date\"")
            
            if historical_data.empty:
                return {
                    'success': False,
                    'error': 'No historical data available for analysis',
                    'question': question,
                    'analysis_type': 'predictive'
                }
            
            question_lower = question.lower()
            
            # Route to appropriate analysis based on question type
            if any(keyword in question_lower for keyword in ['predict', 'forecast', 'next month', 'next quarter', 'will be']):
                return self._handle_forecasting_question(question, historical_data)
            elif any(keyword in question_lower for keyword in ['trend', 'growing', 'declining', 'pattern']):
                return self._handle_trend_analysis_question(question, historical_data)
            elif any(keyword in question_lower for keyword in ['recommend', 'should i', 'restock', 'inventory']):
                return self._handle_recommendation_question(question, historical_data)
            elif any(keyword in question_lower for keyword in ['anomaly', 'unusual', 'spike', 'drop']):
                return self._handle_anomaly_question(question, historical_data)
            elif any(keyword in question_lower for keyword in ['seasonal', 'monthly', 'daily', 'pattern']):
                return self._handle_seasonal_question(question, historical_data)
            else:
                return self._handle_general_predictive_question(question, historical_data)
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error in predictive analysis: {str(e)}",
                'question': question,
                'analysis_type': 'predictive'
            }
    
    def _handle_forecasting_question(self, question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Handle revenue forecasting questions"""
        try:
            # Determine forecast period
            periods = 1
            if 'next quarter' in question.lower() or 'next 3 months' in question.lower():
                periods = 3
            elif 'next year' in question.lower() or 'next 12 months' in question.lower():
                periods = 12
            
            # Get forecast
            forecast_result = self.predictive_analytics.forecast_revenue(data, periods)
            
            if forecast_result.get('error'):
                return {
                    'success': False,
                    'error': forecast_result['error'],
                    'question': question,
                    'analysis_type': 'forecasting'
                }
            
            # Format response
            forecast_amount = forecast_result['forecast']
            confidence = forecast_result['confidence']
            trend = forecast_result['trend']
            pct_change = forecast_result['pct_change']
            
            period_name = "month" if periods == 1 else f"{periods} months"
            
            response = f"🔮 **Revenue Forecast for Next {period_name.title()}:**\n\n"
            response += f"**Predicted Revenue:** {self._format_currency(forecast_amount)}\n"
            response += f"**Confidence Level:** {confidence:.1%}\n"
            response += f"**Range:** {self._format_currency(forecast_result['lower_bound'])} - {self._format_currency(forecast_result['upper_bound'])}\n\n"
            
            if trend == 'growing':
                response += f"📈 **Trend Analysis:** Revenue is growing ({pct_change:.1f}% monthly change)\n"
            elif trend == 'declining':
                response += f"📉 **Trend Analysis:** Revenue is declining ({pct_change:.1f}% monthly change)\n"
            else:
                response += f"📊 **Trend Analysis:** Revenue is stable ({pct_change:.1f}% monthly change)\n"
            
            response += f"\n**Last Month Revenue:** {self._format_currency(forecast_result['last_month_revenue'])}\n"
            response += f"\n**Question:** {question}\n"
            response += f"**Analysis Type:** Revenue Forecasting"
            
            return {
                'success': True,
                'response': response,
                'question': question,
                'analysis_type': 'forecasting',
                'forecast_data': forecast_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Forecasting error: {str(e)}",
                'question': question,
                'analysis_type': 'forecasting'
            }
    
    def _handle_trend_analysis_question(self, question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Handle trend analysis questions"""
        try:
            # Analyze overall revenue trend
            trend_analysis = self.predictive_analytics.analyze_trends(data, 'revenue_calc')
            
            response = f"📊 **Trend Analysis:**\n\n"
            
            if trend_analysis.get('trend') == 'growing':
                response += f"📈 **Revenue Trend:** Growing ({trend_analysis.get('pct_change', 0):.1f}% change)\n"
                response += f"**Confidence:** {trend_analysis.get('confidence', 0):.1%}\n"
            elif trend_analysis.get('trend') == 'declining':
                response += f"📉 **Revenue Trend:** Declining ({trend_analysis.get('pct_change', 0):.1f}% change)\n"
                response += f"**Confidence:** {trend_analysis.get('confidence', 0):.1%}\n"
            else:
                response += f"📊 **Revenue Trend:** Stable ({trend_analysis.get('pct_change', 0):.1f}% change)\n"
                response += f"**Confidence:** {trend_analysis.get('confidence', 0):.1%}\n"
            
            # Analyze product trends
            product_analysis = self.predictive_analytics.analyze_product_performance(data)
            
            if product_analysis.get('declining_products'):
                response += f"\n⚠️ **Declining Products:**\n"
                for sku, metrics in product_analysis['declining_products'][:3]:
                    response += f"• **{sku}:** {metrics['pct_change']:.1f}% decline\n"
            
            if product_analysis.get('top_performers'):
                response += f"\n🏆 **Top Performers:**\n"
                for sku, metrics in product_analysis['top_performers'][:3]:
                    response += f"• **{sku}:** {self._format_currency(metrics['total_revenue'])}\n"
            
            response += f"\n**Question:** {question}\n"
            response += f"**Analysis Type:** Trend Analysis"
            
            return {
                'success': True,
                'response': response,
                'question': question,
                'analysis_type': 'trend_analysis',
                'trend_data': trend_analysis,
                'product_data': product_analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Trend analysis error: {str(e)}",
                'question': question,
                'analysis_type': 'trend_analysis'
            }
    
    def _handle_recommendation_question(self, question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Handle recommendation questions"""
        try:
            product_analysis = self.predictive_analytics.analyze_product_performance(data)
            recommendations = product_analysis.get('recommendations', [])
            
            response = f"💡 **Business Recommendations:**\n\n"
            
            if not recommendations:
                response += "No specific recommendations at this time. All products show stable performance.\n"
            else:
                high_priority = [r for r in recommendations if r['priority'] == 'high']
                medium_priority = [r for r in recommendations if r['priority'] == 'medium']
                
                if high_priority:
                    response += "🚨 **High Priority Actions:**\n"
                    for rec in high_priority[:3]:
                        response += f"• **{rec['sku']}:** {rec['action'].title()} - {rec['reason']}\n"
                
                if medium_priority:
                    response += f"\n⚠️ **Medium Priority Actions:**\n"
                    for rec in medium_priority[:3]:
                        response += f"• **{rec['sku']}:** {rec['action'].title()} - {rec['reason']}\n"
            
            # Add general insights
            response += f"\n📊 **Key Insights:**\n"
            response += f"• Total products analyzed: {len(product_analysis.get('products', {}))}\n"
            response += f"• Products with recommendations: {len(recommendations)}\n"
            
            if product_analysis.get('top_performers'):
                top_sku = product_analysis['top_performers'][0][0]
                top_revenue = product_analysis['top_performers'][0][1]['total_revenue']
                response += f"• Best performing product: {top_sku} ({self._format_currency(top_revenue)})\n"
            
            response += f"\n**Question:** {question}\n"
            response += f"**Analysis Type:** Business Recommendations"
            
            return {
                'success': True,
                'response': response,
                'question': question,
                'analysis_type': 'recommendations',
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Recommendation error: {str(e)}",
                'question': question,
                'analysis_type': 'recommendations'
            }
    
    def _handle_anomaly_question(self, question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Handle anomaly detection questions"""
        try:
            anomalies = self.predictive_analytics.detect_anomalies(data)
            
            response = f"🔍 **Anomaly Detection Results:**\n\n"
            
            if not anomalies:
                response += "✅ **No significant anomalies detected**\n"
                response += "Your business data shows normal patterns with no unusual spikes or drops.\n"
            else:
                response += f"⚠️ **Found {len(anomalies)} anomalies:**\n\n"
                
                for i, anomaly in enumerate(anomalies[:5], 1):
                    response += f"**{i}. {anomaly['type'].replace('_', ' ').title()}**\n"
                    response += f"• **Date:** {anomaly['date']}\n"
                    response += f"• **Value:** {self._format_currency(anomaly['value'])}\n"
                    response += f"• **Expected:** {self._format_currency(anomaly['expected'])}\n"
                    response += f"• **Severity:** {anomaly['severity'].title()}\n"
                    response += f"• **Deviation:** {anomaly['z_score']:.1f} standard deviations\n\n"
            
            response += f"**Question:** {question}\n"
            response += f"**Analysis Type:** Anomaly Detection"
            
            return {
                'success': True,
                'response': response,
                'question': question,
                'analysis_type': 'anomaly_detection',
                'anomalies': anomalies
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Anomaly detection error: {str(e)}",
                'question': question,
                'analysis_type': 'anomaly_detection'
            }
    
    def _handle_seasonal_question(self, question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Handle seasonal pattern questions"""
        try:
            seasonal_data = self.predictive_analytics.get_seasonal_patterns(data)
            patterns = seasonal_data.get('patterns', {})
            insights = seasonal_data.get('insights', [])
            
            response = f"📅 **Seasonal Pattern Analysis:**\n\n"
            
            if not patterns:
                response += "📊 **Insufficient data for seasonal analysis**\n"
                response += "Need at least 30 days of data to identify seasonal patterns.\n"
            else:
                if 'monthly' in patterns:
                    monthly = patterns['monthly']
                    response += f"📆 **Monthly Patterns:**\n"
                    response += f"• **Best Month:** {monthly['best_month']}\n"
                    response += f"• **Worst Month:** {monthly['worst_month']}\n"
                
                if 'daily' in patterns:
                    daily = patterns['daily']
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    response += f"\n📅 **Daily Patterns:**\n"
                    response += f"• **Best Day:** {day_names[daily['best_day']]}\n"
                    response += f"• **Worst Day:** {day_names[daily['worst_day']]}\n"
                
                if insights:
                    response += f"\n💡 **Key Insights:**\n"
                    for insight in insights:
                        response += f"• {insight}\n"
            
            response += f"\n**Question:** {question}\n"
            response += f"**Analysis Type:** Seasonal Pattern Analysis"
            
            return {
                'success': True,
                'response': response,
                'question': question,
                'analysis_type': 'seasonal_patterns',
                'patterns': patterns
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Seasonal analysis error: {str(e)}",
                'question': question,
                'analysis_type': 'seasonal_patterns'
            }
    
    def _handle_general_predictive_question(self, question: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Handle general predictive questions"""
        try:
            # Provide a comprehensive analysis
            forecast_result = self.predictive_analytics.forecast_revenue(data, 1)
            trend_analysis = self.predictive_analytics.analyze_trends(data, 'revenue_calc')
            product_analysis = self.predictive_analytics.analyze_product_performance(data)
            
            response = f"🔮 **Comprehensive Business Analysis:**\n\n"
            
            # Revenue forecast
            if forecast_result.get('forecast', 0) > 0:
                response += f"📈 **Next Month Forecast:** {self._format_currency(forecast_result['forecast'])}\n"
                response += f"**Confidence:** {forecast_result['confidence']:.1%}\n\n"
            
            # Trend analysis
            if trend_analysis.get('trend'):
                response += f"📊 **Revenue Trend:** {trend_analysis['trend'].title()}\n"
                response += f"**Change:** {trend_analysis.get('pct_change', 0):.1f}%\n\n"
            
            # Top recommendations
            recommendations = product_analysis.get('recommendations', [])
            if recommendations:
                response += f"💡 **Top Recommendations:**\n"
                for rec in recommendations[:3]:
                    response += f"• **{rec['sku']}:** {rec['action'].title()} ({rec['reason']})\n"
            
            response += f"\n**Question:** {question}\n"
            response += f"**Analysis Type:** General Predictive Analysis"
            
            return {
                'success': True,
                'response': response,
                'question': question,
                'analysis_type': 'general_predictive',
                'forecast_data': forecast_result,
                'trend_data': trend_analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"General predictive analysis error: {str(e)}",
                'question': question,
                'analysis_type': 'general_predictive'
            }
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions for testing"""
        return [
            # Traditional SQL questions
            "What is the total revenue?",
            "How many orders do we have?",
            "Show me top 5 products by revenue",
            "Which city has the highest revenue?",
            "How many unique SKUs do we have?",
            "What is the average order value?",
            "Show me revenue by city",
            "What are the top 3 cities by sales?",
            
            # Predictive Analytics questions
            "What will be my revenue next month?",
            "Predict my sales for next quarter",
            "Which products are declining in sales?",
            "Should I restock Product ABC?",
            "What's the trend for my business?",
            "Show me unusual patterns in my data",
            "What are the seasonal patterns?",
            "Recommend products to focus on",
            "Forecast revenue for next 3 months",
            "Which products are growing fastest?"
        ]


def test_ai_assistant():
    """Test function to verify AI assistant works independently"""
    print("🧪 Testing AI Assistant Module...")
    print("=" * 50)
    
    try:
        # Initialize AI Assistant
        assistant = AIAssistant()
        
        # Test sample questions
        sample_questions = assistant.get_sample_questions()
        
        print(f"\n📝 Testing with {len(sample_questions)} sample questions:")
        print("-" * 50)
        
        for i, question in enumerate(sample_questions[:3], 1):  # Test first 3 questions
            print(f"\n🔍 Test {i}: {question}")
            result = assistant.ask_question(question)
            
            if result['success']:
                print(f"✅ Success!")
                print(f"📊 Response: {result['response'][:100]}...")
                print(f"🔍 SQL: {result['sql']}")
                print(f"📈 Rows: {result['result_rows']}, Columns: {result['result_columns']}")
            else:
                print(f"❌ Failed: {result['error']}")
            
            print("-" * 30)
        
        print("\n🎉 AI Assistant test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def interactive_mode():
    """Interactive mode for testing questions"""
    print("🤖 AI Assistant Interactive Mode")
    print("Type 'quit' to exit, 'help' for sample questions")
    print("=" * 50)
    
    try:
        assistant = AIAssistant()
        
        while True:
            question = input("\n💬 Ask a question: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print("\n📝 Sample questions:")
                for i, q in enumerate(assistant.get_sample_questions(), 1):
                    print(f"{i}. {q}")
                continue
            elif not question:
                continue
            
            result = assistant.ask_question(question)
            
            if result['success']:
                print(f"\n{result['response']}")
            else:
                print(f"\n❌ Error: {result['error']}")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Assistant for CSV Analytics Dashboard")
    parser.add_argument("--test", action="store_true", help="Run automated tests")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--question", type=str, help="Ask a specific question")
    
    args = parser.parse_args()
    
    if args.test:
        success = test_ai_assistant()
        sys.exit(0 if success else 1)
    elif args.interactive:
        interactive_mode()
    elif args.question:
        try:
            assistant = AIAssistant()
            result = assistant.ask_question(args.question)
            if result['success']:
                print(result['response'])
            else:
                print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Default: run tests
        print("🧪 Running AI Assistant Tests...")
        success = test_ai_assistant()
        if success:
            print("\n💡 Usage examples:")
            print("python ai_assistant.py --test          # Run automated tests")
            print("python ai_assistant.py --interactive   # Interactive mode")
            print('python ai_assistant.py --question "What is the total revenue?"')
        sys.exit(0 if success else 1)
