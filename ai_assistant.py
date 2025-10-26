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

# Import our existing database manager
try:
    from db_manager import query_data, get_row_count, table_exists
except ImportError:
    print("❌ Error: db_manager.py not found. Make sure it's in the same directory.")
    sys.exit(1)

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
        self.database_schema = self._get_database_schema()
        
        # Test Ollama connection
        if not self._test_ollama_connection():
            raise ConnectionError("Cannot connect to Ollama service. Make sure Ollama is running.")
        
        print("✅ AI Assistant initialized successfully")
        print(f"🔗 Connected to Ollama at {ollama_url}")
        print(f"🤖 Using model: {self.model}")
    
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
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
            return False
    
    def _get_database_schema(self) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            if not table_exists('sales'):
                return {}
            
            # Get table schema
            schema_query = "DESCRIBE sales"
            schema_result = query_data(schema_query)
            
            if schema_result.empty:
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
        11. NEVER use window functions (LAG, LEAD, ROW_NUMBER) in HAVING clauses
        12. For trend analysis, use subqueries or CTEs instead of window functions in HAVING
        13. For declining analysis, compare current period vs previous period using separate queries
        14. Return ONLY the SQL query, no explanations or markdown
        
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
        
        Example pattern for declining SKUs:
        WITH current_period AS (
            SELECT "Sku", COUNT(DISTINCT "Invoice Number") as current_orders,
                   SUM("revenue_calc") as current_revenue
            FROM sales 
            WHERE "Invoice Date" >= '2025-10-01'
            GROUP BY "Sku"
        ),
        previous_period AS (
            SELECT "Sku", COUNT(DISTINCT "Invoice Number") as previous_orders,
                   SUM("revenue_calc") as previous_revenue
            FROM sales 
            WHERE "Invoice Date" < '2025-10-01' AND "Invoice Date" >= '2025-09-01'
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
            'EXEC', 'EXECUTE', 'SP_', 'XP_', '--', '/*', '*/', 'UNION',
            'INFORMATION_SCHEMA', 'SYS.', 'PG_', 'MYSQL.'
        ]
        
        # Check for invalid window function usage
        if 'HAVING' in sql_upper and any(func in sql_upper for func in ['LAG(', 'LEAD(', 'ROW_NUMBER(', 'RANK(', 'DENSE_RANK(']):
            return False, "Window functions cannot be used in HAVING clause"
        
        # Check for window functions in WHERE clause (also problematic)
        if 'WHERE' in sql_upper and any(func in sql_upper for func in ['LAG(', 'LEAD(', 'ROW_NUMBER(', 'RANK(', 'DENSE_RANK(']):
            return False, "Window functions cannot be used in WHERE clause - use subqueries or CTEs instead"
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Dangerous keyword '{keyword}' detected"
        
        # Must start with SELECT or WITH (for CTEs)
        if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
            return False, "Query must start with SELECT or WITH"
        
        # Check for proper column name quoting
        if '"' not in sql and any(col in sql_upper for col in ['INVOICE DATE', 'INVOICE AMOUNT', 'INVOICE NUMBER', 'SHIP TO CITY']):
            return False, "Column names must be quoted with double quotes"
        
        return True, ""
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API to generate SQL"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent SQL generation
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to call Ollama: {e}")
    
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
            
            # Generate SQL prompt
            prompt = self._generate_sql_prompt(question)
            
            # Call Ollama to generate SQL
            print("🤖 Generating SQL with Ollama...")
            sql_response = self._call_ollama(prompt)
            
            # Extract SQL from response (remove any markdown or explanations)
            sql = sql_response.strip()
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
            
            # Clean up SQL
            sql = re.sub(r'^SELECT\s+', 'SELECT ', sql, flags=re.IGNORECASE)
            sql = sql.replace('\n', ' ').strip()
            
            print(f"🔍 Generated SQL: {sql}")
            
            # Validate SQL safety
            is_safe, error_msg = self._validate_sql_safety(sql)
            if not is_safe:
                return {
                    'success': False,
                    'error': f"SQL safety validation failed: {error_msg}",
                    'question': question,
                    'sql': sql
                }
            
            # Execute SQL query
            print("📊 Executing SQL query...")
            result = query_data(sql)
            
            # Format response
            formatted_response = self._format_business_response(question, sql, result)
            
            return {
                'success': True,
                'response': formatted_response,
                'question': question,
                'sql': sql,
                'result_rows': len(result),
                'result_columns': len(result.columns) if not result.empty else 0
            }
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'question': question,
                'sql': None
            }
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions for testing"""
        return [
            "What is the total revenue?",
            "How many orders do we have?",
            "Show me top 5 products by revenue",
            "Which city has the highest revenue?",
            "What was the revenue last month?",
            "Compare September vs August performance",
            "How many unique SKUs do we have?",
            "What is the average order value?",
            "Show me revenue by city",
            "What are the top 3 cities by sales?"
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
