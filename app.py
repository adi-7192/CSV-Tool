"""
CSV Analytics Dashboard - Simple & Clean

Upload CSV → Auto-map columns → View KPIs → Chat about data
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Constants
DB_FILE = 'data.db'
TABLE_NAME = 'sales'
MAPPING_FILE = 'data/mapping.json'

# Column synonyms for auto-mapping
SYNONYMS = {
    "order_date": ["order date", "invoice date", "shipment date", "date"],
    "order_id": ["order id", "order number", "invoice id", "order no"],
    "sku": ["sku", "asin", "product id", "item id"],
    "product_name": ["product name", "item description", "title", "product"],
    "quantity": ["quantity", "qty", "units"],
    "revenue_amount": ["invoice amount", "order amount", "total amount", "amount", "revenue"],
    "region": ["region", "city", "market"],
    "status": ["status", "order status"]
}

def normalize_header(header: str) -> str:
    """Normalize header: strip, collapse spaces, lowercase"""
    return re.sub(r'\s+', ' ', header.strip().lower())

def compute_signature(headers: List[str]) -> str:
    """Compute SHA1 signature of sorted headers"""
    return hashlib.sha1("|".join(sorted(headers)).encode()).hexdigest()

def is_date_column(series: pd.Series) -> bool:
    """Check if column contains dates (70% threshold)"""
    sample = series.dropna().head(200)
    if len(sample) == 0: return False
    try:
        parsed = pd.to_datetime(sample, errors='coerce')
        return (parsed.notna().sum() / len(sample)) >= 0.7
    except: return False

def is_numeric_column(series: pd.Series) -> bool:
    """Check if column contains numbers (70% threshold)"""
    sample = series.dropna().head(200)
    if len(sample) == 0: return False
    try:
        cleaned = sample.astype(str).str.replace(r'[₹$,\s]', '', regex=True)
        parsed = pd.to_numeric(cleaned, errors='coerce')
        return (parsed.notna().sum() / len(sample)) >= 0.7
    except: return False

def is_integer_column(series: pd.Series) -> bool:
    """Check if column contains integers (70% threshold)"""
    sample = series.dropna().head(200)
    if len(sample) == 0: return False
    try:
        parsed = pd.to_numeric(sample, errors='coerce')
        return (parsed.notna().sum() / len(sample)) >= 0.7
    except: return False

def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-map CSV columns to standard fields"""
    print(f"Auto-mapping {len(df.columns)} columns...")
    
    mappings = {field: None for field in SYNONYMS.keys()}
    used_headers = set()
    
    for field, synonyms in SYNONYMS.items():
        candidates = []
        
        # Find matching headers
        for header in df.columns:
            if header in used_headers: continue
            norm = normalize_header(header)
            for syn in synonyms:
                if syn in norm or norm in syn:
                    candidates.append(header)
                    break
        
        if len(candidates) == 1:
            mappings[field] = candidates[0]
            used_headers.add(candidates[0])
        elif len(candidates) > 1:
            # Use type inference to disambiguate
            best = None
            if field == "order_date":
                best = next((c for c in candidates if is_date_column(df[c])), None)
            elif field == "quantity":
                best = next((c for c in candidates if is_integer_column(df[c])), None)
            elif field == "revenue_amount":
                # Prioritize "invoice amount"
                invoice_candidates = [c for c in candidates if "invoice" in normalize_header(c)]
                if invoice_candidates:
                    best = next((c for c in invoice_candidates if is_numeric_column(df[c])), None)
                if not best:
                    best = next((c for c in candidates if is_numeric_column(df[c])), None)
            
            if best:
                mappings[field] = best
                used_headers.add(best)
    
    print(f"Mapped {sum(1 for v in mappings.values() if v)} fields")
    return mappings

def save_mapping(headers: List[str], mappings: Dict[str, str]) -> None:
    """Save mapping to file"""
    os.makedirs(os.path.dirname(MAPPING_FILE), exist_ok=True)
    signature = compute_signature(headers)
    
    saved = {}
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                saved = json.load(f)
        except: pass
    
    saved[signature] = {"headers": headers, "mappings": mappings}
    
    with open(MAPPING_FILE, 'w') as f:
        json.dump(saved, f, indent=2)

def load_mapping(headers: List[str]) -> Optional[Dict[str, str]]:
    """Load saved mapping"""
    if not os.path.exists(MAPPING_FILE): return None
    signature = compute_signature(headers)
    
    try:
        with open(MAPPING_FILE, 'r') as f:
            saved = json.load(f)
        return saved.get(signature, {}).get("mappings")
    except: return None

def format_inr(amount: float) -> str:
    """Format amount as INR with thousand separators"""
    return f"₹{amount:,.2f}"

def parse_revenue(series: pd.Series) -> pd.Series:
    """Parse revenue column, stripping currency symbols"""
    cleaned = series.astype(str).str.replace(r'[₹$,\s]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)

def create_sales_table(conn: sqlite3.Connection) -> None:
    """Create sales table with standard schema"""
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            order_date TEXT,
            order_id TEXT,
            sku TEXT,
            product_name TEXT,
            quantity INTEGER,
            revenue_in_inr REAL,
            region TEXT,
            status TEXT
        )
    ''')
    conn.commit()

def store_sales_data(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> str:
    """Store CSV data to sales table"""
    try:
        # Create clean DataFrame
        clean_df = pd.DataFrame()
        
        # Map each field
        if mappings.get("order_date"):
            try:
                clean_df["order_date"] = pd.to_datetime(df[mappings["order_date"]], errors='coerce').dt.strftime('%Y-%m-%d')
            except: clean_df["order_date"] = None
        else: clean_df["order_date"] = None
        
        clean_df["order_id"] = df[mappings["order_id"]].astype(str) if mappings.get("order_id") else None
        clean_df["sku"] = df[mappings["sku"]].astype(str) if mappings.get("sku") else None
        clean_df["product_name"] = df[mappings["product_name"]].astype(str) if mappings.get("product_name") else None
        
        if mappings.get("quantity"):
            clean_df["quantity"] = pd.to_numeric(df[mappings["quantity"]], errors='coerce').fillna(0).astype(int)
        else: clean_df["quantity"] = 0
        
        if mappings.get("revenue_amount"):
            clean_df["revenue_in_inr"] = parse_revenue(df[mappings["revenue_amount"]])
        else: clean_df["revenue_in_inr"] = 0.0
        
        clean_df["region"] = df[mappings["region"]].astype(str) if mappings.get("region") else None
        clean_df["status"] = df[mappings["status"]].astype(str) if mappings.get("status") else None
        
        # Store to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f'DROP TABLE IF EXISTS {TABLE_NAME}')
        create_sales_table(conn)
        clean_df.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        conn.close()
        
        return f"✅ Stored {len(clean_df)} rows successfully!"
        
    except Exception as e:
        return f"❌ Error storing data: {str(e)}"

def compute_kpis() -> Optional[Dict]:
    """Compute KPIs from sales table"""
    if not os.path.exists(DB_FILE): return None
    
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        
        if df.empty: return None
        
        kpis = {}
        kpis['total_revenue'] = df['revenue_in_inr'].sum()
        kpis['total_orders'] = df['order_id'].notna().sum()
        kpis['units_sold'] = df['quantity'].sum()
        
        # Top 5 SKUs with product names
        if df['sku'].notna().sum() > 0:
            # Group by SKU and get product names
            sku_data = df.groupby('sku').agg({
                'revenue_in_inr': 'sum',
                'product_name': 'first'
            }).sort_values('revenue_in_inr', ascending=False).head(5)
            
            # Create human-readable labels
            sku_data['display_name'] = sku_data.apply(
                lambda row: f"{row['product_name']} (SKU: {row.name})" if pd.notna(row['product_name']) and row['product_name'].strip() else f"SKU: {row.name}",
                axis=1
            )
            
            kpis['top_skus'] = sku_data[['display_name', 'revenue_in_inr']].reset_index(drop=True)
            kpis['top_skus'].columns = ['Product', 'Revenue']
        else:
            kpis['top_skus'] = pd.DataFrame(columns=['Product', 'Revenue'])
        
        return kpis
        
    except Exception as e:
        print(f"Error computing KPIs: {e}")
        return None

def run_sql_query(query: str) -> Tuple[List, List[str]]:
    """Run SQL query and return results"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return results, columns
    except Exception as e:
        print(f"SQL Error: {e}")
        return [], []

def answer_numeric_question(question: str) -> str:
    """Answer numeric questions with SQL"""
    question_lower = question.lower()
    
    if "total revenue" in question_lower or "revenue" in question_lower:
        results, _ = run_sql_query("SELECT SUM(revenue_in_inr) as total_revenue FROM sales")
        if results:
            return f"**Total Revenue:** {format_inr(results[0][0])}"
    
    elif "orders" in question_lower or "order count" in question_lower:
        results, _ = run_sql_query("SELECT COUNT(DISTINCT order_id) as total_orders FROM sales")
        if results:
            return f"**Total Orders:** {results[0][0]:,}"
    
    elif "units" in question_lower or "quantity" in question_lower:
        results, _ = run_sql_query("SELECT SUM(quantity) as total_units FROM sales")
        if results:
            return f"**Total Units Sold:** {results[0][0]:,}"
    
    elif "top" in question_lower and ("sku" in question_lower or "product" in question_lower or "item" in question_lower or "best" in question_lower):
        limit = 5
        if "top 3" in question_lower: limit = 3
        elif "top 10" in question_lower: limit = 10
        
        results, columns = run_sql_query(f"""
            SELECT sku, product_name, SUM(revenue_in_inr) as revenue 
            FROM sales 
            WHERE sku IS NOT NULL 
            GROUP BY sku, product_name
            ORDER BY revenue DESC 
            LIMIT {limit}
        """)
        
        if results:
            response = "**Top Products by Revenue:**\n"
            for i, (sku, product_name, revenue) in enumerate(results, 1):
                if product_name and product_name.strip():
                    display_name = f"{product_name} (SKU: {sku})"
                else:
                    display_name = f"SKU: {sku}"
                response += f"{i}. {display_name}: {format_inr(revenue)}\n"
            return response
    
    return "I couldn't understand your question. Try asking about revenue, orders, units, or top products."

def answer_descriptive_question(question: str) -> str:
    """Answer descriptive questions with general insights"""
    kpis = compute_kpis()
    if not kpis: return "No data available for analysis."
    
    question_lower = question.lower()
    
    if "trend" in question_lower or "performance" in question_lower:
        return f"Based on the data: Total revenue is {format_inr(kpis['total_revenue'])}, with {kpis['total_orders']} orders and {kpis['units_sold']} units sold."
    
    elif "declining" in question_lower or "drop" in question_lower:
        return "To identify declining trends, I'd need time-series data. Currently showing overall totals."
    
    elif "best" in question_lower or "top" in question_lower:
        if not kpis['top_skus'].empty:
            top_product = kpis['top_skus'].iloc[0]
            return f"The best performing product is {top_product['Product']} with {format_inr(top_product['Revenue'])} revenue."
        return "No product data available."
    
    return f"Here's a summary: {kpis['total_orders']} orders totaling {format_inr(kpis['total_revenue'])} in revenue."

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="CSV Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'mappings' not in st.session_state:
        st.session_state.mappings = {}
    if 'show_mapping_modal' not in st.session_state:
        st.session_state.show_mapping_modal = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.title("📊 CSV Analytics Dashboard")
    st.markdown("*Upload CSV → Auto-map → View KPIs → Chat about data*")
    
    # Sidebar for CSV upload
    with st.sidebar:
        st.header("📁 Upload CSV")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Check if this is a new file
                is_new = (st.session_state.uploaded_df is None or 
                         list(df.columns) != list(st.session_state.uploaded_df.columns))
                
                if is_new:
                    st.session_state.uploaded_df = df.copy()
                    
                    # Try to load saved mapping
                    saved_mapping = load_mapping(list(df.columns))
                    
                    if saved_mapping:
                        st.session_state.mappings = saved_mapping
                        st.info("✅ Using saved column mapping")
                    else:
                        # Auto-map columns
                        st.session_state.mappings = auto_map_columns(df)
                        
                        # Check for ambiguous fields
                        ambiguous = [field for field, header in st.session_state.mappings.items() 
                                   if header is None and any(normalize_header(h) in 
                                   [normalize_header(s) for s in SYNONYMS.get(field, [])] 
                                   for h in df.columns)]
                        
                        if ambiguous:
                            st.session_state.show_mapping_modal = True
                            st.warning(f"⚠️ {len(ambiguous)} fields need confirmation")
                        else:
                            st.success("✅ All columns mapped automatically!")
                
                # Store data
                if st.button("💾 Store Data", type="primary"):
                    result = store_sales_data(st.session_state.uploaded_df, st.session_state.mappings)
                    st.write(result)
                    
                    if "✅" in result:
                        # Save mapping
                        save_mapping(list(df.columns), st.session_state.mappings)
                        st.session_state.show_mapping_modal = False
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    # Main content area
    if st.session_state.uploaded_df is not None:
        # Show mapping modal if needed
        if st.session_state.show_mapping_modal:
            st.markdown("---")
            st.markdown("### 🤔 Confirm Column Mapping")
            st.markdown("*Please select the correct column for each field:*")
            
            df = st.session_state.uploaded_df
            ambiguous = [field for field, header in st.session_state.mappings.items() 
                        if header is None and any(normalize_header(h) in 
                        [normalize_header(s) for s in SYNONYMS.get(field, [])] 
                        for h in df.columns)]
            
            for field in ambiguous:
                st.markdown(f"**{field.replace('_', ' ').title()}**")
                
                # Get candidates
                candidates = []
                for header in df.columns:
                    norm = normalize_header(header)
                    for syn in SYNONYMS.get(field, []):
                        if syn in norm or norm in syn:
                            candidates.append(header)
                            break
                
                candidates.append("[Skip this field]")
                
                selected = st.selectbox(
                    f"Map {field} to:",
                    options=candidates,
                    key=f"map_{field}"
                )
                
                if selected != "[Skip this field]":
                    st.session_state.mappings[field] = selected
                else:
                    st.session_state.mappings[field] = None
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Confirm", type="primary"):
                    st.session_state.show_mapping_modal = False
                    st.rerun()
            with col2:
                st.caption("Your choices will be saved for future uploads")
        
        # Main tabs
        tab1, tab2 = st.tabs(["📊 KPIs", "💬 Chat"])
        
        with tab1:
            st.header("📊 Key Performance Indicators")
            
            kpis = compute_kpis()
            if kpis:
                # Main metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="💰 Total Revenue",
                        value=format_inr(kpis['total_revenue'])
                    )
                
                with col2:
                    st.metric(
                        label="📦 Total Orders",
                        value=f"{kpis['total_orders']:,}"
                    )
                
                with col3:
                    st.metric(
                        label="📊 Units Sold",
                        value=f"{kpis['units_sold']:,}"
                    )
                
                # Top Products
                st.markdown("---")
                st.markdown("#### 🏆 Top 5 Products by Revenue")
                if not kpis['top_skus'].empty:
                    display_products = kpis['top_skus'].copy()
                    display_products['Revenue'] = display_products['Revenue'].apply(format_inr)
                    st.dataframe(display_products, use_container_width=True, hide_index=True)
                else:
                    st.info("No product data available")
                
                # Data preview
                st.markdown("---")
                st.markdown("#### 📋 Data Preview")
                st.dataframe(st.session_state.uploaded_df.head(10), use_container_width=True)
            else:
                st.info("💡 Store data to view KPIs")
        
        with tab2:
            st.header("💬 Chat About Your Data")
            st.markdown("*Ask questions about revenue, orders, SKUs, and trends*")
            
            # Chat history
            if st.session_state.chat_history:
                st.markdown("### 💬 Chat History")
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {question[:50]}{'...' if len(question) > 50 else ''}", expanded=False):
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Answer:** {answer}")
                st.markdown("---")
            
            # Question input
            question = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., What's the total revenue? Show me top products...",
                key="chat_question"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🚀 Ask", type="primary"):
                    if question.strip():
                        # Determine if numeric or descriptive
                        numeric_keywords = ['total', 'count', 'sum', 'revenue', 'orders', 'units', 'top', 'best']
                        is_numeric = any(keyword in question.lower() for keyword in numeric_keywords)
                        
                        if is_numeric:
                            answer = answer_numeric_question(question)
                        else:
                            answer = answer_descriptive_question(question)
                        
                        # Add to history
                        st.session_state.chat_history.append((question, answer))
                        st.rerun()
                    else:
                        st.warning("Please enter a question.")
            
            with col2:
                st.caption("💡 Try: 'What's the total revenue?', 'Top 5 products', 'How many orders?'")
            
            # Example questions
            st.markdown("---")
            st.markdown("### 💡 Example Questions")
            
            example_cols = st.columns(2)
            with example_cols[0]:
                st.markdown("**Numeric Questions:**")
                st.markdown("- What's the total revenue?")
                st.markdown("- How many orders?")
                st.markdown("- Top 5 products by revenue?")
                st.markdown("- Total units sold?")
            
            with example_cols[1]:
                st.markdown("**Descriptive Questions:**")
                st.markdown("- Show me performance trends")
                st.markdown("- What's our best product?")
                st.markdown("- Any declining trends?")
                st.markdown("- Overall business summary")
    
    else:
        st.info("👆 Upload a CSV file in the sidebar to begin")
        
        # Show sample format
        st.markdown("---")
        st.markdown("### 📝 Expected CSV Format")
        st.markdown("""
        The app automatically recognizes columns like:
        - **Order Date**: Invoice Date, Shipment Date, Order Date, etc.
        - **Order ID**: Order Number, Invoice ID, Order No, etc.
        - **SKU**: SKU, ASIN, Product ID, Item ID, etc.
        - **Product Name**: Item Description, Product Name, Title, etc.
        - **Quantity**: Quantity, Qty, Units, etc.
        - **Revenue**: Invoice Amount, Order Amount, Total Amount, Revenue, etc.
        - **Region**: Region, City, Market, etc.
        - **Status**: Status, Order Status, Fulfillment Status, etc.
        
        Upload any CSV with similar columns and the app will map them automatically!
        """)

if __name__ == "__main__":
    main()