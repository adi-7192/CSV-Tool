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
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

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

def get_date_filtered_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Get filtered data for date range"""
    if not os.path.exists(DB_FILE): return None
    
    try:
        conn = sqlite3.connect(DB_FILE)
        query = f"""
            SELECT * FROM {TABLE_NAME} 
            WHERE order_date >= ? AND order_date <= ?
            ORDER BY order_date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        if df.empty: return None
        
        # Convert order_date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        
        return df
        
    except Exception as e:
        print(f"Error getting filtered data: {e}")
        return None

def compute_business_kpis(df: pd.DataFrame) -> Dict:
    """Compute business-grade KPIs from filtered data"""
    if df is None or df.empty:
        return {}
    
    print(f"Computing KPIs for {len(df)} rows")
    
    kpis = {}
    
    # Basic metrics
    kpis['total_revenue'] = df['revenue_in_inr'].sum()
    kpis['total_orders'] = df['order_id'].nunique() if 'order_id' in df.columns else 0
    kpis['units_sold'] = df['quantity'].sum() if 'quantity' in df.columns else 0
    kpis['aov'] = kpis['total_revenue'] / kpis['total_orders'] if kpis['total_orders'] > 0 else 0
    
    # WoW comparison (if we have enough data)
    if len(df) > 0:
        current_week_start = df['order_date'].max() - timedelta(days=7)
        previous_week_start = current_week_start - timedelta(days=7)
        
        current_week_data = df[df['order_date'] >= current_week_start]
        previous_week_data = df[(df['order_date'] >= previous_week_start) & (df['order_date'] < current_week_start)]
        
        if len(previous_week_data) > 0:
            prev_revenue = previous_week_data['revenue_in_inr'].sum()
            curr_revenue = current_week_data['revenue_in_inr'].sum()
            if prev_revenue > 0:
                kpis['wow_revenue_change'] = ((curr_revenue - prev_revenue) / prev_revenue) * 100
            else:
                kpis['wow_revenue_change'] = 0
        else:
            kpis['wow_revenue_change'] = None
    else:
        kpis['wow_revenue_change'] = None
    
    # Top products by revenue
    if 'sku' in df.columns and 'product_name' in df.columns:
        product_revenue = df.groupby(['sku', 'product_name']).agg({
            'revenue_in_inr': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # Create display names
        product_revenue['display_name'] = product_revenue.apply(
            lambda row: row['product_name'] if pd.notna(row['product_name']) and str(row['product_name']).strip() != '' else f"Unknown (SKU: {row['sku']})",
            axis=1
        )
        
        kpis['top_products_revenue'] = product_revenue.nlargest(10, 'revenue_in_inr')[['display_name', 'revenue_in_inr']]
        kpis['top_products_units'] = product_revenue.nlargest(10, 'quantity')[['display_name', 'quantity']]
    else:
        kpis['top_products_revenue'] = pd.DataFrame(columns=['display_name', 'revenue_in_inr'])
        kpis['top_products_units'] = pd.DataFrame(columns=['display_name', 'quantity'])
    
    # Order status breakdown
    if 'status' in df.columns:
        status_counts = df['status'].value_counts()
        kpis['status_breakdown'] = status_counts
    else:
        kpis['status_breakdown'] = None
    
    # Revenue by region
    if 'region' in df.columns:
        region_revenue = df.groupby('region')['revenue_in_inr'].sum().sort_values(ascending=False).head(10)
        kpis['region_revenue'] = region_revenue
    else:
        kpis['region_revenue'] = None
    
    # Revenue trend
    if 'order_date' in df.columns:
        daily_revenue = df.groupby(df['order_date'].dt.date)['revenue_in_inr'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        kpis['revenue_trend'] = daily_revenue
    else:
        kpis['revenue_trend'] = pd.DataFrame(columns=['date', 'revenue'])
    
    return kpis

def compute_movers_decliners(df: pd.DataFrame, min_units: int = 10) -> Dict:
    """Compute fast movers and decliners"""
    if df is None or df.empty or 'sku' not in df.columns:
        return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}
    
    try:
        # Group by week and SKU
        df['week'] = df['order_date'].dt.to_period('W')
        weekly_data = df.groupby(['week', 'sku', 'product_name']).agg({
            'revenue_in_inr': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # Get current and previous week
        current_week = weekly_data['week'].max()
        previous_week = current_week - 1
        
        current_week_data = weekly_data[weekly_data['week'] == current_week]
        previous_week_data = weekly_data[weekly_data['week'] == previous_week]
        
        if current_week_data.empty or previous_week_data.empty:
            return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}
        
        # Merge for comparison
        comparison = current_week_data.merge(
            previous_week_data[['sku', 'revenue_in_inr', 'quantity']], 
            on='sku', 
            suffixes=('_current', '_previous')
        )
        
        # Calculate WoW change
        comparison['wow_change'] = (
            (comparison['revenue_in_inr_current'] - comparison['revenue_in_inr_previous']) / 
            comparison['revenue_in_inr_previous'] * 100
        )
        
        # Create display names
        comparison['display_name'] = comparison.apply(
            lambda row: row['product_name'] if pd.notna(row['product_name']) and str(row['product_name']).strip() != '' else f"Unknown (SKU: {row['sku']})",
            axis=1
        )
        
        # Decliners: ≥30% drop and ≥N units previous week
        decliners = comparison[
            (comparison['wow_change'] <= -30) & 
            (comparison['quantity_previous'] >= min_units)
        ].sort_values('wow_change')
        
        # Fast movers: ≥30% increase and ≥N units current week
        fast_movers = comparison[
            (comparison['wow_change'] >= 30) & 
            (comparison['quantity_current'] >= min_units)
        ].sort_values('wow_change', ascending=False)
        
        return {
            'decliners': decliners[['display_name', 'revenue_in_inr_current', 'wow_change']],
            'fast_movers': fast_movers[['display_name', 'revenue_in_inr_current', 'wow_change']]
        }
        
    except Exception as e:
        print(f"Error computing movers/decliners: {e}")
        return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}

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
    # Get basic data for descriptive questions
    if not os.path.exists(DB_FILE): return "No data available for analysis."
    
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        
        if df.empty: return "No data available for analysis."
        
        # Basic metrics for descriptive answers
        total_revenue = df['revenue_in_inr'].sum()
        total_orders = df['order_id'].nunique() if 'order_id' in df.columns else 0
        total_units = df['quantity'].sum() if 'quantity' in df.columns else 0
    except:
        return "No data available for analysis."
    
    question_lower = question.lower()
    
    if "trend" in question_lower or "performance" in question_lower:
        return f"Based on the data: Total revenue is {format_inr(total_revenue)}, with {total_orders} orders and {total_units} units sold."
    
    elif "declining" in question_lower or "drop" in question_lower:
        return "To identify declining trends, I'd need time-series data. Currently showing overall totals."
    
    elif "best" in question_lower or "top" in question_lower:
        # Get top product by revenue
        if 'sku' in df.columns and 'product_name' in df.columns:
            product_revenue = df.groupby(['sku', 'product_name'])['revenue_in_inr'].sum().reset_index()
            if not product_revenue.empty:
                top_product = product_revenue.loc[product_revenue['revenue_in_inr'].idxmax()]
                product_name = top_product['product_name'] if pd.notna(top_product['product_name']) else f"Unknown (SKU: {top_product['sku']})"
                return f"The best performing product is {product_name} with {format_inr(top_product['revenue_in_inr'])} revenue."
        return "No product data available."
    
    return f"Here's a summary: {total_orders} orders totaling {format_inr(total_revenue)} in revenue."

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
            st.header("📊 Business Dashboard")
            
            # Date filter
            st.markdown("### 📅 Date Range Filter")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                preset = st.selectbox(
                    "Quick Presets:",
                    ["Custom Range", "Last 7 days", "Last 30 days", "This Month", "Last Month"],
                    key="date_preset"
                )
            
            with col2:
                if preset == "Custom Range":
                    start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                    end_date = st.date_input("End Date", value=datetime.now().date())
                else:
                    today = datetime.now().date()
                    if preset == "Last 7 days":
                        start_date = today - timedelta(days=7)
                        end_date = today
                    elif preset == "Last 30 days":
                        start_date = today - timedelta(days=30)
                        end_date = today
                    elif preset == "This Month":
                        start_date = today.replace(day=1)
                        end_date = today
                    elif preset == "Last Month":
                        first_this_month = today.replace(day=1)
                        start_date = (first_this_month - timedelta(days=1)).replace(day=1)
                        end_date = first_this_month - timedelta(days=1)
                    else:
                        start_date = today - timedelta(days=30)
                        end_date = today
                    
                    st.date_input("Start Date", value=start_date, disabled=True)
                    st.date_input("End Date", value=end_date, disabled=True)
            
            with col3:
                st.markdown("**Range:**")
                st.markdown(f"{start_date} to {end_date}")
            
            # Get filtered data and compute KPIs
            if st.button("🔄 Refresh Dashboard", type="primary"):
                st.rerun()
            
            # Load and compute KPIs
            df = get_date_filtered_data(str(start_date), str(end_date))
            if df is not None and not df.empty:
                kpis = compute_business_kpis(df)
                movers = compute_movers_decliners(df)
                
                # Main KPI cards
                st.markdown("---")
                st.markdown("### 📈 Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    wow_text = ""
                    if kpis.get('wow_revenue_change') is not None:
                        change = kpis['wow_revenue_change']
                        wow_text = f"WoW: {change:+.1f}%" if change != 0 else "WoW: 0%"
                    
                    st.metric(
                        label="💰 Total Revenue",
                        value=format_inr(kpis['total_revenue']),
                        delta=wow_text if wow_text else None
                    )
                
                with col2:
                    st.metric(
                        label="📦 Orders",
                        value=f"{kpis['total_orders']:,}"
                    )
                
                with col3:
                    st.metric(
                        label="📊 Units Sold",
                        value=f"{kpis['units_sold']:,}"
                    )
                
                with col4:
                    st.metric(
                        label="💵 AOV",
                        value=format_inr(kpis['aov'])
                    )
                
                # Revenue Trend Chart
                st.markdown("---")
                st.markdown("### 📈 Revenue Trend")
                if not kpis['revenue_trend'].empty:
                    fig = px.line(
                        kpis['revenue_trend'], 
                        x='date', 
                        y='revenue',
                        title="Daily Revenue Trend",
                        labels={'revenue': 'Revenue (₹)', 'date': 'Date'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No revenue trend data available")
                
                # Top Products
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏆 Top 10 Products by Revenue")
                    if not kpis['top_products_revenue'].empty:
                        display_df = kpis['top_products_revenue'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_in_inr'].apply(format_inr)
                        display_df = display_df[['display_name', 'Revenue (₹)']]
                        display_df.columns = ['Product', 'Revenue']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "top_products_revenue.csv",
                            "text/csv",
                            key="export_revenue"
                        )
                    else:
                        st.info("No product revenue data available")
                
                with col2:
                    st.markdown("#### 📦 Top 10 Products by Units")
                    if not kpis['top_products_units'].empty:
                        display_df = kpis['top_products_units'].copy()
                        display_df['Units'] = display_df['quantity'].apply(lambda x: f"{x:,}")
                        display_df = display_df[['display_name', 'Units']]
                        display_df.columns = ['Product', 'Units']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "top_products_units.csv",
                            "text/csv",
                            key="export_units"
                        )
                    else:
                        st.info("No product units data available")
                
                # Status and Region breakdowns
                col1, col2 = st.columns(2)
                
                with col1:
                    if kpis['status_breakdown'] is not None:
                        st.markdown("#### 📊 Order Status Breakdown")
                        fig = px.pie(
                            values=kpis['status_breakdown'].values,
                            names=kpis['status_breakdown'].index,
                            title="Order Status Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("#### 📊 Order Status Breakdown")
                        st.info("No status data available")
                
                with col2:
                    if kpis['region_revenue'] is not None:
                        st.markdown("#### 🌍 Revenue by Region")
                        fig = px.bar(
                            x=kpis['region_revenue'].values,
                            y=kpis['region_revenue'].index,
                            orientation='h',
                            title="Revenue by Region (Top 10)",
                            labels={'x': 'Revenue (₹)', 'y': 'Region'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("#### 🌍 Revenue by Region")
                        st.info("No region data available")
                
                # Movers & Decliners
                st.markdown("---")
                st.markdown("### 📊 Movers & Decliners")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📉 Decliners (≥30% WoW Drop)")
                    if not movers['decliners'].empty:
                        display_df = movers['decliners'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_in_inr_current'].apply(format_inr)
                        display_df['WoW Change'] = display_df['wow_change'].apply(lambda x: f"{x:.1f}%")
                        display_df = display_df[['display_name', 'Revenue (₹)', 'WoW Change']]
                        display_df.columns = ['Product', 'Revenue', 'WoW Change']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "decliners.csv",
                            "text/csv",
                            key="export_decliners"
                        )
                    else:
                        st.info("No decliners found")
                
                with col2:
                    st.markdown("#### 📈 Fast Movers (≥30% WoW Growth)")
                    if not movers['fast_movers'].empty:
                        display_df = movers['fast_movers'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_in_inr_current'].apply(format_inr)
                        display_df['WoW Change'] = display_df['wow_change'].apply(lambda x: f"+{x:.1f}%")
                        display_df = display_df[['display_name', 'Revenue (₹)', 'WoW Change']]
                        display_df.columns = ['Product', 'Revenue', 'WoW Change']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "fast_movers.csv",
                            "text/csv",
                            key="export_movers"
                        )
                    else:
                        st.info("No fast movers found")
                
                # Data summary
                st.markdown("---")
                st.markdown("#### 📋 Data Summary")
                st.info(f"📊 Showing {len(df):,} rows from {start_date} to {end_date}")
                
            else:
                st.warning("💡 No data found for the selected date range. Please check your date selection or upload data first.")
        
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