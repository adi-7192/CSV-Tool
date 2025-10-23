# CSV Analytics Dashboard

Simple, clean CSV analytics with automatic column mapping and natural language chat.

## Features

✅ **Automatic Column Mapping** - Upload CSV and columns are mapped automatically  
✅ **Business-Grade Dashboard** - Comprehensive KPIs with date filtering and visualizations  
✅ **Date Range Filtering** - Quick presets (Last 7/30 days, This/Last Month) + custom ranges  
✅ **Advanced Analytics** - WoW comparisons, movers & decliners, trend analysis  
✅ **Interactive Charts** - Revenue trends, status breakdowns, regional analysis  
✅ **Export Functionality** - Download CSV reports for all data tables  
✅ **INR Currency** - All amounts displayed in Indian Rupees (₹)  
✅ **Natural Language Chat** - Ask questions about your data  
✅ **Persistent Mappings** - Remembers your column choices for future uploads  

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Upload CSV:**
   - Click "Upload CSV File" in sidebar
   - Select your file
   - Click "Store Data" to process

4. **View Business Dashboard:**
   - Select date range (presets or custom)
   - See comprehensive KPIs with WoW comparisons
   - View interactive charts and trend analysis
   - Export data tables as CSV

5. **Chat about data:**
   - Go to "💬 Chat" tab
   - Ask questions like "What's the total revenue?" or "Show me top products"

## Supported CSV Format

The app automatically recognizes these columns:

| Field | Recognized Headers |
|-------|-------------------|
| **Order Date** | Order Date, Invoice Date, Shipment Date, Date |
| **Order ID** | Order ID, Order Number, Invoice ID, Order No |
| **SKU** | SKU, ASIN, Product ID, Item ID |
| **Product Name** | Product Name, Item Description, Title, Product |
| **Quantity** | Quantity, Qty, Units |
| **Revenue** | Invoice Amount, Order Amount, Total Amount, Revenue |
| **Region** | Region, City, Market |
| **Status** | Status, Order Status |

## Example Questions

### Numeric Questions
- "What's the total revenue?"
- "How many orders?"
- "Top 5 SKUs by revenue?"
- "Total units sold?"

### Descriptive Questions
- "Show me performance trends"
- "What's our best product?"
- "Any declining trends?"
- "Overall business summary"

## How It Works

1. **Upload CSV** → App automatically maps columns using synonyms
2. **Confirm Mapping** → Only if ambiguous (rare)
3. **Store Data** → Saves to SQLite with standardized schema
4. **View KPIs** → Instant metrics and charts
5. **Chat** → Ask questions, get grounded answers

## Data Storage

- **Database:** `data.db` (SQLite)
- **Table:** `sales` with standardized columns
- **Mappings:** `data/mapping.json` (saved for reuse)

## Requirements

- Python 3.8+
- streamlit
- pandas
- plotly
- numpy

## Architecture

- **Single file app** - Everything in `app.py`
- **Auto-mapping** - Uses synonym matching + type inference
- **SQL queries** - For numeric questions (never guesses)
- **Simple chat** - Direct answers based on actual data
- **Clean UI** - Sidebar upload, main tabs for KPIs/Chat

## Troubleshooting

**Issue:** Columns not mapping correctly  
**Solution:** Use "Change Column Mapping" button to adjust

**Issue:** Chat not working  
**Solution:** Make sure data is stored first (click "Store Data")

**Issue:** Wrong currency format  
**Solution:** App uses INR (₹). Revenue column should contain numeric values.

## Support

The app is designed to be simple and self-contained. All logic is in `app.py` with clear comments.

For issues, check the console logs - the app prints debugging information.