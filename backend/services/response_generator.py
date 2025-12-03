"""
Response Generator Service

Uses Gemini to generate natural language responses from query results.
Handles both METRIC and ADVISORY questions.

CRITICAL: Never invents numbers - only uses data from query_results.
"""

import httpx
import logging
import json
from typing import Dict, Any, Optional
from services.api_key_service import APIKeyService
from services.metric_queries import QueryResult

logger = logging.getLogger(__name__)


def format_currency(value: float) -> str:
    """Format value as Indian Rupees"""
    if value >= 10000000:  # 1 crore
        return f"₹{value/10000000:.2f} Cr"
    elif value >= 100000:  # 1 lakh
        return f"₹{value/100000:.2f} L"
    else:
        return f"₹{value:,.2f}"


def format_metric_response(
    question: str,
    query_result: QueryResult,
    intent_type: str,
    metric_type: Optional[str] = None
) -> str:
    """
    Format query results as a clean natural language response.
    This is a FAST fallback when Gemini is not available.
    """
    if not query_result.success:
        return f"Sorry, I couldn't retrieve the data: {query_result.error}"
    
    data = query_result.data
    date_range = ""
    if data.get('start_date') and data.get('end_date'):
        date_range = f" from {data['start_date']} to {data['end_date']}"
    
    # Format based on metric type
    if metric_type == 'revenue' or 'total_revenue' in data:
        revenue = data.get('total_revenue', 0)
        count = data.get('transaction_count', 0)
        return f"Your total revenue{date_range} is {format_currency(revenue)} across {count:,} shipments."
    
    elif 'net_revenue' in data:
        net = data.get('net_revenue', 0)
        gross = data.get('gross_revenue', 0)
        refunds = data.get('refund_amount', 0)
        cancels = data.get('cancel_amount', 0)
        free_repl = data.get('free_replacement_amount', 0)
        
        return (f"Your net revenue{date_range} is {format_currency(net)}.\n\n"
                f"Breakdown:\n"
                f"• Gross revenue: {format_currency(gross)}\n"
                f"• Refunds: -{format_currency(refunds)}\n"
                f"• Cancellations: -{format_currency(cancels)}\n"
                f"• Free replacements: -{format_currency(free_repl)}")
    
    elif 'refund_rate' in data:
        if 'groups' in data:
            lines = [f"Refund rates{date_range}:\n"]
            for g in data['groups'][:10]:
                lines.append(f"• {g['group_key']}: {g['refund_rate']:.1f}% ({g['refund_count']} refunds / {g['shipment_count']} shipments)")
            return "\n".join(lines)
        else:
            rate = data.get('refund_rate', 0)
            refunds = data.get('refund_count', 0)
            shipments = data.get('shipment_count', 0)
            return f"Your refund rate{date_range} is {rate:.1f}% ({refunds:,} refunds out of {shipments:,} shipments)."
    
    elif 'products' in data:
        products = data.get('products', [])
        if not products:
            return f"No products found{date_range}."
        
        lines = [f"Top {len(products)} products by revenue{date_range}:\n"]
        for i, p in enumerate(products, 1):
            lines.append(f"{i}. {p['sku']}: {format_currency(p['total_revenue'])} ({p['transaction_count']:,} orders)")
        return "\n".join(lines)
    
    elif 'regions' in data:
        regions = data.get('regions', [])
        if not regions:
            return f"No region data found{date_range}."
        
        lines = [f"Revenue by region{date_range}:\n"]
        for r in regions[:10]:
            lines.append(f"• {r['region']}: {format_currency(r['total_revenue'])} ({r['transaction_count']:,} orders)")
        return "\n".join(lines)
    
    elif 'transactions' in data:
        transactions = data.get('transactions', [])
        total = data.get('total', 0)
        
        lines = [f"Transaction summary{date_range} (Total: {total:,}):\n"]
        for t in transactions:
            lines.append(f"• {t['transaction_type']}: {t['count']:,}")
        return "\n".join(lines)
    
    elif 'entities' in data:
        # Comparison
        entities = data.get('entities', [])
        if len(entities) >= 2:
            e1, e2 = entities[0], entities[1]
            diff = e1['total_revenue'] - e2['total_revenue']
            return (f"Comparison{date_range}:\n\n"
                    f"• {e1['entity']}: {format_currency(e1['total_revenue'])} ({e1['transaction_count']:,} orders)\n"
                    f"• {e2['entity']}: {format_currency(e2['total_revenue'])} ({e2['transaction_count']:,} orders)\n\n"
                    f"Difference: {format_currency(abs(diff))} in favor of {e1['entity'] if diff > 0 else e2['entity']}")
        elif entities:
            e = entities[0]
            return f"{e['entity']}{date_range}: {format_currency(e['total_revenue'])} ({e['transaction_count']:,} orders)"
    
    # Generic fallback
    return f"Query completed{date_range}. Data: {json.dumps(data, default=str)[:500]}"


async def generate_response_with_gemini(
    question: str,
    query_result: QueryResult,
    intent_type: str,
    user_id: str,
    rag_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Use Gemini to generate a natural language response.
    
    For METRIC: Format numbers nicely, add context
    For ADVISORY: Analyze data, provide business recommendations
    
    CRITICAL: Only uses numbers from query_result, never invents.
    """
    # Get Gemini API key
    gemini_key_info = APIKeyService.get_api_key(user_id, 'gemini', decrypt=True)
    if not gemini_key_info or not gemini_key_info.get('key') or not gemini_key_info.get('enabled', True):
        logger.info("No Gemini key available, using formatted response")
        return format_metric_response(question, query_result, intent_type)
    
    api_key = gemini_key_info['key']
    
    # Build the prompt
    data_json = json.dumps(query_result.data, default=str, indent=2)
    
    if intent_type == "ADVISORY":
        # Advisory prompt: analyze and recommend
        prompt = f"""You are a business data analyst. The user asked: "{question}"

Here is the data from their DuckDB analytics database:

{data_json}

Based on this ACTUAL data (do not invent any numbers), provide:
1. A clear summary of the key findings
2. Identify any concerning trends (high refund rates, declining products, etc.)
3. Provide 2-3 actionable recommendations

CRITICAL RULES:
- Only cite numbers that appear in the data above
- If something looks concerning, explain why
- Be specific and actionable in your recommendations
- Format currency values in Indian Rupees (₹)
- Keep your response concise but insightful"""

    else:
        # Metric prompt: format and explain
        prompt = f"""You are a helpful data assistant. The user asked: "{question}"

Here is the query result from their database:

{data_json}

Please respond in a clear, natural way that:
1. Directly answers their question using the data above
2. Formats numbers nicely (use ₹ for currency, abbreviate large numbers like ₹1.5L or ₹2.3Cr)
3. Adds brief context if helpful (e.g., "across X orders" or "during the selected period")

CRITICAL: Only use numbers from the data above. Do not invent or estimate any values.
Keep your response concise - 2-3 sentences max for simple metrics."""

    # Add RAG context if available
    if rag_context and rag_context.get('statistics'):
        stats = rag_context['statistics']
        prompt += f"\n\nAdditional context: Database has {stats.get('total_records', 0):,} total records."
    
    # Call Gemini
    try:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.3,
            }
        }
        
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        text = candidate['content']['parts'][0].get('text', '')
                        if text:
                            logger.info("Gemini response generated successfully")
                            return text.strip()
            
            logger.warning(f"Gemini API returned {response.status_code}, falling back to formatted response")
            
    except Exception as e:
        logger.error(f"Error calling Gemini for response generation: {e}")
    
    # Fallback to formatted response
    return format_metric_response(question, query_result, intent_type)


def format_advisory_response(
    question: str,
    query_result: QueryResult
) -> str:
    """
    Format advisory data as actionable insights.
    This is the fallback when Gemini is not available.
    """
    if not query_result.success:
        return f"Sorry, I couldn't retrieve the data: {query_result.error}"
    
    data = query_result.data
    summary = data.get('summary', {})
    top_products = data.get('top_products', [])
    top_regions = data.get('top_regions', [])
    high_refunds = data.get('high_refund_products', [])
    
    date_range = ""
    if data.get('start_date') and data.get('end_date'):
        date_range = f" ({data['start_date']} to {data['end_date']})"
    
    lines = [f"**Business Analysis{date_range}**\n"]
    
    # Summary
    if summary:
        gross = summary.get('gross_revenue', 0)
        refunds = summary.get('total_refunds', 0)
        refund_count = summary.get('refund_count', 0)
        shipment_count = summary.get('shipment_count', 0)
        
        lines.append("**Overview:**")
        lines.append(f"• Total Revenue: {format_currency(gross)}")
        lines.append(f"• Total Orders: {shipment_count:,}")
        if shipment_count > 0:
            refund_rate = (refund_count / shipment_count) * 100
            lines.append(f"• Refund Rate: {refund_rate:.1f}% ({refund_count:,} refunds)")
        lines.append("")
    
    # Top performers
    if top_products:
        lines.append("**Top Products:**")
        for i, p in enumerate(top_products[:5], 1):
            lines.append(f"{i}. {p['sku']}: {format_currency(p['revenue'])}")
        lines.append("")
    
    # Top regions
    if top_regions:
        lines.append("**Top Regions:**")
        for r in top_regions[:5]:
            lines.append(f"• {r['region']}: {format_currency(r['revenue'])}")
        lines.append("")
    
    # Concerns
    if high_refunds:
        lines.append("**⚠️ Products with High Refund Rates:**")
        for p in high_refunds:
            lines.append(f"• {p['sku']}: {p['refund_rate']:.1f}% refund rate ({p['refund_count']} refunds)")
        lines.append("")
        lines.append("**Recommendation:** Investigate these products for quality issues or misleading descriptions.")
    
    return "\n".join(lines)


async def generate_conversational_response(
    question: str,
    user_id: str
) -> str:
    """
    Generate a conversational response using Gemini.
    For greetings, help requests, and general chat.
    """
    # Get Gemini API key
    gemini_key_info = APIKeyService.get_api_key(user_id, 'gemini', decrypt=True)
    if not gemini_key_info or not gemini_key_info.get('key') or not gemini_key_info.get('enabled', True):
        return "Hello! I'm your AI data analyst. How can I help you explore your sales data today?"
    
    api_key = gemini_key_info['key']
    
    prompt = f"""You are a friendly AI data analyst assistant. The user said: "{question}"

Respond naturally and helpfully. If they're greeting you, greet them back warmly and briefly mention you can help with their sales data analysis.

If they ask what you can do, explain that you can:
- Answer questions about revenue, sales, refunds
- Show top products and regions
- Identify trends and problem areas  
- Provide business recommendations

Keep responses concise and friendly. Don't be overly formal."""

    try:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 200,
                "temperature": 0.7,
            }
        }
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        text = candidate['content']['parts'][0].get('text', '')
                        if text:
                            return text.strip()
    except Exception as e:
        logger.error(f"Error generating conversational response: {e}")
    
    return "Hello! I'm your AI data analyst. I can help you analyze your sales data - just ask me about revenue, top products, refund rates, or anything else!"
