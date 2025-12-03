"""
RAG Question Analyzer

Analyzes user questions to extract intent, entities, query types, and context requirements.
This helps determine what data context to retrieve for RAG.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from calendar import monthrange

logger = logging.getLogger(__name__)


class QuestionAnalysis:
    """Result of question analysis"""
    
    def __init__(self):
        self.entities: Dict[str, List[str]] = {
            'products': [],      # SKUs, product names
            'regions': [],       # Cities, locations
            'dates': [],         # Date mentions
            'transaction_types': [],  # Shipment, Refund, etc.
        }
        self.query_type: Optional[str] = None  # 'aggregation', 'filtering', 'comparison', 'trend', 'top_n', etc.
        self.has_dates: bool = False
        self.date_range: Optional[Dict[str, Optional[str]]] = None  # {'start': ..., 'end': ...}
        self.month_mentions: List[int] = []  # Month numbers mentioned (1-12)
        self.year_mentions: List[int] = []   # Years mentioned
        self.relevant_columns: Set[str] = set()  # Columns likely needed
        self.keywords: Set[str] = set()  # Important keywords from question
        self.is_net_revenue_query: bool = False
        self.is_temporal_query: bool = False
        self.is_comparison_query: bool = False


def analyze_question(question: str) -> QuestionAnalysis:
    """
    Analyze a user question to extract intent, entities, and context requirements
    
    Args:
        question: User's natural language question
        
    Returns:
        QuestionAnalysis object with extracted information
    """
    analysis = QuestionAnalysis()
    question_lower = question.lower()
    
    # Extract keywords
    analysis.keywords = extract_keywords(question_lower)
    
    # Detect query type
    analysis.query_type = detect_query_type(question_lower)
    
    # Extract entities
    analysis.entities = extract_entities(question_lower)
    
    # Extract temporal information
    analysis.has_dates, analysis.date_range, analysis.month_mentions, analysis.year_mentions = extract_temporal_info(question_lower)
    analysis.is_temporal_query = analysis.has_dates
    
    # Detect relevant columns
    analysis.relevant_columns = detect_relevant_columns(question_lower, analysis.query_type)
    
    # Detect net revenue queries
    analysis.is_net_revenue_query = detect_net_revenue_query(question_lower)
    
    # Detect comparison queries
    analysis.is_comparison_query = detect_comparison_query(question_lower)
    
    logger.debug(f"Question analysis: type={analysis.query_type}, has_dates={analysis.has_dates}, "
                 f"entities={analysis.entities}, columns={analysis.relevant_columns}")
    
    return analysis


def extract_keywords(question_lower: str) -> Set[str]:
    """Extract important keywords from question"""
    # Common business keywords
    keywords = set()
    
    # Revenue-related
    if any(word in question_lower for word in ['revenue', 'sales', 'income', 'earnings']):
        keywords.add('revenue')
    
    # Transaction types
    if 'refund' in question_lower:
        keywords.add('refund')
    if 'shipment' in question_lower or 'order' in question_lower:
        keywords.add('shipment')
    if 'cancel' in question_lower:
        keywords.add('cancel')
    
    # Aggregation keywords
    if any(word in question_lower for word in ['total', 'sum', 'count', 'average', 'avg']):
        keywords.add('aggregation')
    
    # Top N keywords
    if any(word in question_lower for word in ['top', 'best', 'highest', 'most']):
        keywords.add('top_n')
    
    # Comparison keywords
    if any(word in question_lower for word in ['compare', 'versus', 'vs', 'difference', 'more than', 'less than']):
        keywords.add('comparison')
    
    # Temporal keywords
    if any(word in question_lower for word in ['month', 'year', 'week', 'day', 'recent', 'last', 'this']):
        keywords.add('temporal')
    
    return keywords


def detect_query_type(question_lower: str) -> str:
    """Detect the type of query"""
    # Net revenue queries
    if any(phrase in question_lower for phrase in ['net revenue', 'net earnings', 'net profit', 'after refunds', 'after costs']):
        return 'net_revenue'
    
    # Top N queries
    if any(phrase in question_lower for phrase in ['top ', 'best ', 'highest ', 'most ']):
        return 'top_n'
    
    # Comparison queries
    if any(phrase in question_lower for phrase in ['compare', 'versus', 'vs', 'difference between', 'more than', 'less than']):
        return 'comparison'
    
    # Trend queries
    if any(phrase in question_lower for phrase in ['trend', 'over time', 'growth', 'decline', 'change']):
        return 'trend'
    
    # Aggregation queries (default)
    if any(word in question_lower for word in ['total', 'sum', 'count', 'average', 'how many', 'what is']):
        return 'aggregation'
    
    # Filtering queries
    if any(word in question_lower for word in ['show', 'list', 'find', 'which', 'what']):
        return 'filtering'
    
    return 'general'


def extract_entities(question_lower: str) -> Dict[str, List[str]]:
    """Extract entities from question"""
    entities = {
        'products': [],
        'regions': [],
        'dates': [],
        'transaction_types': [],
    }
    
    # Extract transaction types
    if 'shipment' in question_lower or 'order' in question_lower:
        entities['transaction_types'].append('Shipment')
    if 'refund' in question_lower:
        entities['transaction_types'].append('Refund')
    if 'cancel' in question_lower or 'cancellation' in question_lower:
        entities['transaction_types'].append('Cancel')
    if 'free replacement' in question_lower or 'freereplacement' in question_lower:
        entities['transaction_types'].append('FreeReplacement')
    
    # Extract product mentions (SKU patterns, product names)
    # Look for patterns like "GP10_OM2P_STARTRC" or quoted product names
    sku_pattern = r'\b[A-Z0-9_]+[A-Z][A-Z0-9_]*\b'  # Pattern for SKUs
    potential_skus = re.findall(sku_pattern, question_lower.upper())
    if potential_skus:
        entities['products'].extend(potential_skus[:5])  # Limit to 5
    
    # Extract region mentions (common city names)
    common_cities = ['bangalore', 'mumbai', 'delhi', 'chennai', 'hyderabad', 'pune', 'kolkata', 'ahmedabad']
    for city in common_cities:
        if city in question_lower:
            entities['regions'].append(city.title())
    
    return entities


def extract_temporal_info(question_lower: str) -> Tuple[bool, Optional[Dict[str, Optional[str]]], List[int], List[int]]:
    """
    Extract temporal information from question
    
    Returns:
        (has_dates, date_range, month_mentions, year_mentions)
    """
    has_dates = False
    date_range = None
    month_mentions = []
    year_mentions = []
    
    # Month names mapping
    month_map = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sept': 9, 'sep': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12,
    }
    
    # Extract month mentions
    for month_name, month_num in month_map.items():
        if month_name in question_lower:
            month_mentions.append(month_num)
            has_dates = True
    
    # Extract year mentions (4-digit years)
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, question_lower)
    if years:
        year_mentions.extend([int(y) for y in years])
        has_dates = True
    
    # Extract relative dates
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    if 'last month' in question_lower:
        has_dates = True
        last_month = current_month - 1 if current_month > 1 else 12
        last_year = current_year if current_month > 1 else current_year - 1
        last_day = monthrange(last_year, last_month)[1]
        date_range = {
            'start': f'{last_year}-{last_month:02d}-01',
            'end': f'{last_year}-{last_month:02d}-{last_day}'
        }
        month_mentions.append(last_month)
        year_mentions.append(last_year)
    
    elif 'this month' in question_lower:
        has_dates = True
        last_day = monthrange(current_year, current_month)[1]
        date_range = {
            'start': f'{current_year}-{current_month:02d}-01',
            'end': f'{current_year}-{current_month:02d}-{last_day}'
        }
        month_mentions.append(current_month)
        year_mentions.append(current_year)
    
    elif 'last year' in question_lower:
        has_dates = True
        date_range = {
            'start': f'{current_year - 1}-01-01',
            'end': f'{current_year - 1}-12-31'
        }
        year_mentions.append(current_year - 1)
    
    elif 'this year' in question_lower:
        has_dates = True
        date_range = {
            'start': f'{current_year}-01-01',
            'end': f'{current_year}-12-31'
        }
        year_mentions.append(current_year)
    
    # If specific month mentioned, create date range
    if month_mentions and not date_range:
        month = month_mentions[0]
        year = year_mentions[0] if year_mentions else current_year
        last_day = monthrange(year, month)[1]
        date_range = {
            'start': f'{year}-{month:02d}-01',
            'end': f'{year}-{month:02d}-{last_day}'
        }
        has_dates = True
    
    return has_dates, date_range, month_mentions, year_mentions


def detect_relevant_columns(question_lower: str, query_type: str) -> Set[str]:
    """Detect which columns are likely needed for this query"""
    columns = set()
    
    # Revenue-related columns
    if any(word in question_lower for word in ['revenue', 'sales', 'amount', 'income']):
        columns.add('revenue_amount')
    
    # Date columns
    if any(word in question_lower for word in ['date', 'month', 'year', 'day', 'when']):
        columns.add('order_date')
        columns.add('Invoice Date')  # Also check for original column name
    
    # Transaction type
    if any(word in question_lower for word in ['refund', 'shipment', 'cancel', 'transaction']):
        columns.add('transaction_type')
    
    # Product/SKU columns
    if any(word in question_lower for word in ['product', 'sku', 'item', 'asin']):
        columns.add('sku')
    
    # Region columns
    if any(word in question_lower for word in ['city', 'region', 'location', 'where']):
        columns.add('region')
    
    # Quantity
    if 'quantity' in question_lower or 'qty' in question_lower:
        columns.add('quantity')
    
    return columns


def detect_net_revenue_query(question_lower: str) -> bool:
    """Detect if this is a net revenue query"""
    net_revenue_keywords = [
        'net revenue', 'net earnings', 'net profit', 'after refunds',
        'after costs', 'after deductions', 'after cancels', 'after cancellations'
    ]
    return any(keyword in question_lower for keyword in net_revenue_keywords)


def detect_comparison_query(question_lower: str) -> bool:
    """Detect if this is a comparison query"""
    comparison_keywords = [
        'compare', 'versus', 'vs', 'difference', 'more than', 'less than',
        'higher', 'lower', 'better', 'worse'
    ]
    return any(keyword in question_lower for keyword in comparison_keywords)

