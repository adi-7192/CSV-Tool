"""
Intent Classification Service

Classifies user questions into:
- METRIC: Quantitative questions (revenue, count, top N, etc.)
- ADVISORY: Business advice questions (where to focus, trends, recommendations)
- EXPLORATION: Data exploration (show me, list, what products)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple

logger = logging.getLogger(__name__)

IntentType = Literal["METRIC", "ADVISORY", "EXPLORATION", "CONVERSATIONAL"]


@dataclass
class QuestionIntent:
    """Structured representation of question intent"""
    type: IntentType
    metric_type: Optional[str] = None  # revenue, refund_rate, top_products, count, comparison
    entities: Dict[str, List[str]] = field(default_factory=dict)  # products, regions, time
    comparison: Optional[str] = None  # vs, compare, difference
    top_n: Optional[int] = None  # for "top 5", "top 10" queries
    group_by: Optional[str] = None  # region, product, month, etc.


# Pattern definitions for intent classification
METRIC_PATTERNS = {
    'revenue': [
        r'\b(total|gross|net)\s+revenue\b',
        r'\brevenue\b.*\b(total|sum|amount)\b',
        r'\bwhat\s+(was|is|were)\s+(my|the|our)\s+.*revenue\b',
        r'\bhow\s+much\s+.*revenue\b',
        r'\bsales\s+(amount|total|figure)\b',
    ],
    'refund_rate': [
        r'\brefund\s+rate\b',
        r'\breturn\s+rate\b',
        r'\b(percentage|%)\s+of\s+(refunds|returns)\b',
        r'\bhow\s+many\s+(refunds|returns)\b',
    ],
    'top_products': [
        r'\btop\s+\d+\s+(products?|skus?|items?)\b',
        r'\bbest\s+(selling|performing)\s+(products?|skus?)\b',
        r'\bhighest\s+(revenue|sales)\s+(products?|skus?)\b',
    ],
    'count': [
        r'\bhow\s+many\s+(orders?|transactions?|shipments?|products?)\b',
        r'\b(total|number\s+of)\s+(orders?|transactions?|shipments?)\b',
        r'\bcount\s+of\b',
    ],
    'comparison': [
        r'\bvs\.?\b',
        r'\bversus\b',
        r'\bcompare\b',
        r'\bdifference\s+between\b',
        r'\b(\w+)\s+compared\s+to\s+(\w+)\b',
    ],
    'average': [
        r'\baverage\s+(order|revenue|sales)\b',
        r'\baov\b',  # average order value
        r'\bmean\s+(order|revenue)\b',
    ],
}

ADVISORY_PATTERNS = [
    r'\bwhere\s+should\s+(i|we)\s+(focus|invest|improve)\b',
    r'\bany\s+(worrying|concerning|alarming)\s+trends?\b',
    r'\bwhat\s+(should|can)\s+(i|we)\s+do\b',
    r'\brecommend(ation)?s?\b',
    r'\badvice\b',
    r'\binsights?\b',
    r'\bopportunities?\b',
    r'\bproblems?\s+(areas?|products?|regions?)\b',
    r'\bwhat\s+is\s+(going\s+)?(wrong|bad)\b',
    r'\bhow\s+can\s+(i|we)\s+improve\b',
    r'\bwhat\s+are\s+the\s+(issues?|problems?)\b',
    # Analysis patterns - route to ADVISORY for comprehensive data analysis
    r'\b(analyse|analyze)\s+(my|the|our)?\s*(sales?|data|revenue|performance|business)?\b',
    r'\b(tell\s+me|give\s+me)\s+(about|the)\s+(key\s+)?(highlights?|summary|overview)\b',
    r'\bkey\s+(highlights?|points?|takeaways?|metrics?)\b',
    r'\bsummary\s+(of|for)\b',
    r'\boverview\s+(of|for)\b',
    r'\bperformance\s+(analysis|review|summary)\b',
    r'\bhow\s+(am\s+i|are\s+we|is\s+my|did\s+i)\s+(doing|performing)\b',
    r'\bbusiness\s+(health|performance|status)\b',
]

EXPLORATION_PATTERNS = [
    r'\bshow\s+(me|all)\b',
    r'\blist\s+(all|the)?\b',
    r'\bwhat\s+(products?|items?|skus?)\s+(do|did|are)\b',
    r'\bwhich\s+(products?|regions?|items?)\b',
    r'\bgive\s+me\b',
    r'\bdisplay\b',
]

# Add patterns for declining/growing products (should be METRIC, not EXPLORATION)
DECLINING_PATTERNS = [
    r'\b(which|what)\s+(products?|skus?|items?)\s+(are\s+)?(declining|falling|dropping|decreasing|underperforming)\b',
    r'\bdeclining\s+(products?|skus?|items?)\b',
    r'\b(products?|skus?)\s+(are\s+)?(declining|falling|dropping|decreasing)\b',
    r'\bworst\s+(performing\s+)?(products?|skus?)\b',
]

GROWING_PATTERNS = [
    r'\b(which|what)\s+(products?|skus?|items?)\s+(are\s+)?(growing|rising|increasing|improving|best\s+performing)\b',
    r'\bgrowing\s+(products?|skus?|items?)\b',
    r'\b(products?|skus?)\s+(are\s+)?(growing|rising|increasing|improving)\b',
    r'\bbest\s+(performing\s+)?(products?|skus?)\b',
    r'\bmovers\b',
]

# Entity extraction patterns
ENTITY_PATTERNS = {
    'region': [
        r'\bin\s+(bangalore|mumbai|delhi|chennai|hyderabad|kolkata|pune|ahmedabad)\b',
        r'\b(bangalore|mumbai|delhi|chennai|hyderabad|kolkata|pune|ahmedabad)\s+(vs\.?|versus|compared)\b',
    ],
    'time': [
        r'\b(last|this|next)\s+(week|month|quarter|year)\b',
        r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
        r'\b(q1|q2|q3|q4)\b',
    ],
    'product': [
        r'\bsku\s*[:\s]?\s*([A-Za-z0-9\-_]+)\b',
        r'\bproduct\s*[:\s]?\s*([A-Za-z0-9\-_]+)\b',
    ],
}

# Top N extraction pattern
TOP_N_PATTERN = r'\btop\s+(\d+)\b'

# Group by patterns
GROUP_BY_PATTERNS = {
    'region': [r'\bby\s+(region|city|location)\b', r'\b(region|city)\s+wise\b'],
    'product': [r'\bby\s+(product|sku|item)\b', r'\b(product|sku)\s+wise\b'],
    'month': [r'\bby\s+month\b', r'\bmonth\s+wise\b', r'\bmonthly\b'],
    'day': [r'\bby\s+day\b', r'\bdaily\b', r'\bday\s+wise\b'],
}

# Conversational patterns - greetings and non-data questions
CONVERSATIONAL_PATTERNS = [
    r'^(hi|hello|hey|hii|hola|howdy)\b',
    r'^(good\s+(morning|afternoon|evening|night))\b',
    r'\bhow\s+are\s+you\b',
    r'\bwhat\s+are\s+you\b',
    r'\bwho\s+are\s+you\b',
    r'\bwhat\s+can\s+you\s+do\b',
    r'\bhelp\s*me\b',
    r'^(thanks|thank\s+you|thx)\b',
    r'^(bye|goodbye|see\s+you)\b',
    r'\btell\s+me\s+about\s+(yourself|you)\b',
]

# Business/data-related keywords that indicate a valid question
BUSINESS_KEYWORDS = [
    'revenue', 'sales', 'profit', 'margin', 'refund', 'return', 'order', 'product', 'sku',
    'item', 'customer', 'city', 'region', 'location', 'month', 'year', 'date', 'period',
    'compare', 'comparison', 'top', 'best', 'worst', 'highest', 'lowest', 'total', 'sum',
    'count', 'average', 'rate', 'percentage', 'growth', 'decline', 'trend', 'analysis',
    'insight', 'recommend', 'focus', 'strategy', 'business', 'performance', 'metric',
    'mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad'
]


def is_valid_question(question: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a question is meaningful and related to business/data analysis.
    
    Args:
        question: User's input question
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if question is valid, False otherwise
        - error_message: Error message if invalid, None if valid
    """
    question_stripped = question.strip()
    
    # Check if question is too short
    if len(question_stripped) < 2:
        return False, "Your question is too short. Please ask a meaningful question about your sales data."
    
    # Check if question is just repeated characters (e.g., "jjj", "aaa", "111")
    if len(question_stripped) >= 2:
        # Remove spaces and check if all characters are the same
        question_no_spaces = question_stripped.lower().replace(' ', '')
        if len(question_no_spaces) > 0 and len(set(question_no_spaces)) == 1:
            return False, "Your question doesn't make sense. Please ask a meaningful question about your sales data, such as 'What was my revenue last month?' or 'Show me top products'."
    
    # Check if question contains only numbers or special characters (no meaningful words)
    question_alpha = re.sub(r'[^a-zA-Z]', '', question_stripped)
    if len(question_alpha) < 2:
        return False, "Your question needs to contain meaningful words. Please ask about your sales data, products, revenue, or business metrics."
    
    # Check for random character sequences (e.g., "asdf", "qwerty", "zxcv")
    # If question is short and doesn't contain common English words, it's likely invalid
    if len(question_stripped) <= 10:
        # Check if it's a common keyboard pattern
        keyboard_patterns = ['asdf', 'qwerty', 'zxcv', 'hjkl', 'fghj', 'tyui']
        if question_stripped.lower() in keyboard_patterns:
            return False, "Your question doesn't make sense. Please ask a meaningful question about your sales data."
        
        # Check if it's mostly consonants without vowels (likely random typing)
        vowels = set('aeiou')
        question_lower_chars = set(question_stripped.lower())
        if len(question_lower_chars) > 0:
            vowel_ratio = len(question_lower_chars & vowels) / len(question_lower_chars)
            # If less than 20% vowels and no business keywords, likely invalid
            if vowel_ratio < 0.2 and not any(keyword in question_stripped.lower() for keyword in BUSINESS_KEYWORDS):
                return False, "Your question doesn't make sense. Please ask a meaningful question about your sales data."
    
    # Check if question contains any business/data-related keywords
    question_lower = question_stripped.lower()
    has_business_keyword = any(keyword in question_lower for keyword in BUSINESS_KEYWORDS)
    
    # Check for conversational patterns (these are valid even without business keywords)
    has_conversational_pattern = any(
        re.search(pattern, question_lower, re.IGNORECASE) 
        for pattern in CONVERSATIONAL_PATTERNS
    )
    
    # Check for common question words (what, how, which, show, tell, etc.)
    question_words = ['what', 'how', 'which', 'show', 'tell', 'give', 'find', 'list', 
                     'where', 'when', 'why', 'who', 'analyze', 'analyse', 'compare']
    has_question_word = any(word in question_lower for word in question_words)
    
    # If it's conversational, it's valid
    if has_conversational_pattern:
        return True, None
    
    # If it has business keywords, it's valid
    if has_business_keyword:
        return True, None
    
    # If it has question words and is at least 5 characters, it might be valid
    if has_question_word and len(question_stripped) >= 5:
        return True, None
    
    # Check if question is too short even with question words
    if len(question_stripped) < 5:
        return False, "Your question is too short or unclear. Please ask a complete question about your sales data."
    
    # If none of the above, it's likely invalid
    return False, "I couldn't understand your question. Please ask about your sales data, such as:\n- 'What was my revenue last month?'\n- 'Show me top 10 products'\n- 'Compare Mumbai vs Delhi sales'\n- 'Which products are declining?'"


def classify_intent(question: str) -> QuestionIntent:
    """
    Classify a natural language question into structured intent
    
    Args:
        question: User's natural language question
        
    Returns:
        QuestionIntent with type, metric_type, entities, etc.
    """
    question_lower = question.lower().strip()
    
    # Initialize intent
    intent = QuestionIntent(type="EXPLORATION")  # Default
    
    # Check for CONVERSATIONAL patterns first (highest priority)
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            intent.type = "CONVERSATIONAL"
            logger.info(f"Intent classified as CONVERSATIONAL: matched pattern '{pattern}'")
            return intent  # Return immediately for conversational
    
    # Check for ADVISORY patterns (high priority)
    for pattern in ADVISORY_PATTERNS:
        if re.search(pattern, question_lower, re.IGNORECASE):
            intent.type = "ADVISORY"
            logger.info(f"Intent classified as ADVISORY: matched pattern '{pattern}'")
            break
    
    # Check for declining/growing products (METRIC type)
    if intent.type != "ADVISORY":
        for pattern in DECLINING_PATTERNS:
            if re.search(pattern, question_lower, re.IGNORECASE):
                intent.type = "METRIC"
                intent.metric_type = "declining_products"
                logger.info(f"Intent classified as METRIC (declining_products): matched pattern '{pattern}'")
                break
        
        if intent.type != "METRIC":
            for pattern in GROWING_PATTERNS:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    intent.type = "METRIC"
                    intent.metric_type = "growing_products"
                    logger.info(f"Intent classified as METRIC (growing_products): matched pattern '{pattern}'")
                    break
    
    # Check for METRIC patterns
    if intent.type != "ADVISORY" and intent.type != "METRIC":
        for metric_type, patterns in METRIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    intent.type = "METRIC"
                    intent.metric_type = metric_type
                    logger.info(f"Intent classified as METRIC ({metric_type}): matched pattern '{pattern}'")
                    break
            if intent.type == "METRIC":
                break
    
    # Check for EXPLORATION patterns (lowest priority)
    if intent.type == "EXPLORATION":
        for pattern in EXPLORATION_PATTERNS:
            if re.search(pattern, question_lower, re.IGNORECASE):
                logger.info(f"Intent classified as EXPLORATION: matched pattern '{pattern}'")
                break
    
    # Extract entities
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, question_lower, re.IGNORECASE)
            if matches:
                if entity_type not in intent.entities:
                    intent.entities[entity_type] = []
                for match in matches:
                    if isinstance(match, tuple):
                        intent.entities[entity_type].extend([m for m in match if m])
                    else:
                        intent.entities[entity_type].append(match)
    
    # Extract top N
    top_n_match = re.search(TOP_N_PATTERN, question_lower, re.IGNORECASE)
    if top_n_match:
        intent.top_n = int(top_n_match.group(1))
    
    # Extract group by
    for group_type, patterns in GROUP_BY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                intent.group_by = group_type
                break
        if intent.group_by:
            break
    
    # Check for comparison
    for pattern in METRIC_PATTERNS.get('comparison', []):
        if re.search(pattern, question_lower, re.IGNORECASE):
            intent.comparison = "vs"
            if intent.type == "EXPLORATION":
                intent.type = "METRIC"
                intent.metric_type = "comparison"
            break
    
    logger.info(f"Final intent: type={intent.type}, metric={intent.metric_type}, "
                f"entities={intent.entities}, top_n={intent.top_n}, group_by={intent.group_by}")
    
    return intent


def get_intent_summary(intent: QuestionIntent) -> str:
    """Generate a human-readable summary of the intent"""
    parts = [f"Type: {intent.type}"]
    
    if intent.metric_type:
        parts.append(f"Metric: {intent.metric_type}")
    if intent.entities:
        parts.append(f"Entities: {intent.entities}")
    if intent.top_n:
        parts.append(f"Top N: {intent.top_n}")
    if intent.group_by:
        parts.append(f"Group by: {intent.group_by}")
    if intent.comparison:
        parts.append(f"Comparison: {intent.comparison}")
    
    return " | ".join(parts)

