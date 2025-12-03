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
from typing import Dict, List, Optional, Literal

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
    
    # Check for METRIC patterns
    if intent.type != "ADVISORY":
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

