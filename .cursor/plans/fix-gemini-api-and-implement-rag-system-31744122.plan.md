<!-- 31744122-55fe-45a6-9778-cb3bbfdb2d90 18772b1e-51d2-4fb6-bd86-fd2932c518e8 -->
# AI Data Analyst Upgrade Plan

## Current Architecture Summary

### Backend Components

| Component | Location | Purpose |

|-----------|----------|---------|

| Chat Endpoint | [`backend/api/routes/chat.py`](backend/api/routes/chat.py) | `POST /api/chat/ask` - receives question, returns answer |

| AI Service | [`backend/core/ai_service.py`](backend/core/ai_service.py) | SQL generation with LLM, validation, formatting |

| External LLM | [`backend/utils/external_llm.py`](backend/utils/external_llm.py) | Gemini/OpenAI/Anthropic API calls |

| RAG Service | [`backend/services/rag_service.py`](backend/services/rag_service.py) | Context retrieval for prompts |

### Frontend Components

| Component | Location | Purpose |

|-----------|----------|---------|

| AIAnalyst Page | [`frontend/src/pages/AIAnalyst.tsx`](frontend/src/pages/AIAnalyst.tsx) | Chat UI with message bubbles |

| Chat Service | [`frontend/src/services/api.ts`](frontend/src/services/api.ts) | Calls `/api/chat/ask` |

### Current Request/Response

**Request:**

```typescript
{ question: string, context?: { start_date?, end_date? } }
```

**Response:**

```typescript
{ answer: string, sql: string, data: any, execution_time: number, 
  error: string, provider: string, confidence: number }
```

## Current Behavior

1. LLM generates free-form SQL from natural language
2. SQL is validated for safety (no DROP/DELETE)
3. SQL is post-processed to fix date casting issues
4. Query is executed against DuckDB
5. Results formatted as simple text ("The answer is: X")

## Problems Identified

| Issue | Impact | Example |

|-------|--------|---------|

| Date range not passed | Queries ignore selected dates | "Revenue last month" returns NaN |

| Free-form SQL fragile | Wrong columns, aggregations | "hello" returns "1" |

| No intent classification | Cannot distinguish question types | "Where should I focus?" fails |

| Basic answer formatting | No business insight or advice | Just raw numbers |

| Advisory questions unsupported | Cannot analyze trends or advise | "Any worrying trends?" fails |

| LLM can hallucinate | Might invent numbers | SQL result ignored in answer |

## Target Behavior

1. User asks ANY question about their data
2. System classifies intent: METRIC, ADVISORY, or EXPLORATION
3. For metrics: Use verified SQL templates (not free-form)
4. For advisory: Query relevant data, then use Gemini to analyze
5. Gemini generates human answer using ONLY query results
6. Never invent numbers - cite exact values from data

## Transition Plan

### Phase 1: Fix Date Range Passing (Quick Win)

**Frontend** ([`AIAnalyst.tsx`](frontend/src/pages/AIAnalyst.tsx)):

- Get selected date range from the header date picker
- Pass `context: { start_date, end_date }` in chat request

**Backend** ([`chat.py`](backend/api/routes/chat.py)):

- Extract dates from request context
- Pass to SQL generation for filtering

### Phase 2: Intent Classification Service

Create [`backend/services/intent_classifier.py`](backend/services/intent_classifier.py):

```python
class QuestionIntent:
    type: Literal["METRIC", "ADVISORY", "EXPLORATION"]
    metric_type: Optional[str]  # revenue, refund_rate, top_products, etc.
    entities: Dict[str, Any]    # products, regions, time periods
    comparison: Optional[str]   # vs, compare, difference
```

Use pattern matching + lightweight LLM call to classify.

### Phase 3: Safe Metric Query Templates

Create [`backend/services/metric_queries.py`](backend/services/metric_queries.py):

Define verified SQL templates for common metrics:

- `total_revenue(start_date, end_date, region?, product?)`
- `net_revenue(start_date, end_date)` - uses correct formula
- `refund_rate(start_date, end_date, by?)`
- `top_products(n, metric, start_date, end_date)`
- `comparison(metric, period1, period2)`

Map intent types to templates - no free-form SQL for common cases.

### Phase 4: LLM Response Generator

Create [`backend/services/response_generator.py`](backend/services/response_generator.py):

```python
async def generate_response(
    question: str,
    intent: QuestionIntent,
    query_results: Dict[str, Any],
    context: Dict[str, Any]
) -> str:
    """
    Use Gemini to generate natural language response.
    
    For METRIC: Format numbers nicely, add context
    For ADVISORY: Analyze trends, provide recommendations
    
    CRITICAL: Only use numbers from query_results, never invent.
    """
```

### Phase 5: Refactor Chat Endpoint

Update [`backend/api/routes/chat.py`](backend/api/routes/chat.py):

```python
@router.post("/ask")
async def ask_question(request: ChatRequest):
    # 1. Classify intent
    intent = classify_intent(request.question)
    
    # 2. Get query results
    if intent.type == "METRIC":
        # Use safe template
        results = execute_metric_query(intent, context)
    else:
        # Fall back to LLM SQL generation (existing code)
        results = await generate_and_execute_sql(...)
    
    # 3. Generate response with Gemini
    answer = await generate_response(
        question=request.question,
        intent=intent,
        query_results=results,
        context=request.context
    )
    
    return ChatResponse(answer=answer, ...)
```

## What Can Be Reused

- RAG service for context retrieval
- SQL safety validation
- SQL post-processing fixes
- External LLM integration (Gemini)
- Database connection and query execution
- Frontend chat UI (mostly)

## What Needs New Development

- Intent classification service
- Metric query templates
- Response generator with Gemini
- Date range passing from frontend
- Advisory question handling

## Estimated Effort

| Phase | Effort | Priority |

|-------|--------|----------|

| Phase 1: Date range | 30 min | High (quick win) |

| Phase 2: Intent classifier | 2-3 hours | High |

| Phase 3: Metric templates | 2-3 hours | High |

| Phase 4: Response generator | 2-3 hours | High |

| Phase 5: Endpoint refactor | 1-2 hours | High |

Total: ~8-12 hours of focused development

### To-dos

- [ ] Fix Gemini API key validation: improve error handling, add better logging, try additional endpoints, provide clearer error messages
- [ ] Fix Gemini API calls in external_llm.py: improve error handling, add retry logic, better rate limit handling
- [ ] Improve Gemini error display in Settings UI: show helpful troubleshooting steps
- [ ] Create RAG service (rag_service.py): main orchestrator for RAG operations with analyze_question, retrieve_context, enhance_prompt methods
- [ ] Create question analyzer (rag_question_analyzer.py): extract intent, entities, query type, date ranges from user questions
- [ ] Create context retrievers (rag_context_retrievers.py): statistical summaries, sample data, column stats, temporal context, entity context
- [ ] Integrate RAG into ai_service.py: update generate_sql_from_question to use RAG service before SQL generation
- [ ] Integrate RAG into chat.py endpoint: pass RAG context to SQL generation
- [ ] Add database helper functions for efficient context queries (statistics, samples, entity lookups)
- [ ] Test RAG system with various question types and verify context retrieval and SQL generation quality