# Menu Parser System Breakdown

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER / UI                                       │
│                         (ui/index.html)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ POST /process-menu (multipart images)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REST API SERVICE                                   │
│                          (Docker: port 8000)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. OCR Processing (Tesseract)                                       │   │
│  │     - Extract text from each image                                   │   │
│  │     - Memory cleanup after each image                                │   │
│  │                                                                      │   │
│  │  2. Text Parsing (Regex + LLM fallback)                             │   │
│  │     - Pattern: "Item Name ... $XX.XX"                               │   │
│  │     - Extract dish name + price pairs                                │   │
│  │     - Track source_image for each item                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    │ HTTP call to MCP server                 │
│                                    ▼                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ POST /tools/classify-and-calculate
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP SERVER                                         │
│                        (Docker: port 8001)                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Classification Pipeline (per item)                                  │   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │  │   KEYWORD    │ →  │     RAG      │ →  │     LLM      │          │   │
│  │  │   (0.00s)    │    │   (0.02s)    │    │   (1-2s)     │          │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │   │
│  │        │                    │                    │                   │   │
│  │        ▼                    ▼                    ▼                   │   │
│  │   conf ≥ 90%          conf ≥ 70%           combine all              │   │
│  │   → DONE              → DONE               → final result           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Price Calculator                                                    │   │
│  │  - Sum all vegetarian item prices                                   │   │
│  │  - Return total_sum                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Ollama API (host.docker.internal:11434)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OLLAMA (Host Machine)                                │
│                      GPU Acceleration (Metal/CUDA)                           │
│                         Model: llama3.2                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Classification Pipeline

### Layer 1: Keyword Matching (Instant)

**Speed:** ~0ms  
**Confidence:** 95%

```python
POSITIVE_KEYWORDS = [
    "veggie", "vegetarian", "vegan", "tofu", "paneer", 
    "mushroom", "vegetable", "dal", "lentil", "falafel"
]

NEGATIVE_KEYWORDS = [
    "chicken", "beef", "pork", "lamb", "fish", "salmon",
    "shrimp", "bacon", "ham", "anchovy", "caesar"
]
```

**Flow:**
```
"Veggie Burger" → contains "veggie" → ✓ vegetarian (95%)
"Chicken Wings" → contains "chicken" → ✗ non-vegetarian (95%)
"French Fries" → no keyword match → proceed to RAG
```

---

### Layer 2: RAG Retrieval (Fast)

**Speed:** ~20ms  
**Confidence:** Up to 85%

**Components:**
- **Vector Store:** ChromaDB (in-memory)
- **Embeddings:** all-MiniLM-L6-v2 (SentenceTransformer)
- **Knowledge Base:** 66 items (ingredients + dishes)

**Flow:**
```
Query: "Greek Salad"
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Semantic Search in ChromaDB                            │
│                                                         │
│  Top-5 Results:                                         │
│  1. greek salad (veg=True, relevance=0.95)             │
│  2. caprese salad (veg=True, relevance=0.72)           │
│  3. caesar salad (veg=False, relevance=0.68)           │
│  4. garden salad (veg=True, relevance=0.65)            │
│  5. cobb salad (veg=False, relevance=0.60)             │
│                                                         │
│  Score Calculation:                                     │
│  veg_score = 0.95 + 0.72 + 0.65 = 2.32                 │
│  non_veg_score = 0.68 + 0.60 = 1.28                    │
│                                                         │
│  Result: vegetarian (confidence = 0.85)                 │
└─────────────────────────────────────────────────────────┘
```

---

### Layer 3: LLM Classification (Slow but Accurate)

**Speed:** ~1-2s per item (sequential) or ~5s for 8 items (batch)  
**Confidence:** 70-100%

**Provider:** Ollama (llama3.2) on host machine  
**Fallback:** OpenAI API (if configured)

**Batch Processing:**
```
┌─────────────────────────────────────────────────────────┐
│  Items needing LLM: [French Fries, Grilled Cheese,     │
│                      Tiramisu, Mushroom Risotto, ...]   │
│                                                         │
│  Batch Size: 8                                          │
│                                                         │
│  Single LLM Call:                                       │
│  "Classify these 7 dishes as vegetarian:                │
│   1. French Fries                                       │
│   2. Grilled Cheese                                     │
│   3. Tiramisu                                           │
│   ..."                                                  │
│                                                         │
│  Response (JSON array):                                 │
│  [                                                      │
│    {"dish": "French Fries", "is_vegetarian": true, ...}│
│    {"dish": "Grilled Cheese", "is_vegetarian": true,...}│
│    ...                                                  │
│  ]                                                      │
└─────────────────────────────────────────────────────────┘
```

---

### Layer 4: Result Combination

When LLM is used, results from all layers are combined:

```python
weights = {
    "keyword": 0.4,
    "rag": 0.3,
    "llm": 0.3
}

# Weighted voting
veg_probability = weighted_sum / total_weight
is_vegetarian = veg_probability > 0.5
confidence = abs(veg_probability - 0.5) * 2
```

---

## Fallback Policies

### 1. LLM Failure → RAG/Keyword Fallback

```
┌─────────────────────────────────────────────────────────┐
│  If LLM fails (timeout, parse error, 401):              │
│                                                         │
│  1. Log error with details                              │
│  2. Use RAG result if available (conf ≥ 0.3)           │
│  3. Use keyword result if available                     │
│  4. If still uncertain → add to uncertain_items         │
│                                                         │
│  HITL takes over for uncertain items                    │
└─────────────────────────────────────────────────────────┘
```

### 2. Ollama Unavailable → OpenAI Fallback

```python
# In providers.py
def get_llm_provider():
    if settings.llm_provider == "ollama":
        if check_ollama_available():
            return OllamaProvider()
        else:
            logger.warning("Ollama unavailable, falling back to OpenAI")
    return OpenAIProvider()  # Requires OPENAI_API_KEY
```

### 3. Batch Parse Failure → Individual Fallback

```
If batch JSON parsing fails:
  → Each item marked as "Unable to classify"
  → confidence = 0
  → Goes to uncertain_items for HITL review
```

---

## HITL (Human-in-the-Loop) Flow

### Trigger Conditions

```python
HITL_THRESHOLD = 0.5  # configurable

if confidence < HITL_THRESHOLD:
    # Add to uncertain_items
    # Return status: "needs_review"
```

### Review Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. API returns "needs_review" with:                    │
│     - confident_items (high confidence vegetarian)      │
│     - uncertain_items (need human decision)             │
│     - partial_sum (sum of confident items only)         │
│                                                         │
│  2. UI displays uncertain items with ✓/✗ buttons        │
│                                                         │
│  3. User clicks ✓ or ✗ for each item                   │
│                                                         │
│  4. POST /review with corrections:                      │
│     {                                                   │
│       "request_id": "abc-123",                         │
│       "corrections": [                                  │
│         {"name": "French Fries", "is_vegetarian": true}│
│       ]                                                 │
│     }                                                   │
│                                                         │
│  5. Server applies corrections (100% confidence)        │
│     Logs feedback to /tmp/hitl_feedback.jsonl          │
│     Returns final vegetarian_items + total_sum          │
└─────────────────────────────────────────────────────────┘
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM provider (`ollama` or `openai`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OPENAI_API_KEY` | - | OpenAI API key (fallback) |
| `CONFIDENCE_THRESHOLD` | `0.7` | Min confidence for auto-classification |
| `HITL_THRESHOLD` | `0.5` | Below this → needs human review |
| `LLM_BATCH_ENABLED` | `true` | Enable batch LLM processing |
| `LLM_BATCH_SIZE` | `8` | Items per LLM batch call |
| `RAG_TOP_K` | `5` | Number of RAG results to retrieve |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

### Docker Configuration

```bash
# For Mac (Ollama on host):
OLLAMA_BASE_URL=http://host.docker.internal:11434 docker-compose up -d

# For Linux with GPU:
OLLAMA_BASE_URL=http://172.17.0.1:11434 docker-compose up -d
```

---

## Performance Characteristics

### Typical Processing Times (26 items, 4 images)

| Stage | Time | Notes |
|-------|------|-------|
| OCR (4 images) | ~2s | Parallel possible |
| Parsing | ~0.1s | Regex-based |
| Keyword classification | ~0ms | Instant lookup |
| RAG classification | ~0.2s | All items |
| LLM batch (7 items) | ~5s | Single Ollama call |
| **Total** | **~7-8s** | |

### Without Batching

| Stage | Time |
|-------|------|
| LLM sequential (7 items) | ~7-10s |
| **Total** | **~10-12s** |

---

## Observability (Langsmith)

### Traced Operations

```
mcp_classify_and_calculate
├── classifier_tool_execute
│   ├── classify_dish (per item)
│   │   ├── retrieve_evidence (RAG)
│   │   └── llm_classification (if needed)
│   └── llm_batch_classification (if batch enabled)
└── calculator_tool_execute
```

### Logged Metadata

- `fallback_chain`: ["keyword:0.00", "rag:0.30", "llm:1.00"]
- `llm_provider`: "ollama" | "openai"
- `llm_failed`: true | false
- `request_id`: UUID for correlation

---

## Error Handling

### API Errors

| Error | Response | Handling |
|-------|----------|----------|
| No images | 400 | "At least one image required" |
| Invalid image | 400 | "Failed to process image" |
| MCP timeout | 500 | "Classification service unavailable" |
| OCR failure | 500 | "Text extraction failed" |

### Classification Errors

| Error | Handling |
|-------|----------|
| LLM timeout | Use RAG/keyword fallback |
| JSON parse error | Mark as uncertain |
| All methods fail | Add to uncertain_items |

---

## Knowledge Base

### Structure

```python
KNOWLEDGE_BASE = {
    "ingredients": [
        {"name": "tofu", "is_vegetarian": True, "category": "protein"},
        {"name": "chicken", "is_vegetarian": False, "category": "meat"},
        ...
    ],
    "dishes": [
        {"name": "caesar salad", "is_vegetarian": False, 
         "notes": "Traditional dressing contains anchovies"},
        ...
    ],
    "keywords": {
        "vegetarian_positive": ["veggie", "tofu", "paneer", ...],
        "vegetarian_negative": ["chicken", "beef", "caesar", ...]
    }
}
```

### Total Items

- **Ingredients:** 40+
- **Dishes:** 26+
- **Keywords:** 20+ positive, 25+ negative

---

## File Structure

```
menu_parsing/
├── configs/
│   └── settings.py          # Pydantic settings
├── src/
│   ├── api/                  # REST API service
│   │   ├── main.py          # FastAPI app
│   │   ├── routes/
│   │   │   ├── menu.py      # /process-menu endpoint
│   │   │   └── review.py    # /review endpoint (HITL)
│   │   ├── schemas/
│   │   │   └── menu.py      # Pydantic models
│   │   └── services/
│   │       ├── ocr.py       # Tesseract wrapper
│   │       ├── parser.py    # Text → structured data
│   │       └── mcp_client.py # MCP server client
│   └── mcp/                  # MCP server
│       ├── main.py          # FastAPI app
│       ├── llm/
│       │   ├── classifier.py # Classification logic
│       │   └── providers.py  # Ollama/OpenAI providers
│       ├── rag/
│       │   ├── embeddings.py # SentenceTransformer
│       │   ├── vectorstore.py # ChromaDB
│       │   └── data/
│       │       └── knowledge_base.py
│       └── tools/
│           ├── classifier.py # Classification tool
│           └── calculator.py # Price sum tool
├── ui/
│   └── index.html           # Web interface
├── tests/
│   └── data/menus/          # Test images
├── docker/
│   ├── api.Dockerfile
│   └── mcp.Dockerfile
└── docker-compose.yml
```
