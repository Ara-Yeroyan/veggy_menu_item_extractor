import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langsmith import traceable

from src.mcp.tools import ClassifierTool, CalculatorTool
from src.mcp.rag import VectorStore
from configs import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

classifier_tool: ClassifierTool = None
calculator_tool: CalculatorTool = None
vectorstore: VectorStore = None


class ClassifyRequest(BaseModel):
    request_id: str
    items: list[dict] = Field(..., description="Menu items to classify")


class RecomputeRequest(BaseModel):
    request_id: str
    items: list[dict]
    corrections: list[dict]


class ParseRequest(BaseModel):
    prompt: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for initialization."""
    global classifier_tool, calculator_tool, vectorstore
    
    settings = get_settings()
    logger.info(f"Starting MCP Server on {settings.mcp_host}:{settings.mcp_port}")
    
    logger.info("Initializing vector store...")
    vectorstore = VectorStore()
    logger.info(f"Vector store initialized: {vectorstore.get_stats()}")
    
    classifier_tool = ClassifierTool()
    calculator_tool = CalculatorTool()
    
    logger.info("MCP Server ready")
    yield
    logger.info("Shutting down MCP Server")


app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol server for vegetarian classification",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with request ID."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.debug(f"[{request_id}] {request.method} {request.url.path}")
    response = await call_next(request)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.error(f"[{request_id}] Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "request_id": request_id}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = vectorstore.get_stats() if vectorstore else {}
    return {
        "status": "healthy",
        "service": "mcp-server",
        "vectorstore": stats
    }


@app.post("/tools/classify-and-calculate")
@traceable(name="mcp_classify_and_calculate")
async def classify_and_calculate(request: ClassifyRequest) -> dict[str, Any]:
    """
    Main tool: classify menu items and calculate vegetarian total.

    This is the primary endpoint called by the REST API.
    """
    logger.info(
        f"[{request.request_id}] Received classify-and-calculate request "
        f"with {len(request.items)} items"
    )
    
    classification = classifier_tool.execute(
        items=request.items,
        request_id=request.request_id
    )
    
    settings = get_settings()
    
    all_items = classification.get("all_items", [])
    
    if classification["uncertain_items"]:
        confident_items = classification["vegetarian_items"]
        partial_sum = calculator_tool.execute(
            vegetarian_items=confident_items,
            request_id=request.request_id
        )["total_sum"]
        
        return {
            "status": "needs_review",
            "confident_items": confident_items,
            "uncertain_items": classification["uncertain_items"],
            "partial_sum": partial_sum,
            "all_items": all_items
        }
    
    result = calculator_tool.execute(
        vegetarian_items=classification["vegetarian_items"],
        request_id=request.request_id
    )
    
    return {
        "status": "success",
        "vegetarian_items": classification["vegetarian_items"],
        "total_sum": result["total_sum"],
        "all_items": all_items
    }


@app.post("/tools/recompute")
@traceable(name="mcp_recompute")
async def recompute_with_corrections(request: RecomputeRequest) -> dict[str, Any]:
    """
    Recompute classification with human corrections.

    Called after HITL review to apply corrections.
    """
    logger.info(
        f"[{request.request_id}] Recomputing with {len(request.corrections)} corrections"
    )
    
    return calculator_tool.recompute_with_corrections(
        items=request.items,
        corrections=request.corrections,
        request_id=request.request_id
    )


@app.post("/tools/parse-menu")
@traceable(name="mcp_parse_menu")
async def parse_menu_with_llm(request: ParseRequest) -> dict[str, str]:
    """
    Use LLM to parse menu text when regex fails.

    Fallback parsing tool for complex menu formats.
    """
    from src.mcp.llm.providers import get_llm_provider
    
    try:
        llm = get_llm_provider()
        result = llm.generate(request.prompt)
        return {"result": result}
    except Exception as e:
        logger.error(f"LLM parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/search")
@traceable(name="mcp_search")
async def search_knowledge_base(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Search the ingredient knowledge base.

    Useful for debugging and understanding RAG retrieval.
    """
    results = vectorstore.search(query, top_k=top_k)
    return {
        "query": query,
        "results": results
    }
