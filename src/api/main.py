import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid

from src.api.routes import menu_router, review_router
from configs import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    logger.info(f"Starting Menu Parser API on {settings.api_host}:{settings.api_port}")
    logger.info(f"MCP Server URL: {settings.mcp_server_url}")
    yield
    logger.info("Shutting down Menu Parser API")


app = FastAPI(
    title="Menu Parser API",
    description="Extract vegetarian dishes from restaurant menu images",
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
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with logging."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id
        }
    )


app.include_router(menu_router)
app.include_router(review_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "menu-parser-api"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Menu Parser API",
        "version": "1.0.0",
        "endpoints": {
            "process_menu": "POST /process-menu",
            "process_menu_base64": "POST /process-menu/base64",
            "review": "POST /review",
            "health": "GET /health"
        }
    }
