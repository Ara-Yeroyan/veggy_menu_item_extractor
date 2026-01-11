from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="LLM provider to use for classification"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama3.2",
        description="Ollama model name"
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for fallback"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name"
    )
    mcp_server_url: str = Field(
        default="http://localhost:8001",
        description="MCP server URL"
    )
    langchain_tracing_v2: bool = Field(
        default=True,
        description="Enable Langsmith tracing"
    )
    langchain_api_key: str = Field(
        default="",
        description="Langsmith API key"
    )
    langchain_project: str = Field(
        default="menu-parsing",
        description="Langsmith project name"
    )
    confidence_threshold: float = Field(
        default=0.6,
        description="Minimum confidence for definitive classification"
    )
    hitl_threshold: float = Field(
        default=0.4,
        description="Below this threshold, trigger HITL review"
    )
    rag_top_k: int = Field(
        default=5,
        description="Number of RAG results to retrieve"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    llm_batch_enabled: bool = Field(
        default=True,
        description="Enable batch LLM classification for better latency"
    )
    llm_batch_size: int = Field(
        default=8,
        description="Number of items to classify in a single LLM call"
    )
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    mcp_host: str = Field(default="0.0.0.0")
    mcp_port: int = Field(default=8001)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
