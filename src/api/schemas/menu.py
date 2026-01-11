from pydantic import BaseModel, Field
from typing import Optional


class MenuItemSchema(BaseModel):
    """Raw menu item extracted from OCR."""

    name: str = Field(..., description="Dish name")
    price: float = Field(..., description="Dish price")
    source_image: Optional[int] = Field(default=None, description="Source image index (1-based)")


class ClassifiedItemSchema(BaseModel):
    """Menu item with vegetarian classification."""

    name: str
    price: float
    confidence: Optional[float] = Field(
        default=None,
        description="Classification confidence score"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation for classification"
    )
    source_image: Optional[int] = Field(
        default=None,
        description="Source image index (1-based)"
    )
    method: Optional[str] = Field(
        default=None,
        description="Classification method used (keyword, rag, llm, llm_batch)"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence from RAG retrieval"
    )


class DetailedItemSchema(BaseModel):
    """Full item details for data export/mapping."""

    name: str
    price: float
    currency: str = "USD"
    source_image: int
    is_vegetarian: Optional[bool] = None
    confidence: float
    method: str
    reasoning: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)
    related_ingredients: list[str] = Field(default_factory=list)
    category: Optional[str] = None


class UncertainItemSchema(BaseModel):
    """Item with low classification confidence for HITL review."""

    name: str
    price: float
    confidence: float
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence collected during classification"
    )
    suggested_classification: Optional[bool] = Field(
        default=None,
        description="System's best guess"
    )
    source_image: Optional[int] = Field(
        default=None,
        description="Source image index (1-based)"
    )


class ProcessMenuResponse(BaseModel):
    """Response for successful menu processing."""

    request_id: str
    vegetarian_items: list[ClassifiedItemSchema]
    total_sum: float
    status: str = "success"
    all_items: list[DetailedItemSchema] = Field(
        default_factory=list,
        description="Complete data mapping for all extracted items"
    )


class NeedsReviewResponse(BaseModel):
    """Response when items need human review."""

    request_id: str
    status: str = "needs_review"
    message: str = "Some items have low classification confidence"
    uncertain_items: list[UncertainItemSchema]
    confident_items: list[ClassifiedItemSchema] = Field(default_factory=list)
    partial_sum: float = 0.0
    all_items: list[DetailedItemSchema] = Field(
        default_factory=list,
        description="Complete data mapping for all extracted items"
    )


class HITLReviewRequest(BaseModel):
    """Request for human review corrections."""

    request_id: str
    corrections: list[dict] = Field(
        ...,
        description="List of corrections with name and is_vegetarian"
    )


class HITLReviewResponse(BaseModel):
    """Response after applying HITL corrections."""

    request_id: str
    vegetarian_items: list[ClassifiedItemSchema]
    total_sum: float
    applied_corrections: int


class ProcessMenuRequest(BaseModel):
    """Request with base64 encoded images."""

    images: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Base64 encoded menu images"
    )
