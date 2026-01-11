import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends
from langsmith import traceable

from src.api.schemas import HITLReviewRequest, HITLReviewResponse, ClassifiedItemSchema
from src.api.services import MCPClient
from configs import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/review", tags=["review"])

_pending_reviews: dict = {}
_feedback_log_path = Path("/tmp/hitl_feedback.jsonl")


def store_pending_review(request_id: str, items: list, partial_result: dict):
    """Store items pending human review."""
    _pending_reviews[request_id] = {
        "items": items,
        "partial_result": partial_result
    }


def get_pending_review(request_id: str) -> dict | None:
    """Retrieve pending review data."""
    return _pending_reviews.get(request_id)


def clear_pending_review(request_id: str):
    """Remove completed review."""
    _pending_reviews.pop(request_id, None)


def log_feedback(request_id: str, corrections: list[dict]):
    """
    Log human feedback to JSONL file for future analysis/training.
    
    This feedback can be used to:
    1. Improve the knowledge base (add new items)
    2. Fine-tune the LLM classifier
    3. Analyze classification accuracy over time
    """
    try:
        with open(_feedback_log_path, "a") as f:
            for correction in corrections:
                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                    "dish_name": correction.get("name"),
                    "human_label": correction.get("is_vegetarian"),
                    "feedback_type": "hitl_correction"
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"[FEEDBACK] Logged {len(corrections)} corrections to {_feedback_log_path}")
    except Exception as e:
        logger.warning(f"[FEEDBACK] Failed to log: {e}")


@router.get("/feedback/stats")
async def get_feedback_stats():
    """Get statistics about collected HITL feedback."""
    if not _feedback_log_path.exists():
        return {"total_corrections": 0, "unique_dishes": 0, "feedback": []}
    
    feedback = []
    try:
        with open(_feedback_log_path, "r") as f:
            for line in f:
                if line.strip():
                    feedback.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to read feedback: {e}")
    
    dishes = {}
    for item in feedback:
        name = item.get("dish_name", "")
        if name not in dishes:
            dishes[name] = {"veg_count": 0, "non_veg_count": 0}
        if item.get("human_label"):
            dishes[name]["veg_count"] += 1
        else:
            dishes[name]["non_veg_count"] += 1
    
    return {
        "total_corrections": len(feedback),
        "unique_dishes": len(dishes),
        "dish_stats": dishes,
        "recent_feedback": feedback[-20:]
    }


@router.post("", response_model=HITLReviewResponse)
@traceable(name="submit_review_endpoint")
async def submit_review(
    request: HITLReviewRequest,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """
    Submit human corrections for uncertain menu items.

    After receiving a 'needs_review' response, use this endpoint to provide
    corrections and receive updated classification results.
    
    Feedback is logged to enable future improvements:
    - Knowledge base expansion
    - Model fine-tuning
    - Accuracy analysis
    """
    logger.info(
        f"[{request.request_id}] Received HITL review with {len(request.corrections)} corrections"
    )
    
    pending = get_pending_review(request.request_id)
    if not pending:
        raise HTTPException(
            status_code=404,
            detail=f"No pending review found for request_id: {request.request_id}"
        )
    
    log_feedback(request.request_id, request.corrections)
    
    mcp_client = MCPClient(request_id=request.request_id)
    result = await mcp_client.recompute_with_corrections(
        original_items=pending["items"],
        corrections=request.corrections
    )
    
    clear_pending_review(request.request_id)
    
    vegetarian_items = [
        ClassifiedItemSchema(**item)
        for item in result.get("vegetarian_items", [])
    ]
    
    logger.info(
        f"[{request.request_id}] Review complete, {len(vegetarian_items)} vegetarian items"
    )
    
    return HITLReviewResponse(
        request_id=request.request_id,
        vegetarian_items=vegetarian_items,
        total_sum=result.get("total_sum", 0.0),
        applied_corrections=len(request.corrections)
    )
