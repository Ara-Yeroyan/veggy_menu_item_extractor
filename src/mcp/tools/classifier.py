import logging
import time
from typing import Any

from langsmith import traceable

from configs import get_settings
from src.mcp.llm import LLMClassifier
from src.mcp.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class ClassifierTool:
    """
    MCP tool for classifying menu items as vegetarian.
    
    Processes a list of menu items and returns classification results
    with confidence scores and evidence.
    
    Supports batch LLM classification for improved latency.
    """

    def __init__(self):
        self._settings = get_settings()
        self._classifier = LLMClassifier()

    @traceable(name="classifier_tool_execute")
    def execute(
        self,
        items: list[dict],
        request_id: str = None
    ) -> dict[str, Any]:
        """
        Classify menu items as vegetarian or not.

        Parameters
        ----------
        items : list[dict]
            List of menu items with name and price
        request_id : str, optional
            Request ID for tracing

        Returns
        -------
        dict
            Classification results grouped by confidence level
        """
        start_time = time.time()
        batch_enabled = self._settings.llm_batch_enabled
        batch_size = self._settings.llm_batch_size
        
        logger.info(
            f"[{request_id}] Classifying {len(items)} items (batch={batch_enabled}, size={batch_size})",
            extra={"request_id": request_id}
        )
        
        if batch_enabled:
            return self._execute_with_batch(items, request_id, start_time)
        else:
            return self._execute_sequential(items, request_id, start_time)

    def _execute_sequential(
        self,
        items: list[dict],
        request_id: str,
        start_time: float
    ) -> dict[str, Any]:
        """Sequential classification (original behavior)."""
        vegetarian_items = []
        non_vegetarian_items = []
        uncertain_items = []
        all_items = []
        
        for item in items:
            name = item.get("name", "")
            price = item.get("price", 0.0)
            source_image = item.get("source_image", 1)
            
            result = self._classifier.classify(name)
            method = result.get("method", "combined")
            
            classified_item = {
                "name": name,
                "price": price,
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
                "evidence": result.get("evidence", []),
                "source_image": source_image,
                "method": method
            }
            
            is_veg = result.get("is_vegetarian")
            confidence = result.get("confidence", 0.0)
            
            all_items.append({
                **classified_item,
                "is_vegetarian": is_veg,
                "currency": "USD",
                "related_ingredients": [],
                "category": None
            })
            
            if is_veg is None or confidence < self._settings.hitl_threshold:
                classified_item["suggested_classification"] = is_veg
                uncertain_items.append(classified_item)
            elif is_veg:
                vegetarian_items.append(classified_item)
            else:
                non_vegetarian_items.append(classified_item)
        
        elapsed = time.time() - start_time
        logger.info(
            f"[{request_id}] Sequential complete in {elapsed:.2f}s: "
            f"{len(vegetarian_items)} veg, {len(non_vegetarian_items)} non-veg, {len(uncertain_items)} uncertain",
            extra={"request_id": request_id}
        )
        
        EmbeddingService.clear_cache()
        
        return {
            "vegetarian_items": vegetarian_items,
            "non_vegetarian_items": non_vegetarian_items,
            "uncertain_items": uncertain_items,
            "all_items": all_items
        }

    @traceable(name="classifier_batch_execute")
    def _execute_with_batch(
        self,
        items: list[dict],
        request_id: str,
        start_time: float
    ) -> dict[str, Any]:
        """
        Batch classification: keyword/RAG first, then batch LLM for remaining.
        
        This reduces LLM calls by grouping items that need LLM classification.
        """
        vegetarian_items = []
        non_vegetarian_items = []
        uncertain_items = []
        needs_llm = []
        
        all_items = []
        
        for item in items:
            name = item.get("name", "")
            price = item.get("price", 0.0)
            source_image = item.get("source_image", 1)
            
            keyword_result = self._classifier._keyword_classification(name)
            
            if keyword_result["confidence"] >= 0.9:
                classified_item = {
                    "name": name,
                    "price": price,
                    "confidence": keyword_result["confidence"],
                    "reasoning": keyword_result["reasoning"],
                    "evidence": [],
                    "source_image": source_image,
                    "method": "keyword"
                }
                all_items.append({
                    **classified_item,
                    "is_vegetarian": keyword_result["is_vegetarian"],
                    "currency": "USD",
                    "related_ingredients": [],
                    "category": None
                })
                if keyword_result["is_vegetarian"]:
                    vegetarian_items.append(classified_item)
                else:
                    non_vegetarian_items.append(classified_item)
                continue
            
            rag_evidence = self._classifier._retrieve_evidence(name)
            rag_result = self._classifier._analyze_rag_evidence(rag_evidence, name)
            related_ingredients = [
                e.get("metadata", {}).get("name", "")
                for e in rag_evidence[:3]
                if e.get("metadata", {}).get("type") == "ingredient"
            ]
            
            if rag_result["confidence"] >= self._settings.confidence_threshold:
                classified_item = {
                    "name": name,
                    "price": price,
                    "confidence": rag_result["confidence"],
                    "reasoning": rag_result["reasoning"],
                    "evidence": [e["document"] for e in rag_evidence[:3]],
                    "source_image": source_image,
                    "method": "rag"
                }
                all_items.append({
                    **classified_item,
                    "is_vegetarian": rag_result["is_vegetarian"],
                    "currency": "USD",
                    "related_ingredients": related_ingredients,
                    "category": rag_evidence[0].get("metadata", {}).get("category") if rag_evidence else None
                })
                if rag_result["is_vegetarian"]:
                    vegetarian_items.append(classified_item)
                elif rag_result["is_vegetarian"] is False:
                    non_vegetarian_items.append(classified_item)
                else:
                    needs_llm.append({
                        "name": name,
                        "price": price,
                        "source_image": source_image,
                        "evidence": rag_evidence,
                        "rag_result": rag_result,
                        "related_ingredients": related_ingredients
                    })
                continue
            
            needs_llm.append({
                "name": name,
                "price": price,
                "source_image": source_image,
                "evidence": rag_evidence,
                "rag_result": rag_result,
                "related_ingredients": related_ingredients
            })
        
        logger.info(
            f"[{request_id}] Pre-LLM: {len(vegetarian_items)} veg, "
            f"{len(non_vegetarian_items)} non-veg, {len(needs_llm)} need LLM"
        )
        
        if needs_llm:
            batch_size = self._settings.llm_batch_size
            for i in range(0, len(needs_llm), batch_size):
                batch = needs_llm[i:i + batch_size]
                batch_items = [{"name": item["name"], "evidence": item["evidence"]} for item in batch]
                
                logger.info(f"[{request_id}] LLM batch {i//batch_size + 1}: {len(batch)} items")
                llm_results = self._classifier.classify_batch_llm(batch_items)
                
                for item in batch:
                    name = item["name"]
                    llm_result = llm_results.get(name, {})
                    rag_result = item.get("rag_result", {})
                    
                    combined = self._classifier._combine_results(
                        {"is_vegetarian": None, "confidence": 0, "reasoning": ""},
                        rag_result,
                        llm_result
                    )
                    
                    classified_item = {
                        "name": name,
                        "price": item["price"],
                        "confidence": combined.get("confidence", 0.0),
                        "reasoning": combined.get("reasoning", llm_result.get("reasoning", "")),
                        "evidence": [e["document"] for e in item.get("evidence", [])[:3]],
                        "source_image": item["source_image"],
                        "method": "llm_batch"
                    }
                    
                    is_veg = combined.get("is_vegetarian")
                    confidence = combined.get("confidence", 0.0)
                    
                    all_items.append({
                        **classified_item,
                        "is_vegetarian": is_veg,
                        "currency": "USD",
                        "related_ingredients": item.get("related_ingredients", []),
                        "category": None
                    })
                    
                    if is_veg is None or confidence < self._settings.hitl_threshold:
                        classified_item["suggested_classification"] = is_veg
                        uncertain_items.append(classified_item)
                    elif is_veg:
                        vegetarian_items.append(classified_item)
                    else:
                        non_vegetarian_items.append(classified_item)
        
        elapsed = time.time() - start_time
        logger.info(
            f"[{request_id}] Batch complete in {elapsed:.2f}s: "
            f"{len(vegetarian_items)} veg, {len(non_vegetarian_items)} non-veg, {len(uncertain_items)} uncertain",
            extra={"request_id": request_id}
        )
        
        EmbeddingService.clear_cache()
        
        return {
            "vegetarian_items": vegetarian_items,
            "non_vegetarian_items": non_vegetarian_items,
            "uncertain_items": uncertain_items,
            "all_items": all_items
        }
