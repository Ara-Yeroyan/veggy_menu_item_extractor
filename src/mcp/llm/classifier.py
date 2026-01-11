import re
import json
import logging
from typing import Optional

from langsmith import traceable

from configs import get_settings
from .providers import get_llm_provider, BaseLLMProvider
from src.mcp.rag import VectorStore, get_embedding_service
from src.mcp.rag.data.knowledge_base import KNOWLEDGE_BASE

logger = logging.getLogger(__name__)


class LLMClassifier:
    """
    Vegetarian dish classifier using LLM with RAG and keyword fallback.
    
    Combines multiple classification strategies:
    1. Keyword matching (fast, high precision)
    2. RAG retrieval (semantic similarity)
    3. LLM classification (complex cases)
    """

    SYSTEM_PROMPT = """You are a food classification expert. Your task is to determine if a dish is vegetarian.

A dish is VEGETARIAN if it contains NO:
- Meat (beef, pork, chicken, lamb, duck, etc.)
- Poultry
- Fish or seafood
- Hidden meat products (fish sauce, anchovy paste, gelatin, lard, bone broth)

A dish IS vegetarian if it contains:
- Vegetables, fruits, grains, legumes
- Dairy products (milk, cheese, eggs, butter)
- Plant-based proteins (tofu, tempeh, seitan)

Respond ONLY with valid JSON in this exact format:
{"is_vegetarian": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

    BATCH_SYSTEM_PROMPT = """You are a food classification expert. Classify MULTIPLE dishes as vegetarian or not.

A dish is VEGETARIAN if it contains NO meat, poultry, fish, seafood, or hidden animal products (fish sauce, anchovy paste, gelatin, lard, bone broth).
A dish IS vegetarian if it only contains vegetables, fruits, grains, legumes, dairy, eggs, or plant-based proteins.

You will receive a list of dishes. Respond with a JSON array containing one object per dish in the SAME ORDER.
Each object must have: {"dish": "name", "is_vegetarian": true/false, "confidence": 0.0-1.0, "reasoning": "brief"}

IMPORTANT: Return ONLY valid JSON array, no other text."""

    def __init__(self):
        self._settings = get_settings()
        self._vectorstore = VectorStore()
        self._llm: Optional[BaseLLMProvider] = None
        self._positive_keywords = set(
            kw.lower() for kw in KNOWLEDGE_BASE["keywords"]["vegetarian_positive"]
        )
        self._negative_keywords = set(
            kw.lower() for kw in KNOWLEDGE_BASE["keywords"]["vegetarian_negative"]
        )
        self._vegetarian_markers = set(
            kw.lower() for kw in KNOWLEDGE_BASE["keywords"].get("vegetarian_markers", [])
        )

    def _get_llm(self) -> BaseLLMProvider:
        """Lazy-load LLM provider."""
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    @traceable(name="classify_dish")
    def classify(self, dish_name: str) -> dict:
        """
        Classify a dish as vegetarian or not.

        Parameters
        ----------
        dish_name : str
            Name of the dish to classify

        Returns
        -------
        dict
            Classification result with is_vegetarian, confidence, reasoning, evidence
        """
        logger.info(f"Classifying dish: {dish_name}")
        fallback_chain = []
        
        keyword_result = self._keyword_classification(dish_name)
        fallback_chain.append(f"keyword:{keyword_result['confidence']:.2f}")
        if keyword_result["confidence"] >= 0.9:
            logger.info(f"[CLASSIFY] {dish_name} → keyword (conf={keyword_result['confidence']:.2f})")
            keyword_result["fallback_chain"] = fallback_chain
            return keyword_result
        
        rag_evidence = self._retrieve_evidence(dish_name)
        rag_result = self._analyze_rag_evidence(rag_evidence, dish_name)
        fallback_chain.append(f"rag:{rag_result['confidence']:.2f}")
        
        if rag_result["confidence"] >= self._settings.confidence_threshold:
            logger.info(f"[CLASSIFY] {dish_name} → rag (conf={rag_result['confidence']:.2f})")
            rag_result["evidence"] = [e["document"] for e in rag_evidence[:3]]
            rag_result["fallback_chain"] = fallback_chain
            return rag_result
        
        llm_result = self._llm_classification(dish_name, rag_evidence)
        fallback_chain.append(f"llm:{llm_result['confidence']:.2f}")
        
        llm_failed = llm_result.get("confidence", 0) == 0 or llm_result.get("is_vegetarian") is None
        if llm_failed:
            logger.warning(f"[FALLBACK] LLM failed for '{dish_name}', using rag/keyword fallback")
            fallback_chain.append("fallback_to_rag")
        
        combined = self._combine_results(keyword_result, rag_result, llm_result)
        combined["evidence"] = [e["document"] for e in rag_evidence[:3]]
        combined["fallback_chain"] = fallback_chain
        combined["llm_failed"] = llm_failed
        
        logger.info(f"[CLASSIFY] {dish_name} → {combined['method']} (conf={combined['confidence']:.2f}) chain={fallback_chain}")
        return combined

    def _keyword_classification(self, dish_name: str) -> dict:
        """
        Classify using keyword matching with word boundary checks.

        Parameters
        ----------
        dish_name : str
            Dish name to classify

        Returns
        -------
        dict
            Classification result
        """
        name_lower = dish_name.lower()
        
        for marker in self._vegetarian_markers:
            if marker in name_lower:
                return {
                    "is_vegetarian": True,
                    "confidence": 0.95,
                    "reasoning": f"Contains vegetarian marker: '{marker}'",
                    "method": "keyword"
                }
        
        for kw in self._positive_keywords:
            pattern = rf'\b{re.escape(kw)}\b'
            if re.search(pattern, name_lower):
                return {
                    "is_vegetarian": True,
                    "confidence": 0.95,
                    "reasoning": f"Contains vegetarian indicator: '{kw}'",
                    "method": "keyword"
                }
        
        for kw in self._negative_keywords:
            pattern = rf'\b{re.escape(kw)}\b'
            if re.search(pattern, name_lower):
                return {
                    "is_vegetarian": False,
                    "confidence": 0.95,
                    "reasoning": f"Contains non-vegetarian ingredient: '{kw}'",
                    "method": "keyword"
                }
        
        return {
            "is_vegetarian": None,
            "confidence": 0.0,
            "reasoning": "No keyword match",
            "method": "keyword"
        }

    @traceable(name="retrieve_evidence")
    def _retrieve_evidence(self, dish_name: str) -> list[dict]:
        """
        Retrieve relevant evidence from vector store.

        Parameters
        ----------
        dish_name : str
            Dish name to search for

        Returns
        -------
        list[dict]
            Retrieved evidence items
        """
        return self._vectorstore.search(dish_name)

    def _analyze_rag_evidence(self, evidence: list[dict], dish_name: str) -> dict:
        """
        Analyze RAG evidence to determine classification.

        Parameters
        ----------
        evidence : list[dict]
            Retrieved evidence from vector store
        dish_name : str
            Original dish name

        Returns
        -------
        dict
            Classification result based on evidence
        """
        if not evidence:
            return {
                "is_vegetarian": None,
                "confidence": 0.0,
                "reasoning": "No relevant evidence found",
                "method": "rag"
            }
        
        veg_score = 0.0
        non_veg_score = 0.0
        reasons = []
        
        for item in evidence:
            relevance = item.get("relevance_score", 0.5)
            metadata = item.get("metadata", {})
            is_veg = metadata.get("is_vegetarian")
            
            if relevance < 0.3:
                continue
            
            if is_veg is True:
                veg_score += relevance
                reasons.append(f"{metadata.get('name', 'item')} (vegetarian)")
            elif is_veg is False:
                non_veg_score += relevance
                reasons.append(f"{metadata.get('name', 'item')} (non-vegetarian)")
        
        if veg_score > non_veg_score and veg_score > 0.5:
            confidence = min(0.85, veg_score / (veg_score + non_veg_score + 0.1))
            return {
                "is_vegetarian": True,
                "confidence": confidence,
                "reasoning": f"Similar to: {', '.join(reasons[:3])}",
                "method": "rag"
            }
        elif non_veg_score > veg_score and non_veg_score > 0.5:
            confidence = min(0.85, non_veg_score / (veg_score + non_veg_score + 0.1))
            return {
                "is_vegetarian": False,
                "confidence": confidence,
                "reasoning": f"Similar to: {', '.join(reasons[:3])}",
                "method": "rag"
            }
        
        return {
            "is_vegetarian": None,
            "confidence": 0.3,
            "reasoning": "Inconclusive RAG evidence",
            "method": "rag"
        }

    @traceable(name="llm_classification", metadata={"layer": "llm"})
    def _llm_classification(self, dish_name: str, evidence: list[dict]) -> dict:
        """
        Use LLM for classification with retrieved evidence.

        Parameters
        ----------
        dish_name : str
            Dish name to classify
        evidence : list[dict]
            RAG evidence to include in prompt

        Returns
        -------
        dict
            LLM classification result
        """
        llm_provider = self._settings.llm_provider
        logger.info(f"[LLM] Attempting classification with provider={llm_provider}")
        
        try:
            llm = self._get_llm()
            
            evidence_text = "\n".join([
                f"- {e['document']} (vegetarian: {e['metadata'].get('is_vegetarian')})"
                for e in evidence[:5]
            ])
            
            prompt = f"""Classify this dish: "{dish_name}"

Related items from knowledge base:
{evidence_text}

Is this dish vegetarian? Respond with JSON only."""
            
            response = llm.generate(prompt, self.SYSTEM_PROMPT)
            result = self._parse_llm_response(response)
            result["llm_provider"] = llm_provider
            logger.info(f"[LLM] Success: {dish_name} → veg={result.get('is_vegetarian')} conf={result.get('confidence', 0):.2f}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LLM] FAILED provider={llm_provider}: {error_msg}")
            return {
                "is_vegetarian": None,
                "confidence": 0.0,
                "reasoning": f"LLM error: {error_msg}",
                "method": "llm",
                "llm_provider": llm_provider,
                "llm_error": error_msg
            }

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM JSON response."""
        try:
            json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "is_vegetarian": data.get("is_vegetarian"),
                    "confidence": float(data.get("confidence", 0.7)),
                    "reasoning": data.get("reasoning", "LLM classification"),
                    "method": "llm"
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return {
            "is_vegetarian": None,
            "confidence": 0.0,
            "reasoning": "Failed to parse LLM response",
            "method": "llm"
        }

    @traceable(name="llm_batch_classification", metadata={"layer": "llm_batch"})
    def classify_batch_llm(self, items: list[dict]) -> dict[str, dict]:
        """
        Classify multiple dishes in a single LLM call for better latency.

        Parameters
        ----------
        items : list[dict]
            List of items with 'name' and 'evidence' keys

        Returns
        -------
        dict[str, dict]
            Mapping of dish name to classification result
        """
        if not items:
            return {}
        
        llm_provider = self._settings.llm_provider
        batch_size = len(items)
        logger.info(f"[LLM_BATCH] Classifying {batch_size} items with provider={llm_provider}")
        
        try:
            llm = self._get_llm()
            
            dishes_text = "\n".join([
                f"{i+1}. {item['name']}"
                for i, item in enumerate(items)
            ])
            
            prompt = f"""Classify these {batch_size} dishes as vegetarian or not:

{dishes_text}

Return a JSON array with {batch_size} objects, one for each dish in order."""
            
            response = llm.generate(prompt, self.BATCH_SYSTEM_PROMPT)
            results = self._parse_batch_response(response, items)
            
            success_count = sum(1 for r in results.values() if r.get("is_vegetarian") is not None)
            logger.info(f"[LLM_BATCH] Completed: {success_count}/{batch_size} successful")
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LLM_BATCH] FAILED: {error_msg}")
            return {
                item["name"]: {
                    "is_vegetarian": None,
                    "confidence": 0.0,
                    "reasoning": f"Batch LLM error: {error_msg}",
                    "method": "llm_batch",
                    "llm_error": error_msg
                }
                for item in items
            }

    def _parse_batch_response(self, response: str, items: list[dict]) -> dict[str, dict]:
        """Parse batch LLM JSON array response."""
        results = {}
        
        try:
            cleaned = response
            cleaned = re.sub(r"```json\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```", "", cleaned)
            cleaned = cleaned.strip()
            
            start_idx = cleaned.find("[")
            end_idx = cleaned.rfind("]")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = cleaned[start_idx:end_idx + 1]
                data = json.loads(json_str)
                
                if isinstance(data, list):
                    for i, item_result in enumerate(data):
                        dish_name = item_result.get("dish") or item_result.get("name")
                        if not dish_name and i < len(items):
                            dish_name = items[i]["name"]
                        
                        if dish_name:
                            for item in items:
                                if item["name"].lower() in dish_name.lower() or dish_name.lower() in item["name"].lower():
                                    results[item["name"]] = {
                                        "is_vegetarian": item_result.get("is_vegetarian"),
                                        "confidence": float(item_result.get("confidence", 0.7)),
                                        "reasoning": item_result.get("reasoning", "Batch LLM"),
                                        "method": "llm_batch"
                                    }
                                    break
                            else:
                                results[dish_name] = {
                                    "is_vegetarian": item_result.get("is_vegetarian"),
                                    "confidence": float(item_result.get("confidence", 0.7)),
                                    "reasoning": item_result.get("reasoning", "Batch LLM"),
                                    "method": "llm_batch"
                                }
                    
                    logger.debug(f"Batch parse: extracted {len(results)} results from {len(data)} items")
                    return results
                    
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse batch response: {e}, response preview: {response[:200]}")
        
        for item in items:
            if item["name"] not in results:
                results[item["name"]] = {
                    "is_vegetarian": None,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse batch response",
                    "method": "llm_batch"
                }
        
        return results

    def _combine_results(
        self,
        keyword: dict,
        rag: dict,
        llm: dict
    ) -> dict:
        """
        Combine results from all classification methods.

        Parameters
        ----------
        keyword : dict
            Keyword classification result
        rag : dict
            RAG classification result
        llm : dict
            LLM classification result

        Returns
        -------
        dict
            Combined classification result
        """
        results = [
            (keyword, 0.4),
            (rag, 0.3),
            (llm, 0.3)
        ]
        
        valid_results = [
            (r, w) for r, w in results if r.get("is_vegetarian") is not None
        ]
        
        if not valid_results:
            return {
                "is_vegetarian": None,
                "confidence": 0.0,
                "reasoning": "Unable to classify",
                "method": "combined"
            }
        
        weighted_sum = sum(
            (1 if r["is_vegetarian"] else 0) * r["confidence"] * w
            for r, w in valid_results
        )
        total_weight = sum(r["confidence"] * w for r, w in valid_results)
        
        if total_weight == 0:
            return valid_results[0][0]
        
        veg_probability = weighted_sum / total_weight
        is_vegetarian = veg_probability > 0.5
        confidence = abs(veg_probability - 0.5) * 2
        
        reasons = [r["reasoning"] for r, _ in valid_results if r.get("reasoning")]
        
        return {
            "is_vegetarian": is_vegetarian,
            "confidence": round(confidence, 3),
            "reasoning": "; ".join(reasons[:2]),
            "method": "combined"
        }
