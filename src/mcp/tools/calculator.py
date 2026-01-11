import logging
from typing import Any

from langsmith import traceable

logger = logging.getLogger(__name__)


class CalculatorTool:
    """
    MCP tool for calculating sum of vegetarian dish prices.
    
    Simple deterministic calculation with validation.
    """

    @staticmethod
    @traceable(name="calculator_tool_execute")
    def execute(
        vegetarian_items: list[dict],
        request_id: str = None
    ) -> dict[str, Any]:
        """
        Calculate total sum of vegetarian item prices.

        Parameters
        ----------
        vegetarian_items : list[dict]
            List of vegetarian items with prices
        request_id : str, optional
            Request ID for tracing

        Returns
        -------
        dict
            Total sum and item count
        """
        total = 0.0
        valid_count = 0
        
        for item in vegetarian_items:
            price = item.get("price", 0.0)
            if isinstance(price, (int, float)) and price > 0:
                total += price
                valid_count += 1
        
        total = round(total, 2)
        
        logger.info(
            f"[{request_id}] Calculated total: ${total} from {valid_count} items",
            extra={"request_id": request_id}
        )
        
        return {
            "total_sum": total,
            "item_count": valid_count
        }

    @staticmethod
    @traceable(name="recompute_with_corrections")
    def recompute_with_corrections(
        items: list[dict],
        corrections: list[dict],
        request_id: str = None
    ) -> dict[str, Any]:
        """
        Recompute vegetarian sum with human corrections applied.

        Parameters
        ----------
        items : list[dict]
            Original menu items
        corrections : list[dict]
            Human corrections with name and is_vegetarian
        request_id : str, optional
            Request ID for tracing

        Returns
        -------
        dict
            Recomputed results with corrections applied
        """
        correction_map = {
            c["name"].lower(): c["is_vegetarian"]
            for c in corrections
        }
        
        vegetarian_items = []
        
        for item in items:
            name = item.get("name", "")
            name_lower = name.lower()
            
            if name_lower in correction_map:
                if correction_map[name_lower]:
                    vegetarian_items.append({
                        "name": name,
                        "price": item.get("price", 0.0),
                        "confidence": 1.0,
                        "reasoning": "Human verified"
                    })
            elif item.get("is_vegetarian", False):
                vegetarian_items.append(item)
        
        total = round(sum(i.get("price", 0.0) for i in vegetarian_items), 2)
        
        logger.info(
            f"[{request_id}] Recomputed with {len(corrections)} corrections: ${total}",
            extra={"request_id": request_id}
        )
        
        return {
            "vegetarian_items": vegetarian_items,
            "total_sum": total,
            "corrections_applied": len(corrections)
        }
