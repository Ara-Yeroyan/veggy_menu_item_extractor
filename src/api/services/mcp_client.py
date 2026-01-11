import httpx
import logging
from typing import Any
import uuid

from configs import get_settings
from src.api.schemas import MenuItemSchema, ClassifiedItemSchema

logger = logging.getLogger(__name__)


class MCPClient:
    """
    HTTP client for communicating with the MCP server.
    
    Handles tool-calling for classification and calculation.
    """

    def __init__(self, request_id: str = None):
        """
        Initialize MCP client.

        Parameters
        ----------
        request_id : str, optional
            Request ID for tracing, auto-generated if not provided
        """
        self._settings = get_settings()
        self._base_url = self._settings.mcp_server_url
        self._request_id = request_id or str(uuid.uuid4())
        self._timeout = httpx.Timeout(60.0, connect=10.0)

    @property
    def request_id(self) -> str:
        """Get current request ID."""
        return self._request_id

    async def classify_and_calculate(
        self,
        items: list[MenuItemSchema]
    ) -> dict[str, Any]:
        """
        Call MCP server to classify items and calculate total.

        Parameters
        ----------
        items : list[MenuItemSchema]
            Menu items to classify

        Returns
        -------
        dict
            Classification results with vegetarian items and total
        """
        logger.info(
            f"[{self._request_id}] Calling MCP server for classification",
            extra={"request_id": self._request_id, "item_count": len(items)}
        )
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/tools/classify-and-calculate",
                json={
                    "request_id": self._request_id,
                    "items": [item.model_dump() for item in items]
                },
                headers={"X-Request-ID": self._request_id}
            )
            response.raise_for_status()
            result = response.json()
        
        logger.info(
            f"[{self._request_id}] MCP returned {len(result.get('vegetarian_items', []))} vegetarian items",
            extra={"request_id": self._request_id}
        )
        
        return result

    async def recompute_with_corrections(
        self,
        original_items: list[MenuItemSchema],
        corrections: list[dict]
    ) -> dict[str, Any]:
        """
        Recompute classification with human corrections.

        Parameters
        ----------
        original_items : list[MenuItemSchema]
            Original menu items
        corrections : list[dict]
            Human corrections with name and is_vegetarian

        Returns
        -------
        dict
            Updated classification results
        """
        logger.info(
            f"[{self._request_id}] Recomputing with {len(corrections)} corrections",
            extra={"request_id": self._request_id}
        )
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/tools/recompute",
                json={
                    "request_id": self._request_id,
                    "items": [item.model_dump() for item in original_items],
                    "corrections": corrections
                },
                headers={"X-Request-ID": self._request_id}
            )
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> bool:
        """
        Check if MCP server is healthy.

        Returns
        -------
        bool
            True if server is healthy
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return False
