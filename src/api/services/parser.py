import re
import logging
from typing import Optional
import httpx

from src.api.schemas import MenuItemSchema
from configs import get_settings

logger = logging.getLogger(__name__)


class MenuParser:
    """
    Parse raw OCR text into structured menu items.
    
    Uses regex patterns with LLM fallback for complex cases.
    """

    PRICE_PATTERNS = [
        r"(.+?)\s*\$\s*(\d+\.\d{2})\s*$",
        r"(.+?)\s+\$(\d+\.\d{2})",
        r"(.+?)\s*[\.\-–—]{3,}\s*\$?\s*(\d+\.\d{2})",
        r"(.+?)\s{2,}(\d+\.\d{2})\s*$",
        r"^(.+?)\s+(\d+\.\d{2})$",
        r"(.+?)\s*\$\s*(\d+)\s*$",
        r"(.+?)\s*[\.\-–—]{3,}\s*\$?\s*(\d+)\s*$",
    ]

    def __init__(self):
        self._settings = get_settings()
        self._llm_fallback_enabled = True

    def parse(self, raw_text: str) -> list[MenuItemSchema]:
        """
        Parse raw OCR text into menu items.

        Parameters
        ----------
        raw_text : str
            Raw text from OCR extraction

        Returns
        -------
        list[MenuItemSchema]
            List of parsed menu items with names and prices
        """
        items = self._parse_with_regex(raw_text)
        
        if not items and self._llm_fallback_enabled:
            logger.info("Regex parsing failed, attempting LLM fallback")
            items = self._parse_with_llm(raw_text)
        
        logger.info(f"Parsed {len(items)} menu items")
        return items

    def _parse_with_regex(self, text: str) -> list[MenuItemSchema]:
        """
        Parse menu text using regex patterns.

        Parameters
        ----------
        text : str
            Raw OCR text

        Returns
        -------
        list[MenuItemSchema]
            Parsed menu items
        """
        items = []
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            item = self._extract_item_from_line(line)
            if item:
                items.append(item)
        
        return items

    def _extract_item_from_line(self, line: str) -> Optional[MenuItemSchema]:
        """
        Extract a menu item from a single line.

        Parameters
        ----------
        line : str
            Single line from menu text

        Returns
        -------
        Optional[MenuItemSchema]
            Parsed item or None if parsing failed
        """
        line = line.replace("...", " ").replace("..", " ")
        line = re.sub(r"\.{2,}", " ", line)
        
        price_match = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", line)
        if price_match:
            try:
                price = float(price_match.group(1))
                if 0 < price < 1000:
                    name = line[:price_match.start()].strip()
                    name = re.sub(r"[\.\-–—\s]+$", "", name).strip()
                    name = re.sub(r"\s+", " ", name)
                    if len(name) >= 2:
                        return MenuItemSchema(name=name, price=price)
            except ValueError:
                pass
        
        for pattern in self.PRICE_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r"[\.\-–—]+$", "", name).strip()
                name = re.sub(r"\s+", " ", name)
                
                if len(name) < 2:
                    continue
                
                try:
                    price = float(match.group(2))
                    if 0 < price < 1000:
                        return MenuItemSchema(name=name, price=price)
                except ValueError:
                    continue
        
        return None

    def _parse_with_llm(self, text: str) -> list[MenuItemSchema]:
        """
        Use LLM to parse complex menu text.

        Parameters
        ----------
        text : str
            Raw OCR text that regex couldn't parse

        Returns
        -------
        list[MenuItemSchema]
            Parsed menu items from LLM
        """
        try:
            prompt = self._build_parsing_prompt(text)
            response = self._call_mcp_for_parsing(prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return []

    def _build_parsing_prompt(self, text: str) -> str:
        """Build prompt for LLM-based parsing."""
        return f"""Extract menu items from this text. Return ONLY a JSON array.
Each item should have "name" (string) and "price" (number).

Text:
{text}

Return format: [{{"name": "Item Name", "price": 9.99}}]
Return ONLY the JSON array, no other text."""

    def _call_mcp_for_parsing(self, prompt: str) -> str:
        """Call MCP server for LLM parsing assistance."""
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{self._settings.mcp_server_url}/tools/parse-menu",
                json={"prompt": prompt}
            )
            response.raise_for_status()
            return response.json().get("result", "[]")

    def _parse_llm_response(self, response: str) -> list[MenuItemSchema]:
        """Parse LLM response into menu items."""
        import json
        
        try:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                items_data = json.loads(json_match.group())
                return [
                    MenuItemSchema(name=item["name"], price=float(item["price"]))
                    for item in items_data
                    if "name" in item and "price" in item
                ]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
        
        return []
