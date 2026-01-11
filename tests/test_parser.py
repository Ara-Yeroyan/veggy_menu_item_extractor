import pytest
from src.api.services.parser import MenuParser


class TestMenuParser:

    def setup_method(self):
        self.parser = MenuParser()
        self.parser._llm_fallback_enabled = False

    def test_parse_standard_format(self):
        text = "Greek Salad ............. $8.50"
        items = self.parser.parse(text)
        
        assert len(items) == 1
        assert items[0].name == "Greek Salad"
        assert items[0].price == 8.50

    def test_parse_multiple_items(self, sample_ocr_text):
        items = self.parser.parse(sample_ocr_text)
        
        assert len(items) >= 6
        names = [item.name for item in items]
        assert "Greek Salad" in names
        assert "Veggie Burger" in names

    def test_parse_price_formats(self):
        formats = [
            ("Salad - $10.00", 10.00),
            ("Burger $15.50", 15.50),
            ("Pizza  12.99", 12.99),
            ("Soup ... $8.00", 8.00),
        ]
        
        for text, expected_price in formats:
            items = self.parser.parse(text)
            assert len(items) == 1, f"Failed for: {text}"
            assert items[0].price == expected_price

    def test_parse_empty_text(self):
        items = self.parser.parse("")
        assert items == []

    def test_parse_no_prices(self):
        text = "Just some random text without prices"
        items = self.parser.parse(text)
        assert items == []

    def test_parse_filters_invalid_prices(self):
        text = "Item 0.00"
        items = self.parser.parse(text)
        assert items == []
