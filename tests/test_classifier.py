import pytest
from unittest.mock import Mock, patch


class TestKeywordClassification:

    def test_vegetarian_keyword_positive(self):
        from src.mcp.llm.classifier import LLMClassifier
        
        with patch.object(LLMClassifier, '__init__', lambda x: None):
            classifier = LLMClassifier()
            classifier._positive_keywords = {"vegetarian", "veggie", "vegan"}
            classifier._negative_keywords = {"chicken", "beef", "fish"}
            classifier._vegetarian_markers = {"(v)", "[v]"}
            
            result = classifier._keyword_classification("Veggie Burger")
            
            assert result["is_vegetarian"] is True
            assert result["confidence"] >= 0.9

    def test_meat_keyword_negative(self):
        from src.mcp.llm.classifier import LLMClassifier
        
        with patch.object(LLMClassifier, '__init__', lambda x: None):
            classifier = LLMClassifier()
            classifier._positive_keywords = {"vegetarian", "veggie", "vegan"}
            classifier._negative_keywords = {"chicken", "beef", "fish"}
            classifier._vegetarian_markers = {"(v)", "[v]"}
            
            result = classifier._keyword_classification("Chicken Wings")
            
            assert result["is_vegetarian"] is False
            assert result["confidence"] >= 0.9

    def test_no_keyword_match(self):
        from src.mcp.llm.classifier import LLMClassifier
        
        with patch.object(LLMClassifier, '__init__', lambda x: None):
            classifier = LLMClassifier()
            classifier._positive_keywords = {"vegetarian", "veggie", "vegan"}
            classifier._negative_keywords = {"chicken", "beef", "fish"}
            classifier._vegetarian_markers = {"(v)", "[v]"}
            
            result = classifier._keyword_classification("Mushroom Risotto")
            
            assert result["is_vegetarian"] is None
            assert result["confidence"] == 0.0


class TestCalculator:

    def test_calculate_sum(self):
        from src.mcp.tools.calculator import CalculatorTool
        
        items = [
            {"name": "Salad", "price": 8.50},
            {"name": "Burger", "price": 12.00},
        ]
        
        result = CalculatorTool.execute(items)
        
        assert result["total_sum"] == 20.50
        assert result["item_count"] == 2

    def test_calculate_empty_list(self):
        from src.mcp.tools.calculator import CalculatorTool
        
        result = CalculatorTool.execute([])
        
        assert result["total_sum"] == 0.0
        assert result["item_count"] == 0

    def test_calculate_filters_invalid(self):
        from src.mcp.tools.calculator import CalculatorTool
        
        items = [
            {"name": "Valid", "price": 10.00},
            {"name": "Invalid", "price": -5.00},
            {"name": "Zero", "price": 0},
        ]
        
        result = CalculatorTool.execute(items)
        
        assert result["total_sum"] == 10.00
        assert result["item_count"] == 1
