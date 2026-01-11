import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_menu_items():
    return [
        {"name": "Greek Salad", "price": 8.50},
        {"name": "Veggie Burger", "price": 12.00},
        {"name": "Chicken Wings", "price": 11.00},
        {"name": "Mushroom Risotto", "price": 15.50},
        {"name": "Grilled Salmon", "price": 22.00},
        {"name": "Caesar Salad", "price": 9.00},
        {"name": "Margherita Pizza", "price": 14.00},
        {"name": "Beef Steak", "price": 28.00},
    ]


@pytest.fixture
def sample_ocr_text():
    return """
    APPETIZERS
    Greek Salad ............. $8.50
    Caesar Salad ............ $9.00
    Chicken Wings ........... $11.00
    
    MAINS
    Veggie Burger ........... $12.00
    Beef Steak .............. $28.00
    Grilled Salmon .......... $22.00
    Mushroom Risotto ........ $15.50
    Margherita Pizza ........ $14.00
    """


@pytest.fixture
def expected_vegetarian_items():
    return [
        {"name": "Greek Salad", "price": 8.50},
        {"name": "Veggie Burger", "price": 12.00},
        {"name": "Mushroom Risotto", "price": 15.50},
        {"name": "Margherita Pizza", "price": 14.00},
    ]
