#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json


OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "data" / "menus"


SAMPLE_MENUS = [
    {
        "name": "italian_menu",
        "title": "BELLA ITALIA",
        "items": [
            ("Margherita Pizza", 14.00, True),
            ("Pepperoni Pizza", 16.00, False),
            ("Pasta Primavera", 13.50, True),
            ("Chicken Parmesan", 18.00, False),
            ("Mushroom Risotto", 15.50, True),
            ("Tiramisu", 8.00, True),
        ]
    },
    {
        "name": "asian_fusion",
        "title": "ASIAN FUSION",
        "items": [
            ("Vegetable Spring Rolls", 7.50, True),
            ("Pad Thai with Tofu", 12.00, True),
            ("Chicken Satay", 11.00, False),
            ("Tom Yum Soup", 9.00, False),
            ("Veggie Stir Fry", 11.50, True),
            ("Salmon Teriyaki", 19.00, False),
        ]
    },
    {
        "name": "american_diner",
        "title": "CLASSIC DINER",
        "items": [
            ("Greek Salad", 8.50, True),
            ("Caesar Salad", 9.00, False),
            ("Veggie Burger", 12.00, True),
            ("Beef Burger", 14.00, False),
            ("Chicken Wings", 11.00, False),
            ("French Fries", 5.00, True),
            ("Grilled Cheese", 8.00, True),
        ]
    },
    {
        "name": "indian_cuisine",
        "title": "SPICE OF INDIA",
        "items": [
            ("Palak Paneer", 13.00, True),
            ("Dal Makhani", 11.00, True),
            ("Chicken Tikka Masala", 15.00, False),
            ("Vegetable Biryani", 12.50, True),
            ("Lamb Rogan Josh", 17.00, False),
            ("Samosas (Veg)", 6.00, True),
            ("Naan Bread", 3.50, True),
        ]
    },
]


def create_menu_image(menu_data: dict, output_path: Path):
    width = 600
    line_height = 35
    header_height = 80
    padding = 40
    num_items = len(menu_data["items"])
    height = header_height + (num_items * line_height) + (padding * 2) + 40
    
    img = Image.new("RGB", (width, height), color="#FFFEF5")
    draw = ImageDraw.Draw(img)
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        item_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except OSError:
        title_font = ImageFont.load_default()
        item_font = ImageFont.load_default()
    
    draw.rectangle([0, 0, width, header_height], fill="#2C3E50")
    title = menu_data["title"]
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    draw.text(
        ((width - title_width) // 2, 20),
        title,
        fill="#FFFFFF",
        font=title_font
    )
    
    y = header_height + padding
    for name, price, is_veg in menu_data["items"]:
        draw.text((padding, y), name, fill="#333333", font=item_font)
        
        price_text = f"${price:.2f}"
        bbox = draw.textbbox((0, 0), price_text, font=item_font)
        price_width = bbox[2] - bbox[0]
        draw.text(
            (width - padding - price_width, y),
            price_text,
            fill="#333333",
            font=item_font
        )
        
        name_bbox = draw.textbbox((padding, y), name, font=item_font)
        dot_start = name_bbox[2] + 10
        dot_end = width - padding - price_width - 10
        dot_y = y + 12
        for x in range(int(dot_start), int(dot_end), 8):
            draw.ellipse([x, dot_y, x + 2, dot_y + 2], fill="#CCCCCC")
        
        y += line_height
    
    img.save(output_path)
    return output_path


def generate_ground_truth(menus: list) -> dict:
    ground_truth = {}
    for menu in menus:
        veg_items = [
            {"name": name, "price": price}
            for name, price, is_veg in menu["items"]
            if is_veg
        ]
        total = sum(item["price"] for item in veg_items)
        ground_truth[menu["name"]] = {
            "vegetarian_items": veg_items,
            "total_sum": round(total, 2)
        }
    return ground_truth


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for menu in SAMPLE_MENUS:
        output_path = OUTPUT_DIR / f"{menu['name']}.png"
        create_menu_image(menu, output_path)
        print(f"Created: {output_path}")
    
    ground_truth = generate_ground_truth(SAMPLE_MENUS)
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Created: {gt_path}")
    
    print(f"\nGenerated {len(SAMPLE_MENUS)} test menu images")


if __name__ == "__main__":
    main()
