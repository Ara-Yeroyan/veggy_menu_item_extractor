#!/usr/bin/env python3
"""
Evaluation script for the Menu Parser system.

Runs inference on test menu images and compares results against ground truth.
Reports accuracy metrics including precision, recall, and F1 score.

Usage:
    python scripts/evaluate_system.py [--api-url http://localhost:8000]
"""
import argparse
import json
import sys
from pathlib import Path

import httpx


TESTS_DIR = Path(__file__).parent.parent / "tests" / "data" / "menus"
GROUND_TRUTH_PATH = TESTS_DIR / "ground_truth.json"

MENU_FILES = {
    "italian_menu": "italian_menu.png",
    "asian_fusion": "asian_fusion.png",
    "american_diner": "american_diner.png",
    "indian_cuisine": "indian_cuisine.png",
}


def load_ground_truth() -> dict:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def call_api(api_url: str, image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        files = {"files": (image_path.name, f, "image/png")}
        response = httpx.post(
            f"{api_url}/process-menu",
            files=files,
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("  ", " ")


def compare_results(predicted: dict, expected: dict) -> dict:
    pred_items = predicted.get("vegetarian_items", predicted.get("confident_items", []))
    pred_names = {normalize_name(item["name"]) for item in pred_items}
    
    # Build lookup for all items with their details (for reasoning)
    all_items = predicted.get("all_items", [])
    item_details = {normalize_name(item["name"]): item for item in all_items}
    
    exp_items = expected["vegetarian_items"]
    exp_names = {normalize_name(item["name"]) for item in exp_items}
    
    true_positives = pred_names & exp_names
    false_positives = pred_names - exp_names
    false_negatives = exp_names - pred_names
    
    precision = len(true_positives) / len(pred_names) if pred_names else 0
    recall = len(true_positives) / len(exp_names) if exp_names else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    pred_sum = predicted.get("total_sum", predicted.get("partial_sum", 0))
    exp_sum = expected["total_sum"]
    sum_diff = abs(pred_sum - exp_sum)
    sum_accuracy = 1 - (sum_diff / exp_sum) if exp_sum > 0 else (1 if sum_diff == 0 else 0)
    
    # Include reasoning for false negatives
    false_negative_details = []
    for name in false_negatives:
        details = item_details.get(name, {})
        false_negative_details.append({
            "name": name,
            "is_vegetarian": details.get("is_vegetarian"),
            "confidence": details.get("confidence"),
            "method": details.get("method", "unknown"),
            "reasoning": details.get("reasoning", "Not found in OCR output"),
        })
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": false_negative_details,
        "predicted_sum": pred_sum,
        "expected_sum": exp_sum,
        "sum_accuracy": max(0, sum_accuracy),
        "item_details": item_details,
    }


def print_results(menu_name: str, comparison: dict):
    print(f"\n{'='*60}")
    print(f"📋 {menu_name.upper().replace('_', ' ')}")
    print(f"{'='*60}")
    
    print(f"\n📊 Classification Metrics:")
    print(f"   Precision: {comparison['precision']*100:.1f}%")
    print(f"   Recall:    {comparison['recall']*100:.1f}%")
    print(f"   F1 Score:  {comparison['f1']*100:.1f}%")
    
    print(f"\n💰 Sum Comparison:")
    print(f"   Expected:  ${comparison['expected_sum']:.2f}")
    print(f"   Predicted: ${comparison['predicted_sum']:.2f}")
    print(f"   Accuracy:  {comparison['sum_accuracy']*100:.1f}%")
    
    if comparison["true_positives"]:
        print(f"\n✅ Correct ({len(comparison['true_positives'])}):")
        for item in sorted(comparison["true_positives"]):
            print(f"   • {item}")
    
    if comparison["false_positives"]:
        print(f"\n⚠️  False Positives ({len(comparison['false_positives'])}):")
        for item in sorted(comparison["false_positives"]):
            print(f"   • {item} (predicted veg, but not in ground truth)")
    
    if comparison["false_negatives"]:
        print(f"\n❌ Missed ({len(comparison['false_negatives'])}):")
        for item in comparison["false_negatives"]:
            name = item["name"] if isinstance(item, dict) else item
            print(f"   • {name}")
            if isinstance(item, dict):
                if item.get("reasoning"):
                    print(f"     └─ Reason: {item['reasoning']}")
                if item.get("method"):
                    print(f"     └─ Method: {item['method']}, Conf: {item.get('confidence', 'N/A')}")
                if item.get("is_vegetarian") is False:
                    print(f"     └─ System classified as NON-vegetarian")


def print_summary(all_comparisons: dict):
    print(f"\n{'='*60}")
    print(f"📈 OVERALL SUMMARY")
    print(f"{'='*60}")
    
    avg_precision = sum(c["precision"] for c in all_comparisons.values()) / len(all_comparisons)
    avg_recall = sum(c["recall"] for c in all_comparisons.values()) / len(all_comparisons)
    avg_f1 = sum(c["f1"] for c in all_comparisons.values()) / len(all_comparisons)
    avg_sum_acc = sum(c["sum_accuracy"] for c in all_comparisons.values()) / len(all_comparisons)
    
    total_tp = sum(len(c["true_positives"]) for c in all_comparisons.values())
    total_fp = sum(len(c["false_positives"]) for c in all_comparisons.values())
    total_fn = sum(len(c["false_negatives"]) for c in all_comparisons.values())
    
    print(f"\n   Average Precision: {avg_precision*100:.1f}%")
    print(f"   Average Recall:    {avg_recall*100:.1f}%")
    print(f"   Average F1 Score:  {avg_f1*100:.1f}%")
    print(f"   Average Sum Acc:   {avg_sum_acc*100:.1f}%")
    
    print(f"\n   Total Correct:         {total_tp}")
    print(f"   Total False Positives: {total_fp}")
    print(f"   Total Missed:          {total_fn}")
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\n   Overall Precision: {overall_precision*100:.1f}%")
    print(f"   Overall Recall:    {overall_recall*100:.1f}%")
    print(f"   Overall F1 Score:  {overall_f1*100:.1f}%")
    
    # Show all missed items with reasoning
    all_missed = []
    for menu, comp in all_comparisons.items():
        for item in comp["false_negatives"]:
            if isinstance(item, dict):
                item["menu"] = menu
                all_missed.append(item)
    
    if all_missed:
        print(f"\n{'='*60}")
        print(f"🔍 FAILURE ANALYSIS")
        print(f"{'='*60}")
        for item in all_missed:
            print(f"\n   📍 {item['menu'].replace('_', ' ').title()}: {item['name']}")
            if item.get("reasoning"):
                print(f"      Reason: {item['reasoning']}")
            if item.get("method"):
                print(f"      Method: {item['method']}, Confidence: {item.get('confidence', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Menu Parser system")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--menu",
        choices=list(MENU_FILES.keys()),
        help="Evaluate single menu (default: all)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()
    
    if not GROUND_TRUTH_PATH.exists():
        print("❌ Ground truth file not found. Run: make generate-menus")
        sys.exit(1)
    
    ground_truth = load_ground_truth()
    
    menus_to_test = {args.menu: MENU_FILES[args.menu]} if args.menu else MENU_FILES
    
    print(f"🔍 Evaluating Menu Parser System")
    print(f"   API URL: {args.api_url}")
    print(f"   Menus:   {len(menus_to_test)}")
    
    all_comparisons = {}
    all_responses = {}
    
    for menu_name, filename in menus_to_test.items():
        image_path = TESTS_DIR / filename
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            print(f"   Run: make generate-menus")
            continue
        
        print(f"\n⏳ Processing {menu_name}...")
        
        try:
            response = call_api(args.api_url, image_path)
            all_responses[menu_name] = response
            
            if menu_name in ground_truth:
                comparison = compare_results(response, ground_truth[menu_name])
                all_comparisons[menu_name] = comparison
                
                if not args.json:
                    print_results(menu_name, comparison)
            else:
                print(f"   ⚠️ No ground truth for {menu_name}")
                
        except httpx.ConnectError:
            print(f"❌ Cannot connect to API at {args.api_url}")
            print(f"   Make sure the system is running: make docker-up")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing {menu_name}: {e}")
            continue
    
    if args.json:
        output = {
            "comparisons": all_comparisons,
            "responses": all_responses,
        }
        print(json.dumps(output, indent=2))
    elif all_comparisons:
        print_summary(all_comparisons)
    
    if all_comparisons:
        avg_f1 = sum(c["f1"] for c in all_comparisons.values()) / len(all_comparisons)
        sys.exit(0 if avg_f1 >= 0.7 else 1)


if __name__ == "__main__":
    main()
