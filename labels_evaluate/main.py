"""
BART Zero-Shot Classification Demo for Claim Worthiness Testing

Configuration:
- Input file: test_claims.txt (claims and expected labels)
- Strategies file: label_strategies.txt (different label configurations to test)
- Output file: bart_results.txt (detailed results and rankings)

Run with: python bart_demo.py
"""

from transformers import pipeline
import os
from datetime import datetime

# ============================================================================
# FILE CONFIGURATION
# ============================================================================
INPUT_FILE = r"modern_issues_demo\labels_evaluate\input.txt"
STRATEGIES_FILE = r"modern_issues_demo\labels_evaluate\strats.txt"
OUTPUT_FILE = r"modern_issues_demo\labels_evaluate\output.txt"

# ============================================================================
# Model Setup
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BART_PATH = os.path.join(BASE_DIR, 'models', 'bart-large-mnli')

print("Loading BART zero-shot classifier...")
print(f"Model path: {BART_PATH}")

# Check if local model exists
if os.path.exists(BART_PATH):
    print("Loading from local model directory...")
    # Use local_files_only=True to prevent fallback to HuggingFace Hub
    classifier = pipeline("zero-shot-classification", model=BART_PATH, local_files_only=True)
else:
    print(f"Local model not found at: {BART_PATH}")
    print("Downloading from HuggingFace Hub (this may take a while)...")
    # Download from HuggingFace Hub
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print(f"Consider saving the model locally to: {BART_PATH}")

print("✓ BART classifier loaded\n")

# ============================================================================
# File Parsing Functions
# ============================================================================

def parse_claims_file(filepath):
    """
    Parse test_claims.txt file.
    
    Expected format:
    WORTHY
    claim text here
    another worthy claim
    
    UNWORTHY
    unworthy claim here
    """
    claims = []
    current_label = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.upper() == 'WORTHY':
                current_label = True
            elif line.upper() == 'UNWORTHY':
                current_label = False
            elif current_label is not None:
                claims.append((line, current_label))
    
    return claims

def parse_strategies_file(filepath):
    """
    Parse label_strategies.txt file.
    
    Expected format:
    [Strategy Name]
    filter_name | label1, label2 | threshold | reject_index
    
    Example:
    [Simple Binary]
    Factual Check | factual claim, not factual | 0.6 | 1
    """
    strategies = {}
    current_strategy = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Strategy header
            if line.startswith('[') and line.endswith(']'):
                current_strategy = line[1:-1]
                strategies[current_strategy] = []
            # Filter definition
            elif current_strategy and '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 4:
                    filter_name = parts[0]
                    labels = [l.strip() for l in parts[1].split(',')]
                    threshold = float(parts[2])
                    reject_index = int(parts[3])
                    
                    strategies[current_strategy].append({
                        'name': filter_name,
                        'labels': labels,
                        'threshold': threshold,
                        'reject_index': reject_index
                    })
    
    return strategies

# ============================================================================
# Testing Functions
# ============================================================================

def test_strategy(strategy_name, filters, claims, output_lines):
    """Test a single label strategy on all claims."""
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"Testing: {strategy_name}")
    output_lines.append(f"{'='*80}")
    
    correct = 0
    total = len(claims)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    results = []
    
    for claim, is_worthy in claims:
        passed = True
        filter_results = []
        
        # Apply all filters in sequence
        for f in filters:
            result = classifier(claim, candidate_labels=f["labels"])
            
            reject_score = result['scores'][f["reject_index"]]
            filter_passed = reject_score <= f["threshold"]
            
            filter_results.append({
                "name": f["name"],
                "scores": result['scores'],
                "labels": result['labels'],
                "passed": filter_passed
            })
            
            if not filter_passed:
                passed = False
                break
        
        # Evaluate result
        prediction_worthy = passed
        if prediction_worthy == is_worthy:
            correct += 1
            if is_worthy:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if prediction_worthy and not is_worthy:
                false_positives += 1
            else:
                false_negatives += 1
        
        results.append({
            "claim": claim,
            "expected": is_worthy,
            "predicted": prediction_worthy,
            "correct": prediction_worthy == is_worthy,
            "filters": filter_results
        })
    
    # Calculate metrics
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Write to output
    output_lines.append(f"\nOverall Metrics:")
    output_lines.append(f"  Accuracy:  {accuracy:.1%} ({correct}/{total})")
    output_lines.append(f"  Precision: {precision:.1%} (of claims marked worthy, how many were actually worthy)")
    output_lines.append(f"  Recall:    {recall:.1%} (of actually worthy claims, how many were caught)")
    output_lines.append(f"  F1 Score:  {f1:.1%} (harmonic mean of precision & recall)")
    output_lines.append(f"\nConfusion Matrix:")
    output_lines.append(f"  True Positives:  {true_positives:2d} (worthy → worthy)")
    output_lines.append(f"  True Negatives:  {true_negatives:2d} (unworthy → unworthy)")
    output_lines.append(f"  False Positives: {false_positives:2d} (unworthy → worthy) ⚠️")
    output_lines.append(f"  False Negatives: {false_negatives:2d} (worthy → unworthy) ⚠️")
    
    # Show failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        output_lines.append(f"\nFailures ({len(failures)} total):")
        for i, fail in enumerate(failures[:5], 1):
            output_lines.append(f"\n  {i}. \"{fail['claim'][:70]}...\"")
            output_lines.append(f"     Expected: {'WORTHY' if fail['expected'] else 'UNWORTHY'}")
            output_lines.append(f"     Predicted: {'WORTHY' if fail['predicted'] else 'UNWORTHY'}")
            if fail['filters']:
                last_filter = fail['filters'][-1] if not fail['predicted'] else fail['filters'][0]
                output_lines.append(f"     Filter: {last_filter['name']} - {last_filter['labels']}")
                scores_str = ", ".join([f"{s:.3f}" for s in last_filter['scores']])
                output_lines.append(f"     Scores: [{scores_str}]")
    
    return {
        "strategy": strategy_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all strategy tests and compare results."""
    output_lines = []
    
    # Header
    output_lines.append("="*80)
    output_lines.append("BART ZERO-SHOT CLASSIFICATION TEST RESULTS")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("="*80)
    
    # Load data
    print(f"Reading claims from: {INPUT_FILE}")
    claims = parse_claims_file(INPUT_FILE)
    
    print(f"Reading strategies from: {STRATEGIES_FILE}")
    strategies = parse_strategies_file(STRATEGIES_FILE)
    
    worthy_count = sum(1 for _, w in claims if w)
    unworthy_count = sum(1 for _, w in claims if not w)
    
    output_lines.append(f"\nTest Dataset: {len(claims)} claims")
    output_lines.append(f"  - {worthy_count} worthy claims")
    output_lines.append(f"  - {unworthy_count} unworthy claims")
    output_lines.append(f"\nStrategies to test: {len(strategies)}")
    
    print(f"\nTesting {len(claims)} claims across {len(strategies)} strategies")
    print(f"  - {worthy_count} worthy claims")
    print(f"  - {unworthy_count} unworthy claims\n")
    
    all_results = []
    
    # Test each strategy
    for strategy_name, filters in strategies.items():
        print(f"Testing: {strategy_name}...")
        result = test_strategy(strategy_name, filters, claims, output_lines)
        all_results.append(result)
    
    # Final comparison
    output_lines.append(f"\n\n{'='*80}")
    output_lines.append("FINAL COMPARISON - All Strategies Ranked by F1 Score")
    output_lines.append(f"{'='*80}\n")
    
    sorted_results = sorted(all_results, key=lambda x: x['f1'], reverse=True)
    
    output_lines.append(f"{'Rank':<6} {'Strategy':<35} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    output_lines.append("-" * 80)
    
    for rank, result in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        line = (f"{medal} {rank:<4} {result['strategy']:<35} "
                f"{result['accuracy']:.1%}    "
                f"{result['precision']:.1%}    "
                f"{result['recall']:.1%}    "
                f"{result['f1']:.1%}")
        output_lines.append(line)
    
    # Recommendation
    best = sorted_results[0]
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"🏆 RECOMMENDATION: {best['strategy']}")
    output_lines.append(f"{'='*80}")
    output_lines.append(f"This strategy achieved the best F1 score of {best['f1']:.1%}")
    output_lines.append(f"  - Accuracy: {best['accuracy']:.1%}")
    output_lines.append(f"  - Precision: {best['precision']:.1%} (minimizes false positives)")
    output_lines.append(f"  - Recall: {best['recall']:.1%} (catches worthy claims)")
    output_lines.append(f"  - False Positives: {best['false_positives']} (unworthy claims that passed)")
    output_lines.append(f"  - False Negatives: {best['false_negatives']} (worthy claims that failed)")
    
    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Results written to: {OUTPUT_FILE}")
    print(f"\n🏆 Best Strategy: {best['strategy']} (F1: {best['f1']:.1%})")

if __name__ == "__main__":
    main()