"""
Evaluation script for gate detection accuracy.
Computes precision, recall, F1 against manually annotated ground truth.

Usage:
    python scripts/evaluate.py --predictions artifacts/outputs/race1_analysis.json --ground-truth data/annotations/race1_gt.json
"""
import json
import argparse
from pathlib import Path


def evaluate_gate_detection(predicted_gates, ground_truth_gates, threshold=50):
    """
    Calculate precision/recall/F1 for gate detection.

    Args:
        predicted_gates: List of {"center_x": float, "base_y": float}.
        ground_truth_gates: List of {"x": float, "y": float}.
        threshold: Pixel distance to count as a correct match.

    Returns:
        Dictionary with precision, recall, F1, and counts.
    """
    true_positives = 0
    false_positives = 0
    matched = set()

    for pred in predicted_gates:
        px = pred.get("center_x", pred.get("x", 0))
        py = pred.get("base_y", pred.get("y", 0))

        min_dist = float("inf")
        closest_idx = -1

        for i, gt in enumerate(ground_truth_gates):
            if i in matched:
                continue
            gx = gt.get("x", gt.get("center_x", 0))
            gy = gt.get("y", gt.get("base_y", 0))
            dist = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        if min_dist < threshold:
            true_positives += 1
            matched.add(closest_idx)
        else:
            false_positives += 1

    false_negatives = len(ground_truth_gates) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_tracking_coverage(trajectory, total_frames):
    """
    Evaluate what percentage of frames have a tracked position.

    Args:
        trajectory: List of {"frame": int, ...}.
        total_frames: Total frames in the video.

    Returns:
        Coverage percentage.
    """
    tracked_frames = len(trajectory)
    coverage = tracked_frames / total_frames * 100 if total_frames > 0 else 0
    return {
        "tracked_frames": tracked_frames,
        "total_frames": total_frames,
        "coverage_percent": coverage,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection accuracy")
    parser.add_argument("--predictions", required=True, help="Path to analysis JSON")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--threshold", type=float, default=50, help="Match threshold (pixels)")
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # Evaluate gate detection
    gate_results = evaluate_gate_detection(
        predictions["gates"],
        ground_truth.get("gates", []),
        threshold=args.threshold,
    )

    print("\n=== Gate Detection Evaluation ===")
    print(f"  Precision: {gate_results['precision']:.3f}")
    print(f"  Recall:    {gate_results['recall']:.3f}")
    print(f"  F1 Score:  {gate_results['f1']:.3f}")
    print(f"  TP: {gate_results['true_positives']}, "
          f"FP: {gate_results['false_positives']}, "
          f"FN: {gate_results['false_negatives']}")

    # Evaluate tracking coverage
    total_frames = predictions.get("video_info", {}).get("total_frames", 0)
    coverage = evaluate_tracking_coverage(predictions["trajectory_2d"], total_frames)

    print(f"\n=== Tracking Coverage ===")
    print(f"  Tracked: {coverage['tracked_frames']}/{coverage['total_frames']} frames")
    print(f"  Coverage: {coverage['coverage_percent']:.1f}%")

    # Targets
    print(f"\n=== Targets ===")
    if gate_results["f1"] >= 0.85:
        print(f"  ✅ Gate F1 >= 0.85")
    else:
        print(f"  ❌ Gate F1 < 0.85 (current: {gate_results['f1']:.3f})")

    if coverage["coverage_percent"] >= 90:
        print(f"  ✅ Tracking coverage >= 90%")
    else:
        print(f"  ❌ Tracking coverage < 90% (current: {coverage['coverage_percent']:.1f}%)")


if __name__ == "__main__":
    main()
