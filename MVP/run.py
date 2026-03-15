#!/usr/bin/env python3
"""MVP — ski technique analysis.

Runs technique analysis on a video: turn segmentation, coaching tips, overlay video,
and peak-pressure frames for each turn.

Usage:
    python MVP/run.py <video_path>
    python MVP/run.py <video_path> --max-fps 20 --no-overlay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "technique-analysis" / "src"))

from technique_analysis.common.contracts.models import TechniqueRunConfig  # noqa: E402
from technique_analysis.free_ski.pipeline.orchestrator import TechniqueAnalysisRunner  # noqa: E402


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ski technique analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video_path", type=Path, help="Path to the input video.")
    p.add_argument(
        "--pose-engine",
        choices=("mediapipe", "vision"),
        default="mediapipe",
        help=(
            "Pose backend. Use 'vision' on macOS 14+ to run on Apple Vision (often faster on Apple Silicon)."
        ),
    )
    p.add_argument("--max-fps", type=float, default=None,
                   help="Downsample to this FPS for pose analysis (None = auto).")
    p.add_argument("--max-dimension", type=int, default=None,
                   help="Max long-side px for pose extraction (None = auto).")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip rendering the technique overlay video.")
    p.add_argument("--render-max-dimension", type=int, default=None,
                   help="Resize the overlay output to this max dimension.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        return 1

    config = TechniqueRunConfig(
        pose_engine=args.pose_engine,
        max_fps=args.max_fps,
        max_dimension=args.max_dimension,
        render_overlay=not args.no_overlay,
        render_max_dimension=args.render_max_dimension,
    )
    summary = TechniqueAnalysisRunner(config=config).run(video_path)

    mvp_summary = {
        "video": str(video_path),
        "run_directory": summary.run_directory,
        "turns": len(summary.turns),
        "quality": {
            "overall_pose_confidence_mean": round(summary.quality.overall_pose_confidence_mean, 3),
            "low_confidence_fraction": round(summary.quality.low_confidence_fraction, 3),
            "warnings": summary.quality.warnings,
        },
        "coaching_tips": [
            {"severity": t.severity, "title": t.title, "explanation": t.explanation}
            for t in summary.coaching_tips
        ],
        "artifacts": summary.artifacts,
    }
    mvp_path = Path(summary.run_directory) / "mvp_summary.json"
    mvp_path.write_text(json.dumps(mvp_summary, indent=2), encoding="utf-8")

    divider = "─" * 60
    print(f"\n{divider}")
    print(f"Run directory : {summary.run_directory}")
    print(f"Turns found   : {len(summary.turns)}")
    if not args.no_overlay:
        overlay = next((a["path"] for a in summary.artifacts if a["kind"] == "video_overlay"), None)
        if overlay:
            print(f"Overlay video : {overlay}")
    if summary.coaching_tips:
        print(f"\nCoaching tips ({len(summary.coaching_tips)}):")
        for tip in summary.coaching_tips:
            print(f"  [{tip.severity.upper()}] {tip.title}")
    if summary.quality.warnings:
        print(f"\nWarnings:")
        for w in summary.quality.warnings:
            print(f"  ! {w}")
    print(f"\nMVP summary   : {mvp_path}")
    print(divider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
