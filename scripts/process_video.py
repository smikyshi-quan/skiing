"""
Process a ski racing video through the full analysis pipeline.

Usage:
    python scripts/process_video.py data/test_videos/race1.mp4 --gate-model models/gate_detector_best.pt
    python scripts/process_video.py data/test_videos/ --gate-model models/gate_detector_best.pt --discipline giant_slalom
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ski_racing.pipeline import SkiRacingPipeline
from ski_racing.visualize import create_summary_figure, create_demo_video
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process ski racing video(s)")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("--gate-model", required=True, help="Path to gate detector model")
    parser.add_argument("--discipline", default="slalom",
                        choices=["slalom", "giant_slalom", "downhill"])
    parser.add_argument("--gate-spacing", type=float, default=12.0,
                        help="Gate spacing in meters (12 for SL, 25-30 for GS)")
    parser.add_argument("--projection", default="scale",
                        choices=["scale", "homography"],
                        help="Projection mode (scale uses gate spacing; homography uses global H)")
    parser.add_argument("--gate-conf", type=float, default=0.3,
                        help="Gate detection confidence threshold")
    parser.add_argument("--gate-search-frames", type=int, default=150,
                        help="Search first N frames for best gate detection")
    parser.add_argument("--gate-search-stride", type=int, default=5,
                        help="Stride for gate search frames")
    parser.add_argument("--gate-track-frames", type=int, default=120,
                        help="Frames to use for temporal gate stabilization")
    parser.add_argument("--gate-track-stride", type=int, default=3,
                        help="Stride for gate tracking frames")
    parser.add_argument("--gate-track-min-obs", type=int, default=3,
                        help="Minimum observations per gate to keep")
    parser.add_argument("--output-dir", default="artifacts/outputs", help="Output directory")
    parser.add_argument("--no-physics", action="store_true", help="Skip physics validation")
    parser.add_argument("--demo-video", action="store_true", help="Also create demo video with overlay")
    parser.add_argument("--summary", action="store_true", help="Also create summary figure")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Process every Nth frame for tracking (temporal fallback)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit number of frames processed from start")
    parser.add_argument("--max-jump", type=float, default=None,
                        help="Max pixel jump for temporal tracking (override dynamic default)")
    parser.add_argument("--stabilize", action="store_true",
                        help="Enable camera stabilization + Kalman smoothing + dynamic scale (all 4 phases)")
    parser.add_argument("--camera-mode", default="translation",
                        choices=["translation", "affine"],
                        help="Camera motion model for stabilization (translation or affine)")
    args = parser.parse_args()

    pipeline = SkiRacingPipeline(
        gate_model_path=args.gate_model,
        discipline=args.discipline,
        gate_spacing_m=args.gate_spacing,
        stabilize=args.stabilize,
        camera_mode=args.camera_mode,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        videos = sorted(
            list(input_path.glob("*.mp4"))
            + list(input_path.glob("*.avi"))
            + list(input_path.glob("*.mov"))
        )
    else:
        videos = [input_path]

    print(f"Found {len(videos)} video(s) to process\n")

    for video in videos:
        try:
            results = pipeline.process_video(
                video_path=str(video),
                output_dir=args.output_dir,
                validate_physics=not args.no_physics,
                projection=args.projection,
                gate_conf=args.gate_conf,
                gate_search_frames=args.gate_search_frames,
                gate_search_stride=args.gate_search_stride,
                gate_track_frames=args.gate_track_frames,
                gate_track_stride=args.gate_track_stride,
                gate_track_min_obs=args.gate_track_min_obs,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                max_jump=args.max_jump,
            )

            analysis_path = Path(args.output_dir) / f"{video.stem}_analysis.json"

            if args.summary:
                summary_path = Path(args.output_dir) / f"{video.stem}_summary.png"
                create_summary_figure(str(analysis_path), str(summary_path))

            if args.demo_video:
                demo_path = Path(args.output_dir) / f"{video.stem}_demo.mp4"
                create_demo_video(str(video), str(analysis_path), str(demo_path))

            print(f"✓ Successfully processed {video.name}\n")
        except Exception as e:
            print(f"✗ Error processing {video.name}: {e}\n")

    print("Done!")


if __name__ == "__main__":
    main()
