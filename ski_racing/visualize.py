"""
Visualization module for ski racing analysis.
Creates trajectory overlays, bird's-eye views, speed profiles,
and summary figures.
"""
import cv2
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def create_demo_video(video_path, analysis_path, output_path):
    """
    Create demo video with trajectory and gate overlay.

    Args:
        video_path: Path to original video.
        analysis_path: Path to analysis JSON (from pipeline).
        output_path: Path for output video.
    """
    with open(analysis_path, "r") as f:
        analysis = json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build frame -> position lookup (final and raw/debug)
    trajectory = {
        p["frame"]: (int(p["x"]), int(p["y"]))
        for p in analysis.get("trajectory_2d", [])
    }
    raw_source = analysis.get("trajectory_2d_raw") or analysis.get("trajectory_2d_original") or []
    trajectory_raw = {
        p["frame"]: (int(p["x"]), int(p["y"]))
        for p in raw_source
    }
    outlier_frames = set(int(f) for f in analysis.get("outlier_frames", []))

    # Gate positions (static fallback) or per-frame if available
    frame_gate_lookup = None
    last_gate_positions = None
    if "frames" in analysis:
        frame_gate_lookup = {}
        for frame_entry in analysis.get("frames", []):
            frame_idx = frame_entry.get("frame")
            gates = []
            for g in frame_entry.get("gates", []):
                if "center_x" in g and "base_y" in g:
                    gates.append((int(g["center_x"]), int(g["base_y"]), g))
            frame_gate_lookup[frame_idx] = gates
    else:
        gate_positions = [
            (int(g["center_x"]), int(g["base_y"]), g)
            for g in analysis.get("gates", [])
        ]

    frame_num = 0
    trail_points = []
    raw_trail_points = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resolve per-frame gates if available
        if frame_gate_lookup is not None:
            if frame_num in frame_gate_lookup:
                gate_positions = frame_gate_lookup[frame_num]
                last_gate_positions = gate_positions
            elif last_gate_positions is not None:
                gate_positions = last_gate_positions
            else:
                gate_positions = []

        # Draw gates
        for i, (gx, gy, gmeta) in enumerate(gate_positions):
            if isinstance(gmeta, dict) and "class" in gmeta:
                color = (0, 0, 255) if int(gmeta["class"]) % 2 == 0 else (255, 0, 0)
            elif isinstance(gmeta, dict) and "gate_id" in gmeta:
                color = (0, 0, 255) if int(gmeta["gate_id"]) % 2 == 0 else (255, 0, 0)
            else:
                color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)  # Red / Blue
            cv2.circle(frame, (gx, gy), 10, color, -1)
            cv2.circle(frame, (gx, gy), 15, color, 2)

        # Build trail
        if frame_num in trajectory:
            trail_points.append(trajectory[frame_num])
        if frame_num in trajectory_raw:
            raw_trail_points.append(trajectory_raw[frame_num])

        # Draw raw/debug trail first (thin orange)
        for i in range(1, len(raw_trail_points)):
            cv2.line(frame, raw_trail_points[i - 1], raw_trail_points[i], (0, 140, 255), 1)

        # Draw filtered trajectory trail
        for i in range(1, len(trail_points)):
            cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 255, 255), 3)

        # Draw current position
        if frame_num in trajectory:
            pos = trajectory[frame_num]
            cv2.circle(frame, pos, 20, (0, 255, 0), -1)
            cv2.circle(frame, pos, 25, (0, 255, 0), 3)
            if frame_num in outlier_frames:
                cv2.circle(frame, pos, 30, (0, 0, 255), 2)
        if frame_num in outlier_frames and frame_num in trajectory_raw:
            raw_pos = trajectory_raw[frame_num]
            cv2.circle(frame, raw_pos, 10, (0, 0, 255), -1)

        # Text overlay
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Gates: {len(gate_positions)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Outliers: {len(outlier_frames)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 255), 2)

        # Physics status
        physics = analysis.get("physics_validation")
        if physics:
            status = "PASS" if physics["valid"] else f"FAIL ({len(physics['issues'])} issues)"
            color = (0, 255, 0) if physics["valid"] else (0, 0, 255)
            cv2.putText(frame, f"Physics: {status}", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Stabilization indicator
        if analysis.get("stabilized"):
            cv2.putText(frame, "STABILIZED", (width - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    print(f"✓ Demo video saved to {output_path}")


def create_summary_figure(analysis_path, output_path):
    """
    Create 4-panel summary figure:
    1. 2D trajectory
    2. 3D bird's-eye view
    3. Speed profile
    4. Statistics
    """
    with open(analysis_path, "r") as f:
        analysis = json.load(f)

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: 2D trajectory
    ax1 = plt.subplot(2, 2, 1)
    x_2d = [p["x"] for p in analysis["trajectory_2d"]]
    y_2d = [p["y"] for p in analysis["trajectory_2d"]]
    ax1.plot(x_2d, y_2d, "b-", linewidth=2, alpha=0.7)
    if x_2d:
        ax1.scatter(x_2d[0], y_2d[0], c="green", s=150, label="Start", zorder=5)
        ax1.scatter(x_2d[-1], y_2d[-1], c="red", s=150, label="Finish", zorder=5)
    ax1.invert_yaxis()
    ax1.set_title("2D Trajectory (Pixels)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: 3D bird's-eye view
    ax2 = plt.subplot(2, 2, 2)
    x_3d = [p["x"] for p in analysis["trajectory_3d"]]
    y_3d = [p["y"] for p in analysis["trajectory_3d"]]
    ax2.plot(x_3d, y_3d, "r-", linewidth=2, alpha=0.7)
    if x_3d:
        ax2.scatter(x_3d[0], y_3d[0], c="green", s=150, label="Start", zorder=5)
        ax2.scatter(x_3d[-1], y_3d[-1], c="red", s=150, label="Finish", zorder=5)
    ax2.set_title("3D Trajectory (Meters, Bird's Eye)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Panel 3: Speed profile
    ax3 = plt.subplot(2, 2, 3)
    physics = analysis.get("physics_validation")
    if physics and "metrics" in physics:
        speeds = physics["metrics"]["speeds_kmh"]
        # Reconstruct speed array from trajectory
        speed_vals = []
        t3d = analysis["trajectory_3d"]
        fps = analysis.get("video_info", {}).get("fps", 30)
        dt = 1.0 / fps
        for i in range(1, len(t3d)):
            dx = t3d[i]["x"] - t3d[i - 1]["x"]
            dy = t3d[i]["y"] - t3d[i - 1]["y"]
            dist = (dx**2 + dy**2) ** 0.5
            speed_vals.append(dist / dt * 3.6)

        ax3.plot(speed_vals, "g-", linewidth=2)
        ax3.set_title("Speed Profile", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Speed (km/h)")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    # Panel 4: Statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")

    if physics and "metrics" in physics:
        m = physics["metrics"]
        status = "PASS" if physics["valid"] else "FAIL"
        stats_text = (
            f"ANALYSIS SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Gates Detected:     {len(analysis['gates'])}\n"
            f"Trajectory Points:  {len(analysis['trajectory_2d'])}\n"
            f"Total Distance:     {m['total_distance_m']:.1f} m\n"
            f"Duration:           {m['duration_s']:.1f} s\n\n"
            f"Avg Speed:          {m['speeds_kmh']['mean']:.1f} km/h\n"
            f"Max Speed:          {m['speeds_kmh']['max']:.1f} km/h\n"
            f"Max G-force:        {m['g_forces']['max']:.2f} G\n"
            f"Min Turn Radius:    {m['turn_radii_m']['min']:.1f} m\n\n"
            f"Physics Check:      {status}\n"
        )
        if analysis.get("stabilized"):
            stats_text += "Mode:               STABILIZED\n"
        stats_text += f"Video: {Path(analysis['video']).name}"
    else:
        stats_text = (
            f"ANALYSIS SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Gates Detected:     {len(analysis['gates'])}\n"
            f"Trajectory Points:  {len(analysis['trajectory_2d'])}\n"
            f"Video: {Path(analysis['video']).name}"
        )

    ax4.text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
             verticalalignment="center", transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Summary saved to {output_path}")


def plot_trajectory_comparison(analysis_paths, output_path):
    """
    Compare trajectories from multiple runs (bird's-eye view overlay).

    Args:
        analysis_paths: List of analysis JSON file paths.
        output_path: Path for output image.
    """
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, len(analysis_paths)))

    for i, path in enumerate(analysis_paths):
        with open(path, "r") as f:
            analysis = json.load(f)

        x = [p["x"] for p in analysis["trajectory_3d"]]
        y = [p["y"] for p in analysis["trajectory_3d"]]
        label = Path(analysis["video"]).stem

        ax.plot(x, y, color=colors[i], linewidth=2, alpha=0.8, label=label)

    ax.set_title("Trajectory Comparison (Bird's Eye View)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Comparison plot saved to {output_path}")
