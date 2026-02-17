#!/usr/bin/env python3
"""
Track D study runner:
1) Kalman Q sweep across regression videos.
2) Pipeline ordering comparison (camera compensation pre-smoothing).
3) Markdown reports in tracks/D_tracking_outlier/reports.
"""

import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ski_racing.pipeline import SkiRacingPipeline  # noqa: E402


VIDEO_IDS = ("2907", "2909", "2911")
Q_VALUES = (200, 400, 800, 1600, 3200)

REGRESSION_DIR = PROJECT_ROOT / "tracks" / "E_evaluation_ci" / "regression_videos"
REPORTS_DIR = PROJECT_ROOT / "tracks" / "D_tracking_outlier" / "reports"


def find_video(video_id):
    matches = sorted(p for p in REGRESSION_DIR.glob("*.mp4") if video_id in p.stem)
    if not matches:
        raise FileNotFoundError(f"Missing regression video for id={video_id} in {REGRESSION_DIR}")
    return matches[0]


def paired_points(raw, smooth):
    raw_map = {int(p["frame"]): p for p in raw}
    smooth_map = {int(p["frame"]): p for p in smooth}
    frames = sorted(set(raw_map.keys()) & set(smooth_map.keys()))
    return [(raw_map[f], smooth_map[f]) for f in frames]


def rms_diff_px(raw, smooth):
    pairs = paired_points(raw, smooth)
    if not pairs:
        return 0.0
    sq = []
    for r, s in pairs:
        dx = float(s["x"]) - float(r["x"])
        dy = float(s["y"]) - float(r["y"])
        sq.append(dx * dx + dy * dy)
    return math.sqrt(sum(sq) / len(sq))


def jump_metrics_px(trajectory):
    if len(trajectory) < 2:
        return {"max_jump_px": 0.0, "p95_jump_px": 0.0, "mean_jump_px": 0.0}

    traj = sorted(trajectory, key=lambda p: int(p["frame"]))
    jumps = []
    for i in range(1, len(traj)):
        dx = float(traj[i]["x"]) - float(traj[i - 1]["x"])
        dy = float(traj[i]["y"]) - float(traj[i - 1]["y"])
        jumps.append(math.sqrt(dx * dx + dy * dy))

    jumps_sorted = sorted(jumps)
    p95_idx = min(len(jumps_sorted) - 1, int(0.95 * (len(jumps_sorted) - 1)))
    return {
        "max_jump_px": float(max(jumps)),
        "p95_jump_px": float(jumps_sorted[p95_idx]),
        "mean_jump_px": float(sum(jumps) / len(jumps)),
    }


def run_pipeline(video_path, output_dir, kalman_q=None, precomp=False):
    pipeline = SkiRacingPipeline(
        gate_model_path=str(PROJECT_ROOT / "models" / "gate_detector_best.pt"),
        discipline="slalom",
        gate_spacing_m=12.0,
        stabilize=True,
        camera_mode="affine",
        camera_pitch_deg=6.0,
    )
    results = pipeline.process_video(
        video_path=str(video_path),
        output_dir=str(output_dir),
        validate_physics=True,
        projection="scale",
        gate_conf=0.35,
        gate_iou=0.55,
        skier_conf=0.25,
        gate_search_frames=150,
        gate_search_stride=5,
        gate_track_frames=120,
        gate_track_stride=1,
        gate_track_min_obs=3,
        frame_stride=1,
        kalman_process_noise=kalman_q,
        camera_compensate_before_smoothing=precomp,
    )
    return results


def run_q_sweep(stamp):
    artifact_root = REPORTS_DIR / f"kalman_sweep_artifacts_{stamp}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_id in VIDEO_IDS:
        video_path = find_video(video_id)
        for q in Q_VALUES:
            print(f"[Q-SWEEP] video={video_id} q={q}")
            out_dir = artifact_root / f"{video_id}_q{q}"
            out_dir.mkdir(parents=True, exist_ok=True)
            res = run_pipeline(video_path, out_dir, kalman_q=q, precomp=False)

            raw = res.get("trajectory_2d_raw") or res.get("trajectory_2d") or []
            smooth = res.get("trajectory_2d") or []
            rms_px = rms_diff_px(raw, smooth)
            jump = jump_metrics_px(smooth)
            physics = (res.get("physics_validation") or {}).get("metrics") or {}
            speeds = physics.get("speeds_kmh") or {}
            g_forces = physics.get("g_forces") or {}
            total_frames = int((res.get("video_info") or {}).get("total_frames") or 0)
            outlier_count = int(res.get("outlier_count") or 0)

            rows.append({
                "video_id": video_id,
                "video_file": video_path.name,
                "q": float(q),
                "rms_diff_px": float(rms_px),
                "max_speed_kmh": float(speeds.get("max") or 0.0),
                "max_g_force": float(g_forces.get("max") or 0.0),
                "max_jump_px": float(jump["max_jump_px"]),
                "p95_jump_px": float(jump["p95_jump_px"]),
                "bytetrack_coverage": float(res.get("bytetrack_coverage") or 0.0),
                "track_id_switches": int(res.get("track_id_switches") or 0),
                "outlier_count": outlier_count,
                "outlier_ratio": float(outlier_count / total_frames) if total_frames else 0.0,
                "analysis_json": str(out_dir / f"{video_path.stem}_analysis.json"),
            })

    json_path = REPORTS_DIR / f"kalman_tuning_{stamp}.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    grouped = {q: [] for q in Q_VALUES}
    by_video = {video_id: [] for video_id in VIDEO_IDS}
    for row in rows:
        grouped[int(row["q"])].append(row)
        by_video[row["video_id"]].append(row)

    def agg_metric(rows_for_q, key):
        vals = [float(r[key]) for r in rows_for_q]
        return float(sum(vals) / len(vals)) if vals else 0.0

    aggregate_rows = []
    for q in Q_VALUES:
        q_rows = grouped[q]
        aggregate_rows.append({
            "q": q,
            "mean_rms_diff_px": agg_metric(q_rows, "rms_diff_px"),
            "mean_max_speed_kmh": agg_metric(q_rows, "max_speed_kmh"),
            "mean_max_g_force": agg_metric(q_rows, "max_g_force"),
            "mean_max_jump_px": agg_metric(q_rows, "max_jump_px"),
        })

    # Prefer minimizing physics extremes first, then smoothing displacement.
    best_q_row = min(
        aggregate_rows,
        key=lambda r: (r["mean_max_g_force"], r["mean_max_speed_kmh"], r["mean_rms_diff_px"]),
    )

    md_lines = []
    md_lines.append(f"# Kalman Tuning ({stamp})")
    md_lines.append("")
    md_lines.append("- Videos: `2907`, `2909`, `2911`")
    md_lines.append(f"- Q sweep: `{list(Q_VALUES)}`")
    md_lines.append(f"- Recommended Q (aggregate): **{int(best_q_row['q'])}**")
    md_lines.append("")
    md_lines.append("## Aggregate Across 3 Videos")
    md_lines.append("")
    md_lines.append("| Q | Mean RMS diff (px) | Mean max speed (km/h) | Mean max G | Mean max jump (px) |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for row in aggregate_rows:
        md_lines.append(
            f"| {int(row['q'])} | {row['mean_rms_diff_px']:.2f} | "
            f"{row['mean_max_speed_kmh']:.2f} | {row['mean_max_g_force']:.2f} | "
            f"{row['mean_max_jump_px']:.2f} |"
        )
    md_lines.append("")

    for video_id in VIDEO_IDS:
        md_lines.append(f"## Video {video_id}")
        md_lines.append("")
        md_lines.append(
            "| Q | RMS diff (px) | Max speed (km/h) | Max G | Max jump (px) | "
            "ByteTrack coverage | Track-ID switches | Outlier ratio |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        video_rows = sorted(by_video[video_id], key=lambda r: r["q"])
        for row in video_rows:
            md_lines.append(
                f"| {int(row['q'])} | {row['rms_diff_px']:.2f} | {row['max_speed_kmh']:.2f} | "
                f"{row['max_g_force']:.2f} | {row['max_jump_px']:.2f} | "
                f"{row['bytetrack_coverage']:.3f} | {row['track_id_switches']} | "
                f"{row['outlier_ratio']:.3%} |"
            )
        md_lines.append("")

    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append(
        "- Physics extremes remain dominated by geometry/scale instability; Track D changes improved 2D continuity "
        "and coverage but cannot fully fix 3D explosions alone."
    )
    md_lines.append("- Raw per-run metrics are saved in the JSON companion report.")
    md_lines.append("")
    md_lines.append(f"JSON artifact: `{json_path}`")

    md_path = REPORTS_DIR / f"kalman_tuning_{stamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return md_path, json_path, best_q_row


def run_ordering_check(stamp, q_recommended):
    video_id = "2911"
    video_path = find_video(video_id)
    artifact_root = REPORTS_DIR / f"ordering_study_artifacts_{stamp}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    print(f"[ORDERING] video={video_id} baseline")
    base_res = run_pipeline(video_path, artifact_root / "baseline", kalman_q=q_recommended, precomp=False)
    print(f"[ORDERING] video={video_id} precomp-before-smoothing")
    alt_res = run_pipeline(video_path, artifact_root / "precomp", kalman_q=q_recommended, precomp=True)

    def summarize(res):
        traj = res.get("trajectory_2d") or []
        jump = jump_metrics_px(traj)
        physics = (res.get("physics_validation") or {}).get("metrics") or {}
        speeds = physics.get("speeds_kmh") or {}
        g_forces = physics.get("g_forces") or {}
        return {
            "max_jump_px": float(jump["max_jump_px"]),
            "p95_jump_px": float(jump["p95_jump_px"]),
            "mean_jump_px": float(jump["mean_jump_px"]),
            "bytetrack_coverage": float(res.get("bytetrack_coverage") or 0.0),
            "track_id_switches": int(res.get("track_id_switches") or 0),
            "outlier_count": int(res.get("outlier_count") or 0),
            "max_speed_kmh": float(speeds.get("max") or 0.0),
            "max_g_force": float(g_forces.get("max") or 0.0),
        }

    base = summarize(base_res)
    alt = summarize(alt_res)

    md_lines = []
    md_lines.append(f"# Pipeline Ordering Study ({stamp})")
    md_lines.append("")
    md_lines.append(f"- Video: `{video_path.name}`")
    md_lines.append(f"- Kalman Q: `{q_recommended}`")
    md_lines.append("- Compared:")
    md_lines.append("  - Baseline: `track -> smooth -> transform(camera compensation)`")
    md_lines.append("  - Alternative: `track -> camera compensate 2D -> smooth -> transform`")
    md_lines.append("")
    md_lines.append("| Metric | Baseline | Alternative | Delta (alt-baseline) |")
    md_lines.append("|---|---:|---:|---:|")
    for key, label, fmt in [
        ("max_jump_px", "Max jump (px)", "{:.2f}"),
        ("p95_jump_px", "P95 jump (px)", "{:.2f}"),
        ("mean_jump_px", "Mean jump (px)", "{:.2f}"),
        ("bytetrack_coverage", "ByteTrack coverage", "{:.3f}"),
        ("track_id_switches", "Track-ID switches", "{:d}"),
        ("outlier_count", "Outlier count", "{:d}"),
        ("max_speed_kmh", "Max speed (km/h)", "{:.2f}"),
        ("max_g_force", "Max G", "{:.2f}"),
    ]:
        b = base[key]
        a = alt[key]
        d = a - b
        if isinstance(b, int):
            md_lines.append(f"| {label} | {b:d} | {a:d} | {d:+d} |")
        else:
            md_lines.append(f"| {label} | {fmt.format(b)} | {fmt.format(a)} | {d:+.2f} |")
    md_lines.append("")
    md_lines.append("## Recommendation")
    md_lines.append("")
    md_lines.append(
        "- Keep the baseline ordering as default for now. The alternative should remain an experiment flag until "
        "Track C geometry stabilization is finalized and cross-track validation is repeated."
    )

    md_path = REPORTS_DIR / f"pipeline_ordering_{stamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return md_path


def main():
    stamp = datetime.now().strftime("%Y%m%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    q_md, q_json, best_q = run_q_sweep(stamp)
    ordering_md = run_ordering_check(stamp, int(best_q["q"]))

    print(f"[DONE] {q_md}")
    print(f"[DONE] {q_json}")
    print(f"[DONE] {ordering_md}")


if __name__ == "__main__":
    main()
