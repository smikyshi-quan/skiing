#!/usr/bin/env python3
"""Local MacBook worker — polls Supabase for queued jobs and runs technique analysis.

Setup:
    pip install supabase python-dotenv

    Create MVP/.env.worker (copy .env.worker.example and fill in values).

Run:
    python MVP/worker.py            # continuous loop
    python MVP/worker.py --once     # process one job then exit
    python MVP/worker.py --interval 5   # poll every 5 s
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = Path(__file__).resolve().parent  # MVP/
RUN_SCRIPT = WORKER_DIR / "run.py"

# Load secrets from MVP/.env.worker (not committed to repo)
load_dotenv(WORKER_DIR / ".env.worker")

import os  # noqa: E402  (after dotenv)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

from supabase import create_client  # noqa: E402

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_running = True


def _handle_sigterm(signum, frame):  # noqa: ANN001
    global _running
    print("\n[worker] Shutting down...")
    _running = False


signal.signal(signal.SIGINT, _handle_sigterm)
signal.signal(signal.SIGTERM, _handle_sigterm)

# ── Job processing ────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_status(job_id: str, status: str, **extra) -> None:
    supabase.table("jobs").update(
        {"status": status, "updated_at": _now_iso(), **extra}
    ).eq("id", job_id).execute()


def _claim_job() -> dict | None:
    """Claim one queued job by atomically flipping its status to 'running'."""
    result = (
        supabase.table("jobs")
        .select("*")
        .eq("status", "queued")
        .order("created_at")
        .limit(1)
        .execute()
    )
    if not result.data:
        return None

    job = result.data[0]

    # Guard against a second worker claiming the same row
    update = (
        supabase.table("jobs")
        .update({"status": "running", "updated_at": _now_iso()})
        .eq("id", job["id"])
        .eq("status", "queued")  # only succeeds if still queued
        .execute()
    )
    return update.data[0] if update.data else None


def _run_analysis(local_video: Path, job_config: dict) -> tuple[Path, dict, dict | None]:
    """Invoke MVP/run.py and return (run_dir, mvp_summary_dict, analysis_summary_dict_or_None)."""
    pose_engine = job_config.get("pose_engine", "mediapipe")
    max_fps = job_config.get("max_fps", 12)
    max_dimension = job_config.get("max_dimension", None)
    render_overlay = bool(job_config.get("render_overlay", True))
    render_max_dimension = job_config.get("render_max_dimension", None)

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        str(local_video),
        "--pose-engine", str(pose_engine),
        "--max-fps", str(max_fps),
    ]
    if max_dimension is not None:
        cmd.extend(["--max-dimension", str(max_dimension)])
    if render_max_dimension is not None:
        cmd.extend(["--render-max-dimension", str(render_max_dimension)])
    if not render_overlay:
        cmd.append("--no-overlay")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    # Always print analysis output so warnings (e.g. overlay rendering failures) are visible
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"  [run.py] {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-30:]:
            print(f"  [run.py stderr] {line}", file=sys.stderr)

    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"run.py exited {result.returncode}:\n{stderr_tail}")

    # Parse run directory from stdout line: "Run directory : /path/to/dir"
    run_dir: Path | None = None
    for line in result.stdout.splitlines():
        if "Run directory :" in line:
            run_dir = Path(line.split("Run directory :")[1].strip())
            break

    if run_dir is None or not run_dir.exists():
        raise RuntimeError(
            f"Could not locate run directory in output.\nstdout:\n{result.stdout[-1000:]}"
        )

    summary_path = run_dir / "mvp_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"mvp_summary.json not found in {run_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    analysis_summary_path = run_dir / "summary" / "summary.json"
    analysis_summary = None
    if analysis_summary_path.exists():
        analysis_summary = json.loads(analysis_summary_path.read_text(encoding="utf-8"))
    else:
        print(f"  [warn] analysis summary not found at {analysis_summary_path}", file=sys.stderr)

    return run_dir, summary, analysis_summary


def _upload_bytes(
    *,
    bucket: str,
    remote_path: str,
    content: bytes,
    content_type: str,
) -> None:
    try:
        res = supabase.storage.from_(bucket).upload(
            path=remote_path,
            file=content,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        # storage3 may return error info without raising in some versions
        if hasattr(res, "error") and res.error:
            raise RuntimeError(f"Storage error: {res.error}")
    except Exception as exc:
        raise RuntimeError(
            f"Upload to {bucket}/{remote_path} failed: {exc}"
        ) from exc


def _upload_file(
    *,
    bucket: str,
    remote_path: str,
    local_path: Path,
    content_type: str | None = None,
) -> None:
    guessed, _ = mimetypes.guess_type(str(local_path))
    size_mb = local_path.stat().st_size / 1_048_576
    print(f"  → {local_path.name} ({size_mb:.1f} MB) → {bucket}/{remote_path}")
    _upload_bytes(
        bucket=bucket,
        remote_path=remote_path,
        content=local_path.read_bytes(),
        content_type=content_type or guessed or "application/octet-stream",
    )
    print(f"  ✓ {local_path.name} uploaded")


def _extract_cool_moment_photos(
    *,
    video_path: Path,
    turns: list[dict],
    output_dir: Path,
    limit: int = 24,
) -> list[tuple[Path, dict]]:
    """Extract one 'cool-moment' frame per turn (midpoint) and return [(path, meta)]."""
    try:
        import cv2  # type: ignore
    except Exception:
        print("  [warn] cv2 not available, skipping cool-moment extraction", file=sys.stderr)
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] could not open video for cool-moment extraction: {video_path}", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, dict]] = []

    try:
        for turn in (turns or [])[:limit]:
            turn_idx = turn.get("turn_idx")
            side = turn.get("side")
            try:
                start_s = float(turn.get("start_s", 0.0))
                end_s = float(turn.get("end_s", start_s))
            except Exception:
                continue

            ts = (start_s + end_s) / 2.0 if end_s > start_s else start_s
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            safe_side = str(side or "turn")
            try:
                safe_side = "".join(ch for ch in safe_side if ch.isalnum() or ch in ("-", "_"))[:12]
            except Exception:
                safe_side = "turn"

            idx_str = f"{int(turn_idx):02d}" if isinstance(turn_idx, int) else "xx"
            filename = f"cool_{idx_str}_{safe_side}_{ts:.1f}s.jpg"
            out_path = output_dir / filename

            ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                continue

            results.append(
                (
                    out_path,
                    {
                        "turn_idx": turn_idx,
                        "side": side,
                        "timestamp_s": ts,
                    },
                )
            )
    finally:
        cap.release()

    return results


def _upload_artifacts(
    *,
    job_id: str,
    run_dir: Path,
    summary: dict,
    analysis_summary: dict | None,
    local_video: Path,
) -> list[dict]:
    """Upload overlay video + cool-moment photos + summary to Supabase Storage and return DB rows."""
    rows: list[dict] = []

    # summary JSON
    summary_local = run_dir / "mvp_summary.json"
    summary_remote = f"jobs/{job_id}/summary.json"
    _upload_file(
        bucket="artifacts",
        remote_path=summary_remote,
        local_path=summary_local,
        content_type="application/json",
    )
    rows.append(
        {
            "job_id": job_id,
            "kind": "summary_json",
            "object_path": summary_remote,
            "meta": {},
        }
    )

    # overlay video (if present)
    overlay_local: Path | None = None
    for artifact in summary.get("artifacts", []):
        if artifact.get("kind") == "video_overlay" and artifact.get("path"):
            overlay_local = Path(str(artifact["path"]))
            break
    if overlay_local is None:
        candidate = run_dir / "videos" / "overlay.mp4"
        if candidate.exists():
            overlay_local = candidate

    if overlay_local and overlay_local.exists():
        overlay_remote = f"jobs/{job_id}/overlay{overlay_local.suffix or '.mp4'}"
        _upload_file(
            bucket="artifacts",
            remote_path=overlay_remote,
            local_path=overlay_local,
            content_type="video/mp4",
        )
        rows.append(
            {
                "job_id": job_id,
                "kind": "video_overlay",
                "object_path": overlay_remote,
                "meta": {},
            }
        )
    else:
        expected = overlay_local or (run_dir / "videos" / "overlay.mp4")
        print(f"  [warn] overlay video not found at {expected} — skipping", file=sys.stderr)

    # metrics CSV (if present)
    if analysis_summary:
        for artifact in analysis_summary.get("artifacts", []) or []:
            if artifact.get("kind") != "metrics_csv" or not artifact.get("path"):
                continue
            metrics_local = Path(str(artifact["path"]))
            if not metrics_local.exists():
                continue
            metrics_remote = f"jobs/{job_id}/metrics.csv"
            _upload_file(
                bucket="artifacts",
                remote_path=metrics_remote,
                local_path=metrics_local,
                content_type="text/csv",
            )
            rows.append(
                {
                    "job_id": job_id,
                    "kind": "metrics_csv",
                    "object_path": metrics_remote,
                    "meta": {},
                }
            )
            break

    # cool-moment photos (midpoint per turn)
    turns = (analysis_summary or {}).get("turns") if analysis_summary else None
    if isinstance(turns, list) and turns:
        print(f"  extracting cool-moment frames for {len(turns)} turn(s)...")
        source_video = overlay_local if overlay_local and overlay_local.exists() else local_video
        extracted = _extract_cool_moment_photos(
            video_path=source_video,
            turns=turns,
            output_dir=run_dir / "cool_moments",
        )
        print(f"  extracted {len(extracted)} cool-moment photo(s)")
        for local_path, meta in extracted:
            remote_path = f"jobs/{job_id}/cool_moments/{local_path.name}"
            _upload_file(
                bucket="artifacts",
                remote_path=remote_path,
                local_path=local_path,
                content_type="image/jpeg",
            )
            rows.append(
                {
                    "job_id": job_id,
                    "kind": "cool_moment_photo",
                    "object_path": remote_path,
                    "meta": meta or {},
                }
            )
    elif analysis_summary is not None:
        print("  [warn] no turns in analysis summary — no cool-moment photos", file=sys.stderr)

    # any legacy peak-pressure frames if present
    for artifact in summary.get("artifacts", []):
        kind = artifact.get("kind", "")
        if kind not in ("peak_pressure_frame", "peak_pressure_frame_enhanced"):
            continue

        local_path = Path(str(artifact.get("path") or ""))
        if not local_path.exists():
            print(f"  [warn] Frame not found locally, skipping: {local_path}")
            continue

        remote_path = f"jobs/{job_id}/frames/{local_path.name}"
        _upload_file(
            bucket="artifacts",
            remote_path=remote_path,
            local_path=local_path,
            content_type="image/jpeg",
        )
        rows.append(
            {
                "job_id": job_id,
                "kind": kind,
                "object_path": remote_path,
                "meta": {
                    "turn_idx": artifact.get("turn_idx"),
                    "side": artifact.get("side"),
                    "timestamp_s": artifact.get("timestamp_s"),
                },
            }
        )

    return rows


def process_job(job: dict) -> None:
    job_id: str = job["id"]
    video_path_in_storage: str | None = job.get("video_object_path")
    config: dict = dict(job.get("config") or {})

    def _set_progress(note: str) -> None:
        """Write a progress note into jobs.config so the UI can show it."""
        config["progress_note"] = note
        supabase.table("jobs").update(
            {"config": config, "updated_at": _now_iso()}
        ).eq("id", job_id).execute()

    print(f"[{job_id[:8]}] Starting job (video: {video_path_in_storage})")

    if not video_path_in_storage:
        _set_status(job_id, "error", error="video_object_path is empty")
        return

    with tempfile.TemporaryDirectory(prefix="skicoach_") as tmpdir:
        try:
            # 1. Download video
            _set_progress("Downloading video...")
            print(f"[{job_id[:8]}] Downloading video from Storage...")
            video_bytes: bytes = supabase.storage.from_("videos").download(
                video_path_in_storage
            )
            suffix = Path(video_path_in_storage).suffix or ".mp4"
            local_video = Path(tmpdir) / f"video{suffix}"
            local_video.write_bytes(video_bytes)
            size_mb = len(video_bytes) / 1_048_576
            print(f"[{job_id[:8]}] Downloaded {size_mb:.1f} MB")

            # 2. Run technique analysis
            _set_progress("Running pose analysis...")
            print(f"[{job_id[:8]}] Running technique analysis...")
            run_dir, summary, analysis_summary = _run_analysis(local_video, config)
            n_turns = summary.get("turns", 0)
            print(f"[{job_id[:8]}] Analysis done — {n_turns} turn(s) detected")

            # Warn if quality flags were set
            warnings = summary.get("quality", {}).get("warnings", [])
            for w in warnings:
                print(f"  [quality warn] {w}", file=sys.stderr)

            # 3. Upload results
            _set_progress(f"Uploading results ({n_turns} turn(s) found)...")
            print(f"[{job_id[:8]}] Uploading artifacts...")
            rows = _upload_artifacts(
                job_id=job_id,
                run_dir=run_dir,
                summary=summary,
                analysis_summary=analysis_summary,
                local_video=local_video,
            )
            if rows:
                supabase.table("artifacts").insert(rows).execute()
            print(f"[{job_id[:8]}] Uploaded {len(rows)} artifact(s)")

            # 4. Mark done
            config.pop("progress_note", None)
            _set_status(
                job_id,
                "done",
                config=config,
                result_prefix=f"jobs/{job_id}/",
            )
            print(f"[{job_id[:8]}] Done.")

        except Exception as exc:  # noqa: BLE001
            msg = str(exc)[:1000]
            print(f"[{job_id[:8]}] ERROR: {msg}", file=sys.stderr)
            _set_status(job_id, "error", error=msg)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="SkiCoach local worker.")
    p.add_argument(
        "--once", action="store_true", help="Process one job then exit."
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        metavar="SECS",
        help="Seconds between polls when idle (default 10).",
    )
    args = p.parse_args()

    print(f"[worker] Starting — polling {SUPABASE_URL}")
    print(f"[worker] Using {RUN_SCRIPT}")
    print("[worker] Press Ctrl-C to stop.\n")

    while _running:
        job = _claim_job()
        if job:
            process_job(job)
            if args.once:
                break
        else:
            if args.once:
                print("[worker] No queued jobs found.")
                break
            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
