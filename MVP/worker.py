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


def _run_analysis(local_video: Path, job_config: dict) -> tuple[Path, dict]:
    """Invoke MVP/run.py and return (run_dir, mvp_summary_dict)."""
    pose_engine = job_config.get("pose_engine", "vision")
    max_fps = job_config.get("max_fps", 12)

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        str(local_video),
        "--pose-engine", str(pose_engine),
        "--max-fps", str(max_fps),
        "--no-overlay",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

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
    return run_dir, summary


def _upload_artifacts(job_id: str, run_dir: Path, summary: dict) -> list[dict]:
    """Upload peak frames + summary JSON to Supabase Storage and return DB rows."""
    rows: list[dict] = []

    # summary JSON
    summary_local = run_dir / "mvp_summary.json"
    summary_remote = f"jobs/{job_id}/summary.json"
    supabase.storage.from_("artifacts").upload(
        path=summary_remote,
        file=summary_local.read_bytes(),
        file_options={"content-type": "application/json", "upsert": "true"},
    )
    rows.append(
        {
            "job_id": job_id,
            "kind": "summary_json",
            "object_path": summary_remote,
            "meta": {},
        }
    )

    # peak frames
    for artifact in summary.get("artifacts", []):
        kind = artifact.get("kind", "")
        if kind not in ("peak_pressure_frame", "peak_pressure_frame_enhanced"):
            continue

        local_path = Path(artifact["path"])
        if not local_path.exists():
            print(f"  [warn] Frame not found locally, skipping: {local_path}")
            continue

        remote_path = f"jobs/{job_id}/frames/{local_path.name}"
        supabase.storage.from_("artifacts").upload(
            path=remote_path,
            file=local_path.read_bytes(),
            file_options={"content-type": "image/jpeg", "upsert": "true"},
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
    config: dict = job.get("config") or {}

    print(f"[{job_id[:8]}] Starting job (video: {video_path_in_storage})")

    if not video_path_in_storage:
        _set_status(job_id, "error", error="video_object_path is empty")
        return

    with tempfile.TemporaryDirectory(prefix="skicoach_") as tmpdir:
        try:
            # 1. Download video
            print(f"[{job_id[:8]}] Downloading video from Storage...")
            video_bytes: bytes = supabase.storage.from_("videos").download(
                video_path_in_storage
            )
            suffix = Path(video_path_in_storage).suffix or ".mp4"
            local_video = Path(tmpdir) / f"video{suffix}"
            local_video.write_bytes(video_bytes)
            print(
                f"[{job_id[:8]}] Downloaded {len(video_bytes) / 1_048_576:.1f} MB"
            )

            # 2. Run technique analysis
            print(f"[{job_id[:8]}] Running technique analysis...")
            run_dir, summary = _run_analysis(local_video, config)
            n_turns = summary.get("turns", 0)
            print(f"[{job_id[:8]}] Analysis done — {n_turns} turn(s) detected")

            # 3. Upload results
            print(f"[{job_id[:8]}] Uploading artifacts...")
            rows = _upload_artifacts(job_id, run_dir, summary)
            if rows:
                supabase.table("artifacts").insert(rows).execute()
            print(f"[{job_id[:8]}] Uploaded {len(rows)} artifact(s)")

            # 4. Mark done
            _set_status(
                job_id,
                "done",
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
