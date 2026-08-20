#!/usr/bin/env python3
"""Local MacBook worker — polls Supabase for queued jobs and runs technique analysis.

Setup:
    pip install supabase python-dotenv boto3

    Create MVP/.env.worker (copy .env.worker.example and fill in values).

Run:
    python MVP/worker.py            # continuous loop
    python MVP/worker.py --once     # process one job then exit
    python MVP/worker.py --interval 5   # poll every 5 s
    python MVP/worker.py --recover  # recover stale running jobs then exit
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
from typing import Callable

from dotenv import load_dotenv

# Path setup

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = Path(__file__).resolve().parent  # MVP/
RUN_SCRIPT = WORKER_DIR / "run.py"

# Load secrets from MVP/.env.worker (not committed to repo)
load_dotenv(WORKER_DIR / ".env.worker")

import os  # noqa: E402

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
POLL_INTERVAL_S = float(os.environ.get("WORKER_POLL_INTERVAL", "10"))
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_VIDEOS_BUCKET = os.environ.get("R2_VIDEOS_BUCKET")
R2_ARTIFACTS_BUCKET = os.environ.get("R2_ARTIFACTS_BUCKET") or R2_VIDEOS_BUCKET
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL") or "http://localhost:1234"
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL")

# How long (seconds) a job can stay in 'running' without a heartbeat before it
# is considered stale and re-queued.
STALE_THRESHOLD_S = int(os.environ.get("WORKER_STALE_THRESHOLD", "600"))

from supabase import create_client  # noqa: E402
import boto3  # noqa: E402

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
_r2_client = None

# Graceful shutdown

_running = True


def _handle_sigterm(signum, frame):  # noqa: ANN001
    global _running
    print("\n[worker] Shutting down...")
    _running = False


signal.signal(signal.SIGINT, _handle_sigterm)
signal.signal(signal.SIGTERM, _handle_sigterm)

# Helpers


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_status(job_id: str, status: str, **extra) -> None:
    supabase.table("jobs").update(
        {"status": status, "updated_at": _now_iso(), **extra}
    ).eq("id", job_id).execute()


def _set_progress(
    job_id: str,
    config: dict,
    note: str,
    *,
    step: int | None = None,
    total: int | None = None,
    stage: str | None = None,
) -> None:
    """Write a progress note, structured step info, and heartbeat into jobs.config."""
    config["progress_note"] = note
    config["heartbeat_at"] = _now_iso()
    if step is not None:
        config["progress_step"] = step
    if total is not None:
        config["progress_total"] = total
    if stage is not None:
        config["progress_stage"] = stage
    supabase.table("jobs").update(
        {"config": config, "updated_at": _now_iso()}
    ).eq("id", job_id).execute()


def _write_heartbeat(job_id: str, config: dict) -> None:
    """Update heartbeat_at without changing the progress note."""
    config["heartbeat_at"] = _now_iso()
    supabase.table("jobs").update(
        {"config": config, "updated_at": _now_iso()}
    ).eq("id", job_id).execute()


def _video_storage_provider(config: dict) -> str:
    return "r2" if config.get("video_storage_provider") == "r2" else "supabase"


def _expected_video_size_bytes(config: dict) -> int | None:
    value = config.get("video_file_size_bytes")
    if isinstance(value, bool):  # bool is a subclass of int
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    return None


def _get_r2_client():
    global _r2_client

    if _r2_client is not None:
        return _r2_client

    missing = [
        name
        for name, value in (
            ("R2_ACCOUNT_ID", R2_ACCOUNT_ID),
            ("R2_ACCESS_KEY_ID", R2_ACCESS_KEY_ID),
            ("R2_SECRET_ACCESS_KEY", R2_SECRET_ACCESS_KEY),
            ("R2_VIDEOS_BUCKET", R2_VIDEOS_BUCKET),
        )
        if not value
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(f"Missing required R2 environment variable(s): {missing_names}")

    from botocore.config import Config as BotoConfig

    _r2_client = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=BotoConfig(
            read_timeout=300,
            connect_timeout=30,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )
    return _r2_client


def _download_video_bytes(
    remote_path: str,
    provider: str,
    *,
    expected_size_bytes: int | None = None,
    max_retries: int = 3,
) -> bytes:
    if provider == "r2":
        for attempt in range(1, max_retries + 1):
            try:
                response = _get_r2_client().get_object(Bucket=R2_VIDEOS_BUCKET, Key=remote_path)
                remote_size = response.get("ContentLength")
                if expected_size_bytes is not None and remote_size not in (None, expected_size_bytes):
                    body = response["Body"]
                    body.close()
                    raise RuntimeError(
                        f"Stored video size mismatch before download: expected {expected_size_bytes} bytes, got {remote_size} bytes"
                    )

                # Stream in 1 MB chunks instead of a single .read()
                chunks = bytearray()
                body = response["Body"]
                while True:
                    chunk = body.read(1_048_576)  # 1 MB
                    if not chunk:
                        break
                    chunks.extend(chunk)
                body.close()
                video_bytes = bytes(chunks)
                actual_size = len(video_bytes)

                if actual_size == 0:
                    raise RuntimeError("Downloaded video is empty")
                if isinstance(remote_size, int) and actual_size != remote_size:
                    raise RuntimeError(
                        f"Downloaded truncated video from R2: expected {remote_size} bytes, got {actual_size} bytes"
                    )
                if expected_size_bytes is not None and actual_size != expected_size_bytes:
                    raise RuntimeError(
                        f"Downloaded video size mismatch: expected {expected_size_bytes} bytes, got {actual_size} bytes"
                    )

                return video_bytes
            except Exception as exc:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  [retry] download attempt {attempt} failed: {exc} — retrying in {wait}s", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise

    video_bytes = supabase.storage.from_("videos").download(remote_path)
    actual_size = len(video_bytes)
    if actual_size == 0:
        raise RuntimeError("Downloaded video is empty")
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise RuntimeError(
            f"Downloaded video size mismatch: expected {expected_size_bytes} bytes, got {actual_size} bytes"
        )
    return video_bytes


def _upload_file_to_r2(
    *,
    bucket: str,
    remote_path: str,
    local_path: Path,
    content_type: str | None = None,
) -> None:
    guessed, _ = mimetypes.guess_type(str(local_path))
    size_mb = local_path.stat().st_size / 1_048_576
    print(f"  -> {local_path.name} ({size_mb:.1f} MB) -> r2://{bucket}/{remote_path}")

    extra_args = {"ContentType": content_type or guessed or "application/octet-stream"}
    try:
        _get_r2_client().upload_file(str(local_path), bucket, remote_path, ExtraArgs=extra_args)
    except Exception as exc:
        raise RuntimeError(f"Upload to r2://{bucket}/{remote_path} failed: {exc}") from exc

    print(f"  OK {local_path.name} uploaded to R2")


# Stale job recovery


def recover_stale_jobs() -> int:
    """Requeue running jobs whose heartbeat (or updated_at) is older than STALE_THRESHOLD_S."""
    result = (
        supabase.table("jobs")
        .select("id, config, updated_at")
        .eq("status", "running")
        .execute()
    )
    if not result.data:
        return 0

    now = datetime.now(timezone.utc)
    recovered = 0

    for job in result.data:
        config = job.get("config") or {}
        heartbeat_str = config.get("heartbeat_at")

        ref_time = None
        if heartbeat_str:
            try:
                ref_time = datetime.fromisoformat(heartbeat_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if ref_time is None:
            try:
                ref_time = datetime.fromisoformat(job["updated_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

        age_s = (now - ref_time).total_seconds()
        if age_s > STALE_THRESHOLD_S:
            config.pop("heartbeat_at", None)
            config["progress_note"] = "Recovered from stale running state — requeued"
            supabase.table("jobs").update(
                {
                    "status": "queued",
                    "config": config,
                    "error": None,
                    "updated_at": _now_iso(),
                }
            ).eq("id", job["id"]).execute()
            print(f"  [recovery] Requeued stale job {job['id']} (idle {age_s:.0f}s)")
            recovered += 1

    return recovered


# Job claiming


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

    update = (
        supabase.table("jobs")
        .update({"status": "running", "updated_at": _now_iso()})
        .eq("id", job["id"])
        .eq("status", "queued")
        .execute()
    )
    return update.data[0] if update.data else None


# Analysis


def _run_analysis(
    local_video: Path,
    job_config: dict,
    heartbeat: Callable[[], None] | None = None,
) -> tuple[Path, dict, dict | None]:
    """Invoke MVP/run.py and return (run_dir, mvp_summary_dict, full_analysis_summary_or_None)."""
    max_fps = job_config.get("max_fps", None)
    max_dimension = job_config.get("max_dimension", None)
    render_overlay = bool(job_config.get("render_overlay", True))
    render_max_dimension = job_config.get("render_max_dimension", None)

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        str(local_video),
    ]
    if max_fps is not None:
        cmd.extend(["--max-fps", str(max_fps)])
    if max_dimension is not None:
        cmd.extend(["--max-dimension", str(max_dimension)])
    if render_max_dimension is not None:
        cmd.extend(["--render-max-dimension", str(render_max_dimension)])
    if not render_overlay:
        cmd.append("--no-overlay")

    heartbeat_deadline = time.monotonic() + 60.0

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )

        while process.poll() is None:
            if heartbeat and time.monotonic() >= heartbeat_deadline:
                heartbeat()
                heartbeat_deadline = time.monotonic() + 60.0
            time.sleep(1.0)

        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()

    class _Completed:
        def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    result = _Completed(process.returncode, stdout, stderr)

    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"  [run.py] {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-30:]:
            print(f"  [run.py stderr] {line}", file=sys.stderr)

    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"run.py exited {result.returncode}:\n{stderr_tail}")

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

    full_summary_path = run_dir / "summary" / "summary.json"
    full_summary = None
    if full_summary_path.exists():
        full_summary = json.loads(full_summary_path.read_text(encoding="utf-8"))
    else:
        print(f"  [warn] full pipeline summary not found at {full_summary_path}", file=sys.stderr)

    return run_dir, summary, full_summary


# Uploads


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
        if hasattr(res, "error") and res.error:
            raise RuntimeError(f"Storage error: {res.error}")
    except Exception as exc:
        raise RuntimeError(f"Upload to {bucket}/{remote_path} failed: {exc}") from exc


def _upload_file(
    *,
    bucket: str,
    remote_path: str,
    local_path: Path,
    content_type: str | None = None,
) -> None:
    guessed, _ = mimetypes.guess_type(str(local_path))
    size_mb = local_path.stat().st_size / 1_048_576
    print(f"  -> {local_path.name} ({size_mb:.1f} MB) -> {bucket}/{remote_path}")
    _upload_bytes(
        bucket=bucket,
        remote_path=remote_path,
        content=local_path.read_bytes(),
        content_type=content_type or guessed or "application/octet-stream",
    )
    print(f"  OK {local_path.name} uploaded")


def _extract_cool_moment_photos(
    *,
    video_path: Path,
    turns: list[dict],
    output_dir: Path,
    limit: int = 24,
) -> list[tuple[Path, dict]]:
    """Extract one cool-moment frame per turn and return [(path, meta)]."""
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
    full_summary: dict | None,
    local_video: Path,
    coaching_local: Path | None = None,
) -> list[dict]:
    """Upload artifacts and return DB rows."""
    rows: list[dict] = []

    full_summary_local = run_dir / "summary" / "summary.json"
    if full_summary_local.exists():
        summary_remote = f"jobs/{job_id}/summary.json"
        _upload_file(
            bucket="artifacts",
            remote_path=summary_remote,
            local_path=full_summary_local,
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
    else:
        print(f"  [warn] full summary not found at {full_summary_local}", file=sys.stderr)

    overlay_local: Path | None = None
    if full_summary:
        for artifact in full_summary.get("artifacts", []):
            if artifact.get("kind") == "video_overlay" and artifact.get("path"):
                overlay_local = Path(str(artifact["path"]))
                break
    if overlay_local is None:
        candidate = run_dir / "videos" / "overlay.mp4"
        if candidate.exists():
            overlay_local = candidate

    if overlay_local and overlay_local.exists():
        overlay_remote = f"artifacts/jobs/{job_id}/overlay{overlay_local.suffix or '.mp4'}"
        overlay_bucket = R2_ARTIFACTS_BUCKET
        if not overlay_bucket:
            raise RuntimeError("R2 artifact bucket is not configured")
        _upload_file_to_r2(
            bucket=overlay_bucket,
            remote_path=overlay_remote,
            local_path=overlay_local,
            content_type="video/mp4",
        )
        rows.append(
            {
                "job_id": job_id,
                "kind": "video_overlay",
                "object_path": overlay_remote,
                "meta": {
                    "storage_provider": "r2",
                    "storage_bucket": overlay_bucket,
                },
            }
        )
    else:
        print("  [warn] overlay video not found — skipping", file=sys.stderr)

    if full_summary:
        for artifact in full_summary.get("artifacts", []) or []:
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

    if coaching_local and coaching_local.exists():
        coaching_remote = f"jobs/{job_id}/ai_coaching.json"
        _upload_file(
            bucket="artifacts",
            remote_path=coaching_remote,
            local_path=coaching_local,
            content_type="application/json",
        )
        rows.append(
            {
                "job_id": job_id,
                "kind": "ai_coaching",
                "object_path": coaching_remote,
                "meta": {},
            }
        )

    turns = (full_summary or {}).get("turns") if full_summary else None
    if isinstance(turns, list) and turns:
        print(f"  extracting cool-moment frames for {len(turns)} turn(s)...")
        source_video = local_video
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
    elif full_summary is not None:
        print("  [warn] no turns in analysis summary — no cool-moment photos", file=sys.stderr)

    return rows


# Job processing


def _judge_unavailable_payload(error: Exception, language: str) -> dict:
    if language == "zh":
        summary = "LLM 评审暂不可用。"
        observation = "分析已成功完成，但本地 LLM 评审调用失败。这里不会显示规则生成的替代反馈。"
    else:
        summary = "LLM judge unavailable."
        observation = (
            "The analysis completed successfully, but the local LLM judge call failed. "
            "No rule-generated fallback feedback is shown."
        )

    return {
        "judge_status": "unavailable",
        "judge_kind": "metrics_only_llm",
        "judge_error": str(error)[:500],
        "coach_summary": summary,
        "coaching_points": [],
        "additional_observations": [observation],
    }


def process_job(job: dict) -> None:
    job_id: str = job["id"]
    video_path_in_storage: str | None = job.get("video_object_path")
    config: dict = dict(job.get("config") or {})
    preferred_language = str(config.get("preferred_language") or "en").lower()
    if preferred_language not in {"en", "zh"}:
        preferred_language = "en"
    provider = _video_storage_provider(config)
    expected_size_bytes = _expected_video_size_bytes(config)

    print(f"[{job_id[:8]}] Starting job (provider: {provider}, video: {video_path_in_storage})")

    if not video_path_in_storage:
        _set_status(job_id, "error", error="video_object_path is empty")
        return

    with tempfile.TemporaryDirectory(prefix="skicoach_") as tmpdir:
        try:
            _set_progress(job_id, config, "Downloading your video...", step=1, total=5, stage="Preparing your video")
            print(f"[{job_id[:8]}] Downloading video from {provider}...")
            video_bytes = _download_video_bytes(
                video_path_in_storage,
                provider,
                expected_size_bytes=expected_size_bytes,
            )
            suffix = Path(video_path_in_storage).suffix or ".mp4"
            local_video = Path(tmpdir) / f"video{suffix}"
            local_video.write_bytes(video_bytes)
            size_mb = len(video_bytes) / 1_048_576
            print(f"[{job_id[:8]}] Downloaded {size_mb:.1f} MB")

            _set_progress(job_id, config, "Analyzing your technique...", step=2, total=5, stage="Analyzing your technique")
            print(f"[{job_id[:8]}] Running technique analysis...")
            run_dir, mvp_summary, full_summary = _run_analysis(
                local_video,
                config,
                heartbeat=lambda: _write_heartbeat(job_id, config),
            )
            n_turns = mvp_summary.get("turns", 0)
            print(f"[{job_id[:8]}] Analysis done — {n_turns} turn(s) detected")

            for warning in (mvp_summary.get("quality") or {}).get("warnings", []):
                print(f"  [quality warn] {warning}", file=sys.stderr)

            _write_heartbeat(job_id, config)

            if not full_summary:
                raise RuntimeError("Detailed summary was not produced, so LLM judge feedback could not be generated.")

            coaching_path: Path | None = None
            _set_progress(job_id, config, "Writing your LLM judge feedback...", step=3, total=5, stage="Writing your LLM judge feedback")
            try:
                from lmstudio_coaching import generate_coaching

                print(f"[{job_id[:8]}] Calling LM Studio for metrics-only judge feedback...")
                coaching_result = generate_coaching(
                    full_summary,
                    base_url=LMSTUDIO_BASE_URL,
                    api_key=LMSTUDIO_API_KEY,
                    model=LMSTUDIO_MODEL,
                    language=preferred_language,
                )
                coaching_path = run_dir / "summary" / "ai_coaching.json"
                coaching_path.parent.mkdir(parents=True, exist_ok=True)
                coaching_path.write_text(json.dumps(coaching_result, indent=2))
                print(f"[{job_id[:8]}] LLM judge feedback ready")
            except Exception as exc:
                print(
                    f"[{job_id[:8]}] WARN: LLM judge unavailable, continuing with explicit unavailable state: {exc}",
                    file=sys.stderr,
                )
                coaching_path = run_dir / "summary" / "ai_coaching.json"
                coaching_path.parent.mkdir(parents=True, exist_ok=True)
                coaching_path.write_text(json.dumps(_judge_unavailable_payload(exc, preferred_language), indent=2))

            _set_progress(job_id, config, f"Publishing your recap ({n_turns} turn(s) found)...", step=4, total=5, stage="Publishing your recap")
            print(f"[{job_id[:8]}] Uploading recap assets...")
            rows = _upload_artifacts(
                job_id=job_id,
                run_dir=run_dir,
                full_summary=full_summary,
                local_video=local_video,
                coaching_local=coaching_path,
            )
            if rows:
                supabase.table("artifacts").insert(rows).execute()
            print(f"[{job_id[:8]}] Uploaded {len(rows)} recap asset(s)")

            _set_progress(job_id, config, "Finalizing your recap...", step=5, total=5, stage="Finishing up")

            config.pop("progress_note", None)
            config.pop("heartbeat_at", None)
            config.pop("progress_step", None)
            config.pop("progress_total", None)
            config.pop("progress_stage", None)
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


# Entry point


def main() -> int:
    p = argparse.ArgumentParser(description="SkiCoach local worker.")
    p.add_argument("--once", action="store_true", help="Process one job then exit.")
    p.add_argument(
        "--recover",
        action="store_true",
        help="Recover stale running jobs then exit.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL_S,
        metavar="SECS",
        help="Seconds between polls when idle.",
    )
    args = p.parse_args()

    print(f"[worker] Starting — polling {SUPABASE_URL}")
    print(f"[worker] Using {RUN_SCRIPT}")
    if all((R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_VIDEOS_BUCKET)):
        print(f"[worker] R2 video bucket configured: {R2_VIDEOS_BUCKET}")
        if R2_ARTIFACTS_BUCKET:
            print(f"[worker] R2 artifact bucket configured: {R2_ARTIFACTS_BUCKET}")
    else:
        print("[worker] R2 video bucket not configured — only Supabase-backed jobs can be downloaded")

    n = recover_stale_jobs()
    if n:
        print(f"[worker] Recovered {n} stale job(s)")

    if args.recover:
        return 0

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
