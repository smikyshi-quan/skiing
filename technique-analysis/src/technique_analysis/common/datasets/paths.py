"""Filesystem helpers for the technique-analysis session."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    # technique-analysis/src/technique_analysis/common/datasets/paths.py
    # parents: [datasets, common, technique_analysis, src, technique-analysis, project_root]
    return Path(__file__).resolve().parents[5]


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("-._")
    return normalized or "video"


@dataclass(slots=True)
class SessionPaths:
    repo_root: Path
    session_root: Path
    runs_root: Path
    docs_root: Path

    def ensure(self) -> None:
        for path in (self.runs_root, self.docs_root):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class RunPaths:
    run_id: str
    run_dir: Path
    videos_dir: Path
    metrics_dir: Path
    summary_dir: Path
    debug_dir: Path
    overlay_path: Path
    metrics_csv_path: Path
    summary_json_path: Path

    def ensure(self) -> None:
        for path in (
            self.videos_dir,
            self.metrics_dir,
            self.summary_dir,
            self.debug_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_session_paths() -> SessionPaths:
    root = _repo_root()
    session_root = root / "technique-analysis"
    return SessionPaths(
        repo_root=root,
        session_root=session_root,
        runs_root=session_root / "artifacts" / "runs",
        docs_root=session_root / "docs",
    )


def create_run_paths(video_path: Path) -> RunPaths:
    session_paths = get_session_paths()
    session_paths.ensure()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{_safe_stem(video_path.stem)}"
    run_dir = session_paths.runs_root / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{timestamp}_{_safe_stem(video_path.stem)}_{suffix:02d}"
        run_dir = session_paths.runs_root / run_id
        suffix += 1
    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        videos_dir=run_dir / "videos",
        metrics_dir=run_dir / "metrics",
        summary_dir=run_dir / "summary",
        debug_dir=run_dir / "debug",
        overlay_path=run_dir / "videos" / "overlay.mp4",
        metrics_csv_path=run_dir / "metrics" / "metrics.csv",
        summary_json_path=run_dir / "summary" / "summary.json",
    )
