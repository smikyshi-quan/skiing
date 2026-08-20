"""Generate personalised coaching feedback for a single run using LM Studio.

Reads the technique-analysis summary JSON, sends it to LM Studio's local
OpenAI-compatible chat completions endpoint, and returns structured coaching
output.

Usage (standalone test):
    python MVP/lmstudio_coaching.py path/to/summary.json

Requires:
    pip install requests
    LM Studio server running locally
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env.worker"))

# ---------------------------------------------------------------------------
# Drill library — keep in sync with MVP/web/lib/drills.ts
# ---------------------------------------------------------------------------

DRILLS = [
    {"id": "traverse-outside-ski", "title": "Traverse on outside ski", "category": "balance"},
    {"id": "equal-rhythm-turns", "title": "Equal rhythm turns", "category": "rhythm"},
    {"id": "hockey-stops", "title": "Hockey stops both sides", "category": "edging"},
    {"id": "hands-forward-quiet-poles", "title": "Hands forward, quiet poles", "category": "movement"},
    {"id": "inside-ski-lift", "title": "Lift inside ski in turns", "category": "balance"},
    {"id": "short-turns-corridor", "title": "Short turns in a corridor", "category": "edging"},
    {"id": "no-poles-balance", "title": "Ski without poles", "category": "balance"},
    {"id": "side-slip-falling-leaf", "title": "Side-slip & falling leaf", "category": "edging"},
    {"id": "hold-finish-pause", "title": "Pause at turn finish", "category": "balance"},
]

DRILL_IDS = [d["id"] for d in DRILLS]
DEFAULT_BASE_URL = "http://localhost:1234"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a metrics-only LLM judge for alpine ski technique. You are reviewing
a single run analysis produced by a computer-vision pipeline. The input is a
curated structured-metrics payload, not raw video.

Your job is to write personalised, actionable coaching feedback for this
specific run. Write as a friendly but direct coach speaking to the skier.

IMPORTANT GUIDELINES:
- Base your feedback ONLY on the metrics provided. Do not invent observations
  about things you cannot see (you do not have the video).
- Be specific: reference actual numbers from the data (e.g., "your left-right
  knee asymmetry averaged 18 degrees, aim for under 10 degrees").
- Treat reliability fields as first-class evidence. If score_reliability is
  limited or insufficient, include an explicit caveat and avoid precise
  degree-by-degree prescriptions that the data cannot support.
- Keep the tone encouraging but honest.
- Do not mention deterministic rules or imply the feedback came from video
  inspection. It came from structured metrics only.

OUTPUT FORMAT — respond with valid JSON only, no markdown fences:
{
  "coach_summary": "A 2-4 sentence overall assessment of the run.",
  "coaching_points": [
    {
      "title": "Short title (5-8 words)",
      "feedback": "2-3 sentences of specific, actionable coaching.",
      "category": "balance|edging|rhythm|movement",
      "severity": "action|warn|info",
      "recommended_drill_id": "<drill_id from the list below, or null if none fit>"
    }
  ],
  "additional_observations": [
    "Any extra observations that don't map to the drills above (text only)."
  ]
}

Produce 2-4 coaching_points (the most important ones). If an observation fits
one of the available drills, set recommended_drill_id. If none fit, set it to null.

AVAILABLE DRILLS:
"""


def _normalize_language(language: str | None) -> str:
    value = (language or "en").strip().lower()
    if value.startswith("zh"):
        return "zh"
    return "en"


def _language_instruction(language: str) -> str:
    if language == "zh":
        return """

LANGUAGE REQUIREMENT:
- Write every natural-language value in Simplified Chinese.
- Keep the JSON keys in English exactly as specified.
- Keep enum values for category, severity, and recommended_drill_id exactly in English.
"""

    return """

LANGUAGE REQUIREMENT:
- Write every natural-language value in English.
- Keep the JSON keys, enum values, and recommended_drill_id exactly as specified.
"""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return _mean([(value - avg) ** 2 for value in values]) ** 0.5


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smaller_is_better(value: float, best: float, worst: float) -> float:
    if value <= best:
        return 100.0
    if value >= worst:
        return 0.0
    return _clamp(100.0 - ((value - best) / (worst - best)) * 100.0, 0.0, 100.0)


def _closeness_score(value: float, target: float, spread: float) -> float:
    return _clamp(100.0 - (abs(value - target) / spread) * 100.0, 0.0, 100.0)


def _positive_score(value: float, floor: float, ceiling: float) -> float:
    if value <= floor:
        return 0.0
    if value >= ceiling:
        return 100.0
    return _clamp(((value - floor) / (ceiling - floor)) * 100.0, 0.0, 100.0)


def _number(value, digits: int | None = None):  # noqa: ANN001
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return round(numeric, digits) if digits is not None else numeric


def _numbers(values) -> list[float]:  # noqa: ANN001
    result: list[float] = []
    for value in values:
        numeric = _number(value)
        if numeric is not None:
            result.append(numeric)
    return result


def _score_label(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 78:
        return "Dialed"
    if score >= 62:
        return "Good"
    if score >= 45:
        return "Building"
    return "Focus"


def _compute_summary_score(summary: dict) -> int | None:
    turns = summary.get("turns") if isinstance(summary.get("turns"), list) else []
    if not turns:
        return None

    quality_scores = _numbers([turn.get("quality_score") for turn in turns if isinstance(turn, dict)])
    if not quality_scores:
        return None

    smoothness_scores = _numbers([turn.get("smoothness_score") for turn in turns if isinstance(turn, dict)])
    edge_angles = _numbers([turn.get("avg_edge_angle") for turn in turns if isinstance(turn, dict)])
    stance_widths = _numbers([turn.get("avg_stance_width_ratio") for turn in turns if isinstance(turn, dict)])
    asymmetry: list[float] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        value = _number(turn.get("avg_knee_flexion_diff"))
        if value is not None:
            asymmetry.append(abs(value))
    lean_angles = _numbers([turn.get("avg_lean_angle") for turn in turns if isinstance(turn, dict)])
    quietness = _numbers([turn.get("avg_upper_body_quietness") for turn in turns if isinstance(turn, dict)])
    com_shift = _numbers([turn.get("avg_com_shift_3d") for turn in turns if isinstance(turn, dict)])
    durations = _numbers([turn.get("duration_s") for turn in turns if isinstance(turn, dict)])

    quality = _quality_source(summary)
    pose_confidence = round((_number(quality.get("overall_pose_confidence_mean")) or 0.0) * 100.0)
    overall_score = round(_mean(quality_scores))
    smoothness_score = round(_mean(smoothness_scores)) if smoothness_scores else None
    edge_angle = round(_mean(edge_angles), 1)
    stance_width = round(_mean(stance_widths), 2)
    knee_asymmetry = round(_mean(asymmetry), 1)
    lean_angle = round(_mean(lean_angles), 1)
    quietness_mean = _mean(quietness)
    com_shift_mean = round(_mean(com_shift), 2)
    duration_drift = round(_stddev(durations), 2)
    best_turn_score = round(max(quality_scores or [0.0]))

    balance_score = round(_mean([
        _smaller_is_better(knee_asymmetry, 6, 28),
        _closeness_score(stance_width, 1.45, 1.35),
        _closeness_score(lean_angle, 24, 18),
    ]))
    edging_score = round(_mean([
        _closeness_score(edge_angle, 47, 24),
        _closeness_score(com_shift_mean, 0.28, 0.26),
        _positive_score(pose_confidence, 55, 92),
    ]))
    rhythm_score = round(_mean([
        smoothness_score if smoothness_score is not None else overall_score,
        _smaller_is_better(duration_drift, 0.1, 1.7),
        _closeness_score(_mean(durations), 1.7, 1.4),
    ]))
    movement_score = round(_mean([
        _smaller_is_better(quietness_mean, 0.002, 0.02),
        overall_score,
        _positive_score(best_turn_score, 35, 82),
    ]))

    return round(_mean([balance_score, edging_score, rhythm_score, movement_score]))


def _quality_source(summary: dict) -> dict:
    quality_report = summary.get("quality_report")
    if isinstance(quality_report, dict):
        return quality_report
    quality = summary.get("quality")
    return quality if isinstance(quality, dict) else {}


def _list_of_strings(value) -> list[str]:  # noqa: ANN001
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def _normalized_quality_report(summary: dict) -> dict:
    source = _quality_source(summary)
    reliability = source.get("score_reliability")
    if reliability not in {"reliable", "limited", "insufficient"}:
        reliability = "limited"
    score_counts = source.get("score_counts_for_progress")
    if not isinstance(score_counts, bool):
        score_counts = reliability != "insufficient"

    return {
        "score_reliability": reliability,
        "score_counts_for_progress": score_counts,
        "quality_warnings": _list_of_strings(source.get("quality_warnings")),
        "legacy_warnings": _list_of_strings(source.get("warnings")),
        "stance_measurable": source.get("stance_measurable") if isinstance(source.get("stance_measurable"), bool) else True,
        "stance_visibility_fraction": _number(source.get("stance_visibility_fraction"), 3),
        "wedge_likely": source.get("wedge_likely") is True,
        "overall_pose_confidence_mean": _number(source.get("overall_pose_confidence_mean"), 3),
        "overall_pose_confidence_min": _number(source.get("overall_pose_confidence_min"), 3),
        "low_confidence_fraction": _number(source.get("low_confidence_fraction"), 3),
        "viewpoint_warning": source.get("viewpoint_warning") if isinstance(source.get("viewpoint_warning"), str) else None,
        "jitter_score_mean": _number(source.get("jitter_score_mean"), 3),
        "resolved_max_fps": _number(source.get("resolved_max_fps"), 2),
        "resolved_max_dimension": _number(source.get("resolved_max_dimension"), 0),
    }


def _boundary_reliability(summary: dict, turn_idx) -> str | None:  # noqa: ANN001
    candidates = [
        summary.get("boundary_reliability_by_turn"),
        (summary.get("diagnostics") or {}).get("boundary_reliability_by_turn")
        if isinstance(summary.get("diagnostics"), dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in (turn_idx, str(turn_idx)):
            value = candidate.get(key)
            if isinstance(value, str):
                return value
    return None


def _phase_metrics(turn: dict):
    for key in ("phase_metrics", "phases", "phase_summary", "phase_summaries"):
        value = turn.get(key)
        if isinstance(value, (dict, list)):
            return value
    return None


def _cool_moment_timestamp(turn: dict):
    start_s = _number(turn.get("start_s"), 2)
    end_s = _number(turn.get("end_s"), 2)
    if start_s is None:
        return None
    if end_s is None or end_s <= start_s:
        return start_s
    return round((start_s + end_s) / 2.0, 2)


def _compact_turn(summary: dict, turn: dict) -> dict:
    turn_idx = turn.get("turn_idx")
    payload = {
        "turn_idx": turn_idx,
        "side": turn.get("side"),
        "start_s": _number(turn.get("start_s"), 2),
        "end_s": _number(turn.get("end_s"), 2),
        "duration_s": _number(turn.get("duration_s"), 2),
        "quality_score": _number(turn.get("quality_score"), 1),
        "smoothness_score": _number(turn.get("smoothness_score"), 1),
        "avg_pose_confidence": _number(turn.get("avg_pose_confidence"), 3),
        "boundary_reliability": _boundary_reliability(summary, turn_idx),
        "cool_moment_timestamp_s": _cool_moment_timestamp(turn),
        "metrics": {
            "avg_knee_flexion_L": _number(turn.get("avg_knee_flexion_L"), 1),
            "avg_knee_flexion_R": _number(turn.get("avg_knee_flexion_R"), 1),
            "avg_knee_flexion_diff": _number(turn.get("avg_knee_flexion_diff"), 1),
            "avg_stance_width_ratio": _number(turn.get("avg_stance_width_ratio"), 2),
            "avg_upper_body_quietness": _number(turn.get("avg_upper_body_quietness"), 5),
            "avg_lean_angle": _number(turn.get("avg_lean_angle"), 1),
            "avg_edge_angle": _number(turn.get("avg_edge_angle"), 1),
            "avg_com_shift_3d": _number(turn.get("avg_com_shift_3d"), 3),
            "peak_lateral_shift": _number(turn.get("peak_lateral_shift"), 3),
            "amplitude": _number(turn.get("amplitude"), 3),
        },
    }
    phases = _phase_metrics(turn)
    if phases is not None:
        payload["phase_metrics"] = phases
    return payload


def build_metrics_judge_input(summary: dict) -> dict:
    """Return the compact structured payload sent to the metrics-only LLM judge."""
    quality = _normalized_quality_report(summary)
    turns = [turn for turn in summary.get("turns", []) if isinstance(turn, dict)]
    score = _number(summary.get("score"), 0)
    if score is None:
        score = _compute_summary_score(summary)
    video_metadata = summary.get("video_metadata") if isinstance(summary.get("video_metadata"), dict) else {}
    segments = summary.get("segments") if isinstance(summary.get("segments"), list) else []

    return {
        "judge_input_version": "improvements_v2_metrics_only",
        "run": {
            "score": score,
            "score_label": _score_label(score),
            "score_reliability": quality["score_reliability"],
            "score_counts_for_progress": quality["score_counts_for_progress"],
            "turns_detected": len(turns),
            "video_duration_s": _number(video_metadata.get("duration_s"), 2),
        },
        "quality_report": quality,
        "video_metadata": {
            "fps": _number(video_metadata.get("fps"), 2),
            "width": _number(video_metadata.get("width"), 0),
            "height": _number(video_metadata.get("height"), 0),
            "frame_count": _number(video_metadata.get("frame_count"), 0),
        },
        "tracking_segments": [
            {
                "idx": segment.get("idx"),
                "start_s": _number(segment.get("start_s"), 2),
                "end_s": _number(segment.get("end_s"), 2),
                "n_confident_frames": _number(segment.get("n_confident_frames"), 0),
                "mean_confidence": _number(segment.get("mean_confidence"), 3),
                "n_turns": _number(segment.get("n_turns"), 0),
                "is_primary": segment.get("is_primary") is True,
            }
            for segment in segments
            if isinstance(segment, dict)
        ],
        "turns": [_compact_turn(summary, turn) for turn in turns],
    }


def _build_messages(summary: dict, language: str = "en") -> list[dict]:
    drill_list = "\n".join(
        f'- id: "{d["id"]}" | title: "{d["title"]}" | category: {d["category"]}'
        for d in DRILLS
    )
    system = SYSTEM_PROMPT + drill_list + _language_instruction(_normalize_language(language))
    user_msg = json.dumps(build_metrics_judge_input(summary), indent=2, default=str)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def _coaching_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ski_coaching_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "coach_summary": {"type": "string"},
                    "coaching_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "feedback": {"type": "string"},
                                "category": {
                                    "type": "string",
                                    "enum": ["balance", "edging", "rhythm", "movement"],
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["action", "warn", "info"],
                                },
                                "recommended_drill_id": {
                                    "anyOf": [
                                        {"type": "string", "enum": DRILL_IDS},
                                        {"type": "null"},
                                    ]
                                },
                            },
                            "required": [
                                "title",
                                "feedback",
                                "category",
                                "severity",
                                "recommended_drill_id",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "additional_observations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "coach_summary",
                    "coaching_points",
                    "additional_observations",
                ],
                "additionalProperties": False,
            },
        },
    }


def _normalize_base_url(base_url: str | None) -> str:
    raw = (base_url or os.environ.get("LMSTUDIO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("LMSTUDIO_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _resolve_model(base_url: str, headers: dict[str, str], model: str | None) -> str:
    configured = model or os.environ.get("LMSTUDIO_MODEL")
    if configured:
        return configured

    response = requests.get(f"{base_url}/models", headers=headers, timeout=15)
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")
        raise RuntimeError(f"LM Studio model lookup failed ({response.status_code}): {detail[:400]}")

    data = response.json().get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(
            "LM Studio returned no models. Load a model in LM Studio or set LMSTUDIO_MODEL explicitly."
        )

    first_id = data[0].get("id")
    if not first_id:
        raise RuntimeError("LM Studio /v1/models response did not include a usable model id")
    return str(first_id)


def _extract_text(response_json: dict) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LM Studio response did not include choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LM Studio response did not include a message object")

    content = message.get("content", "")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        text = "\n".join(part.strip() for part in texts if part.strip()).strip()
    else:
        text = ""

    if not text:
        raise RuntimeError("LM Studio returned no text content")
    return text


def _parse_json_text(text: str) -> dict:
    raw = text.strip()
    candidates = [raw]

    if raw.startswith("```"):
        stripped = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        candidates.append(stripped)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    raise last_error or json.JSONDecodeError("No JSON object found", raw, 0)


def _sanitize_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise RuntimeError("LM Studio coaching payload must be a JSON object")

    result.setdefault("coach_summary", "")
    result.setdefault("coaching_points", [])
    result.setdefault("additional_observations", [])
    result["judge_status"] = "available"
    result["judge_kind"] = "metrics_only_llm"

    if not isinstance(result["coaching_points"], list):
        result["coaching_points"] = []
    if not isinstance(result["additional_observations"], list):
        result["additional_observations"] = []

    for point in result["coaching_points"]:
        if not isinstance(point, dict):
            continue
        drill_id = point.get("recommended_drill_id")
        if drill_id and drill_id not in DRILL_IDS:
            point["recommended_drill_id"] = None

    return result


def generate_coaching(
    summary: dict,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    language: str = "en",
    timeout: int = 120,
) -> dict:
    """Call LM Studio to generate coaching feedback for a run summary."""
    normalized_base_url = _normalize_base_url(base_url)
    headers = _headers(api_key)
    resolved_model = _resolve_model(normalized_base_url, headers, model)
    messages = _build_messages(summary, language=language)

    response = requests.post(
        f"{normalized_base_url}/chat/completions",
        headers=headers,
        json={
            "model": resolved_model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 1400,
            "stream": False,
            "response_format": _coaching_schema(),
        },
        timeout=timeout,
    )

    if not response.ok:
        detail = response.text.strip().replace("\n", " ")
        raise RuntimeError(f"LM Studio API error ({response.status_code}): {detail[:400]}")

    text = _extract_text(response.json())
    return _sanitize_result(_parse_json_text(text))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <summary.json>", file=sys.stderr)
        sys.exit(1)

    summary_path = Path(sys.argv[1])
    summary = json.loads(summary_path.read_text())

    result = generate_coaching(summary)
    print(json.dumps(result, indent=2))
