"""Google AI (Gemini) photo enhancement for technique-analysis peak-pressure frames."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import cv2
import numpy as np


_MODEL_NAME = "gemini-3.1-flash-image-preview"

_ENHANCE_PROMPT = (
    "This is a peak-carve moment from alpine skiing — the skier is at maximum edge "
    "pressure and deepest lean in the turn. Enhance this action frame for visual impact: "
    "lift shadows on the skier to reveal detail in their suit and body position, "
    "selectively boost contrast to separate the skier from the background, increase color "
    "vibrancy to make the ski suit pop against the snow, gently warm the white balance, "
    "and recover snow texture in any overexposed areas. Do not over-sharpen — preserve "
    "the natural motion energy and sense of speed. Make the skier the undeniable hero "
    "of the frame."
)

# Send at up to 1536px on the long side — gives Gemini enough detail for an action frame
# without exceeding typical API limits.
_MAX_INPUT_DIM = 1536


def _repo_root() -> Path:
    # enhance_api.py lives at:
    # technique-analysis/src/technique_analysis/common/rendering/enhance_api.py
    # parents[0]=rendering, [1]=common, [2]=technique_analysis, [3]=src,
    #          [4]=technique-analysis, [5]=repo_root
    return Path(__file__).resolve().parents[5]


def _load_api_key() -> str | None:
    key = os.environ.get("GOOGLE_AI_API_KEY", "").strip()
    if key:
        return key
    # Fall back to cool-moment's api_config.json in the same repo
    config_path = _repo_root() / "cool-moment" / "data" / "api_config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            key = str(payload.get("google_ai_api_key") or "").strip()
            return key if key else None
        except Exception:
            pass
    return None


def enhance_peak_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Enhance a peak-pressure ski frame using Google AI Gemini.

    Returns (enhanced_bgr, used_api). Falls back to (None, False) on any failure
    so callers can always save the raw frame as a fallback.
    """
    api_key = _load_api_key()
    if not api_key:
        return None, False

    try:
        from google import genai
        from google.genai import types
        from PIL import Image
    except ImportError:
        return None, False

    try:
        client = genai.Client(api_key=api_key)

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        if max(h, w) > _MAX_INPUT_DIM:
            scale = _MAX_INPUT_DIM / max(h, w)
            pil_image = pil_image.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.LANCZOS,
            )

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=92)

        response = client.models.generate_content(
            model=_MODEL_NAME,
            contents=[
                types.Content(parts=[
                    types.Part(text=_ENHANCE_PROMPT),
                    types.Part(inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=buf.getvalue(),
                    )),
                ])
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        for candidate in (response.candidates or []):
            for part in (getattr(candidate.content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                if inline is None:
                    continue
                data = getattr(inline, "data", None)
                if not data:
                    continue
                result_pil = Image.open(io.BytesIO(data)).convert("RGB")
                # Scale back to original frame size if Gemini changed dimensions
                if result_pil.size != (w, h):
                    result_pil = result_pil.resize((w, h), Image.LANCZOS)
                return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR), True

    except Exception:
        pass

    return None, False
