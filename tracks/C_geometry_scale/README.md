# Track C: Geometry / Scale Stabilization

## Owner
Assigned to: _______________

## Goal
Fix the 2D-to-3D coordinate transform so auto-calibration correction factors drop from 80–243x to under 5x.

## Your Spec Document
**Read this first:** `Track_C_Geometry_Scale_Stabilization.docx` (in this folder)

## Files You Own (edit these)
| File | What it does |
|------|-------------|
| `ski_racing/transform.py` | HomographyTransform, CameraMotionCompensator, DynamicScaleTransform |
| `ski_racing/pipeline.py` | `_auto_calibrate_scale()` method, pipeline ordering |
| `scripts/process_video.py` | Add CLI flags (--gate-spacing, --camera-pitch) |

## Files You Read (do NOT edit)
| File | Why you need it |
|------|----------------|
| `ski_racing/detection.py` | Gate positions feed into your transforms |
| `ski_racing/physics.py` | Physics validator tells you if your geometry is working |

## Your Reports Folder
`tracks/C_geometry_scale/reports/` — contains:
- Video regression results (current baseline for max speed, G-force, auto-cal)
- Previous iteration analysis (outputs_fixed3, outputs_fixed4)

## Key Commands
```bash
# Run pipeline on regression video and inspect auto-cal factor
python scripts/process_video.py data/test_videos_unseen/VIDEO.mp4 \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --summary

# Quick test: vary gate spacing
# Edit pipeline.py gate_spacing_m or add CLI flag, then run above

# Check output JSON for auto_calibration.correction_factor
python -c "import json; d=json.load(open('artifacts/latest/VIDEO_analysis.json')); print(d.get('auto_calibration',{}))"
```

## Dependencies
- **Coordinate with Track D**: Your geometry changes affect their trajectory metrics, and vice versa
- **Track B helps you** (soft): Better gate detection = better transform inputs
