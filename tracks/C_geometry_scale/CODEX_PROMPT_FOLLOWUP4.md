# Track C Follow-up #4 — Fix sin(pitch) Scale Bug

## The Bug

The `_compute_effective_spacing()` function in `ski_racing/transform.py` multiplies `gate_spacing_m` by `sin(camera_pitch_deg)`. This is geometrically wrong and is the root cause of all remaining speed errors.

With `pitch=6°` and `gate_spacing_m=9.5`:
```
effective_spacing = 9.5 * sin(6°) = 9.5 * 0.1045 = 0.99 m
```

The pipeline thinks gates are ~1m apart instead of 9.5m. This makes pixel→meter conversion 9.5× too compressed, producing speeds 3-4× too high (after other corrections partially compensate).

**Why this is wrong:** Camera pitch describes the camera's viewing angle relative to the slope. It does NOT change the physical distance between gates on the slope. The dynamic per-frame scale already handles perspective foreshortening by measuring actual pixel gaps each frame. Multiplying by sin(pitch) double-corrects and destroys the scale.

## What to Fix

### Fix 1: `transform.py` — DynamicScaleTransform._compute_effective_spacing() (line ~350)

**Before:**
```python
def _compute_effective_spacing(self):
    if self.camera_pitch_deg is None:
        return self.gate_spacing_m
    pitch_rad = np.deg2rad(float(self.camera_pitch_deg))
    sin_pitch = float(np.sin(pitch_rad))
    if sin_pitch <= 1e-6:
        print("⚠️  camera_pitch_deg is near 0°. Using raw gate spacing.")
        return self.gate_spacing_m
    return self.gate_spacing_m * sin_pitch
```

**After:**
```python
def _compute_effective_spacing(self):
    # Camera pitch does NOT compress gate spacing.
    # Gates are a fixed physical distance apart on the slope.
    # Perspective foreshortening is handled by per-frame pixel gap measurement.
    # Pitch is retained as metadata for future slope-angle corrections only.
    return self.gate_spacing_m
```

### Fix 2: `transform.py` — _calculate_scale_from_gates() (line ~1048)

**Before:**
```python
effective_spacing_m = self.gate_spacing_m
if self.camera_pitch_deg is not None:
    pitch_rad = np.deg2rad(float(self.camera_pitch_deg))
    sin_pitch = float(np.sin(pitch_rad))
    if sin_pitch > 1e-6:
        effective_spacing_m = self.gate_spacing_m * sin_pitch
    else:
        print("⚠️  camera_pitch_deg is near 0°. Using raw gate spacing.")
self.effective_gate_spacing_m = effective_spacing_m
```

**After:**
```python
# Camera pitch does NOT modify gate spacing.
# Physical gate distance is fixed regardless of viewing angle.
effective_spacing_m = self.gate_spacing_m
self.effective_gate_spacing_m = effective_spacing_m
```

### Fix 3: Keep camera_pitch_deg parameter alive but unused for spacing

Do NOT remove camera_pitch_deg from CLI or constructor — it will be used later for slope-grade altitude corrections. Just disconnect it from the spacing calculation.

## After Fixing — Verify

Run eval on all 3 regression videos:

```bash
python scripts/run_eval.py \
  --gate-model models/gate_detector_best.pt \
  --videos tracks/E_evaluation_ci/regression_videos/IMG_2907.mp4 \
          tracks/E_evaluation_ci/regression_videos/IMG_2909.mp4 \
          tracks/E_evaluation_ci/regression_videos/IMG_2911.mp4 \
  --discipline slalom --stabilize --summary \
  --output-dir tracks/C_geometry_scale/reports/eval_post_sinfix_$(date +%Y%m%d)
```

Also run with `--discipline giant_slalom` on at least one video to confirm GS spacing (27m) still produces reasonable numbers.

## Pass Criteria

After removing sin(pitch):
1. **P90 speed (pre-auto-cal) must be 15–70 km/h** on all 3 regression videos
2. **Auto-cal correction factor must be < 2.0×** (close to 1.0 means the raw scale is correct)
3. **No physics validation failures** for speed > 120 km/h pre-autocal

If P90 is now reasonable but correction is still >2×, the gate_spacing_m value may need tuning — try 9.5, 10.0, 11.0 to find the best fit.

## Files you own
- `ski_racing/transform.py` (both fix locations)

## Do NOT modify
- `ski_racing/tracking.py`
- `ski_racing/detection.py`
- `scripts/run_eval.py`

## Save report to
`tracks/C_geometry_scale/reports/sinfix_eval_YYYYMMDD.md`
