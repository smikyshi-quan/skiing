# Track C Follow-up #5 — Fix Pole-vs-Gate Scale Error (ROOT CAUSE)

## The Real Problem

We've been chasing the wrong bug. The sin(pitch) fix was correct but minor. The **actual root cause** of speeds being 50× too high is that the scale calculation confuses **individual gate poles** with **separate gates**.

### How the bug works

The YOLOv8 gate detector detects individual **poles** (each slalom gate has 2 poles — a turning pole and an outside pole). In video 2907, it detects 7 poles. The `_cluster_gates_by_y()` function in `pipeline.py` tries to merge nearby poles, but its threshold (15px) is too low — poles of the same gate can be 18–32px apart in Y.

After clustering (which doesn't merge enough), the scale code in `transform.py` uses a "minimum gap" heuristic to find `one_gate_px` (the pixel distance for one gate interval). It picks **32px** — but that's the gap between two poles of the **same gate**, not between different gates. The real between-gate interval is **55–70px**.

Result: `ppm_y = 32 / 12.0 = 2.67 px/m` when it should be ~7–13 px/m. Everything is 5–10× too large in meters.

### Evidence from video 2907

```
7 pole detections (sorted by Y):
  (267, 226)  ← pole A of gate 1
  (337, 245)  ← pole B of gate 1 (dY=19px, dX=70px — SAME GATE)
  (258, 305)  ← gate 2
  (433, 356)  ← gate 3
  (318, 389)  ← gate 4
  (654, 441)  ← pole A of gate 5
  (761, 473)  ← pole B of gate 5 (dY=32px, dX=108px — SAME GATE)

After proper grouping: 5 gates at Y ≈ 235, 305, 356, 389, 457
Real intervals: 70, 51, 33, 68 px → median ≈ 59 px
ppm_y = 59 / 9.5 = 6.2 px/m (vs current 2.67)
```

### TWO fixes needed

There are two independent problems. Fix both.

## Fix 1: Better pole clustering in `pipeline.py`

The `_cluster_gates_by_y()` method at line ~1009 uses `y_thresh=15.0` and `dynamic_thresh = max(y_thresh, 0.35 * median_gap)`.

**Change the clustering to be frame-height-aware.** Two detections are the same gate if their Y-distance is less than 8% of frame height (38px for 480p). This accounts for the fact that two poles of one slalom gate can be up to 2m apart on the slope, appearing 20–40px apart in typical footage.

**Modify `_cluster_gates_by_y`** to accept `frame_height` parameter:

```python
def _cluster_gates_by_y(self, gates, y_thresh=12.0, frame_height=None):
    if not gates:
        return gates

    gates_sorted = sorted(gates, key=lambda g: g["base_y"])

    # Adaptive threshold: 8% of frame height (two poles of same gate
    # can be ~20-40px apart vertically in typical footage)
    if frame_height and frame_height > 0:
        adaptive_thresh = 0.08 * frame_height
    else:
        adaptive_thresh = y_thresh

    # Also compute gap-based threshold
    gaps = []
    for i in range(1, len(gates_sorted)):
        dy = gates_sorted[i]["base_y"] - gates_sorted[i - 1]["base_y"]
        if dy > 1e-3:
            gaps.append(dy)
    if gaps:
        median_gap = float(np.median(gaps))
        gap_thresh = 0.45 * median_gap  # was 0.35, too conservative
    else:
        gap_thresh = y_thresh

    # Use the LARGER of the two thresholds
    dynamic_thresh = max(adaptive_thresh, gap_thresh)

    # ... rest of clustering logic stays the same ...
```

**Update all call sites** to pass `frame_height`:
- Line ~236: `gates = self._cluster_gates_by_y(gates, y_thresh=15.0, frame_height=frame_height)` — you'll need to get frame_height from the video capture (it's already available at that point in `process_video`).
- Line ~295: same.

## Fix 2: Update `regression_defaults.yaml`

The config file `configs/regression_defaults.yaml` hardcodes `gate_spacing_m: 12.0`. This should be removed so the pipeline uses the discipline-based default (9.5m for slalom).

**Change `configs/regression_defaults.yaml`:**
```yaml
# REMOVE the gate_spacing_m line entirely, or set to null
# gate_spacing_m: 12.0  ← DELETE THIS
```

Also remove `camera_pitch_deg: 6.0` — it's no longer used for spacing calculation and the auto-estimation is unreliable. Set it to null or remove it.

**And update `scripts/run_eval.py`** line ~562:
```python
# Change from:
gate_spacing_m=float(regression_config["gate_spacing_m"]),
# To:
gate_spacing_m=safe_float(regression_config.get("gate_spacing_m"), None),
```

This way, if gate_spacing_m is absent from the config, None is passed, and the pipeline uses its discipline default (9.5m for slalom).

## Fix 3: Also fix `_calculate_scale_from_gates` min-gap heuristic

In `transform.py` `_calculate_scale_from_gates()` (line ~1066), the 40% filter for the minimum gap is not aggressive enough. After the clustering fix, this should be less of a problem, but add a safety check:

After computing `one_gate_px`, add a sanity check:
```python
# Sanity: one_gate_px should be at least 5% of the Y-span of all gates
total_y_span = gates_sorted[-1][1] - gates_sorted[0][1]
if total_y_span > 0 and one_gate_px < 0.15 * total_y_span:
    # one_gate_px is suspiciously small relative to the gate field
    # Fall back to total_span / (n_gates - 1)
    one_gate_px = total_y_span / max(1, len(gates_sorted) - 1)
    print(f"  ⚠️  one_gate_px too small, using average: {one_gate_px:.1f}px")
```

## After Fixing — Verify

Run eval:
```bash
python scripts/run_eval.py \
  --model models/gate_detector_best.pt \
  --output-root tracks/C_geometry_scale/reports
```

## Pass Criteria

1. **P90 speed (pre-auto-cal) must be 15–70 km/h** on all 3 regression videos
2. **Auto-cal correction factor must be < 3.0×** (ideally < 2.0×)
3. **ppm_y must be > 5.0 px/m** (sanity: frame shouldn't cover >100m)
4. **3D course length should be 50–200m** (not 1400m like current)

## Files you own
- `ski_racing/pipeline.py` — `_cluster_gates_by_y()` fix, pass frame_height
- `ski_racing/transform.py` — `_calculate_scale_from_gates()` sanity check
- `configs/regression_defaults.yaml` — remove hardcoded gate_spacing_m and camera_pitch_deg
- `scripts/run_eval.py` — handle missing gate_spacing_m gracefully

## Do NOT modify
- `ski_racing/tracking.py`
- `ski_racing/detection.py`

## Save report to
`tracks/C_geometry_scale/reports/pole_clustering_fix_YYYYMMDD.md`

Include in the report:
- Number of gates before/after clustering for each video
- ppm_y value for each video
- P90 speed pre-auto-cal for each video
- 3D course length for each video
