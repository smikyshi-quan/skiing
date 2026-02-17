# Analysis Report: `outputs_fixed3` Results

## Overview

Three slalom skiing videos were analyzed with the updated pipeline. All three **fail the physics validation check**. While some metrics have improved compared to what would be expected from earlier iterations (e.g., auto-calibration is active, Kalman smoothing is applied), several fundamental issues persist.

| Metric | Video 2907 | Video 2909 | Video 2911 |
|--------|-----------|-----------|-----------|
| Gates detected | 7 | 8 | 8 |
| Trajectory points | 515 | 254 | 185 |
| Duration (s) | 17.2 | 8.4 | 6.1 |
| Total distance (m) | 146.8 | 101.0 | 43.4 |
| Avg speed (km/h) | 30.7 | 43.1 | 25.5 |
| Max speed (km/h) | 57.3 | 55.0 | 55.0 |
| Max G-force | **91.7 G** | **87.5 G** | **8.7 G** |
| Min turn radius (m) | **0.01** | **0.04** | **0.16** |
| Max acceleration (m/s²) | **444.2** | **299.9** | **106.8** |
| Physics check | FAIL | FAIL | FAIL |
| Auto-cal correction | 9.4x | 10.0x | 11.3x |
| Soft-clamped points | 0 | **454** (out of 254!) | 0 |

---

## Problem 1: Extreme Auto-Calibration Corrections (9–11x)

The auto-calibration system is applying massive scale corrections to force the p90 speed down to the 55 km/h target.

- **2907**: Pre-calibration median speed was **263 km/h**, p90 was **517 km/h** → corrected by **9.4x**
- **2909**: Pre-calibration median speed was **502 km/h**, p90 was **550 km/h** → corrected by **10.0x**
- **2911**: Pre-calibration median speed was **218 km/h**, p90 was **619 km/h** → corrected by **11.3x**

**What this means**: The raw pixel-to-meter conversion is off by roughly an order of magnitude. The "scale" projection method using gate spacing alone is not producing accurate world coordinates. The auto-calibration is masking the root problem rather than fixing it — it rescales distances but cannot fix the *shape* of the trajectory (which is why G-forces and turn radii remain unrealistic).

---

## Problem 2: Trajectory Only Covers the Upper Part of the Course

This is perhaps the most critical issue. In all three videos, the skier's 2D trajectory only passes near the **top 2–3 gates** and never reaches the lower gates:

**Video 2907** (7 gates):
- Gates 0–1 (y ≈ 268–281): trajectory passes within 4–9 px ✓
- Gates 2–6 (y ≈ 316–475): **closest trajectory point is 47–253 px away** ✗

**Video 2909** (8 gates):
- Gate 1 (y ≈ 242): trajectory is 58 px away (borderline)
- Gates 2–7 (y ≈ 261–427): **closest point is 150–305 px away** ✗

**Video 2911** (8 gates):
- Gates 0–2 (y ≈ 214–291): trajectory passes within 4–10 px ✓
- Gates 3–7 (y ≈ 336–455): **closest point is 95–216 px away** ✗

**Why this matters**: The trajectory appears to track the skier only while they're in the upper portion of the frame. Once the skier approaches the camera and moves into the lower half, the trajectory either stops tracking or the skier has already exited the tracked region. This means the 3D projection is being computed from incomplete data — the system only sees a fraction of the actual slalom run.

---

## Problem 3: Unrealistic G-Forces and Turn Radii

All three videos show physically impossible dynamics:

- **G-forces**: 8.7–91.7 G (limit should be ~3–5 G for alpine skiing)
- **Turn radii**: 0.01–0.16 m minimum (a slalom turn radius should be at minimum ~6–10 m)
- **Accelerations**: 107–444 m/s² (should be under ~15 m/s²)

These extreme values persist *after* auto-calibration, which means the trajectory *shape* itself is problematic — not just the scale. The 3D trajectory has sudden, sharp direction changes that don't correspond to smooth skiing turns.

Looking at the 3D trajectory oscillation data:
- **2907**: 38 X-direction changes with 34 "major" turns — far too many for 7 gates. This suggests noisy, jittery tracking rather than clean slalom arcs.
- **2909**: 11 X-direction changes for 8 gates — reasonable count, but the clamping issue (454/254 points clamped) suggests the raw data was highly erratic.
- **2911**: Only 1 X-direction change for 8 gates — the 3D trajectory is essentially a straight line with one kink, completely missing the slalom pattern.

---

## Problem 4: Inconsistent Gate Spacing in Pixels

If all gates are truly 12 m apart (as specified), the pixel spacing between consecutive gates should follow a consistent perspective scaling pattern (increasing as gates get closer to camera). Instead, the spacings are highly irregular:

**Video 2907**: Gate spacings (dy) = 13, 35, 54, 29, 43, 34 px → implied scale jumps from 1.1 to 4.5 px/m

**Video 2909**: Gate spacings (dy) = 37, 19, 45, 20, 36, 43, 22 px → alternates between large and small, suggesting alternating left/right gate detection where some "gates" may be poles from the same gate pair.

**Video 2911**: Gate spacings (dy) = 26, 51, 45, 30, 41, 34, 14 px → last pair only 14 px apart, likely a duplicate detection.

**Key concern for 2907**: Gates 0 and 1 are only **13 px apart** vertically and nearly identical in X (225.7 vs 225.4). These are almost certainly the same gate detected twice, which throws off the entire spacing model.

---

## Problem 5: 3D Trajectory Shape Doesn't Match Slalom Geometry

Looking at the summary plots:

- **2907 3D plot**: Shows a chaotic, zigzagging trajectory spanning only ~23 m × 14 m, with many overlapping loops. This doesn't resemble a slalom run at all.
- **2909 3D plot**: Shows a very compressed shape (~50 m × 6 m), essentially a narrow zigzag band. The Y-span of only 5.8 m for a run with 8 gates at 12 m spacing (expected ~96 m downhill) indicates the downhill dimension is severely compressed.
- **2911 3D plot**: Shows a roughly triangular path spanning 25 m × 15 m with almost no lateral oscillation — doesn't capture the slalom turns.

---

## Problem 6: Speed Profile Anomalies

From the speed profile plots:

- **2907**: Speed oscillates wildly between near-zero and ~50 km/h with many sudden drops. Real skiing speed profiles should be much smoother with gradual speed variations through turns.
- **2909**: Many sudden drops to zero or near-zero, suggesting tracking loss or stabilization artifacts. The speed is clamped at 55 km/h for long stretches (visible as flat plateaus).
- **2911**: Speed is near-zero for the majority of the run, then jumps suddenly. This suggests the skier is barely moving in the tracked frames, which contradicts what should be an active slalom run.

---

## Suggestions

### 1. Fix the Gate Detection (Deduplication)

The gate detector is producing near-duplicate detections (e.g., 2907 gates 0–1 are only 13 px apart). Implement a **non-maximum suppression** step for gates: if two detected gates are within a threshold distance (e.g., < 25 px vertically), merge them. This will fix the scale model foundation.

### 2. Rethink the Scale Projection Model

The "scale" projection with a single `gate_spacing_m` parameter assumes uniform perspective, but the actual pixel-to-meter ratio varies by 2–4x across the frame. Consider:
- Using a **per-gate local scale** derived from the known 12 m spacing between consecutive gate pairs, interpolated across the image.
- Implementing a simple **homography** using the gate positions as control points (if gate layout is known) instead of a single global scale factor.
- If the camera angle is roughly known, using a **ground-plane homography** to properly account for perspective foreshortening.

### 3. Extend Trajectory Tracking to Cover More Gates

The trajectory currently only covers the upper portion of the frame (the first 2–3 gates). This needs investigation:
- Is the skier detector losing the target when they get larger/closer to camera?
- Is the stabilization cropping out the lower portion of the frame?
- Consider whether the tracking window or ROI needs to be expanded.

As a diagnostic step: overlay the 2D trajectory on a frame of the video to verify where the trajectory sits relative to the actual skier path.

### 4. Smooth the 3D Trajectory Before Computing Dynamics

Even after Kalman smoothing, the trajectory has too many sharp direction changes (especially 2907 with 38 reversals). Consider:
- Applying a **stronger low-pass filter** to the 3D trajectory (e.g., Savitzky-Golay with a wider window, or a moving average over 10–15 frames).
- Computing G-forces and turn radii from the **smoothed** 3D trajectory rather than the raw trajectory.
- Using a **minimum turn radius constraint** as a physical prior during trajectory fitting.

### 5. Rethink the Auto-Calibration Strategy

A 9–11x correction factor signals that something fundamental is wrong upstream. Instead of force-fitting the speed distribution:
- Use the auto-calibration factor as a **diagnostic signal**: if correction > 3x, flag the result and investigate the projection/gate-detection pipeline.
- Consider calibrating the scale from gate-to-gate distances directly (if 12 m is known, and you have gate positions in pixels, you can compute the local scale at each gate level).

### 6. Validate Against Known Course Parameters

For a slalom course with 12 m gate spacing and ~7–8 gates:
- Expected downhill distance: ~84–96 m
- Expected run time through visible gates: ~5–8 seconds
- Expected average speed: ~40–55 km/h
- Expected max lateral displacement: ~3–5 m from center

Compare these benchmarks against your computed metrics. Currently, 2907 shows 147 m distance in 17 s (too long/slow for 7 gates), and 2911 shows only 43 m in 6 s (too short).

### 7. Consider Camera Motion Compensation

The `camera_motion_frames` values (452, 586, 566) suggest substantial camera panning. Verify that the stabilization is correctly compensating for this — if the camera follows the skier, the stabilized trajectory in the reference frame should show the skier moving downhill. The current trajectories staying in the upper frame region suggest the stabilization may not be fully accounting for downhill panning.
