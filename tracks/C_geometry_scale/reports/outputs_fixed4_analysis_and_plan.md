# Analysis: `outputs_fixed4` + Professor's Feedback → Final Action Plan

---

## Part 1: What Changed from fixed3 → fixed4

### Improved
- **Trajectory coverage** jumped dramatically: 61% → 99% (2907), 28% → 97% (2909), 24% → 96% (2911). The tracker now runs across nearly the entire video instead of a short segment.
- **Gate deduplication** is working: 2907 went from 7 gates (with a duplicate pair 13 px apart) down to 6 clean gates. Similarly 2909 and 2911 dropped from 8 to 6 gates.
- **3D X-direction changes** in 2907 improved from 38 noisy reversals to 5 — much closer to the expected ~6 for a slalom with 6 gates. The slalom oscillation pattern is more plausible now.

### Got Worse
- **Max G-forces** are still extreme and in some cases worse: 2909 jumped from 87.5 G to 247.6 G. Video 2911 went from 8.7 G to 81.0 G.
- **Trajectory jumps** got worse: 2909 now has a 16.7 m single-frame jump (was 0.51 m). Video 2911 has a 4.0 m jump (was 0.51 m).
- **Soft-clamped points** exploded: 2909 has 1,092 clamped points out of 879 trajectory points; 2911 has 766/748. This means essentially *every* point is being corrected by the sanitizer.
- **Max speed** escaped the cap in 2909 (162 km/h) and 2911 (109 km/h), suggesting the auto-calibration isn't catching extreme outliers.
- **Auto-calibration correction** is still ~9.5–9.9x — the fundamental scale problem remains unchanged.

### Unchanged
- All three videos still **FAIL** physics validation.
- The **core "tail problem"**: gate detection drops to zero and stays at zero for the final 25–43% of the video (2907: frames 482–843, 2909: frames 616–910, 2911: frames 590–776). The trajectory continues tracking the skier, but with no gate anchors for stabilization or scale reference.
- The trajectory still only passes near the **top 2–3 gates** in pixel space. Even though we now have ~800+ trajectory points, the Y-range of the trajectory (145–283 px) never reaches the lower gates (y > 300 px). The skier is tracked in the upper frame while the gates sit in the lower frame.

---

## Part 2: Evaluating the Professor's Suggestions Against the Actual Data and Code

### A. "Premature Tracking Termination" — CONFIRMED, IMPORTANT

The professor's observation is spot-on. The data proves it conclusively:

| Video | Detection span | Dead tail | % of video lost |
|-------|---------------|-----------|-----------------|
| 2907 | frames 0–481 | frames 482–843 | **42.9%** |
| 2909 | frames 0–615 | frames 616–910 | **32.4%** |
| 2911 | frames 0–589 | frames 590–776 | **24.1%** |

However, the professor's diagnosis ("lacks memory or predictive state") is **partially wrong** — you already have a `TemporalGateTracker` with Kalman-style tracking. The real issue is likely that the gates move out of the detectable region (below frame, too close to camera, or occluded by finish area), and the tracker correctly stops when confidence drops to zero for too long. The professor's suggested fix (predict via Kalman when detection is empty) is reasonable but needs a limit — you can't predict gate positions indefinitely with no observations.

**Verdict: The observation is correct and important. The suggested fix (Kalman prediction to bridge short gaps) already exists in your tracker. The real fix is understanding *why* gates vanish — likely the camera has panned past them.**

### B. "Low Confidence Thresholds (0.38–0.57)" — CONFIRMED, BUT NUANCED

The per-frame confidence data confirms this:

| Video | Median conf | < 0.5 | ≥ 0.7 |
|-------|-------------|-------|-------|
| 2907 | 0.623 | 36.9% | 43.2% |
| 2909 | 0.517 | 47.4% | 31.8% |
| 2911 | 0.511 | 47.5% | 21.5% |

The professor is correct that ~37–47% of detections have confidence < 0.5, which is low for red/blue gates on snow.

However, the professor's suggestion (CLAHE on L-channel) is **somewhat outdated for a YOLO-based detector**. YOLO models trained on properly augmented data shouldn't need external preprocessing. The more likely issue is that the training data doesn't include enough variety of lighting conditions, camera distances, or gate occlusion states.

**Verdict: The observation is valid. CLAHE is a cheap experiment worth trying, but the real fix is probably improving the training data with harder examples (far gates, partial occlusion, shadow conditions).**

### C. "Static Geometric Assumptions (12 m spacing, translation mode)" — PARTIALLY VALID

The professor's critique has two parts:

1. **"Hardcoded 12 m spacing is wrong"** — This is **debatable**. FIS slalom regulations specify gate spacing within a specific range, but the professor is right that real courses vary. However, you probably can't estimate spacing dynamically from a single monocular video without additional information. The 12 m assumption is a reasonable approximation if your videos come from regulation courses. The bigger problem isn't the value of 12 m — it's that the scale projection itself is flawed (hence the 9.5x correction).

2. **"Translation-only camera model ignores rotation"** — This is **correct and important**. Your `CameraMotionCompensator` has both `translation` and `affine` modes, but you're using `translation`. In skiing footage, the camera pans and tilts to follow the racer. Translation-only compensation will leave residual rotation artifacts in the stabilized trajectory, which directly contributes to the jittery paths and extreme G-forces. Your `affine` mode (Euclidean: rotation + translation without scale) already exists in the code — you should be using it.

**Verdict: The spacing critique is minor (you can't easily fix it). The camera rotation critique is important and actionable — switch from `translation` to `affine` camera mode.**

### D. "Implement Kalman Filter for tracking" — ALREADY DONE

You already have a full Rauch-Tung-Striebel smoother (`KalmanSmoother` in tracking.py) with discipline-tuned parameters. The professor seems to be reviewing your output data without seeing the code. This suggestion can be skipped.

**Verdict: Already implemented. No action needed.**

### E. "Dynamic Homography Estimation" — VALID, PARTIALLY IMPLEMENTED

Your code already has `HomographyTransform` with both full homography and piecewise scale modes, plus `DynamicScaleTransform` for per-frame scaling. However, the current approach computes the homography from the first/best frame only (static). The professor's suggestion to make it dynamic (recompute as gates move through the frame) is valid.

The more practical version of this: since you track gates per-frame, you could recompute the local scale at each frame using the currently-visible gate pairs, which is exactly what `DynamicScaleTransform` is designed to do. The question is whether it's being used effectively.

**Verdict: The infrastructure exists but may not be fully exploited. Ensure `DynamicScaleTransform` is driving the per-frame 2D→3D conversion, not a single static scale.**

### F. "RANSAC for outlier rejection" — ALREADY IMPLEMENTED

Your `HomographyTransform.calculate_from_gates()` already uses `cv2.findHomography(..., cv2.RANSAC, 5.0)` and your camera motion estimation uses `cv2.estimateAffinePartial2D(..., method=cv2.RANSAC)`.

**Verdict: Already implemented. No action needed.**

---

## Part 3: Final Filtered Action Plan

Based on combining the professor's valid observations with what the data actually shows, here are the actionable fixes ranked by expected impact:

### Priority 1: Fix the Scale/Projection Pipeline (ROOT CAUSE)

**The fundamental problem**: The auto-calibration consistently needs a ~9.5–10x correction, which means the raw pixel-to-meter conversion is off by an order of magnitude. Rescaling by 10x fixes speed magnitudes but cannot fix the *shape* of the trajectory — which is why G-forces and turn radii remain extreme.

**Actions:**
1. **Switch camera mode from `translation` to `affine`** — This is a one-line config change and addresses the professor's valid rotation critique. Camera pan/tilt is the dominant motion in skiing footage; ignoring it corrupts the stabilization.
2. **Verify the piecewise scale mapping** — Print/log the per-frame `ppm_y` values from `DynamicScaleTransform` and check if they're reasonable. A typical slalom gate in a 480p video at moderate distance might subtend ~30–50 px, implying ~2.5–4.2 px/m. If the scale is wildly different, trace back to gate spacing in pixels.
3. **Investigate why pixel gate spacing varies so much** — The implied scale ranges from 1.1 to 8.6 px/m across consecutive gate pairs. This either means the gates aren't at uniform 12 m spacing (possible), or some detected "gates" are wrong (false positives or wrong gate poles). Add a sanity check: if any gate pair implies a scale more than 2x different from its neighbors, flag it.

### Priority 2: Address the Trajectory–Gate Mismatch

**The problem**: The 2D trajectory (y = 145–283 px) never reaches the lower gates (y > 300 px). The skier is being tracked in the upper portion of the frame while the gates sit in the lower portion.

**Actions:**
1. **Overlay the 2D trajectory on actual video frames** as a diagnostic — verify whether the tracked trajectory actually follows the skier or is drifting.
2. **Check if the stabilization is shifting the trajectory** — The camera motion compensation may be incorrectly translating the trajectory upward. Compare raw (`trajectory_2d_raw`) vs stabilized (`trajectory_2d`) positions.
3. **If the skier genuinely doesn't reach the lower gates in the stabilized frame**, this is expected — the camera follows the skier, so in the stabilized reference frame, the skier stays roughly centered while the gates move. But then the scale mapping needs to account for where the *gates* are relative to the skier, not where the skier is relative to the original gate positions.

### Priority 3: Reduce Trajectory Noise (Fix G-forces and Turn Radii)

**The problem**: Even with Kalman smoothing, the trajectory has sudden jumps (up to 16.7 m) and extreme instantaneous direction changes.

**Actions:**
1. **Apply a post-hoc smoothing pass on the 3D trajectory before physics validation** — A Savitzky-Golay filter (order 3, window ~15–21 frames ≈ 0.5–0.7s) would remove high-frequency jitter while preserving the slalom turn shape.
2. **Clamp the max per-frame displacement** — At 30 fps and 70 km/h max slalom speed, the maximum reasonable displacement is ~0.65 m/frame. Any jump > 1.0 m between consecutive frames should be interpolated, not just clamped to the boundary.
3. **Compute physics metrics (G-force, turn radius) from a more heavily smoothed trajectory** — Use a separate, smoother version of the trajectory for dynamics computation. The position trajectory can remain detailed, but curvature-based quantities need more smoothing to be meaningful.

### Priority 4: Improve Gate Detection Confidence (Professor's CLAHE Suggestion)

**This is lower priority because it's a training/model issue, not a pipeline logic issue.** But it's cheap to try:

1. **Try CLAHE preprocessing** as the professor suggests — apply to the L-channel in LAB space before feeding frames to the YOLOv8 detector. If confidence scores improve by >0.1 on average, keep it.
2. **Longer-term: retrain the gate detector** with more diverse training data including far-field gates, partial occlusion, and shadow conditions.

### Priority 5: Handle the Detection Tail (Less Urgent)

The gate detection tail (final 25–43% of video with no gates) is a real issue but less critical than the others because:
- The skier tracker continues independently of gate detection
- The trajectory already covers these frames
- The main impact is that camera stabilization and dynamic scaling lack anchors in the tail

**Actions:**
1. **In the tail region, freeze the last known camera motion estimate and scale** rather than letting them drift or reset.
2. **Don't extend gate tracking via pure prediction** — predicting gate positions for 200+ frames with no observations would introduce more error than using a frozen estimate.

### NOT Recommended (Professor's suggestions to skip)

1. **"Add a regression head for vanishing point estimation"** — This is a significant architecture change to the YOLO model with uncertain payoff. The piecewise scale approach you already have is more practical.
2. **"Implement RANSAC for gate fitting"** — Already done in the code.
3. **"Implement Kalman Filter"** — Already done (RTS smoother, discipline-tuned).
4. **"Replace static detection with tracking"** — Already done (`TemporalGateTracker`).

---

## Summary Table

| Professor's Suggestion | Already Done? | Valid? | Action |
|----------------------|:---:|:---:|--------|
| A. Detection tail (Kalman predict) | Partially | Yes | Freeze last-known state in tail; don't predict indefinitely |
| B. CLAHE preprocessing | No | Worth trying | Low-cost experiment; apply to L-channel |
| C1. Don't hardcode 12 m | No | Minor | Keep 12 m but add gate-pair sanity checks |
| C2. Use rotation camera model | Code exists | **Yes, important** | **Switch `translation` → `affine`** |
| D. Kalman Filter for tracking | **Yes** | N/A | Skip |
| E. Dynamic homography | Partially | Yes | Ensure `DynamicScaleTransform` is active per-frame |
| F. RANSAC outlier rejection | **Yes** | N/A | Skip |

**The single highest-impact change is switching from `translation` to `affine` camera mode.** This directly addresses the root cause of trajectory distortion, which cascades into the G-force, turn radius, and acceleration problems.
