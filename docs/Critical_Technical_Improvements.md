# Critical Technical Improvements - Implementation Notes

**Based on expert feedback, these are MUST-HAVE additions to the plan:**

---

## 1. Domain-Specific Metrics Over ML Metrics

### ❌ Wrong Approach:
"My gate detector achieves 92% mAP!"

### ✅ Right Approach:
"My system estimates speeds within 5 km/h of TV overlay, detects gate positions within 50cm of known spacing, and produces G-forces in the realistic 2-4G range."

### Why This Matters:
- 95% detection accuracy means nothing if all detections are 2 meters off
- Coaches don't care about mAP - they care about "Is this useful?"
- Physics validation catches scaling/timing errors early

### Implementation:

```python
def validate_against_reality(trajectory_3d, fps=30):
    """
    Critical validation step
    """
    # 1. Speed validation (if TV overlay available)
    estimated_speeds = calculate_speeds(trajectory_3d, fps)
    # Compare to TV overlay or known course records
    # GS typically: 60-80 km/h
    # SL typically: 40-60 km/h
    
    # 2. G-force validation
    g_forces = calculate_g_forces(trajectory_3d, fps)
    max_g = max(g_forces)
    
    if max_g > 6:
        print(f"⚠ PHYSICS ERROR: {max_g:.1f}G is unrealistic")
        print("Check: FPS assumption, gate spacing, or homography")
        return False
    
    # 3. Turn radius validation
    radii = calculate_turn_radii(trajectory_3d)
    if min(radii) < 3:
        print(f"⚠ PHYSICS ERROR: {min(radii):.1f}m radius impossible")
        return False
    
    return True

# Run this on EVERY trajectory you extract
# If it fails, your 2D→3D transform is broken
```

**Test case for Week 4:**
- Find race video with TV speed overlay
- Extract trajectory
- Compare your calculated speed to TV speed
- Target: Within 5 km/h

---

## 2. Occlusion Handling (The Slalom Problem)

### The Problem:
Slalom racers intentionally hit gates (cross-blocking). Poles bend, snap back, disappear temporarily. Naive detection will:
- Lose gate tracking when racer hits it
- Get confused by overlapping gates
- Produce jumpy trajectories

### Solution: Temporal Consistency

```python
class TemporalGateTracker:
    """
    Remember gate positions even when they disappear
    """
    def __init__(self):
        self.gate_memory = {}
        self.missing_frames = {}
    
    def update(self, detected_gates, frame_num):
        """
        Update gates with temporal tracking
        """
        for gate_id, last_pos in self.gate_memory.items():
            # Try to match to detection
            matched = find_closest_gate(last_pos, detected_gates, max_dist=50)
            
            if matched:
                self.gate_memory[gate_id] = matched
                self.missing_frames[gate_id] = 0
            else:
                # Gate disappeared - use last known position
                self.missing_frames[gate_id] += 1
                
                if self.missing_frames[gate_id] < 15:  # 0.5 sec grace period
                    # Keep using last position
                    pass
                else:
                    # Gate permanently gone
                    del self.gate_memory[gate_id]
        
        return list(self.gate_memory.values())
```

### Testing Strategy:
1. Find aggressive slalom video (racer hitting many gates)
2. Count how many gates disappear during contact
3. Verify: Does trajectory stay smooth?
4. Target: System handles 20% temporary gate loss

**Add this to Week 6 as dedicated task.**

---

## 3. Robust Person Tracking (Multiple People Problem)

### The Problem:
Race videos contain:
- Spectators near course
- Course workers
- Other racers
- Camera operators

Naive "largest bounding box" approach WILL fail.

### Solution: Proper Tracking Algorithm

```python
class RobustSkierTracker:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
    
    def track_video(self, video_path):
        """
        Use YOLOv8's built-in ByteTrack
        """
        racer_id = None
        trajectory = []
        
        # ByteTrack maintains consistent IDs across frames
        results = self.model.track(
            source=video_path,
            classes=[0],  # person
            persist=True,
            tracker="bytetrack.yaml"
        )
        
        for frame_idx, result in enumerate(results):
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                
                # First frame: identify racer
                if racer_id is None:
                    # Heuristic: racer is centered in frame
                    # OR: Let user click on racer in first frame
                    racer_id = select_centered_person(boxes, frame.shape)
                
                # Track that specific person
                racer_box = boxes[ids == racer_id]
                if len(racer_box) > 0:
                    trajectory.append(extract_center(racer_box[0]))
        
        return trajectory
```

### Alternative: Temporal Consistency

If ByteTrack fails:
- Track position from previous frame
- Find box closest to previous position
- Reject jumps >100 pixels per frame
- Skier motion should be continuous

**Update Week 2 implementation with this approach.**

---

## 4. Class Imbalance in Training Data

### The Problem:
- Skier: Appears in 800 frames
- Gates: 50 gates × 5 frames each = 250 gate appearances
- **3:1 imbalance** → Model learns skier well, gates poorly

### Solutions:

**1. Balanced Frame Extraction:**
```python
def extract_frames_balanced(video_path):
    """
    Oversample frames with gates
    """
    cap = cv2.VideoCapture(video_path)
    
    # Dense sampling first 10 seconds (gates visible)
    # Sparse sampling rest of race
    
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num < 300:  # First 10 sec
            if frame_num % 10 == 0:  # Every 10th frame
                save_frame(frame)
        else:
            if frame_num % 30 == 0:  # Every 1 sec
                save_frame(frame)
        
        frame_num += 1
```

**2. Check Class Distribution:**
```python
# Before training, verify balance
gate_count = count_annotations('red_gate') + count_annotations('blue_gate')
skier_count = count_annotations('skier')

ratio = gate_count / (gate_count + skier_count)
print(f"Gate representation: {ratio*100:.1f}%")

if ratio < 0.3:
    print("⚠ Warning: Gates underrepresented")
    print("  Add more gate-heavy frames or use augmentation")
```

**3. Augmentation:**
```yaml
# In data.yaml
augment: true
copy_paste: 0.1  # Copy gates to different frames
mosaic: 1.0      # Mix multiple images
```

**Add this warning to Week 1 training section.**

---

## 5. Speed Estimation Validation

### Critical Test Case:

Many race broadcasts show real-time speed overlays. Use these as ground truth!

```python
def validate_speed_estimation():
    """
    Week 4 validation task
    """
    # 1. Find race video with TV speed overlay
    # Example: World Cup broadcasts often show speed
    
    # 2. Manually record displayed speeds at specific timestamps
    tv_speeds = {
        'frame_100': 68.5,  # km/h
        'frame_200': 72.1,
        'frame_300': 65.8
    }
    
    # 3. Calculate speeds from your trajectory
    trajectory = extract_trajectory('race_with_speed.mp4')
    calculated_speeds = calculate_speeds(trajectory, fps=30)
    
    # 4. Compare
    for frame, tv_speed in tv_speeds.items():
        frame_num = int(frame.split('_')[1])
        calc_speed = calculated_speeds[frame_num]
        error = abs(calc_speed - tv_speed)
        
        print(f"Frame {frame_num}: TV={tv_speed}, Calc={calc_speed:.1f}, Error={error:.1f} km/h")
    
    # 5. Target: Error < 5 km/h
```

**Add this as mandatory Week 4 validation.**

---

## 6. Physics Sanity Checks

### Must-Have Validation Function:

```python
def physics_sanity_check(trajectory_3d, fps=30):
    """
    Run on every trajectory to catch errors
    """
    issues = []
    
    # 1. Speed check
    speeds = calculate_speeds(trajectory_3d, fps)
    max_speed = max(speeds)
    if max_speed > 120:  # km/h
        issues.append(f"Speed {max_speed:.1f} km/h > downhill speed")
    if max_speed < 20:
        issues.append(f"Speed {max_speed:.1f} km/h too slow for racing")
    
    # 2. G-force check
    g_forces = []
    for i in range(1, len(trajectory_3d)-1):
        # Calculate centripetal acceleration
        v = speeds[i] / 3.6  # Convert to m/s
        r = turn_radius(trajectory_3d[i-1:i+2])
        g = (v**2 / r) / 9.81  # G-force
        g_forces.append(g)
    
    max_g = max(g_forces)
    if max_g > 6:
        issues.append(f"G-force {max_g:.1f}G exceeds human limits")
    
    # 3. Turn radius check
    radii = calculate_turn_radii(trajectory_3d)
    min_radius = min(radii)
    if min_radius < 3:  # meters
        issues.append(f"Turn radius {min_radius:.1f}m physically impossible")
    
    # 4. Trajectory smoothness
    # Large jumps indicate detection errors
    for i in range(1, len(trajectory_3d)):
        dx = trajectory_3d[i]['x'] - trajectory_3d[i-1]['x']
        dy = trajectory_3d[i]['y'] - trajectory_3d[i-1]['y']
        jump = (dx**2 + dy**2)**0.5
        
        if jump > 5:  # meters per frame at 30fps = 150 m/s = 540 km/h
            issues.append(f"Frame {i}: Jump of {jump:.1f}m is unrealistic")
    
    return issues

# ALWAYS run this
issues = physics_sanity_check(trajectory)
if issues:
    print("⚠ PHYSICS VALIDATION FAILED:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nLikely causes:")
    print("  1. Wrong gate spacing in homography")
    print("  2. Wrong FPS assumption")
    print("  3. Detection errors causing jumps")
```

**Add this to every trajectory extraction script.**

---

## 7. Test Dataset Requirements

### What Your Test Set MUST Include:

**Challenging conditions (not just easy cases):**
1. **Occlusions**
   - Racer hitting gates
   - Gates behind trees/structures
   - Tight gate sections

2. **Multiple people**
   - Spectators visible
   - Course workers
   - Other racers

3. **Varied visibility**
   - Bright sun (overexposure)
   - Fog/snow
   - Backlit conditions
   - Low light

4. **Different camera angles**
   - Side view
   - Head-on
   - From below course
   - Close vs. far

5. **Different disciplines**
   - Slalom (tight turns, gate contact)
   - Giant Slalom (higher speeds, wider turns)

**Don't just test on "perfect" videos - that's not reality.**

---

## Summary: Key Additions to Your Plan

### Week 1:
- ✅ Add class imbalance warning to training section
- ✅ Add balanced frame extraction strategy

### Week 2:
- ✅ Replace naive tracking with ByteTrack
- ✅ Add temporal consistency fallback

### Week 4:
- ✅ Add speed validation against TV overlays
- ✅ Add physics sanity checks as mandatory step

### Week 6:
- ✅ Add occlusion handling as dedicated task
- ✅ Test on aggressive slalom videos

### Ongoing:
- ✅ Run physics_sanity_check() on EVERY trajectory
- ✅ Report domain-specific metrics, not just ML metrics
- ✅ Test on challenging conditions, not just easy videos

---

## Final Reminder

**Good enough for Week 8 MVP:**
- 80% detection accuracy IF physics checks pass
- Occasional missed gates IF trajectory stays smooth
- Simple tracking IF it works on clean videos

**Not acceptable even for MVP:**
- Speeds 2x too fast/slow
- G-forces >6G
- Turn radii <3m
- Trajectories with >5m jumps per frame

**Physics validation catches errors early. Use it religiously.**

