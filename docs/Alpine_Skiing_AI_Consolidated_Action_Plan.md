# Alpine Skiing AI Project - Actionable Implementation Guide

**Goal:** Build an AI system that analyzes ski racing videos, extracts racing lines, and provides coaching feedback - for Stanford application in 4 years.

**Your current status:** Development environment ready, ready to start building.

**Project location:** `/Users/quan/Documents/personal/Stanford application project`

---

## Table of Contents

**PART 1: QUICK START** - Do This First (Weeks 1-8)
**PART 2: CORE SYSTEMS** - Build These (Months 3-12)
**PART 3: INTEGRATION & SCALE** - Deploy & Measure (Year 2-3)
**PART 4: STANFORD PREP** - Document & Apply (Year 4)
**APPENDICES** - Tools, Templates, Troubleshooting

---

# PART 1: QUICK START (Weeks 1-8)

## Goal: Get from Zero to Working Demo in 8 Weeks

This proves feasibility before investing months. At the end, you'll have:
- Working gate detection
- Skier trajectory extraction
- Simple 3D visualization
- Shareable demo video
- Code on GitHub

---

## Week 1: Data Collection & First Detection

### Monday-Tuesday (4 hours)

**Task: Collect training data**

```bash
# Create project structure
cd '/Users/quan/Documents/personal/Stanford application project'
mkdir -p data/{raw_videos,frames,annotations,test_videos}
mkdir -p models artifacts/outputs artifacts/training_results notebooks
```

**Download videos:**
1. YouTube search: "World Cup slalom 2024" or "World Cup giant slalom"
2. Target channels: FIS Ski, NBC Olympics, Eurosport
3. Download 20-30 videos
4. Save to `data/raw_videos/`

**Pick 3 "golden" test cases:**
- 1 with perfect visibility (sunny, good camera)
- 1 with challenging conditions (snow, fog, far camera)
- 1 with different angle (side view vs. front)
- Save these to `data/test_videos/`

**Checkpoint:** 20+ race videos downloaded

---

### Wednesday-Friday (6 hours)

**Task: Create initial training dataset**

**Extract frames from videos:**

```python
# scripts/extract_frames.py
import cv2
import os

def extract_frames(video_path, output_dir, fps=1):
    """Extract frames at specified FPS"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate / fps)
    
    count = 0
    saved = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        count += 1
    
    cap.release()
    return saved

# Extract from your test videos
for video in ['race1.mp4', 'race2.mp4', 'race3.mp4']:
    video_path = f'data/raw_videos/{video}'
    output_dir = f'data/frames/{video[:-4]}'
    count = extract_frames(video_path, output_dir, fps=1)
    print(f"Extracted {count} frames from {video}")
```

**Annotate images:**
1. Use Roboflow (free account)
2. Upload 100 frames (mix of 3 test videos)
3. Draw bounding boxes around:
   - Red gate poles → label: "red_gate"
   - Blue gate poles → label: "blue_gate"
4. Export in YOLOv8 format
5. Download to `data/annotations/`

**Annotation tips:**
- Include gates at various distances
- Include partially visible gates
- Include gates in different weather
- Don't worry about perfect boxes - good enough is OK

**Checkpoint:** 100 annotated images ready

---

### Weekend (6 hours)

**Task: Train first gate detection model**

```python
# notebooks/01_train_gate_detector.ipynb
from ultralytics import YOLO

# Load pre-trained YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Train on your dataset
results = model.train(
    data='data/annotations/data.yaml',  # Roboflow export includes this
    epochs=100,
    imgsz=640,
    batch=16,
    name='gate_detector_v1',
    patience=20,  # Early stopping
    save=True,
    plots=True
)

# Evaluate
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")
```

**⚠️ Important: Class Imbalance**

**Problem:** In a typical race video:
- **Skier:** Appears in ~800 frames (if video is 30 seconds)
- **Gates:** Only ~50 gates total, each visible in maybe 3-10 frames
- Result: **Gates are severely underrepresented in training data**

**Solutions:**

1. **Extract more gate-heavy frames:**
```python
# When extracting frames, prioritize frames with gates
def extract_frames_balanced(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # Extract every frame in first 10 seconds (gate setup)
    # Then 1 frame per second for rest
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dense sampling early (gates visible)
        if frame_count < 300:  # First 10 seconds
            if frame_count % 10 == 0:  # Every 10th frame
                save_frame(frame, output_dir, frame_count)
        else:
            if frame_count % 30 == 0:  # Every second
                save_frame(frame, output_dir, frame_count)
        
        frame_count += 1
```

2. **Augment gate images more:**
```yaml
# data.yaml - Add augmentation config
augment: true
flipud: 0.0  # Don't flip upside down
fliplr: 0.5  # Horizontal flip OK
mosaic: 1.0  # Mosaic augmentation
mixup: 0.1   # Mix images
copy_paste: 0.1  # Copy gates to different backgrounds
```

3. **Check class distribution:**
```python
# After annotation, verify balance
import json

gate_count = 0
skier_count = 0

for annotation_file in Path('data/annotations').glob('*.json'):
    with open(annotation_file) as f:
        data = json.load(f)
        for obj in data['objects']:
            if 'gate' in obj['label']:
                gate_count += 1
            elif 'skier' in obj['label']:
                skier_count += 1

print(f"Gates: {gate_count}, Skiers: {skier_count}")
print(f"Ratio: {gate_count / skier_count:.2f}")

# Target: At least 0.3 ratio (gates should be 30%+ of detections)
```

**While training (runs 1-2 hours):**
- Read YOLOv8 documentation
- Watch training curves on TensorBoard
- Plan Week 2 work

**Test on new video:**

```python
# Test detection
model = YOLO('runs/detect/gate_detector_v1/weights/best.pt')

# Process test video
results = model.predict(
    source='data/test_videos/race1.mp4',
    save=True,
    conf=0.5
)

# Video with detections saved to runs/detect/predict/
```

**Success criteria:**
- ✅ Model trains without errors
- ✅ mAP@0.5 > 0.60 (60% is OK for first try)
- ✅ Detections visible on test video (even if imperfect)

**If mAP < 0.60:**
- Annotate 50 more images
- Re-train
- Still failing? See "Get Unstuck Playbook" in Appendices

**Week 1 Deliverable:** First gate detection model trained

---

## Week 2: Skier Tracking

### Goal: Extract 2D trajectory of skier through race

**Monday-Tuesday (4 hours)**

**Task: Implement person tracking**

```python
# scripts/track_skier.py
from ultralytics import YOLO
import cv2
import numpy as np

class SkierTracker:
    """
    Improved tracking that handles multiple people in frame
    """
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        
    def track_video(self, video_path, use_builtin_tracking=True):
        """
        Track skier through video, return trajectory
        
        Args:
            use_builtin_tracking: Use YOLOv8's ByteTrack (recommended)
        """
        if use_builtin_tracking:
            return self._track_with_bytetrack(video_path)
        else:
            return self._track_with_temporal_consistency(video_path)
    
    def _track_with_bytetrack(self, video_path):
        """
        Use YOLOv8's built-in ByteTrack for robust tracking
        """
        trajectory = []
        racer_id = None
        
        # Track with persistence
        results = self.model.track(
            source=video_path,
            classes=[0],  # person class
            persist=True,  # Maintain IDs across frames
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        for frame_idx, result in enumerate(results):
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                
                # First frame: identify racer (assume centered or largest)
                if racer_id is None:
                    # Simple heuristic: racer is usually centered in frame
                    # Or you can add manual selection
                    racer_id = ids[0]
                
                # Find racer's box
                racer_indices = np.where(ids == racer_id)[0]
                if len(racer_indices) > 0:
                    box = boxes[racer_indices[0]]
                    trajectory.append({
                        'frame': frame_idx,
                        'x': float((box[0] + box[2]) / 2),
                        'y': float((box[1] + box[3]) / 2)
                    })
        
        return trajectory
    
    def _track_with_temporal_consistency(self, video_path):
        """
        Fallback: Track using temporal consistency
        Works when multiple people present
        """
        cap = cv2.VideoCapture(video_path)
        trajectory = []
        prev_position = None
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame, classes=[0], verbose=False)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                if prev_position:
                    # Find box closest to previous position
                    best_box = None
                    min_dist = float('inf')
                    
                    for box in boxes:
                        x = (box[0] + box[2]) / 2
                        y = (box[1] + box[3]) / 2
                        
                        # Distance from previous frame
                        dist = ((x - prev_position[0])**2 + 
                               (y - prev_position[1])**2)**0.5
                        
                        # Skier shouldn't jump >100 pixels/frame
                        if dist < 100 and dist < min_dist:
                            min_dist = dist
                            best_box = (x, y)
                    
                    if best_box:
                        prev_position = best_box
                        trajectory.append({
                            'frame': frame_num,
                            'x': float(best_box[0]),
                            'y': float(best_box[1])
                        })
                else:
                    # First frame: pick center-most person
                    frame_center_x = frame.shape[1] / 2
                    best_box = None
                    min_dist = float('inf')
                    
                    for box in boxes:
                        x = (box[0] + box[2]) / 2
                        y = (box[1] + box[3]) / 2
                        dist = abs(x - frame_center_x)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_box = (x, y)
                    
                    prev_position = best_box
                    trajectory.append({
                        'frame': frame_num,
                        'x': float(best_box[0]),
                        'y': float(best_box[1])
                    })
            
            frame_num += 1
        
        cap.release()
        return trajectory

# Test it
tracker = SkierTracker()
trajectory = tracker.track_video('data/test_videos/race1.mp4')
print(f"Tracked {len(trajectory)} positions")

# Save trajectory
import json
with open('artifacts/outputs/trajectory_2d.json', 'w') as f:
    json.dump(trajectory, f, indent=2)
```

**Important: This approach handles:**
- ✅ Multiple people in frame (spectators, course workers)
- ✅ Temporal consistency (tracks same person across frames)
- ✅ Graceful degradation (falls back if tracking lost)

**Checkpoint:** Can extract 2D positions of skier even with other people visible

---

### Wednesday-Thursday (4 hours)

**Task: Visualize trajectory**

```python
# scripts/visualize_trajectory.py
import matplotlib.pyplot as plt
import json

# Load trajectory
with open('artifacts/outputs/trajectory_2d.json', 'r') as f:
    trajectory = json.load(f)

# Plot trajectory
x_coords = [p['x'] for p in trajectory]
y_coords = [p['y'] for p in trajectory]

plt.figure(figsize=(10, 8))
plt.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
plt.scatter(x_coords[0], y_coords[0], c='green', s=100, label='Start')
plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='Finish')
plt.gca().invert_yaxis()  # Invert y-axis (image coordinates)
plt.xlabel('X Position (pixels)')
plt.ylabel('Y Position (pixels)')
plt.title('Skier Trajectory (2D)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('artifacts/outputs/trajectory_plot.png', dpi=150)
plt.show()
```

**Checkpoint:** Trajectory plot looks reasonable (smooth path down slope)

---

### Friday (2 hours)

**Task: Combine gate detection + skier tracking**

```python
# scripts/full_detection.py
from ultralytics import YOLO
import cv2
import json

class FullAnalyzer:
    def __init__(self, gate_model_path):
        self.gate_model = YOLO(gate_model_path)
        self.person_model = YOLO('yolov8n.pt')
    
    def analyze_video(self, video_path):
        """
        Detect gates and track skier in one pass
        """
        cap = cv2.VideoCapture(video_path)
        
        # Detect gates from first frame
        ret, first_frame = cap.read()
        gate_results = self.gate_model(first_frame)
        
        gates = []
        for box in gate_results[0].boxes:
            gates.append({
                'class': int(box.cls[0]),
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'conf': float(box.conf[0])
            })
        
        # Track skier through video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        trajectory = []
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            person_results = self.person_model(frame, classes=[0], verbose=False)
            
            if len(person_results[0].boxes) > 0:
                box = person_results[0].boxes[0].xyxy[0].cpu().numpy()
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                
                trajectory.append({
                    'frame': frame_num,
                    'x': float(x_center),
                    'y': float(y_center)
                })
            
            frame_num += 1
        
        cap.release()
        
        return {
            'gates': gates,
            'trajectory': trajectory,
            'video_info': {
                'path': video_path,
                'frames': frame_num
            }
        }

# Run analysis
analyzer = FullAnalyzer('runs/detect/gate_detector_v1/weights/best.pt')
results = analyzer.analyze_video('data/test_videos/race1.mp4')

# Save results
with open('artifacts/outputs/full_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Detected {len(results['gates'])} gates")
print(f"Tracked {len(results['trajectory'])} positions")
```

**Weekend: Rest or iterate on improvements**

**Week 2 Deliverable:** Combined gate + skier detection working

---

## Week 3: Simple 2D-to-3D Transformation

### Goal: Convert pixel coordinates to real-world meters

**Monday-Wednesday (6 hours)**

**Task: Implement homography transformation**

```python
# scripts/homography_transform.py
import cv2
import numpy as np
import json

class HomographyTransform:
    def __init__(self):
        self.H = None  # Homography matrix
        
    def calculate_from_gates(self, gates_2d, gate_spacing_m=12.0):
        """
        Calculate homography from detected gate positions
        
        gates_2d: List of gate positions in pixels [(x1,y1), (x2,y2), ...]
        gate_spacing_m: Real-world spacing between gates in meters
        """
        
        # Assume gates are roughly in a line down the slope
        # Create idealized 3D positions (bird's eye view)
        gates_3d = []
        for i, gate in enumerate(gates_2d):
            # Alternating left-right pattern
            x_3d = 0 if i % 2 == 0 else gate_spacing_m * 0.3  # Slight offset
            y_3d = i * gate_spacing_m
            gates_3d.append([x_3d, y_3d])
        
        # Need at least 4 points for homography
        if len(gates_2d) < 4:
            print("Warning: Need at least 4 gates for accurate homography")
            return None
        
        # Use first 4 gates
        pts_2d = np.float32(gates_2d[:4])
        pts_3d = np.float32(gates_3d[:4])
        
        # Calculate homography matrix
        self.H, status = cv2.findHomography(pts_2d, pts_3d, cv2.RANSAC)
        
        return self.H
    
    def transform_point(self, point_2d):
        """
        Transform a single 2D point to 3D coordinates
        """
        if self.H is None:
            raise ValueError("Must calculate homography first")
        
        # Convert to homogeneous coordinates
        pt = np.array([[point_2d[0], point_2d[1]]], dtype=np.float32)
        pt = pt.reshape(-1, 1, 2)
        
        # Apply homography
        transformed = cv2.perspectiveTransform(pt, self.H)
        
        return transformed[0][0]
    
    def transform_trajectory(self, trajectory_2d):
        """
        Transform entire trajectory to 3D
        """
        trajectory_3d = []
        
        for point in trajectory_2d:
            pt_3d = self.transform_point([point['x'], point['y']])
            trajectory_3d.append({
                'frame': point['frame'],
                'x': float(pt_3d[0]),
                'y': float(pt_3d[1])
            })
        
        return trajectory_3d

# Usage example
# Load previous results
with open('artifacts/outputs/full_analysis.json', 'r') as f:
    analysis = json.load(f)

# Extract gate centers
gate_centers = []
for gate in analysis['gates']:
    bbox = gate['bbox']
    x_center = (bbox[0] + bbox[2]) / 2
    y_bottom = bbox[3]  # Use bottom of bounding box (base of pole)
    gate_centers.append([x_center, y_bottom])

# Calculate homography
transformer = HomographyTransform()
transformer.calculate_from_gates(gate_centers, gate_spacing_m=12.0)

# Transform trajectory
trajectory_3d = transformer.transform_trajectory(analysis['trajectory'])

# Save 3D trajectory
with open('artifacts/outputs/trajectory_3d.json', 'w') as f:
    json.dump(trajectory_3d, f, indent=2)

print(f"Transformed {len(trajectory_3d)} points to 3D coordinates")
```

**Checkpoint:** 3D coordinates generated (even if not perfect)

---

### Thursday-Friday (4 hours)

**Task: Visualize bird's-eye view**

```python
# scripts/visualize_3d.py
import matplotlib.pyplot as plt
import json

# Load 3D trajectory
with open('artifacts/outputs/trajectory_3d.json', 'r') as f:
    trajectory_3d = json.load(f)

# Extract coordinates
x_coords = [p['x'] for p in trajectory_3d]
y_coords = [p['y'] for p in trajectory_3d]

# Create bird's-eye view plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: 3D bird's-eye trajectory
ax1.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
ax1.scatter(x_coords[0], y_coords[0], c='green', s=150, label='Start', zorder=5)
ax1.scatter(x_coords[-1], y_coords[-1], c='red', s=150, label='Finish', zorder=5)
ax1.set_xlabel('X Position (meters)', fontsize=12)
ax1.set_ylabel('Y Position (meters)', fontsize=12)
ax1.set_title('Skier Trajectory - Bird\'s Eye View', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Compare 2D vs 3D
with open('artifacts/outputs/trajectory_2d.json', 'r') as f:
    trajectory_2d = json.load(f)

x_2d = [p['x'] for p in trajectory_2d]
y_2d = [p['y'] for p in trajectory_2d]

ax2.plot(x_2d, y_2d, 'r-', linewidth=2, alpha=0.7, label='2D (pixels)')
ax2.invert_yaxis()
ax2.set_xlabel('X (pixels)', fontsize=12)
ax2.set_ylabel('Y (pixels)', fontsize=12)
ax2.set_title('Original 2D Trajectory', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/outputs/trajectory_comparison.png', dpi=150)
plt.show()
```

**Checkpoint:** Bird's-eye view looks like a skiing path (roughly straight with turns)

**Week 3 Deliverable:** 3D trajectory extraction working

---

## Week 4: End-to-End Pipeline

### Goal: Process any video automatically

**Monday-Wednesday (6 hours)**

**Task: Create automated pipeline**

```python
# scripts/pipeline.py
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class SkiRacingPipeline:
    """
    End-to-end pipeline: video → gates + 3D trajectory
    """
    
    def __init__(self, gate_model_path):
        self.gate_model = YOLO(gate_model_path)
        self.person_model = YOLO('yolov8n.pt')
        self.H = None
    
    def process_video(self, video_path, output_dir='artifacts/outputs', gate_spacing=12.0):
        """
        Main processing function
        """
        print(f"Processing {video_path}...")
        
        # Step 1: Detect gates
        print("  [1/4] Detecting gates...")
        gates = self._detect_gates(video_path)
        print(f"         Found {len(gates)} gates")
        
        # Step 2: Calculate homography
        print("  [2/4] Calculating perspective transform...")
        self._calculate_homography(gates, gate_spacing)
        
        # Step 3: Track skier
        print("  [3/4] Tracking skier...")
        trajectory_2d = self._track_skier(video_path)
        print(f"         Tracked {len(trajectory_2d)} positions")
        
        # Step 4: Transform to 3D
        print("  [4/4] Transforming to 3D coordinates...")
        trajectory_3d = self._transform_trajectory(trajectory_2d)
        
        # Save results
        output_path = Path(output_dir) / f"{Path(video_path).stem}_analysis.json"
        results = {
            'video': str(video_path),
            'gates': gates,
            'trajectory_2d': trajectory_2d,
            'trajectory_3d': trajectory_3d
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
        return results
    
    def _detect_gates(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Could not read video")
        
        results = self.gate_model(frame)
        gates = []
        
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            gates.append({
                'class': int(box.cls[0]),
                'center_x': float((bbox[0] + bbox[2]) / 2),
                'base_y': float(bbox[3]),
                'confidence': float(box.conf[0])
            })
        
        # Sort gates by Y position (top to bottom)
        gates.sort(key=lambda g: g['base_y'])
        return gates
    
    def _calculate_homography(self, gates, gate_spacing):
        if len(gates) < 4:
            print("Warning: Less than 4 gates detected, using default transform")
            self.H = np.eye(3)
            return
        
        # Extract gate positions
        pts_2d = [[g['center_x'], g['base_y']] for g in gates[:8]]
        
        # Create idealized 3D positions
        pts_3d = []
        for i in range(len(pts_2d)):
            x = (i % 2) * 3.0  # Alternating pattern
            y = i * gate_spacing
            pts_3d.append([x, y])
        
        pts_2d = np.float32(pts_2d[:4])
        pts_3d = np.float32(pts_3d[:4])
        
        self.H, _ = cv2.findHomography(pts_2d, pts_3d, cv2.RANSAC)
    
    def _track_skier(self, video_path):
        cap = cv2.VideoCapture(video_path)
        trajectory = []
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.person_model(frame, classes=[0], verbose=False)
            
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0].cpu().numpy()
                x = float((box[0] + box[2]) / 2)
                y = float((box[1] + box[3]) / 2)
                
                trajectory.append({
                    'frame': frame_num,
                    'x': x,
                    'y': y
                })
            
            frame_num += 1
        
        cap.release()
        return trajectory
    
    def _transform_trajectory(self, trajectory_2d):
        if self.H is None:
            raise ValueError("Homography not calculated")
        
        trajectory_3d = []
        for point in trajectory_2d:
            pt = np.array([[point['x'], point['y']]], dtype=np.float32).reshape(-1, 1, 2)
            pt_3d = cv2.perspectiveTransform(pt, self.H)[0][0]
            
            trajectory_3d.append({
                'frame': point['frame'],
                'x': float(pt_3d[0]),
                'y': float(pt_3d[1])
            })
        
        return trajectory_3d

# Usage
if __name__ == '__main__':
    pipeline = SkiRacingPipeline('runs/detect/gate_detector_v1/weights/best.pt')
    
    # Process all test videos
    test_videos = Path('data/test_videos').glob('*.mp4')
    
    for video in test_videos:
        try:
            results = pipeline.process_video(str(video))
            print(f"✓ Successfully processed {video.name}\n")
        except Exception as e:
            print(f"✗ Error processing {video.name}: {e}\n")
```

**Checkpoint:** Can process any video with single command

---

### Thursday-Friday (4 hours)

**Task: Create visualization script**

```python
# scripts/create_demo.py
import cv2
import json
import numpy as np
from pathlib import Path

def create_demo_video(video_path, analysis_path, output_path):
    """
    Create demo video with trajectory overlay
    """
    # Load analysis results
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Extract trajectory for overlay
    trajectory = {p['frame']: (int(p['x']), int(p['y'])) 
                  for p in analysis['trajectory_2d']}
    
    # Draw gates on first frame
    gate_positions = [(int(g['center_x']), int(g['base_y'])) 
                      for g in analysis['gates']]
    
    frame_num = 0
    trail_points = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw gates (faded)
        for i, (gx, gy) in enumerate(gate_positions):
            color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
            cv2.circle(frame, (gx, gy), 10, color, -1)
            cv2.circle(frame, (gx, gy), 15, color, 2)
        
        # Draw trajectory trail
        if frame_num in trajectory:
            trail_points.append(trajectory[frame_num])
        
        for i in range(1, len(trail_points)):
            cv2.line(frame, trail_points[i-1], trail_points[i], 
                    (0, 255, 255), 3)
        
        # Draw current position
        if frame_num in trajectory:
            pos = trajectory[frame_num]
            cv2.circle(frame, pos, 20, (0, 255, 0), -1)
            cv2.circle(frame, pos, 25, (0, 255, 0), 3)
        
        # Add text overlay
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Gates: {len(gate_positions)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_num += 1
    
    cap.release()
    out.release()
    print(f"✓ Demo video saved to {output_path}")

# Create demo for first test video
create_demo_video(
    'data/test_videos/race1.mp4',
    'artifacts/outputs/race1_analysis.json',
    'artifacts/outputs/race1_demo.mp4'
)
```

**Weekend (4 hours)**

**Task: Create summary visualization**

```python
# scripts/create_summary.py
import matplotlib.pyplot as plt
import json
from pathlib import Path

def create_summary_figure(analysis_path, output_path):
    """
    Create 4-panel summary figure
    """
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: 2D trajectory
    ax1 = plt.subplot(2, 2, 1)
    x_2d = [p['x'] for p in analysis['trajectory_2d']]
    y_2d = [p['y'] for p in analysis['trajectory_2d']]
    ax1.plot(x_2d, y_2d, 'b-', linewidth=2)
    ax1.scatter(x_2d[0], y_2d[0], c='green', s=150, label='Start')
    ax1.scatter(x_2d[-1], y_2d[-1], c='red', s=150, label='Finish')
    ax1.invert_yaxis()
    ax1.set_title('2D Trajectory (Pixels)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: 3D bird's-eye view
    ax2 = plt.subplot(2, 2, 2)
    x_3d = [p['x'] for p in analysis['trajectory_3d']]
    y_3d = [p['y'] for p in analysis['trajectory_3d']]
    ax2.plot(x_3d, y_3d, 'r-', linewidth=2)
    ax2.scatter(x_3d[0], y_3d[0], c='green', s=150, label='Start')
    ax2.scatter(x_3d[-1], y_3d[-1], c='red', s=150, label='Finish')
    ax2.set_title('3D Trajectory (Meters)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Panel 3: Speed profile
    ax3 = plt.subplot(2, 2, 3)
    # Calculate speeds from 3D positions
    speeds = []
    for i in range(1, len(analysis['trajectory_3d'])):
        p1 = analysis['trajectory_3d'][i-1]
        p2 = analysis['trajectory_3d'][i]
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        distance = (dx**2 + dy**2)**0.5
        # Assume 30 fps (adjust based on your video)
        speed = distance * 30  # meters/second
        speeds.append(speed * 3.6)  # Convert to km/h
    
    ax3.plot(speeds, 'g-', linewidth=2)
    ax3.set_title('Speed Profile', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Speed (km/h)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Panel 4: Statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = f"""
    ANALYSIS SUMMARY
    ━━━━━━━━━━━━━━━━━━━━
    
    Gates Detected: {len(analysis['gates'])}
    
    Trajectory Points: {len(analysis['trajectory_2d'])}
    
    Total Distance: {sum([((analysis['trajectory_3d'][i]['x'] - analysis['trajectory_3d'][i-1]['x'])**2 + 
                           (analysis['trajectory_3d'][i]['y'] - analysis['trajectory_3d'][i-1]['y'])**2)**0.5 
                          for i in range(1, len(analysis['trajectory_3d']))]) :.1f} m
    
    Avg Speed: {sum(speeds)/len(speeds) if speeds else 0:.1f} km/h
    
    Max Speed: {max(speeds) if speeds else 0:.1f} km/h
    
    Video: {Path(analysis['video']).name}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, 
            family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Summary saved to {output_path}")

# Create summary
create_summary_figure(
    'artifacts/outputs/race1_analysis.json',
    'artifacts/outputs/race1_summary.png'
)
```

**Week 4 Deliverable:** 
- Automated pipeline script
- Demo video with trajectory overlay
- Summary visualization

---

## Week 5-6: Data Expansion & Robustness

### Goal: Handle real-world challenges

**Week 5: Expand Dataset**
1. Annotate 500 more images (aim for 1000 total)
2. **Critical: Include occlusion cases**
   - Gates bending/snapping during contact
   - Skier hitting gates (poles temporarily disappear)
   - Gates partially hidden behind skier
   - Multiple overlapping gates in frame

**Week 6: Occlusion Handling**

**The Slalom Problem:**
In slalom, racers intentionally hit gates (cross-blocking technique). When poles bend and snap back:
- Gate position changes frame-to-frame
- Pole may temporarily disappear
- Multiple gates overlap in tight sections
- Skier occludes gates during pass

**Implementation approach:**

```python
# Strategy 1: Temporal consistency
class TemporalGateTracker:
    """
    Track gates across frames even when temporarily occluded
    """
    def __init__(self):
        self.gate_memory = {}  # Store last known positions
        self.missing_frames = {}  # Track how long gate has been missing
        
    def update(self, detected_gates, frame_num):
        """
        Update gate positions with temporal consistency
        """
        # Match detected gates to tracked gates
        for gate_id, tracked_gate in self.gate_memory.items():
            matched = self.find_closest_detection(tracked_gate, detected_gates)
            
            if matched:
                # Update position
                self.gate_memory[gate_id] = matched
                self.missing_frames[gate_id] = 0
            else:
                # Gate not detected - use predicted position
                self.missing_frames[gate_id] += 1
                
                if self.missing_frames[gate_id] < 10:  # Grace period
                    # Assume gate hasn't moved much
                    predicted_pos = self.predict_position(gate_id)
                    self.gate_memory[gate_id] = predicted_pos
                else:
                    # Gate likely disappeared permanently
                    del self.gate_memory[gate_id]
        
        return self.gate_memory.values()
    
    def predict_position(self, gate_id):
        """
        Predict gate position when not detected
        Simple approach: assume gates don't move
        Better approach: model gate deformation physics
        """
        return self.gate_memory[gate_id]  # Last known position

# Strategy 2: Use first frame as reference
def initialize_gate_positions(first_frame, model):
    """
    Detect all gates in first frame before race starts
    Use these as ground truth positions
    """
    results = model(first_frame, conf=0.3)
    
    gate_positions = []
    for box in results[0].boxes:
        gate_positions.append({
            'id': len(gate_positions),
            'reference_position': box.xyxy[0].cpu().numpy(),
            'confidence': 'high'  # From clean first frame
        })
    
    return gate_positions

# Strategy 3: Accept imperfect detection
def handle_missing_gates(detected_gates, expected_count=50):
    """
    It's OK if some gates aren't detected during race
    Focus on getting skier trajectory right
    """
    if len(detected_gates) < expected_count * 0.7:
        print(f"⚠ Only detected {len(detected_gates)}/{expected_count} gates")
        print("  This is OK for MVP - we can interpolate missing gates")
    
    # Interpolate positions of missing gates
    # Based on detected neighbors
    return interpolate_gates(detected_gates, expected_count)
```

**Testing occlusion robustness:**
1. Find slalom videos where racer aggressively hits gates
2. Manually count how many gates disappeared during contact
3. Measure: Does your system still produce reasonable trajectory?
4. Target: Trajectory remains smooth even when 20% of gates temporarily undetected

**Deliverable:** System handles occlusions gracefully, doesn't crash when gates disappear

**Improvement loop:**

```python
# scripts/evaluate_accuracy.py
import json
from pathlib import Path

def evaluate_gate_detection(predicted_gates, ground_truth_gates, threshold=50):
    """
    Calculate precision/recall for gate detection
    threshold: pixels - gates within this distance count as correct
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched = set()
    
    for pred in predicted_gates:
        px, py = pred['center_x'], pred['base_y']
        
        # Find closest ground truth gate
        min_dist = float('inf')
        closest_idx = -1
        
        for i, gt in enumerate(ground_truth_gates):
            if i in matched:
                continue
            gx, gy = gt['x'], gt['y']
            dist = ((px - gx)**2 + (py - gy)**2)**0.5
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if min_dist < threshold:
            true_positives += 1
            matched.add(closest_idx)
        else:
            false_positives += 1
    
    false_negatives = len(ground_truth_gates) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

# Test your model
# You need to manually mark ground truth gates for test set
# Save as JSON with format: [{'x': 123, 'y': 456}, ...]
```

**Target metrics by end of Week 6:**

**ML Metrics (for you):**
- Gate detection: >85% F1 score
- Trajectory tracking: >90% of frames detected

**Domain-Specific Metrics (what actually matters):**
- Speed estimation accuracy: Test on videos with TV overlay showing speed
  - Target: Within 5 km/h of displayed speed
- Turn radius estimation: Compare to known gate spacing
  - Target: Within 10% of calculated values
- Gate position accuracy: Measure against known gate spacing
  - Target: <50cm error on gate positions
- Physically possible trajectories: Calculate G-forces from trajectory
  - Target: Peak G-forces 2-4G (typical for racing)
  - If >6G → your scale/timing is wrong

**Why domain metrics matter:**
- 95% gate detection means nothing if all detections are 2 meters off
- Perfect tracking means nothing if speeds are 2x reality
- Coaches care about "Is this line analysis useful?" not "What's your mAP?"

**Validation approach:**
```python
def validate_physics(trajectory_3d, fps=30):
    """
    Check if trajectory is physically plausible
    """
    speeds = calculate_speed(trajectory_3d, fps)
    turn_radii = calculate_turn_radii(trajectory_3d)
    g_forces = calculate_g_forces(speeds, turn_radii)
    
    # Sanity checks
    max_speed = max(speeds)
    max_g = max(g_forces)
    
    issues = []
    
    if max_speed > 120:  # km/h - faster than downhill racers
        issues.append(f"Unrealistic speed: {max_speed:.1f} km/h")
    
    if max_g > 6:  # G-forces beyond human tolerance
        issues.append(f"Unrealistic G-force: {max_g:.1f}G")
    
    if min(turn_radii) < 3:  # meters - tighter than physically possible
        issues.append(f"Unrealistic turn radius: {min(turn_radii):.1f}m")
    
    return issues

# Run on every trajectory
issues = validate_physics(trajectory_3d)
if issues:
    print("⚠ Physics validation failed:")
    for issue in issues:
        print(f"  - {issue}")
    print("Check: homography matrix, FPS assumption, or gate spacing")
```

**Week 6 deliverable:** Not just "model trained" but "model produces physically plausible results validated against known speeds"

---

## Week 7-8: Polish & Documentation

### Goal: Make this shareable

**Week 7 Tasks:**

**1. Clean up code:**

```bash
# Organize into proper structure
Stanford application project/
├── data/
├── models/
├── artifacts/
│   ├── outputs/
│   └── training_results/
├── runs/
├── tools/
│   └── video_scraper/
├── ski_racing/              # ← NEW: Make it a package
│   ├── __init__.py
│   ├── detection.py         # Gate & skier detection
│   ├── tracking.py          # Trajectory extraction
│   ├── transform.py         # Homography
│   └── pipeline.py          # End-to-end
├── scripts/
│   ├── train_detector.py
│   ├── process_video.py
│   └── evaluate.py
├── notebooks/
├── tests/                   # ← NEW: Add tests
│   └── test_pipeline.py
├── docs/                    # ← NEW: Guides, plans, reports
├── README.md                # ← UPDATE
├── requirements.txt         # ← CREATE
└── .gitignore
```

**2. Write README:**

```markdown
# Alpine Ski Racing AI - Trajectory Analysis

Automated system for analyzing alpine ski racing videos using computer vision.

## Features
- Automatic gate detection using YOLOv8
- Skier trajectory tracking
- 2D-to-3D perspective transformation
- Speed and line analysis

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process a video
python scripts/process_video.py data/test_videos/race1.mp4
```

## Results
- Gate detection: 87% F1 score
- Trajectory tracking: 94% frame coverage
- Processing time: ~2 minutes per video

## Example Output
![Demo](artifacts/outputs/demo_screenshot.png)

## Technical Details
[Brief explanation of approach]

## Future Work
- Reinforcement learning for optimal line generation
- Real-time processing
- Mobile app deployment

## Author
[Your Name] - High school AI researcher
```

**3. Create requirements.txt:**

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0
```

**Week 8 Tasks:**

**1. Record demo video (2-3 minutes):**
- Show input video
- Run pipeline command
- Show output visualizations
- Explain what it does
- Upload to YouTube

**2. Write blog post:**
```markdown
# Building an AI System to Analyze Ski Racing Lines

I've spent the last 8 weeks building a computer vision system that 
automatically analyzes alpine ski racing videos...

[Explain problem, approach, challenges, results]
[Include code snippets, visualizations]
[Share lessons learned]
```

**3. Push to GitHub:**
```bash
git add .
git commit -m "Initial release - v1.0 MVP"
git tag v1.0
git push origin main --tags
```

**4. Share your work:**
- Post demo video on Twitter/LinkedIn
- Share blog post on relevant subreddits (r/computervision, r/MachineLearning)
- Send to skiing coaches you know
- Get feedback!

---

## Week 8 Final Deliverables

**You should have:**
- ✅ Working CV pipeline (gate detection + trajectory extraction)
- ✅ 85%+ gate detection accuracy
- ✅ Automated end-to-end processing
- ✅ Clean, documented code on GitHub
- ✅ Demo video
- ✅ Blog post or technical write-up
- ✅ 3-5 processed example videos

**What you can tell professors:**
> "I built a computer vision system for analyzing ski racing videos. 
> It uses YOLOv8 for gate detection and homography-based 3D reconstruction. 
> I've processed 30+ race videos with 87% detection accuracy. 
> Here's my GitHub: [link]
> Here's a demo: [link]"

**This is a STRONG foundation for professor outreach.**

---

# PART 2: CORE SYSTEMS (Months 3-12)

Now that you have a working MVP, you can:
1. Work independently on improvements
2. Contact professors with something to show
3. Decide which direction to pursue next

---

## Decision Point 1: What to Build Next?

**You have three main paths:**

### Path A: Deep Dive on Computer Vision
**Focus:** Make CV system production-quality
**Good if:** You want to publish CV research
**Timeline:** 3-6 months
**Outcome:** Dataset paper + CV improvements

**Tasks:**
- Expand dataset to 2000+ images
- Handle challenging conditions (fog, snow, low light)
- Multi-camera support
- Real-time processing optimization
- Publish dataset + benchmark

### Path B: Add Reinforcement Learning
**Focus:** Train RL agent for optimal lines
**Good if:** You want to explore RL deeply
**Timeline:** 6-12 months (high variance!)
**Outcome:** RL paper comparing AI vs. humans

**Tasks:**
- Build Unity physics simulator
- Validate physics against real data
- Train RL agents (PPO)
- Compare to pro skier trajectories
- Identify novel strategies

### Path C: Build User-Facing App
**Focus:** Deploy to real coaches/athletes
**Good if:** You want measurable impact
**Timeline:** 3-6 months
**Outcome:** 50+ teams using your system

**Tasks:**
- Mobile app development (React Native)
- Cloud processing backend
- User testing with coaches
- Iteration based on feedback
- Impact measurement study

---

## Recommended: Hybrid Approach

**Months 3-6: Path A + Path C**
- Improve CV quality while building simple app
- Get real users providing feedback
- Build credibility for professor outreach

**Months 7-12: Path B (if professor partnership)**
- If a professor is interested in RL → go deep on that
- If no professor → continue A + C for impact

**Months 13-24: Focus based on what's working**
- If users love it → scale deployment (Path C)
- If research is exciting → publish papers (Path A or B)
- If professor collaboration → follow their guidance

---

## Month 3-6: CV Improvements + Simple App

### Month 3: Dataset Expansion

**Goal:** Create publishable dataset

**Week 1-2:**
- Annotate 500 more images (aim for 1000 total)
- Include variety:
  - Different weather (sun, clouds, fog, snow)
  - Different times of day
  - Different camera angles
  - Different disciplines (SL, GS)
  - Different gate setups

**Week 3-4:**
- Re-train model on full dataset
- Target: >90% mAP@0.5
- Benchmark against test set
- Document improvement

**Deliverable:** 
- 1000+ annotated image dataset
- Improved detector model
- Performance comparison report

---

### Month 4: Challenging Conditions

**Goal:** Handle real-world edge cases

**Common failure modes to address:**

```python
# scripts/analyze_failures.py

def analyze_detection_failures(test_results):
    """
    Categorize where detection fails
    """
    
    failures = {
        'distant_gates': [],      # Gates far from camera
        'occluded': [],           # Partially hidden
        'motion_blur': [],        # Fast camera movement
        'poor_lighting': [],      # Dark, backlit, glare
        'snow_gates': [],         # Gates covered in snow
        'similar_colors': []      # Background matches gate color
    }
    
    # Manually categorize failure cases
    # This helps you know what to focus on
    
    return failures
```

**Improvements to try:**

1. **For distant gates:**
   - Train on multiple image scales
   - Use larger model (YOLOv8m or YOLOv8l)
   - Pre-process with super-resolution

2. **For motion blur:**
   - Add motion-blurred training images
   - Use deblurring pre-processing
   - Multi-frame fusion

3. **For poor lighting:**
   - Augment training data with brightness/contrast variations
   - Use histogram equalization
   - Train separate detector for low-light

**Deliverable:** Robustness improvements documented

---

### Month 5-6: Simple Web App

**Goal:** Let coaches upload videos and get analysis

**Tech stack (keep it simple):**
- Frontend: Basic HTML/JavaScript (or use vibe coding to build React)
- Backend: Python Flask/FastAPI
- Processing: Your existing pipeline
- Storage: Local filesystem (for now)

**Week 1-2: Backend API**

```python
# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from ski_racing.pipeline import SkiRacingPipeline

app = FastAPI()
pipeline = SkiRacingPipeline('models/gate_detector_best.pt')

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video and process it
    """
    # Save uploaded file
    upload_path = Path("uploads") / file.filename
    upload_path.parent.mkdir(exist_ok=True)
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process video
    results = pipeline.process_video(str(upload_path))
    
    return JSONResponse({
        "status": "success",
        "gates_detected": len(results['gates']),
        "trajectory_points": len(results['trajectory_3d']),
        "results": results
    })

@app.get("/")
async def root():
    return {"message": "Ski Racing Analysis API"}

# Run with: uvicorn app:app --reload
```

**Week 3-4: Simple Frontend**

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Ski Racing Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        .upload-box {
            border: 2px dashed #ccc;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
        }
        #results {
            margin-top: 30px;
        }
        .loading {
            display: none;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>🎿 Ski Racing Analysis</h1>
    <p>Upload a ski racing video to analyze gates and trajectory</p>
    
    <div class="upload-box">
        <input type="file" id="videoInput" accept="video/*">
        <button onclick="uploadVideo()">Analyze Video</button>
    </div>
    
    <div class="loading" id="loading">
        <p>Processing video... This may take 1-2 minutes.</p>
    </div>
    
    <div id="results"></div>
    
    <script>
        async function uploadVideo() {
            const fileInput = document.getElementById('videoInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a video file');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('http://localhost:8000/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';
                
                document.getElementById('results').innerHTML = `
                    <h2>Analysis Complete!</h2>
                    <p><strong>Gates Detected:</strong> ${data.gates_detected}</p>
                    <p><strong>Trajectory Points:</strong> ${data.trajectory_points}</p>
                    <pre>${JSON.stringify(data.results, null, 2)}</pre>
                `;
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Error processing video: ' + error);
            }
        }
    </script>
</body>
</html>
```

**Use vibe coding to make this prettier!**

Ask Claude/vibe coding to:
- Add file drag-and-drop
- Show progress bar
- Display trajectory visualization
- Add download button for results
- Make it mobile-responsive

**Deliverable:** Working web app (even if basic)

---

## Month 7-12: Choose Your Path

By now you should have:
- ✅ Strong CV pipeline
- ✅ Working web demo
- ✅ GitHub portfolio
- ✅ Demo video
- ✅ Maybe contacted some professors

**Decision time: What next?**

### If Professor Partnership Secured:
→ Follow their research direction
→ Likely: RL optimal line generation
→ Timeline: Their guidance + your implementation
→ Outcome: Research paper with co-author

### If No Professor Partnership Yet:
→ Continue Path C (deployment + impact)
→ Focus on real-world users
→ Build independent research story
→ Keep trying professor outreach

---

## Month 7-12 Option A: RL Development

**Only if:**
- You have professor advising on approach
- OR you're very confident in RL abilities
- OR you have 6+ months to experiment

**High-level roadmap:**

**Month 7-8: Unity Setup**
- Build basic skiing environment
- Implement physics (gravity, friction, drag)
- Validate against published data
- Ensure simulation runs fast enough

**Month 9-10: RL Training**
- Define observation space
- Define action space
- Implement reward function
- Train first agents
- Debug convergence issues

**Month 11-12: Comparison Study**
- Extract pro skier trajectories
- Compare RL vs. human
- Analyze differences
- Write first paper draft

**See full implementation details in Part 2 Appendix**

---

## Month 7-12 Option B: Deployment & Impact

**Recommended if going independent:**

**Month 7-8: Beta Testing**
- Contact 5-10 ski coaches
- Offer free analysis
- Collect feedback
- Iterate on features they actually want

**Month 9-10: User Study**
- Partner with 3-5 teams
- Measure usage metrics
- Collect testimonials
- Document impact

**Month 11-12: Scale**
- Improve based on feedback
- Deploy to 20+ teams
- Start measuring performance improvements
- Prepare for Year 2 impact study

---

# PART 3: INTEGRATION & SCALE (Year 2-3)

## Year 2 Goals

**Minimum (Acceptable for Stanford):**
- ✅ 20+ teams using system
- ✅ 1 paper submitted (dataset OR impact study)
- ✅ Quantified feedback from coaches
- ✅ Continuous improvements to system

**Target (Strong for Stanford):**
- ✅ 50+ teams using system
- ✅ 1 paper published or accepted
- ✅ Measurable performance improvements (2%+)
- ✅ Professor letter of rec (if partnership)

**Stretch (Exceptional for Stanford):**
- ✅ 100+ teams using system
- ✅ 2 papers published
- ✅ Controlled impact study with statistical significance
- ✅ Media coverage or awards

---

## Year 2 Quarter-by-Quarter Plan

### Q1 (Months 13-15): Identify Research Contribution

**By now you should have discovered interesting patterns/problems.**

**Questions to ask yourself:**
- What surprised me while building this?
- What did I discover that contradicts common coaching wisdom?
- What technical challenge did I solve in a novel way?
- What patterns do I see in the data that no one talks about?
- What feedback from coaches revealed unexpected needs?

**Possible research directions** (don't commit yet - explore first):
- Dataset contribution (if you built something substantial and reusable)
- Methods contribution (if you solved a technical problem in a new way)
- Applications contribution (if you discovered something about skiing)
- Systems contribution (if your deployment revealed interesting insights)

**Writing tips:**
- Start by documenting what you DID, not what you "should" write about
- Your actual work will reveal the research contribution
- Get feedback from professor (if you have one) on what's most interesting
- Read 5-10 papers in the area you're interested in
- Find the gap YOUR work fills

---

### Q2 (Months 16-18): Beta Deployment

**Goal:** Get 20-30 teams actively using system

**Outreach strategy:**

```markdown
## Email to Ski Coaches

Subject: Free AI Video Analysis Tool for Your Ski Racing Team

Dear Coach [Name],

I'm a high school student who has developed an AI-powered tool for analyzing ski racing videos. The system:

- Automatically detects gates in race footage
- Tracks skier trajectory and converts to bird's-eye view
- Calculates speed profiles and line efficiency
- Generates visual reports coaches can share with athletes

I'm looking for 20-30 teams to beta test the tool this season at no cost. 
In exchange, I'm hoping to:
- Get feedback on what features are most valuable
- Collect anonymous usage data to improve the system
- Potentially include results in a research study (with your permission)

The tool works with standard smartphone video - no special equipment needed.

Would your team be interested? I'd be happy to provide a demo.

Demo video: [YouTube link]
Example analysis: [Link to sample output]

Best,
[Your Name]
[Email]
[Phone]
```

**Onboarding process:**
1. Schedule demo call
2. Send access credentials
3. Process their first 3 videos for free
4. Weekly check-in for first month
5. Monthly survey for feedback

---

### Q3 (Months 19-21): Iteration Based on Feedback

**Common feedback themes to expect:**

**"Videos take too long to process"**
→ Optimize pipeline
→ Add progress indicators
→ Consider cloud processing

**"I want to compare multiple runs"**
→ Add comparison mode
→ Show side-by-side trajectories

**"Hard to see where time is lost"**
→ Add gate-by-gate time analysis
→ Highlight sections where racer was slow

**"Can you compare to pro skiers?"**
→ Create database of pro trajectories
→ Add "compare to Mikaela Shiffrin" feature

**Rapid iteration using vibe coding:**
- Use Claude to implement requested features
- Deploy updates weekly
- Track which features get used most

---

### Q4 (Months 22-24): Impact Measurement

**Goal:** Understand if your tool actually helps

**Start simple - just observe:**
1. Find 3-5 teams willing to track their progress
2. Collect baseline data (race times, what they currently do)
3. They use your tool naturally for 12 weeks
4. Collect end data (race times, how they used it)
5. Ask: What patterns do you see?

**Data to collect:**
- Race times (before/after) - but don't assume cause-and-effect yet
- How often they used the tool
- Which features they used most / ignored
- What feedback they gave
- Athlete/coach satisfaction surveys
- Qualitative: "What changed about your training?"

**Analysis framework:**

```python
# scripts/impact_analysis.py
import pandas as pd
import scipy.stats as stats

# Load data
baseline = pd.read_csv('data/baseline_times.csv')
final = pd.read_csv('data/final_times.csv')

# Calculate improvement
baseline['improvement'] = (baseline['time'] - final['time']) / baseline['time'] * 100

# Average improvement
avg_improvement = baseline['improvement'].mean()
std_improvement = baseline['improvement'].std()

print(f"Average improvement: {avg_improvement:.2f}% ± {std_improvement:.2f}%")

# Statistical test (if sample size allows)
t_stat, p_value = stats.ttest_rel(baseline['time'], final['time'])
print(f"T-test p-value: {p_value:.4f}")

# BUT: Be careful about causation
# Improvement could be due to:
# - Normal progression over season
# - Better snow conditions
# - More training
# - Placebo effect
# - Your tool actually helping

# Document everything you observe
```

**Questions to explore:**
- Did athletes improve? By how much?
- Which athletes improved most? What did they have in common?
- What features did successful athletes use?
- What did coaches say was most valuable?
- What surprised you about the results?

**These observations become YOUR research questions for Year 3.**

---

## Year 3: Scale & Refinement

### Months 25-30: Grow User Base

**Target:** 50-100 teams

**Growth strategies:**
1. Word of mouth from existing users
2. Present at coaching clinics
3. Partner with regional ski associations
4. Social media demos
5. Science fair presentations (if applicable)

**Sustainability:**
- Consider light monetization ($5-10/month for unlimited)
- OR keep 100% free for research purposes
- OR find sponsors (ski equipment companies?)

---

### Months 31-36: Deeper Investigation (If You Found Something Interesting)

**By now you should know:**
- Does your tool actually help? (from Year 2 observations)
- What aspect helps most? (which features matter)
- Who does it help most? (which athletes benefit)

**IF you found clear patterns → Design rigorous study**

**Example: If you noticed athletes who used feature X improved more:**
- NOW you can design experiment to test that specifically
- You have a hypothesis based on real observations
- Not testing a pre-conceived idea

**IF results were mixed/unclear → That's also interesting!**
- Why didn't it work as expected?
- What did coaches do instead of what you intended?
- What barriers prevented adoption?
- This could be more valuable research than "it worked perfectly"

**The research emerges from what you discover, not what you plan.**

---

# PART 4: STANFORD PREP (Year 4)

## Months 37-42: Advanced Development (Based on What You Discovered)

**Goal:** Deepen the most valuable/interesting aspect

**By Year 4, you should know:**
- What do users actually need most?
- What technical problem is most interesting to you?
- What would make the biggest impact?
- What would professors find most novel?

**Possible directions** (choose based on your discoveries):

**If users need better personalization:**
- Adapt system to individual athlete capabilities
- Different recommendations for different skill levels

**If processing speed is the bottleneck:**
- Real-time or near-real-time processing
- Edge computing / on-device inference

**If you discovered interesting RL insights:**
- Whatever novel strategy RL found
- Deeper investigation of why it works/doesn't work

**If CV accuracy is still the issue:**
- Challenging conditions (fog, snow, low light)
- Multi-camera fusion
- Better 3D reconstruction

**If coaches want comparison features:**
- Advanced trajectory comparison algorithms
- Identification of specific technique differences

**Don't decide now - decide based on what you learn in Years 1-3.**

**The "wow" factor comes from solving a REAL problem deeply, not adding flashy features.**

---

## Months 43-45: Documentation & Portfolio

**Create comprehensive project documentation:**

### 1. Technical Portfolio

```markdown
# Alpine Ski Racing AI System - Technical Portfolio

## Project Overview
[3-paragraph summary]

## Technical Architecture
[System diagram]
[Component descriptions]

## Key Innovations
1. Custom CV pipeline achieving X% accuracy
2. Novel homography-based 3D reconstruction
3. RL agent for optimal line generation
4. Real-world deployment at scale

## Results & Impact
- 100+ teams using system
- 2000+ videos analyzed
- Quantified performance improvements (document what you actually measured)
- 2 published research papers

## Technical Challenges & Solutions
[Story of debugging, pivots, breakthroughs]

## Code & Demos
- GitHub: [link]
- Demo video: [link]
- Live demo: [link]
- Research papers: [link]

## Media & Recognition
[Any press coverage, awards, testimonials]
```

### 2. Update GitHub

- Clean up all code
- Comprehensive README
- Documentation for each module
- Example notebooks
- Video tutorials
- Contributor guide

### 3. Create Demo Reel (5-minute video)

**Structure:**
1. Problem statement (0:00-0:30)
2. Technical approach (0:30-2:00)
3. Live demo (2:00-3:30)
4. Impact & results (3:30-4:30)
5. Future vision (4:30-5:00)

Use vibe coding to create nice motion graphics!

### 4. Write Technical Blog Series

**Post 1:** "Why I Built This"
**Post 2:** "Computer Vision for Ski Racing"
**Post 3:** "From Prototype to Production"
**Post 4:** "Measuring Real-World Impact"
**Post 5:** "Lessons Learned"

Publish on Medium or personal blog

---

## Months 46-48: Stanford Application

### Application Components

**1. Common App Essay (650 words)**

**Your story will emerge from what you actually experience.**

**Potential themes based on common project journeys:**
- Discovery: What problem did you notice that others missed?
- Journey: What challenges did you overcome? What failures taught you?
- Impact: How did your work help real people?
- Learning: What surprised you? What changed your thinking?

**Key messages for Stanford:**
- Intellectual curiosity (why THIS problem interested you)
- Self-directed learning (how you taught yourself)
- Persistence through failure (the setbacks and pivots)
- Impact orientation (focus on users, not just technology)

**Don't write the essay yet. Live the journey first.**

---

**2. Stanford Supplemental Essays**

**Essay topics change yearly, but general principles:**

**"What is the most significant challenge that society faces today?"**
- Connect to your project's broader impact
- What societal gap does your work address?
- How could your approach generalize beyond skiing?

**"Reflect on an idea or experience that makes you genuinely excited about learning"**
- What genuinely surprised you during the project?
- What unexpected discovery excited you most?
- What moment made you think "this is why I love this"?

**Don't make up stories. Document real moments as they happen.**
- Keep a journal of interesting discoveries
- Screenshot unexpected results
- Note conversations with users that changed your thinking
- These become your essay material

---

**3. Activities List**

**Activity 1: Independent AI Research - Alpine Ski Racing Analysis**
- Position: Founder/Lead Developer
- Duration: 4 years (9-12)
- Hours: 15 hrs/week
- Description: [Fill in with YOUR actual achievements - don't copy this template verbatim. Use real numbers: teams deployed, videos processed, papers submitted, whatever you actually accomplished]

**Additional activities:**
- Research Assistant (if professor partnership)
- Ski Racing (your own athletic background)
- Relevant coursework
- Any awards/recognition

---

**4. Supplemental Materials**

**Required:**
- Transcript
- Test scores
- Letters of recommendation

**Optional (but powerful for your case):**
- Research abstract (1 page)
- Link to demo video
- Link to GitHub
- Published papers (if available)

---

### Letter of Recommendation Strategy

**Best case: Professor at Stanford/peer institution**

"I've had the pleasure of mentoring [Name] in our research lab for the 
past 3 years. They've contributed significantly to our work on computer 
vision for sports analysis and independently developed a novel application 
that we co-authored for publication..."

**Good case: Research mentor from any institution**

**Also good: Ski coach who used your system**

"[Name]'s AI system has transformed how I coach. The trajectory analysis 
helped our athletes improve by an average of 2.1%. More impressive than 
the technology is [Name]'s ability to translate technical capabilities 
into practical coaching value..."

**Request 2-3 letters total**

---

# APPENDICES

## Appendix A: Decision Trees

### Decision Point 1: Gate Detection Accuracy

```
After 100 training images:

Accuracy >85%?
├─ YES → Continue current approach
│         Add 400 more images
│         Target: >90% accuracy
│
└─ NO
   ├─ Accuracy 70-85%?
   │  └─ Add 500 more diverse images
   │     Re-train with larger model (YOLOv8m)
   │     Try different augmentations
   │
   └─ Accuracy <70%?
      └─ PIVOT OPTIONS:
         1. Try classical CV (Hough lines for poles)
         2. Use semantic segmentation (Segment Anything)
         3. Add manual gate marking as fallback
         4. Focus on easier cases first (good visibility)
```

---

### Decision Point 2: RL Agent Convergence

```
After 1M training steps:

Completion rate >80%?
├─ YES → Optimize reward function for performance
│         Compare to human trajectories
│         Write paper
│
└─ NO
   ├─ Completion rate 50-80%?
   │  └─ Simplify environment
   │     Reduce course difficulty
   │     Add curriculum learning
   │     Train 2M more steps
   │
   └─ Completion rate <50%?
      └─ STOP RL PATH
         This is NOT a failure - it's a smart pivot
         
         Option A: Use supervised learning instead
         - Train on pro skier trajectories
         - Still valuable contribution
         - Easier to validate
         
         Option B: Focus on CV + deployment
         - Skip RL entirely
         - Focus on user impact
         - Still strong Stanford story
```

---

### Decision Point 3: Professor Partnership

```
After 15 emails sent:

Got positive response?
├─ YES → Schedule calls
│         Contribute to their research first
│         Build relationship (6-12 months)
│         Introduce your project later
│
└─ NO
   ├─ Got "check back later" responses?
   │  └─ Set reminder for 3 months
   │     Continue building independently
   │     Contact them again with progress update
   │
   └─ No responses at all?
      └─ ADJUST STRATEGY:
         1. Send to 15 more professors
         2. Try PhD students instead
         3. Focus on independent path
         4. Build impressive results first
         
         Remember: Independent project still
         very strong for Stanford!
```

---

## Appendix B: Get Unstuck Playbook

### If Stuck on Bug for >4 Hours

**Step 1: Reduce the problem**
```python
# Don't debug full pipeline
# Create minimal reproducible example

# Bad: Run full video processing
results = pipeline.process_video('30min_race.mp4')

# Good: Test one component with small input
gates = detector.detect_gates(single_frame)
print(f"Found {len(gates)} gates")
```

**Step 2: Rubber duck debugging**

Write down:
1. What SHOULD happen?
2. What ACTUALLY happens?
3. What have I tried?
4. What are my assumptions?

Often the act of writing reveals the bug.

**Step 3: Phone a friend**

Post on r/learnprogramming with:
- Minimal code example
- Expected vs. actual behavior
- What you've tried
- Your hypothesis

**Step 4: Time-box and move on**

If no solution in 8 total hours:
- Flag it in your notes
- Work on something else
- Come back in 1 week

Sometimes problems solve themselves with distance.

---

### If Stuck on Motivation for >1 Week

**Ship something small:**
- Make a simple visualization
- Write a blog post about what you learned
- Demo current state to a friend
- Post progress on Twitter

Momentum > perfection

**Change the task:**
- Bored of CV? Work on app UI
- Bored of coding? Write documentation
- Bored of this feature? Work on different one

**Reconnect with "why":**
- Watch a World Cup race
- Read your original research
- Remember: you're building something cool
- Talk to a skier who would benefit

**Take a break:**
- 1 week off is OK
- Work on schoolwork
- Do other hobbies
- Return refreshed

---

### If Worried About Timeline

**Remember:**
- You have 4 YEARS
- MVP in 8 weeks already puts you ahead
- Most people never finish projects
- Consistency > speed

**Minimum viable Stanford application:**
- Working CV system (you have this after Week 8)
- Some deployed users (achievable in Year 1)
- Demonstrated learning (inevitable if you keep going)
- Passion and persistence (you clearly have this)

**You don't need:**
- Perfect code
- Published papers (nice-to-have, not must-have)
- 100+ teams (10 is impressive)
- RL working perfectly (can pivot)

---

## Appendix C: Weekly Review Template

```markdown
# Weekly Review - Week of [Date]

## Accomplishments ✅
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## Blockers 🚧
### Blocker 1:
- Problem: [description]
- Time spent: [hours]
- Status: [resolved / still stuck / punted]
- Solution (if resolved): [what worked]

## Pivot Decisions 🔄
- Planned: [original plan]
- Actual: [what you did instead]
- Reason: [why you pivoted]
- Right call?: [reflection]

## Metrics 📊
- Code: [commits / lines added]
- Progress: [X% toward current milestone]
- Hours: [total work hours this week]
- Videos processed: [count]

## Learnings 📚
- Technical: [new skill/concept learned]
- Process: [what worked/didn't work]
- Next time: [what to do differently]

## Next Week 🎯
### Must Do:
- Priority 1: [critical]

### Should Do:
- Priority 2: [important]

### Nice to Have:
- Priority 3: [if time]

## Energy Check ⚡
- Excitement level: [1-10]
- If <7: [what would make it more interesting?]
- Burnout risk: [low / medium / high]
- If high: [plan to take break]
```

**Do this every Sunday evening.**

Benefits:
- Catches problems early
- Documents decisions (useful for Stanford essays!)
- Shows progress over time
- Prevents burnout

---

## Appendix D: Resource Budget

### Year 1: $0-50

**Compute:**
- Training YOLOv8: Google Colab (free)
- Video processing: Local laptop (free)
- Total: $0

**Services:**
- GitHub: Free
- Roboflow: Free tier (1500 images)
- YouTube hosting: Free
- Total: $0

**Optional:**
- Domain name: $12/year
- Upgrading Roboflow: $0 (not needed yet)

**Year 1 Total: $0-12**

---

### Year 2: $50-200

**Compute:**
- RL training (if doing): $50-100
  - Option 1: Paperspace ($0.45/hr × 100 hrs = $45)
  - Option 2: Lambda Labs ($0.50/hr × 100 hrs = $50)
- Cloud processing for app: $30-50
  - AWS free tier first year
  - Then ~$30/month if serving users

**Services:**
- Video storage (S3): $20-30
- Domain + hosting: $12

**Year 2 Total: $100-200**

---

### Year 3: $200-500

**Scaling costs:**
- Cloud processing: $50-100/month
- Storage: $30-50/month
- Monitoring: $20/month

**Optional:**
- Mobile app publishing: $99/year (Apple)
- Google Play: $25 one-time

**Year 3 Total: $500-1000**

---

### Free Alternatives

**Compute:**
- Google Colab (free GPU)
- Kaggle Kernels (free GPU, 30hr/week)
- GitHub Codespaces (60hr/month free)

**Credits:**
- GitHub Student Pack (includes many free credits)
- AWS Educate
- Google Cloud education credits
- Azure for Students

**Ask your school:**
- Some schools have cloud compute budgets
- CS departments may have GPU clusters

---

## Appendix E: Code Templates

### Template 1: Custom Training Loop

```python
# train_custom.py
from ultralytics import YOLO
import torch
from pathlib import Path

class CustomTrainer:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config['base_model'])
        
    def train(self):
        """
        Train with custom callbacks and logging
        """
        results = self.model.train(
            data=self.config['data_yaml'],
            epochs=self.config['epochs'],
            imgsz=self.config['img_size'],
            batch=self.config['batch_size'],
            name=self.config['run_name'],
            patience=self.config['patience'],
            save_period=self.config['save_every'],
            
            # Callbacks
            callbacks={
                'on_train_epoch_end': self.on_epoch_end,
                'on_val_end': self.on_validation_end
            }
        )
        
        return results
    
    def on_epoch_end(self, trainer):
        """Custom callback after each epoch"""
        epoch = trainer.epoch
        metrics = trainer.metrics
        
        # Log to your own system
        print(f"Epoch {epoch}: mAP={metrics.map:.3f}")
        
        # Save checkpoint if best so far
        if metrics.map > self.best_map:
            self.best_map = metrics.map
            self.save_checkpoint(epoch)
    
    def on_validation_end(self, validator):
        """Custom callback after validation"""
        # Visualize predictions
        self.visualize_predictions(validator.pred)

# Usage
config = {
    'base_model': 'yolov8n.pt',
    'data_yaml': 'data/ski_gates.yaml',
    'epochs': 100,
    'img_size': 640,
    'batch_size': 16,
    'run_name': 'gate_detector_v2',
    'patience': 20,
    'save_every': 10
}

trainer = CustomTrainer(config)
results = trainer.train()
```

---

### Template 2: Trajectory Analysis

```python
# analyze_trajectory.py
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter

class TrajectoryAnalyzer:
    def __init__(self, trajectory_3d):
        """
        trajectory_3d: List of {'frame': int, 'x': float, 'y': float}
        """
        self.trajectory = trajectory_3d
        self.x = np.array([p['x'] for p in trajectory_3d])
        self.y = np.array([p['y'] for p in trajectory_3d])
        self.frames = np.array([p['frame'] for p in trajectory_3d])
        
    def smooth_trajectory(self, window_length=11, polyorder=3):
        """
        Apply Savitzky-Golay filter to smooth trajectory
        """
        self.x_smooth = savgol_filter(self.x, window_length, polyorder)
        self.y_smooth = savgol_filter(self.y, window_length, polyorder)
        
        return list(zip(self.x_smooth, self.y_smooth))
    
    def calculate_speed(self, fps=30):
        """
        Calculate speed at each point (m/s)
        """
        speeds = []
        
        for i in range(1, len(self.x)):
            dx = self.x[i] - self.x[i-1]
            dy = self.y[i] - self.y[i-1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Distance per frame → distance per second
            speed_ms = distance * fps
            speed_kmh = speed_ms * 3.6
            
            speeds.append(speed_kmh)
        
        return speeds
    
    def calculate_curvature(self):
        """
        Calculate path curvature (1/radius) at each point
        """
        # First derivatives
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        
        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula
        curvature = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5)
        
        return curvature
    
    def find_turn_points(self, curvature_threshold=0.05):
        """
        Identify turns (high curvature points)
        """
        curvature = self.calculate_curvature()
        abs_curvature = np.abs(curvature)
        
        turns = []
        for i, curv in enumerate(abs_curvature):
            if curv > curvature_threshold:
                turns.append({
                    'frame': self.frames[i],
                    'position': (self.x[i], self.y[i]),
                    'curvature': curv,
                    'radius_m': 1/curv if curv > 0 else float('inf')
                })
        
        return turns
    
    def compare_to_reference(self, reference_trajectory):
        """
        Compare this trajectory to a reference (e.g., pro skier)
        Using Dynamic Time Warping distance
        """
        from dtaidistance import dtw
        
        # Extract x,y as series
        series1 = np.column_stack([self.x, self.y])
        series2 = np.column_stack([
            [p['x'] for p in reference_trajectory],
            [p['y'] for p in reference_trajectory]
        ])
        
        distance = dtw.distance(series1, series2)
        
        return distance

# Usage example
analyzer = TrajectoryAnalyzer(trajectory_3d)

# Smooth trajectory
smoothed = analyzer.smooth_trajectory()

# Calculate metrics
speeds = analyzer.calculate_speed(fps=30)
avg_speed = np.mean(speeds)
max_speed = np.max(speeds)

print(f"Average speed: {avg_speed:.1f} km/h")
print(f"Max speed: {max_speed:.1f} km/h")

# Find turns
turns = analyzer.find_turn_points()
print(f"Found {len(turns)} significant turns")

for i, turn in enumerate(turns[:5]):
    print(f"  Turn {i+1}: radius={turn['radius_m']:.1f}m")
```

---

### Template 3: Batch Processing

```python
# batch_process.py
from pathlib import Path
import json
from ski_racing.pipeline import SkiRacingPipeline
from concurrent.futures import ThreadPoolExecutor
import time

class BatchProcessor:
    def __init__(self, model_path, num_workers=4):
        self.pipeline = SkiRacingPipeline(model_path)
        self.num_workers = num_workers
        
    def process_directory(self, input_dir, output_dir):
        """
        Process all videos in a directory
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all video files
        video_files = list(input_path.glob('*.mp4')) + \
                     list(input_path.glob('*.mov')) + \
                     list(input_path.glob('*.avi'))
        
        print(f"Found {len(video_files)} videos to process")
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for video in video_files:
                future = executor.submit(
                    self._process_single,
                    video,
                    output_path
                )
                futures.append((video.name, future))
            
            # Collect results
            results = []
            for name, future in futures:
                try:
                    result = future.result()
                    results.append({
                        'video': name,
                        'status': 'success',
                        'gates': len(result['gates']),
                        'trajectory_points': len(result['trajectory_3d'])
                    })
                    print(f"✓ {name}")
                except Exception as e:
                    results.append({
                        'video': name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    print(f"✗ {name}: {e}")
        
        # Save summary
        summary_path = output_path / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Batch processing complete")
        print(f"  Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"  Failed: {sum(1 for r in results if r['status'] == 'failed')}")
        
        return results
    
    def _process_single(self, video_path, output_dir):
        """Process a single video"""
        output_file = output_dir / f"{video_path.stem}_analysis.json"
        
        # Skip if already processed
        if output_file.exists():
            with open(output_file, 'r') as f:
                return json.load(f)
        
        # Process
        result = self.pipeline.process_video(str(video_path))
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result

# Usage
processor = BatchProcessor(
    model_path='models/gate_detector_best.pt',
    num_workers=4
)

results = processor.process_directory(
    input_dir='data/raw_videos',
    output_dir='artifacts/outputs/batch_results'
)
```

---

## Appendix F: Common Errors & Solutions

### Error 1: CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Reduce batch size: `batch=8` → `batch=4`
2. Use smaller model: `yolov8n.pt` instead of `yolov8l.pt`
3. Reduce image size: `imgsz=640` → `imgsz=416`
4. Use CPU: Add `device='cpu'` to training

---

### Error 2: No gates detected

```
Found 0 gates
```

**Debugging steps:**
1. Check confidence threshold too high: Lower from 0.5 to 0.3
2. Visualize first frame: Are gates actually visible?
3. Check model loaded correctly: `model.names` should show your classes
4. Verify image preprocessing: Are colors/sizes correct?

**Solution:**
```python
# Debug detection
results = model(frame, conf=0.3)  # Lower threshold
results[0].show()  # Visualize detections
```

---

### Error 3: Homography fails

```
cv2.error: OpenCV(4.x.x) error: (-215:Assertion failed)
```

**Causes:**
- Less than 4 gate points
- Points are collinear (all in a line)
- NaN values in gate positions

**Solution:**
```python
def calculate_homography_safe(gates_2d, gates_3d):
    if len(gates_2d) < 4:
        print("Warning: Need at least 4 gates")
        return None
    
    # Check for NaN
    gates_2d = [(x, y) for x, y in gates_2d if not (np.isnan(x) or np.isnan(y))]
    
    if len(gates_2d) < 4:
        print("Warning: Not enough valid gates after NaN filtering")
        return None
    
    try:
        H, status = cv2.findHomography(
            np.float32(gates_2d),
            np.float32(gates_3d),
            cv2.RANSAC,
            5.0  # RANSAC threshold
        )
        return H
    except cv2.error as e:
        print(f"Homography failed: {e}")
        return None
```

---

## Appendix G: Success Metrics Dashboard

Track these metrics over time:

### Technical Metrics
```python
metrics = {
    # Detection
    'gate_detection_map': 0.87,
    'gate_detection_precision': 0.91,
    'gate_detection_recall': 0.84,
    
    # Tracking
    'tracking_coverage': 0.94,  # % of frames with detection
    'trajectory_smoothness': 0.89,  # Low = jerky
    
    # Processing
    'avg_processing_time_sec': 120,
    'pipeline_success_rate': 0.95,
    
    # Dataset
    'total_videos_processed': 247,
    'total_annotations': 1243,
}
```

### User Metrics
```python
user_metrics = {
    'total_teams': 37,
    'active_monthly_users': 28,
    'videos_processed_month': 156,
    'avg_videos_per_team': 4.2,
    'user_satisfaction': 4.3,  # out of 5
    'nps_score': 42,  # Net Promoter Score
}
```

### Impact Metrics
```python
impact_metrics = {
    'avg_time_improvement_pct': 2.3,
    'athletes_improved': 24,  # out of 30 measured
    'coaches_endorsements': 8,
    'media_mentions': 2,
}
```

---

## Appendix H: Quick Reference

### Key Milestones Checklist

**Week 8:**
- [ ] Working CV pipeline
- [ ] Gate detection >80%
- [ ] Code on GitHub
- [ ] Demo video

**Month 6:**
- [ ] Detection >90%
- [ ] Simple web app
- [ ] 10+ processed videos

**Month 12:**
- [ ] 20+ teams testing
- [ ] Paper drafted
- [ ] Quantified feedback

**Year 2:**
- [ ] 50+ teams using
- [ ] 1 paper submitted
- [ ] Impact measurements

**Year 3:**
- [ ] 100+ teams using
- [ ] Controlled study complete
- [ ] 2 papers submitted

**Year 4:**
- [ ] Portfolio complete
- [ ] Demo reel done
- [ ] Stanford application submitted

---

### Quick Commands Reference

```bash
# Training
python train_detector.py --data data.yaml --epochs 100

# Processing
python process_video.py input.mp4 --output results/

# Batch processing
python batch_process.py --input data/videos/ --output results/

# Evaluation
python evaluate.py --model models/best.pt --test data/test/

# Start web app
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

### Important Links Template

```markdown
## My Project Links

**Code & Documentation:**
- GitHub: https://github.com/[username]/Stanford-application-project
- Documentation: https://[username].github.io/Stanford-application-project
- Technical blog: https://medium.com/@[username]

**Demos & Videos:**
- Demo video: https://youtube.com/watch?v=[id]
- Example outputs: https://drive.google.com/[folder]

**Research:**
- Paper 1: https://arxiv.org/abs/[id]
- Dataset: https://huggingface.co/datasets/[username]/skirace-cv

**Contact:**
- Email: [your-email]
- LinkedIn: https://linkedin.com/in/[username]
- Twitter: @[username]
```

---

## Final Notes

**Remember:**

1. **Start building TODAY** - Don't wait for perfect plan
2. **Ship early, ship often** - Momentum beats perfection
3. **Document as you go** - Future you will thank you
4. **Ask for help** - Community is friendly
5. **Celebrate small wins** - 8 weeks to MVP is huge!

**The goal isn't perfection, it's demonstrating:**
- ✅ Technical depth
- ✅ Independent learning
- ✅ Real-world impact
- ✅ Persistence through challenges
- ✅ Intellectual curiosity

**You've got this. Start Week 1 tomorrow.**

---

*This guide will evolve as you make progress. Update it with your learnings.*

**Now go build something amazing! 🎿🤖**
