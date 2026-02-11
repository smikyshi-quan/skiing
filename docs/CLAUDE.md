# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Alpine Ski Racing AI** system that analyzes ski race videos to:
1. Detect ski racing gates (red and blue poles) using YOLOv8
2. Track the skier's trajectory through the course
3. Transform 2D video coordinates to 3D world coordinates using homography
4. Validate results against physics constraints (speed, G-forces, turn radii)

**Key Philosophy**: Domain-specific metrics matter more than ML metrics. An 95% accurate gate detector is useless if gates are 2m off or speeds are 2x too fast. Always validate outputs against physics.

## Project Structure

```
ski_racing/              # Core Python package with analysis modules
├── detection.py        # Gate detection (YOLOv8) + temporal gate tracking
├── tracking.py         # Skier tracking using ByteTrack + fallback
├── transform.py        # 2D→3D coordinate transformation (homography)
├── physics.py          # Physics validation engine (speeds, G-forces, turn radii)
├── pipeline.py         # End-to-end pipeline orchestration
└── visualize.py        # Visualization and output generation

scripts/                 # Command-line tools
├── extract_frames.py   # Extract video frames (with balancing for class imbalance)
├── train_detector.py   # Train YOLOv8 gate detector
├── process_video.py    # Full pipeline runner (main entry point)
└── evaluate.py         # Evaluate accuracy against ground truth

tests/                   # Unit tests
└── test_physics.py     # Physics validation module tests

data/                    # Dataset directory (not in git)
├── annotations/        # Annotated dataset in YOLOv8 format
├── raw_videos/         # Source videos
└── test_videos/        # Test videos for validation

models/                  # Trained weights (not in git)
├── gate_detector_best.pt   # Fine-tuned YOLOv8s gate detector
└── yolov8s.pt             # Base YOLOv8 weights

artifacts/                # Generated outputs (not in git)
├── outputs/              # Analysis results
└── training_results/     # Training artifacts

runs/                     # Ultralytics run outputs (not in git)
docs/                     # Guides, plans, reports
tools/                    # Data acquisition helpers
└── video_scraper/        # Video scrape helpers + source lists
```

## Key Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run physics tests
python tests/test_physics.py
```

### Dataset Preparation
```bash
# Extract frames from raw videos (with balanced class sampling)
python scripts/extract_frames.py data/raw_videos/ --output-dir data/frames --balanced

# Prepare dataset for training (organize annotations)
python scripts/prepare_dataset.py data/annotations/
```

### Training
```bash
# Train gate detector (modify data.yaml path if needed)
python scripts/train_detector.py --data data/annotations/data.yaml

# Copy best weights to models/
cp runs/detect/*/weights/best.pt models/gate_detector_best.pt
```

### Processing & Analysis
```bash
# Process single video (full pipeline)
python scripts/process_video.py VIDEO_PATH \
  --gate-model models/gate_detector_best.pt \
  --summary --demo-video

# Process all test videos
python scripts/process_video.py data/test_videos/ \
  --gate-model models/gate_detector_best.pt

# Evaluate against ground truth
python scripts/evaluate.py \
  --predictions artifacts/outputs/race1_analysis.json \
  --ground-truth data/annotations/race1_gt.json
```

### Direct Python API
```python
from ski_racing.pipeline import SkiRacingPipeline

pipeline = SkiRacingPipeline(
    gate_model_path="models/gate_detector_best.pt",
    discipline="slalom",
    gate_spacing_m=12.0
)

results = pipeline.process_video(
    "race.mp4",
    output_dir="artifacts/outputs",
    validate_physics=True
)
```

## Architecture & Key Concepts

### 1. Gate Detection (detection.py)

**GateDetector**: Wraps YOLOv8 to detect red and blue gate poles.
- Input: Video frames (BGR images)
- Output: List of gate detections with class, center, confidence, and **base_y** (bottom of pole)
- Classes: 0=red_gate, 1=blue_gate (or similar)

**TemporalGateTracker**: Handles occlusion (gates disappearing when racer hits them).
- Remembers gate positions across frames
- Allows ~0.5 second (15 frame) grace periods for temporary disappearance
- Crucial for slalom races where racers intentionally contact gates

**Key Design Pattern**: Gates are indexed by their base_y position (distance down the slope). First-frame detection finds ~4-8 gates; temporal tracking stabilizes them; then clustering merges duplicate detections.

### 2. Skier Tracking (tracking.py)

**SkierTracker**: Uses ByteTrack for robust multi-person tracking.
- Detects all people using YOLOv8 person class
- Assigns consistent IDs across frames
- Filters by position (centered person likely the racer) and temporal consistency
- Fallback: If track is lost, finds closest person to previous position

**Why ByteTrack?** Naive "largest bounding box" fails because:
- Spectators/workers may be in frame
- Racer size changes due to perspective
- Need consistent ID assignment across frames

**Frame Stride Options**: Can skip frames (stride=5) for speed; trajectory is interpolated.

### 3. Coordinate Transformation (transform.py)

**HomographyTransform**: Converts 2D pixel coordinates to 3D world coordinates.

Two methods:
1. **Scale-based** (simpler): Assumes level slope, uses gate spacing to calibrate pixel→meter ratio
2. **Homography-based** (more robust): Fits perspective transform using 4+ gate base positions

**Critical for accuracy**: This is where most errors occur. If gates are detected wrong or spacing is wrong, the 3D trajectory will be physically impossible (detected by physics validator).

### 4. Physics Validation (physics.py)

**PhysicsValidator**: The "sanity checker" for trajectories.

Validates by discipline (slalom/giant_slalom/downhill):
- **Speed**: 30-70 km/h (slalom), 50-100 km/h (GS), 80-150 km/h (downhill)
- **G-forces**: Peak 2-4G typical, max 5-6G brief
- **Turn radius**: 6m+ (slalom), 20m+ (GS)
- **Smoothness**: No >5m jumps per frame (indicates detection errors)

**Usage Pattern**: If physics validation fails, the issue is almost always:
1. Wrong gate spacing parameter
2. Wrong FPS assumption
3. Detection errors (missing/misplaced gates)
4. Wrong discipline setting

### 5. End-to-End Pipeline (pipeline.py)

**SkiRacingPipeline**: Orchestrates the full analysis.

Processing steps:
1. Detect gates (first frame + optional temporal search + temporal tracking)
2. Cluster duplicate gate detections
3. Calculate pixel-to-meter transform
4. Track skier across video
5. Transform trajectory to 3D
6. Run physics validation
7. Save results as JSON

**Key Parameters**:
- `gate_conf`: Detection confidence threshold (0.3 default, increase for precision)
- `gate_search_frames`: If <4 gates found, scan this many frames (150 default)
- `frame_stride`: Skip frames for speed (1=no skip)
- `discipline`: Determines physics limits
- `gate_spacing_m`: Critical parameter—must be accurate

## Important Design Decisions

### Class Imbalance Problem
Training data naturally has imbalance: gates appear in ~25% of frames. Solution implemented:
- `extract_frames.py --balanced`: Dense sampling in gate-heavy regions, sparse in long sections
- Check distribution before training: gates should be ~30-40% of annotations

### Occlusion Handling
Slalom racers hit gates intentionally. System handles this via:
- Initial detection from first frame (before contact)
- Temporal tracking through early frames (stabilizes gate positions)
- Grace period: remembers gates for 0.5s even if undetected
- If permanently lost, uses last known position

### Multiple Person Problem
Videos contain spectators, workers, camera crew. Handled by:
- Using person detector + ByteTrack (maintains consistent IDs)
- Selecting centered person as racer (heuristic, or can be manually selected)
- Rejecting jumps >100 pixels/frame (temporal consistency)

## Physics Validation Thresholds

These are per-discipline and critical for validation:

**Slalom**: 30-70 km/h, 5G max, 6m min turn radius
**Giant Slalom**: 50-100 km/h, 4.5G max, 20m min turn radius
**Downhill**: 80-150 km/h, 4G max, 25m min turn radius

If validation fails, adjust:
1. `--gate-spacing`: Try ±10% if speeds seem wrong
2. `--discipline`: Verify correct discipline selected
3. Check FPS in video metadata (must be accurate)

## Testing & Validation

**Unit tests** (test_physics.py):
- Run before making changes: `python tests/test_physics.py`
- Tests physics validator against known trajectories

**Integration test** (manual):
1. Pick race video with visible TV speed overlay
2. Run full pipeline: `python scripts/process_video.py race.mp4 --gate-model models/gate_detector_best.pt`
3. Check output JSON for estimated max speed
4. Compare to TV speed: should be within 5 km/h
5. If off: adjust gate spacing and re-run

**Quality checks**:
- Are ≥4 gates detected in first frame? (If not, increase `--gate-search-frames`)
- Does trajectory stay smooth? (Look at `_demo_overlay.mp4`)
- Do physics metrics look realistic? (Check console output)

## Data Files & Annotations

**Dataset Location**: `data/annotations/ski-gate-detection.v2i.yolov8/`

**Quality Analysis**: See `README_QUALITY_ANALYSIS.md` for detailed dataset metrics.
- 420 total images, split into train/valid/test
- 4 images with critical quality issues (completely black)
- 224 images with low contrast (likely fog/snow scenes—keep these for robustness)

**To Remove Bad Images**:
```bash
# Completely black images (CRITICAL)
rm data/annotations/ski-gate-detection.v2i.yolov8/train/images/GS__Lucas-GS-T2__frame_0007_jpg.rf.8e675b93243457020496ae1e497948d3.jpg
rm data/annotations/ski-gate-detection.v2i.yolov8/train/images/GS__MO-GS-T3__frame_0040_jpg.rf.7179bf5df51f3b5c7f358f6b047fda11.jpg

# Nearly black images (RECOMMENDED)
rm data/annotations/ski-gate-detection.v2i.yolov8/train/images/GS__Lucas-GS-T2__frame_0000_jpg.rf.b6ba779e51e2ba550356bd86850e6604.jpg
rm data/annotations/ski-gate-detection.v2i.yolov8/train/images/GS__Lucas-GS-T2__frame_0001_jpg.rf.d3303e09618e2eb1894b686bf4733dee.jpg
```

## Common Debugging Patterns

### Low gate detection rate
- Increase `--gate-conf` (e.g., 0.3→0.4) to require higher confidence
- Increase `--gate-search-frames` to scan more frames
- Check: Is model trained? Is model path correct?

### Physics validation failures
- Check speed: Too high? Wrong gate spacing or wrong FPS
- Check G-forces: Too high? Bad gate detection or wrong homography
- Check trajectory smoothness: Jumps indicate missed/bad detections

### Skier tracking lost
- Likely multiple people in frame; verify center person is racer
- Check `max_jump` parameter (default: None = no limit)
- Increase `--frame-stride` (skip fewer frames) for better continuity

### Homography errors
- Need ≥4 gates with correct base_y positions
- If gates clustered vertically, homography will be singular
- Try `--projection scale` (simpler method) if homography fails

## Development Notes

### When to Use Which Pipeline Features
- `--summary`: Generates PNG visualization of trajectory overlay
- `--demo-video`: Outputs video file with trajectory drawn
- `--no-physics`: Skip validation (faster, but risky)
- `--projection scale`: Use if geometry is unclear
- `--projection homography`: Use for multi-perspective footage

### Model Training Details
- YOLOv8s (small) is good starting point; yolov8n (nano) if speed needed
- Training uses data augmentation: mosaic, rotation, color jitter
- Typical mAP target: >0.60 for basic use, >0.85 for production
- Training dataset should have 400+ images for robust results

### Output Format
All results saved as JSON in `artifacts/outputs/` with structure:
```json
{
  "video": "path/to/video.mp4",
  "gates": [...],
  "trajectory_2d": [...],
  "trajectory_3d": [...],
  "physics_validation": {...}
}
```

The JSON is human-readable and can be loaded in Python or visualized.
