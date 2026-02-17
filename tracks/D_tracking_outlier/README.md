# Track D: Tracking / Outlier Handling

## Owner
Assigned to: _______________

## Goal
Reduce 2D trajectory noise: fewer jumps, fewer tracker switches, smoother paths before physics checks.

## Your Spec Document
**Read this first:** `Track_D_Tracking_Outlier_Handling.docx` (in this folder)

## Files You Own (edit these)
| File | What it does |
|------|-------------|
| `ski_racing/tracking.py` | SkierTracker (ByteTrack), KalmanSmoother (RTS filter) |
| `ski_racing/pipeline.py` | Pipeline ordering (if you move camera compensation before smoothing) |
| `ski_racing/visualize.py` | Trajectory overlay for visual debugging |

## Files You Read (do NOT edit)
| File | Why you need it |
|------|----------------|
| `ski_racing/detection.py` | Gate tracking feeds camera compensation |
| `ski_racing/transform.py` | Understand how 2D noise amplifies in 3D |

## Your Reports Folder
`tracks/D_tracking_outlier/reports/` — contains:
- Video regression results (current max jump, G-force baselines)

## Key Commands
```bash
# Process video and generate demo overlay for visual inspection
python scripts/process_video.py data/test_videos_unseen/VIDEO.mp4 \
  --gate-model models/gate_detector_best.pt \
  --discipline slalom --stabilize --demo-video --summary

# Run physics tests
python -m pytest tests/test_physics.py -v

# Inspect trajectory in output JSON
python -c "
import json
d = json.load(open('artifacts/latest/VIDEO_analysis.json'))
traj = d['trajectory_2d']
jumps = []
for i in range(1, len(traj)):
  dx = traj[i]['x'] - traj[i-1]['x']
  dy = traj[i]['y'] - traj[i-1]['y']
  jumps.append((dx**2 + dy**2)**0.5)
print(f'Max jump: {max(jumps):.1f}px, Mean: {sum(jumps)/len(jumps):.1f}px')
"
```

## Dependencies
- **Coordinate with Track C**: Trajectory smoothness and geometry quality are entangled
- **Track E**: Use their pipeline to measure before/after
