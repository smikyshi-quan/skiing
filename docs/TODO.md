# Ski Racing Line Analysis App — To-Do List

The goal: build an app that takes a ski racing video, detects gates, tracks the skier's line, converts it to a bird's-eye view, and validates it against real physics.

Each **Step** is a batch of tasks you can do **at the same time**. Finish all tasks in a Step before moving to the next one.

---

## Step 1 — Setup (all at the same time)

| # | Task | Who | Time |
|---|------|-----|------|
| 1A | Install dependencies: `pip install -r requirements.txt` | You | 10 min |
| 1B | Create a free Roboflow account at roboflow.com | You | 5 min |
| 1C | Create a GitHub repo (e.g. `alpine-ski-racing-ai`) and push the current code | You | 15 min |
| 1D | Download 20–30 World Cup race videos from YouTube. Save to `data/raw_videos/` | You | 1–2 hrs |

**Done when:** Dependencies installed, Roboflow account ready, videos downloaded, code on GitHub.

---

## Step 2 — Prepare Videos & Annotate (all at the same time)

| # | Task | Who | Time |
|---|------|-----|------|
| 2A | Pick 3 "golden" test videos and copy to `data/test_videos/`: (1) clear/sunny, (2) bad weather, (3) different camera angle | You | 15 min |
| 2B | Extract frames: `python scripts/extract_frames.py data/raw_videos/ --output-dir data/frames --balanced` | You (run command) | 10 min |
| 2C | While frames extract — read YOLOv8 docs so you understand what the training script does | You | 30 min |

**Done when:** 3 test videos selected, frames extracted, you understand the training flow.

---

## Step 3 — Annotate (this is the bottleneck, no shortcuts)

| # | Task | Who | Time |
|---|------|-----|------|
| 3A | Upload ~100 frames to Roboflow | You | 10 min |
| 3B | Draw bounding boxes: red poles → `red_gate`, blue poles → `blue_gate`. Include gates at different distances, partially visible, bad weather | You | 2–4 hrs |
| 3C | Export in **YOLOv8 format**, download to `data/annotations/` | You | 5 min |

**Tip:** This is the slowest step. Put on music. Aim for 100 well-annotated images — quality matters more than quantity right now.

**Done when:** 100+ annotated images exported in YOLOv8 format.

---

## Step 4 — Train & Test First Model (all at the same time)

| # | Task | Who | Time |
|---|------|-----|------|
| 4A | Train gate detector: `python scripts/train_detector.py --data data/annotations/data.yaml` | You (run command) | 1–2 hrs (GPU) |
| 4B | **While training runs** — watch the training curves in TensorBoard or the console output | You | passive |
| 4C | **While training runs** — start annotating 50 more "hard" images in Roboflow (occlusion, bad lighting) as backup in case mAP < 0.60 | You | 1 hr |

**After training finishes:**

| # | Task | Who | Time |
|---|------|-----|------|
| 4D | Check mAP@0.5. Target: **> 0.60** | You | 5 min |
| 4E | If below 0.60 → export those 50 extra annotations, re-train | You | 1–2 hrs |
| 4F | Copy best model: `cp runs/detect/*/weights/best.pt models/gate_detector_best.pt` | You (run command) | 1 min |

**Done when:** You have `models/gate_detector_best.pt` with mAP@0.5 > 0.60.

---

## Step 5 — Run Full Pipeline on All 3 Test Videos (all at the same time)

| # | Task | Who | Time |
|---|------|-----|------|
| 5A | Process all test videos with summary + demo: `python scripts/process_video.py data/test_videos/ --gate-model models/gate_detector_best.pt --summary --demo-video` | You (run command) | 5–15 min |
| 5B | While processing — run physics tests: `python tests/test_physics.py` | You (run command) | 10 sec |

**After processing finishes:**

| # | Task | Who | Time |
|---|------|-----|------|
| 5C | Read the physics validation report for each video. Check: speeds 30–60 km/h, G-forces < 5G, no teleportation jumps | You | 15 min |
| 5D | Look at the summary PNG and demo MP4 in `artifacts/outputs/` — does the trajectory overlay look right? | You | 15 min |
| 5E | If physics fails → check that ≥ 4 gates detected in first frame. If not, annotate more gate-heavy frames and go back to Step 4 | You | varies |

**Done when:** All 3 test videos produce physically plausible results. You have demo videos and summary figures.

---

## Step 6 — Improve to Production Quality (two tracks in parallel)

**Track A: More Data**

| # | Task | Who | Time |
|---|------|-----|------|
| 6A-1 | Annotate 500 more images in Roboflow. Focus on: gates being hit (occlusion), bad weather, different disciplines (SL vs GS) | You | 8–15 hrs (spread over days) |
| 6A-2 | Export full dataset, re-train: target **mAP@0.5 > 0.85** | You | 1–2 hrs |
| 6A-3 | Re-run pipeline on test videos, compare old vs new physics reports | You | 15 min |

**Track B: Validate Against Real Data (do alongside Track A)**

| # | Task | Who | Time |
|---|------|-----|------|
| 6B-1 | Find a race video that shows the TV speed overlay on screen | You | 30 min |
| 6B-2 | Run pipeline on that video | You (run command) | 5 min |
| 6B-3 | Compare your estimated max speed to the TV overlay. Target: **within 5 km/h** | You | 15 min |
| 6B-4 | If off by more than 5 km/h → adjust gate spacing parameter (`--gate-spacing`) and re-run | You | 15 min |

**Done when:** mAP > 0.85, speed estimates match reality within 5 km/h.

---

## Step 7 — Add Slope Estimation & Polish (two tracks in parallel)

**Track A: Slope Module**

| # | Task | Who | Time |
|---|------|-----|------|
| 7A-1 | Add `SlopeEstimator` class that infers slope angle from gate spacing compression | Claude builds | 1 hr |
| 7A-2 | Integrate into pipeline, re-validate physics with slope-corrected speeds | Claude builds | 30 min |
| 7A-3 | Write unit tests for slope estimation | Claude builds | 30 min |

**Track B: Share Your Work (do alongside Track A)**

| # | Task | Who | Time |
|---|------|-----|------|
| 7B-1 | Push latest code to GitHub | You | 5 min |
| 7B-2 | Record a 2–3 min demo video: input video → run command → show output | You | 1 hr |
| 7B-3 | Write a short blog post about building it | You | 1–2 hrs |
| 7B-4 | Share on Twitter/LinkedIn and r/computervision, r/MachineLearning | You | 15 min |

**Done when:** Slope estimation working, demo video recorded, project shared publicly.

---

## Quick Reference — Key Commands

```bash
# Extract frames from videos
python scripts/extract_frames.py data/raw_videos/ --balanced

# Train gate detector
python scripts/train_detector.py --data data/annotations/data.yaml

# Process a single video
python scripts/process_video.py VIDEO_PATH --gate-model models/gate_detector_best.pt --summary --demo-video

# Process all test videos
python scripts/process_video.py data/test_videos/ --gate-model models/gate_detector_best.pt --summary

# Evaluate accuracy against ground truth
python scripts/evaluate.py --predictions artifacts/outputs/race1_analysis.json --ground-truth data/annotations/race1_gt.json

# Run physics tests
python tests/test_physics.py
```

## Project Structure

```
Stanford application project/
├── ski_racing/           # Core Python package
│   ├── detection.py      #   Gate detection + temporal occlusion handling
│   ├── tracking.py       #   Skier tracking (ByteTrack + fallback)
│   ├── transform.py      #   2D → 3D homography
│   ├── physics.py        #   Physics validation engine
│   ├── pipeline.py       #   End-to-end processing
│   └── visualize.py      #   Trajectory overlay + summary figures
├── scripts/              # CLI tools
│   ├── extract_frames.py #   Frame extraction from videos
│   ├── train_detector.py #   Model training
│   ├── process_video.py  #   Full pipeline runner
│   └── evaluate.py       #   Accuracy evaluation
├── tests/
│   └── test_physics.py   #   Physics engine unit tests
├── docs/                 #   Guides, plans, reports
├── tools/
│   └── video_scraper/    #   Video scrape helpers + source lists
├── data/                 #   Your videos and annotations (not in git)
├── models/               #   Trained model weights (not in git)
├── artifacts/
│   ├── outputs/          #   Analysis results (not in git)
│   └── training_results/ #   Training artifacts (not in git)
├── runs/                 #   Ultralytics run outputs (not in git)
└── requirements.txt
```
