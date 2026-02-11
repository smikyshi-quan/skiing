"""
Gate detection module for ski racing analysis.
Uses YOLOv8 for detecting red and blue gates in race footage.
"""
import cv2
import numpy as np
from ultralytics import YOLO


class GateDetector:
    """
    Detect ski racing gates using a fine-tuned YOLOv8 model.
    """

    def __init__(self, model_path):
        """
        Args:
            model_path: Path to trained YOLOv8 weights (.pt file).
        """
        self.model = YOLO(model_path)

    def detect_in_frame(self, frame, conf=0.3):
        """
        Detect gates in a single frame.

        Args:
            frame: BGR image (numpy array).
            conf: Minimum confidence threshold.

        Returns:
            List of gate detections, each with class, center, base, confidence.
        """
        results = self.model(frame, conf=conf, verbose=False)
        gates = []

        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy()
            gates.append({
                "class": int(box.cls[0]),
                "class_name": self.model.names[int(box.cls[0])],
                "center_x": float((bbox[0] + bbox[2]) / 2),
                "center_y": float((bbox[1] + bbox[3]) / 2),
                "base_y": float(bbox[3]),  # Bottom of bounding box = base of pole
                "bbox": bbox.tolist(),
                "confidence": float(box.conf[0]),
            })

        # Sort gates by Y position (top to bottom = far to near)
        gates.sort(key=lambda g: g["base_y"])
        return gates

    def detect_from_first_frame(self, video_path, conf=0.3):
        """
        Detect gates from the first frame of a video.
        Best used before the race starts when all gates are fully visible.

        Args:
            video_path: Path to video file.
            conf: Minimum confidence threshold.

        Returns:
            List of gate detections.
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read video: {video_path}")

        return self.detect_in_frame(frame, conf=conf)

    def detect_from_best_frame(self, video_path, conf=0.3, max_frames=150, stride=5):
        """
        Scan early frames and return the frame with the most detected gates.

        Args:
            video_path: Path to video file.
            conf: Confidence threshold.
            max_frames: Max frames to scan from the start.
            stride: Process every Nth frame.

        Returns:
            List of gate detections from the best frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        best_gates = []
        best_frame_idx = -1
        frame_idx = 0
        stride = max(1, int(stride))
        max_frames = max(1, int(max_frames))

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            gates = self.detect_in_frame(frame, conf=conf)
            if len(gates) > len(best_gates):
                best_gates = gates
                best_frame_idx = frame_idx

            frame_idx += 1

        cap.release()

        if best_frame_idx >= 0:
            print(f"         Best gate frame: {best_frame_idx} with {len(best_gates)} gates")

        return best_gates


class TemporalGateTracker:
    """
    Track gates across frames with temporal consistency.
    Handles the Slalom Problem: gates temporarily disappearing when
    racers hit them (cross-blocking technique).
    """

    def __init__(self, max_missing_frames=10, match_threshold=50.0):
        """
        Args:
            max_missing_frames: How many frames a gate can be missing before removal.
            match_threshold: Max pixel distance to consider a detection matching a tracked gate.
        """
        self.gate_memory = {}       # gate_id -> last known position
        self.missing_frames = {}    # gate_id -> frames since last detection
        self.confidence = {}        # gate_id -> current confidence level
        self.max_missing = max_missing_frames
        self.match_threshold = match_threshold
        self.next_id = 0
        self.frame_history = {}     # frame_idx -> {gate_id: (center_x, base_y)}

    def initialize(self, gates):
        """
        Initialize tracker with gates detected from the first (clean) frame.

        Args:
            gates: List of gate detections from GateDetector.
        """
        for gate in gates:
            self.gate_memory[self.next_id] = {
                "center_x": gate["center_x"],
                "base_y": gate["base_y"],
                "class": gate["class"],
                "class_name": gate.get("class_name", "unknown"),
            }
            self.missing_frames[self.next_id] = 0
            self.confidence[self.next_id] = 1.0  # High confidence from clean frame
            self.next_id += 1

    def update(self, detected_gates):
        """
        Update tracked gates with new detections.
        Handles occlusion gracefully by maintaining last known positions.

        Args:
            detected_gates: List of gate detections from current frame.

        Returns:
            List of all tracked gates (including interpolated missing ones).
        """
        matched_detections = set()
        matched_tracked = set()

        # Match detected gates to tracked gates (nearest neighbor)
        for gate_id, tracked in self.gate_memory.items():
            best_match = None
            best_dist = float("inf")

            for i, det in enumerate(detected_gates):
                if i in matched_detections:
                    continue
                dist = (
                    (tracked["center_x"] - det["center_x"]) ** 2
                    + (tracked["base_y"] - det["base_y"]) ** 2
                ) ** 0.5

                if dist < self.match_threshold and dist < best_dist:
                    best_dist = dist
                    best_match = i

            if best_match is not None:
                # Update with fresh detection
                det = detected_gates[best_match]
                self.gate_memory[gate_id]["center_x"] = det["center_x"]
                self.gate_memory[gate_id]["base_y"] = det["base_y"]
                self.missing_frames[gate_id] = 0
                self.confidence[gate_id] = min(1.0, self.confidence[gate_id] + 0.1)
                matched_detections.add(best_match)
                matched_tracked.add(gate_id)
            else:
                # Gate not detected this frame
                self.missing_frames[gate_id] += 1
                # Decay confidence while missing
                self.confidence[gate_id] = max(
                    0.1, self.confidence[gate_id] - 0.05
                )

        # Remove gates that have been missing too long
        to_remove = [
            gid
            for gid, missed in self.missing_frames.items()
            if missed > self.max_missing
        ]
        for gid in to_remove:
            del self.gate_memory[gid]
            del self.missing_frames[gid]
            del self.confidence[gid]

        # Return all tracked gates with confidence info
        result = []
        for gate_id, pos in self.gate_memory.items():
            result.append({
                "gate_id": gate_id,
                "center_x": pos["center_x"],
                "base_y": pos["base_y"],
                "class": pos["class"],
                "class_name": pos["class_name"],
                "confidence": self.confidence[gate_id],
                "is_interpolated": self.missing_frames[gate_id] > 0,
                "missing_frames": self.missing_frames[gate_id],
            })

        return result

    def update_with_frame_idx(self, detected_gates, frame_idx):
        """
        Update tracked gates and record per-frame positions.

        Args:
            detected_gates: List of gate detections from current frame.
            frame_idx: Current frame index (for history recording).

        Returns:
            List of all tracked gates (same as update()).
        """
        result = self.update(detected_gates)

        # Record current positions of all tracked gates for this frame
        frame_positions = {}
        for gate_id, pos in self.gate_memory.items():
            if self.missing_frames.get(gate_id, 0) == 0:
                # Only record gates that were actually detected this frame
                frame_positions[gate_id] = (pos["center_x"], pos["base_y"])
        self.frame_history[frame_idx] = frame_positions

        return result

    def get_frame_history(self):
        """
        Get per-frame gate positions.

        Returns:
            Dict: {frame_idx: {gate_id: (center_x, base_y)}}
        """
        return self.frame_history

    def get_status(self):
        """Get summary of tracking status."""
        total = len(self.gate_memory)
        missing = sum(1 for m in self.missing_frames.values() if m > 0)
        return {
            "total_tracked": total,
            "currently_visible": total - missing,
            "currently_interpolated": missing,
            "avg_confidence": (
                sum(self.confidence.values()) / total if total > 0 else 0
            ),
        }
