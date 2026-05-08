"""
koi_video_edge_tracker.py

Prototype computer-vision pipeline for tracking koi movement and position from video.

Current method:
- OpenCV frame preprocessing
- Canny edge detection
- Morphological cleanup
- Contour detection
- Centroid tracking with simple nearest-neighbor association
- Rule-based behavior flags:
    - PIPING: fish remains near the surface/top region with low movement
    - DIVING: fish moves downward rapidly over a short time window

This is a baseline, not a clinical health classifier.
A later ML version should replace contour detection with segmentation/detection
and train behavior classifiers from labeled koi footage.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    centroid: Point
    bbox: BBox
    area: float


@dataclass
class Track:
    track_id: int
    centroid: Point
    bbox: BBox
    area: float
    missed_frames: int = 0
    history: Deque[Point] = field(default_factory=lambda: deque(maxlen=60))
    behavior_flags: Dict[str, bool] = field(default_factory=dict)

    def update(self, detection: Detection) -> None:
        self.centroid = detection.centroid
        self.bbox = detection.bbox
        self.area = detection.area
        self.missed_frames = 0
        self.history.append(detection.centroid)

    def mark_missed(self) -> None:
        self.missed_frames += 1


class KoiEdgeDetector:
    def __init__(
        self,
        min_contour_area: int = 400,
        max_contour_area: int = 50_000,
        canny_low: int = 40,
        canny_high: int = 120,
    ) -> None:
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.canny_low = canny_low
        self.canny_high = canny_high

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Return candidate koi detections and the edge/debug mask."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reduce high-frequency water shimmer and camera noise.
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Optional contrast normalization helps in uneven lighting.
        equalized = cv2.equalizeHist(blurred)

        edges = cv2.Canny(equalized, self.canny_low, self.canny_high)

        # Connect broken edge fragments into larger fish-like blobs.
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(h, 1)

            # Loose shape filter. Koi can appear long, angled, or partially occluded.
            if aspect_ratio < 0.25 or aspect_ratio > 6.0:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            detections.append(Detection((cx, cy), (x, y, w, h), area))

        return detections, dilated


class SimpleTracker:
    def __init__(self, max_distance: float = 80.0, max_missed_frames: int = 20) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())
        matches: List[Tuple[int, int]] = []

        # Greedy nearest-neighbor matching. Adequate for prototype use.
        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for det_idx, detection in enumerate(detections):
                dist = self._distance(track.centroid, detection.centroid)
                if dist <= self.max_distance:
                    candidates.append((dist, track_id, det_idx))

        candidates.sort(key=lambda item: item[0])

        for _, track_id, det_idx in candidates:
            if track_id in unmatched_tracks and det_idx in unmatched_detections:
                matches.append((track_id, det_idx))
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(det_idx)

        for track_id, det_idx in matches:
            self.tracks[track_id].update(detections[det_idx])

        for track_id in list(unmatched_tracks):
            self.tracks[track_id].mark_missed()
            if self.tracks[track_id].missed_frames > self.max_missed_frames:
                del self.tracks[track_id]

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track = Track(
                track_id=self.next_track_id,
                centroid=detection.centroid,
                bbox=detection.bbox,
                area=detection.area,
            )
            track.history.append(detection.centroid)
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        return self.tracks


class BehaviorAnalyzer:
    def __init__(
        self,
        surface_fraction: float = 0.20,
        piping_min_frames: int = 45,
        piping_max_motion_px: float = 6.0,
        diving_window: int = 15,
        diving_min_drop_px: float = 45.0,
    ) -> None:
        self.surface_fraction = surface_fraction
        self.piping_min_frames = piping_min_frames
        self.piping_max_motion_px = piping_max_motion_px
        self.diving_window = diving_window
        self.diving_min_drop_px = diving_min_drop_px

    @staticmethod
    def _mean_step_distance(points: List[Point]) -> float:
        if len(points) < 2:
            return 0.0
        distances = [
            math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
            for i in range(1, len(points))
        ]
        return float(np.mean(distances))

    def analyze(self, track: Track, frame_height: int) -> Dict[str, bool]:
        history = list(track.history)
        flags = {
            "piping": False,
            "diving": False,
        }

        if len(history) >= self.piping_min_frames:
            recent = history[-self.piping_min_frames :]
            surface_y = frame_height * self.surface_fraction
            near_surface_ratio = sum(1 for _, y in recent if y <= surface_y) / len(recent)
            mean_motion = self._mean_step_distance(recent)

            # Piping proxy: prolonged near-surface presence with limited motion.
            if near_surface_ratio >= 0.75 and mean_motion <= self.piping_max_motion_px:
                flags["piping"] = True

        if len(history) >= self.diving_window:
            recent = history[-self.diving_window :]
            y_start = recent[0][1]
            y_end = recent[-1][1]
            vertical_drop = y_end - y_start

            # In image coordinates, larger y means lower in the frame.
            if vertical_drop >= self.diving_min_drop_px:
                flags["diving"] = True

        track.behavior_flags = flags
        return flags


def draw_overlay(
    frame: np.ndarray,
    tracks: Dict[int, Track],
    frame_height: int,
    surface_fraction: float,
) -> np.ndarray:
    output = frame.copy()
    surface_y = int(frame_height * surface_fraction)
    cv2.line(output, (0, surface_y), (frame.shape[1], surface_y), (255, 255, 255), 2)
    cv2.putText(
        output,
        "surface zone",
        (10, max(surface_y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    for track in tracks.values():
        x, y, w, h = track.bbox
        cx, cy = track.centroid
        flags = track.behavior_flags

        color = (0, 255, 0)
        label_flags = []
        if flags.get("piping"):
            color = (0, 0, 255)
            label_flags.append("PIPING")
        if flags.get("diving"):
            color = (0, 165, 255)
            label_flags.append("DIVING")

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.circle(output, (cx, cy), 4, color, -1)

        label = f"ID {track.track_id}"
        if label_flags:
            label += " " + ",".join(label_flags)

        cv2.putText(
            output,
            label,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        points = list(track.history)
        for i in range(1, len(points)):
            cv2.line(output, points[i - 1], points[i], color, 1)

    return output


def process_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv2.VideoWriter] = None
    if args.output:
        output_path = Path(args.output)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detector = KoiEdgeDetector(
        min_contour_area=args.min_area,
        max_contour_area=args.max_area,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )
    tracker = SimpleTracker(max_distance=args.max_track_distance)
    analyzer = BehaviorAnalyzer(
        surface_fraction=args.surface_fraction,
        piping_min_frames=args.piping_min_frames,
        piping_max_motion_px=args.piping_max_motion_px,
        diving_window=args.diving_window,
        diving_min_drop_px=args.diving_min_drop_px,
    )

    frame_index = 0
    behavior_counts = defaultdict(int)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections, edge_mask = detector.detect(frame)
        tracks = tracker.update(detections)

        for track in tracks.values():
            flags = analyzer.analyze(track, height)
            if flags.get("piping"):
                behavior_counts["piping"] += 1
            if flags.get("diving"):
                behavior_counts["diving"] += 1

        overlay = draw_overlay(frame, tracks, height, args.surface_fraction)

        if args.show_edges:
            edge_bgr = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
            overlay = np.hstack([overlay, edge_bgr])

        cv2.imshow("Koi behavior tracker", overlay)
        if writer is not None:
            # If showing edges side-by-side, keep output as overlay only to preserve size.
            writer.write(draw_overlay(frame, tracks, height, args.surface_fraction))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Processing complete.")
    print(f"Frames processed: {frame_index}")
    print(f"Piping frame-flags: {behavior_counts['piping']}")
    print(f"Diving frame-flags: {behavior_counts['diving']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track koi movement using edge detection.")
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output", default=None, help="Optional path for annotated output video.")

    parser.add_argument("--min-area", type=int, default=400, help="Minimum contour area.")
    parser.add_argument("--max-area", type=int, default=50000, help="Maximum contour area.")
    parser.add_argument("--canny-low", type=int, default=40, help="Lower Canny threshold.")
    parser.add_argument("--canny-high", type=int, default=120, help="Upper Canny threshold.")
    parser.add_argument(
        "--max-track-distance",
        type=float,
        default=80.0,
        help="Maximum centroid distance for matching detections to tracks.",
    )

    parser.add_argument(
        "--surface-fraction",
        type=float,
        default=0.20,
        help="Top fraction of the frame treated as surface zone.",
    )
    parser.add_argument(
        "--piping-min-frames",
        type=int,
        default=45,
        help="Number of recent frames used for piping detection.",
    )
    parser.add_argument(
        "--piping-max-motion-px",
        type=float,
        default=6.0,
        help="Maximum average frame-to-frame motion for piping flag.",
    )
    parser.add_argument(
        "--diving-window",
        type=int,
        default=15,
        help="Number of recent frames used for diving detection.",
    )
    parser.add_argument(
        "--diving-min-drop-px",
        type=float,
        default=45.0,
        help="Minimum downward pixel displacement for diving flag.",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Display edge mask beside annotated frame.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    process_video(parser.parse_args())
