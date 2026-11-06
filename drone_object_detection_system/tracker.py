"""
ByteTrack Object Tracking Module.
Maintains continuous object tracking, assigns persistent tracking IDs,
records motion trajectories, and computes unique object statistics without duplicate counts.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict, deque

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False


class DroneByteTracker:
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh
        ) if HAS_SUPERVISION else None

        self.trajectory_history = defaultdict(lambda: deque(maxlen=40))
        self.unique_track_ids = set()
        self.track_classes = {}
        self._fallback_next_id = 1
        self._fallback_tracks = {}

    def reset(self):
        if HAS_SUPERVISION:
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh
            )
        self.trajectory_history.clear()
        self.unique_track_ids.clear()
        self.track_classes.clear()
        self._fallback_next_id = 1
        self._fallback_tracks.clear()

    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        tracked_objects = []
        if HAS_SUPERVISION:
            if not detections:
                empty_det = sv.Detections.empty()
                self.byte_tracker.update_with_detections(empty_det)
                return []
            xyxy = np.array([d["bbox"] for d in detections])
            confidence = np.array([d["confidence"] for d in detections])
            class_id = np.array([d["class_id"] for d in detections])
            det_obj = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = self.byte_tracker.update_with_detections(det_obj)

            for i in range(len(tracked)):
                t_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i + 1
                box = [float(b) for b in tracked.xyxy[i]]
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.85
                c_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                c_name = next((d["class_name"] for d in detections if d["class_id"] == c_id), "object")
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0

                self.unique_track_ids.add(t_id)
                self.track_classes[t_id] = c_name
                self.trajectory_history[t_id].append((cx, cy))
                tracked_objects.append({
                    "track_id": t_id,
                    "bbox": box,
                    "class_name": c_name,
                    "confidence": conf,
                    "center": (cx, cy),
                    "trajectory": list(self.trajectory_history[t_id])
                })
        else:
            tracked_objects = self._fallback_update(detections)
        return tracked_objects

    def _fallback_update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        unmatched_dets = list(range(len(detections)))
        for t_id, t_info in list(self._fallback_tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for idx in unmatched_dets:
                iou = self._calculate_iou(t_info["bbox"], detections[idx]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= 0.3 and best_idx != -1:
                det = detections[best_idx]
                self._fallback_tracks[t_id]["bbox"] = det["bbox"]
                self._fallback_tracks[t_id]["lost"] = 0
                unmatched_dets.remove(best_idx)
                cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
                self.trajectory_history[t_id].append((cx, cy))
                self.unique_track_ids.add(t_id)
                self.track_classes[t_id] = det["class_name"]
                results.append({
                    "track_id": t_id, "bbox": det["bbox"], "class_name": det["class_name"],
                    "confidence": det["confidence"], "center": (cx, cy), "trajectory": list(self.trajectory_history[t_id])
                })
            else:
                self._fallback_tracks[t_id]["lost"] += 1
                if self._fallback_tracks[t_id]["lost"] > self.track_buffer:
                    del self._fallback_tracks[t_id]

        for idx in unmatched_dets:
            det = detections[idx]
            new_id = self._fallback_next_id
            self._fallback_next_id += 1
            self._fallback_tracks[new_id] = {"bbox": det["bbox"], "lost": 0, "class": det["class_name"]}
            cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
            cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
            self.trajectory_history[new_id].append((cx, cy))
            self.unique_track_ids.add(new_id)
            self.track_classes[new_id] = det["class_name"]
            results.append({
                "track_id": new_id, "bbox": det["bbox"], "class_name": det["class_name"],
                "confidence": det["confidence"], "center": (cx, cy), "trajectory": list(self.trajectory_history[new_id])
            })
        return results

    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1]) + max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1]) - inter
        return inter / union if union > 0 else 0.0

    def get_unique_counts(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for t_id, c_name in self.track_classes.items():
            counts[c_name] += 1
        return dict(counts)

    def get_total_unique(self) -> int:
        return len(self.unique_track_ids)
