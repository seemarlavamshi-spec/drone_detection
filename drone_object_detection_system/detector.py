"""
YOLO Object Detection Module for Drone Aerial Imagery.
Supports YOLOv8 and YOLO11 models from Ultralytics.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

DEFAULT_DRONE_CLASSES = [
    "person", "car", "motorcycle", "bus", "truck", "bicycle", "boat", "airplane"
]


class DroneYOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35, iou_thresh: float = 0.45):
        self.model_name = model_name
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.model = YOLO(model_name)
        self.class_name_to_id = {name: idx for idx, name in self.model.names.items()}
        self.selected_class_ids = [
            self.class_name_to_id[c] for c in DEFAULT_DRONE_CLASSES if c in self.class_name_to_id
        ]

    def set_target_classes(self, class_names: List[str]):
        self.selected_class_ids = [
            self.class_name_to_id[c] for c in class_names if c in self.class_name_to_id
        ]

    def set_confidence(self, conf: float):
        self.confidence = max(0.05, min(0.95, conf))

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_thresh,
            classes=self.selected_class_ids if self.selected_class_ids else None,
            verbose=False
        )
        detections = []
        if not results or len(results) == 0:
            return detections
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            c_id = cls_ids[i]
            c_name = self.model.names.get(c_id, f"class_{c_id}")
            detections.append({
                "bbox": [float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])],
                "confidence": float(confs[i]),
                "class_id": int(c_id),
                "class_name": str(c_name)
            })
        return detections
