"""
Pose detection module with YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple
from dataclasses import dataclass

from config import PoseConfig


@dataclass
class DetectedPerson:
    """Information of a detected person"""
    person_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: np.ndarray  # (17, 3) - x, y, confidence
    confidence: float


class PoseDetector:
    """Human pose detection with YOLOv8-Pose"""
    
    def __init__(self, config: PoseConfig = None):
        self.config = config or PoseConfig()
        
        print(f"📦 Loading model: {self.config.model_name}")
        self.model = YOLO(self.config.model_name)
        print("✅ Model loaded successfully")
    
    def detect(self, frame: np.ndarray) -> List[DetectedPerson]:
        """
        Detect people and their keypoints in a frame

        Args:
            frame: BGR image

        Returns:
            List of detected people
        """
        results = self.model(
            frame, 
            verbose=False,
            conf=self.config.confidence_threshold
        )
        
        detected_persons = []
        
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            
            boxes = result.boxes
            keypoints = result.keypoints
            
            for idx, (box, kps) in enumerate(zip(boxes, keypoints)):
                # Extract bounding box
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # Extract keypoints
                kps_data = kps.data[0].cpu().numpy()
                
                person = DetectedPerson(
                    person_id=idx,
                    bbox=tuple(bbox),
                    keypoints=kps_data,
                    confidence=conf
                )
                
                detected_persons.append(person)
        
        return detected_persons
