"""
Display and drawing module
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from datetime import datetime

from config import KeypointIndex, Colors
from core.fall_analyzer import PersonState, AnalysisResult
from utils.keypoint_utils import get_keypoint


# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    # torso
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),

    # left arm
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),

    # right arm
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),

    # left leg
    (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
    (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),

    # right leg
    (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
    (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE),
]


def get_state_color(state: PersonState) -> Tuple[int, int, int]:
    """Color matching the state"""
    color_map = {
        PersonState.NORMAL: Colors.GREEN,
        PersonState.WARNING: Colors.YELLOW,
        PersonState.FALLING: Colors.ORANGE,
        PersonState.FALLEN: Colors.RED,
        PersonState.IMMOBILE: Colors.RED,
    }
    return color_map.get(state, Colors.WHITE)


def get_state_text(state: PersonState) -> str:
    """Status text"""
    text_map = {
        PersonState.NORMAL: "Normal",
        PersonState.WARNING: "Warning",
        PersonState.FALLING: "FALLING!",
        PersonState.FALLEN: "FALL DETECTED!",
        PersonState.IMMOBILE: "IMMOBILE - ALERT!",
    }
    return text_map.get(state, "Unknown")


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = Colors.SKELETON,
    point_radius: int = 5,
    line_thickness: int = 2
):
    """
    Draw body skeleton on image
    """
    # Draw points
    for i in range(17):
        point = get_keypoint(keypoints, i)
        if point:
            cv2.circle(
                frame, 
                (int(point[0]), int(point[1])), 
                point_radius, 
                color, 
                -1
            )
    
    # Draw connection lines
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start = get_keypoint(keypoints, start_idx)
        end = get_keypoint(keypoints, end_idx)
        
        if start and end:
            cv2.line(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                color,
                line_thickness
            )


def draw_person_info(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    result: AnalysisResult,
    person_id: int
):
    """
    Draw person information (bbox, status, metrics)
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = get_state_color(result.state)
    
    # Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Text background
    text_bg_height = 70
    cv2.rectangle(
        frame, 
        (x1, y1 - text_bg_height), 
        (x2, y1), 
        color, 
        -1
    )
    
    # Status text
    status_text = get_state_text(result.state)
    cv2.putText(
        frame, status_text,
        (x1 + 5, y1 - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        Colors.WHITE, 2
    )
    
    # Metrics
    metrics = []
    if result.body_angle is not None:
        metrics.append(f"Angle: {result.body_angle:.0f}°")
    if result.aspect_ratio is not None:
        metrics.append(f"Ratio: {result.aspect_ratio:.2f}")
    
    metrics_text = " | ".join(metrics)
    cv2.putText(
        frame, metrics_text,
        (x1 + 5, y1 - 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        Colors.WHITE, 1
    )
    
    # Confidence
    conf_text = f"Conf: {result.confidence:.0%}"
    cv2.putText(
        frame, conf_text,
        (x1 + 5, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        Colors.WHITE, 1
    )


def draw_dashboard(
    frame: np.ndarray,
    total_persons: int,
    fall_count: int,
    fps: float
):
    """
    Draw general information dashboard
    """
    h, w = frame.shape[:2]

    # Dashboard background
    dashboard_height = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, dashboard_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(
        frame, "FALL DETECTION DEMO - Phase 1",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        Colors.WHITE, 2
    )

    # Time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame, timestamp,
        (w - 200, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        Colors.WHITE, 1
    )
    
    # Statistics
    stats = f"Persons: {total_persons} | Falls: {fall_count} | FPS: {fps:.1f}"
    cv2.putText(
        frame, stats,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        Colors.GREEN, 2
    )
    
    # Guide
    cv2.putText(
        frame, "Press 'Q' to quit | 'R' to reset",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        Colors.WHITE, 1
    )
