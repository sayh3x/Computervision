"""
Fall Detection System Settings
Demo Phase 1 - Proof of Concept
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraConfig:
    """Camera Settings"""
    source: int | str = 0          # 0 = webcam, or video file path
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class PoseConfig:
    """Pose Detection Settings"""
    model_name: str = "yolov8n-pose.pt"  # Lightweight model for demo
    confidence_threshold: float = 0.5
    keypoint_confidence: float = 0.3


@dataclass
class FallDetectionConfig:
    """Fall Detection Settings"""
    # Body angle threshold (degrees from vertical)
    angle_threshold_warning: float = 45.0   # warning
    angle_threshold_fall: float = 60.0      # fall

    # Aspect ratio threshold
    aspect_ratio_threshold: float = 1.3

    # Fall speed threshold (pixels per frame)
    vertical_velocity_threshold: float = 12.0

    # Sudden angle change threshold (degrees)
    sudden_angle_change: float = 25.0

    # Number of frames for analysis
    history_frames: int = 30

    # Confidence threshold for fall detection
    fall_confidence_threshold: float = 0.55


@dataclass
class ImmobilityConfig:
    """Immobility Detection Settings"""
    # Suspicious immobility duration (seconds)
    threshold_seconds: float = 5.0

    # Movement threshold (pixels)
    movement_threshold: float = 15.0

    # Number of frames to check
    check_frames: int = 20


# COCO Keypoint Mapping
class KeypointIndex:
    """Keypoint indices in COCO format"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # Grouping for calculations
    UPPER_BODY = [LEFT_SHOULDER, RIGHT_SHOULDER]
    LOWER_BODY = [LEFT_HIP, RIGHT_HIP]
    CORE = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]


# Display Colors
class Colors:
    GREEN = (0, 255, 0)      # normal
    YELLOW = (0, 255, 255)   # warning
    ORANGE = (0, 165, 255)   # suspicious
    RED = (0, 0, 255)        # fall
    WHITE = (255, 255, 255)
    SKELETON = (255, 200, 100)
