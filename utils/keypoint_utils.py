"""
Helper functions for working with Keypoints
Pure computational logic - without ML
"""

import numpy as np
from typing import Optional, Tuple, List
from config import KeypointIndex


def get_keypoint(
    keypoints: np.ndarray,
    index: int,
    min_confidence: float = 0.3
) -> Optional[Tuple[float, float]]:
    """
    Extract coordinates of a keypoint with confidence check

    Args:
        keypoints: Keypoint array (17, 3) or (17, 2)
        index: Keypoint index
        min_confidence: Minimum acceptable confidence

    Returns:
        (x, y) or None
    """
    if keypoints is None or len(keypoints) <= index:
        return None
    
    kp = keypoints[index]

    # If we have confidence, check it
    if len(kp) >= 3:
        if kp[2] < min_confidence:
            return None
        return (float(kp[0]), float(kp[1]))

    # If we only have x, y
    if len(kp) >= 2 and kp[0] > 0 and kp[1] > 0:
        return (float(kp[0]), float(kp[1]))
    
    return None


def get_midpoint(
    keypoints: np.ndarray,
    idx1: int,
    idx2: int,
    min_confidence: float = 0.3
) -> Optional[Tuple[float, float]]:
    """
    Calculate midpoint between two keypoints
    """
    p1 = get_keypoint(keypoints, idx1, min_confidence)
    p2 = get_keypoint(keypoints, idx2, min_confidence)
    
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def calculate_center_of_mass(
    keypoints: np.ndarray,
    min_confidence: float = 0.3
) -> Optional[Tuple[float, float]]:
    """
    Calculate approximate center of mass of body
    Using shoulders and hips

    📐 Logic:
    Center of mass ≈ average of core body points (shoulders + hips)
    """
    core_points = []
    
    for idx in KeypointIndex.CORE:
        point = get_keypoint(keypoints, idx, min_confidence)
        if point:
            core_points.append(point)
    
    if len(core_points) < 2:
        return None
    
    x = np.mean([p[0] for p in core_points])
    y = np.mean([p[1] for p in core_points])
    
    return (x, y)


def calculate_body_angle(
    keypoints: np.ndarray,
    min_confidence: float = 0.3
) -> Optional[float]:
    """
    Calculate body angle relative to vertical

    📐 Logic:
    - 0° = completely standing (vertical)
    - 90° = completely lying down (horizontal)

    Body vector = from hip midpoint to shoulder midpoint
    Angle = deviation of this vector from Y axis
    """
    # Shoulder midpoint
    shoulder_mid = get_midpoint(
        keypoints,
        KeypointIndex.LEFT_SHOULDER,
        KeypointIndex.RIGHT_SHOULDER,
        min_confidence
    )

    # Hip midpoint
    hip_mid = get_midpoint(
        keypoints,
        KeypointIndex.LEFT_HIP,
        KeypointIndex.RIGHT_HIP,
        min_confidence
    )
    
    if shoulder_mid is None or hip_mid is None:
        return None

    # Body vector
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]  # In image, Y is positive downward

    # Angle from Y axis (vertical)
    # arctan2(dx, -dy) because:
    # - dx: horizontal deviation
    # - -dy: vertical direction (positive upward)
    angle_rad = np.arctan2(abs(dx), -dy)
    angle_deg = np.degrees(angle_rad)
    
    return abs(angle_deg)


def calculate_body_aspect_ratio(
    keypoints: np.ndarray,
    min_confidence: float = 0.3
) -> Optional[float]:
    """
    Calculate body aspect ratio (width to height)

    📐 Logic:
    - ratio < 1: standing (height > width)
    - ratio > 1: lying down (width > height)

    Using bounding box of skeleton points
    """
    valid_points = []
    
    for i in range(17):
        point = get_keypoint(keypoints, i, min_confidence)
        if point:
            valid_points.append(point)
    
    if len(valid_points) < 4:
        return None
    
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if height < 20:  # Prevent division by zero
        return float('inf')
    
    return width / height


def calculate_vertical_velocity(
    positions: List[Tuple[float, float]],
    frame_count: int = 10
) -> Optional[float]:
    """
    Calculate vertical velocity of center of mass

    📐 Logic:
    Velocity = Y change in last N frames
    Positive = movement downward (fall)
    Negative = movement upward (rising)
    """
    if len(positions) < frame_count:
        return None
    
    recent = positions[-frame_count:]
    
    y_start = recent[0][1]
    y_end = recent[-1][1]
    
    velocity = (y_end - y_start) / frame_count
    
    return velocity


def calculate_angle_change_rate(
    angles: List[float],
    frame_count: int = 10
) -> Optional[float]:
    """
    Calculate body angle change rate

    📐 Logic:
    First half vs second half of range
    Positive change = going towards horizontal
    """
    if len(angles) < frame_count:
        return None
    
    recent = [a for a in angles[-frame_count:] if a is not None]
    
    if len(recent) < frame_count // 2:
        return None
    
    mid = len(recent) // 2
    first_half = np.mean(recent[:mid])
    second_half = np.mean(recent[mid:])
    
    return second_half - first_half


def calculate_movement_amount(
    positions: List[Tuple[float, float]],
    frame_count: int = 20
) -> Optional[float]:
    """
    Calculate movement amount (for immobility detection)

    📐 Logic:
    Position standard deviation = movement indicator
    Low = immobile
    High = moving
    """
    if len(positions) < frame_count:
        return None
    
    recent = positions[-frame_count:]
    
    std_x = np.std([p[0] for p in recent])
    std_y = np.std([p[1] for p in recent])
    
    return std_x + std_y
