"""
Fall analysis module
Pure computational logic based on Keypoints
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum

from config import FallDetectionConfig, ImmobilityConfig
from utils.keypoint_utils import (
    calculate_center_of_mass,
    calculate_body_angle,
    calculate_body_aspect_ratio,
    calculate_vertical_velocity,
    calculate_angle_change_rate,
    calculate_movement_amount
)


class PersonState(Enum):
    """Person state"""
    NORMAL = "normal"
    WARNING = "warning"
    FALLING = "falling"
    FALLEN = "fallen"
    IMMOBILE = "immobile"


@dataclass
class AnalysisResult:
    """Result of analyzing one frame"""
    state: PersonState
    confidence: float
    body_angle: Optional[float]
    aspect_ratio: Optional[float]
    vertical_velocity: Optional[float]
    is_fall_event: bool = False      # Is it the moment of fall?
    is_immobile_event: bool = False  # Is there suspicious immobility?


@dataclass
class PersonHistory:
    """History of a person for temporal analysis"""
    positions: deque = field(default_factory=lambda: deque(maxlen=60))
    angles: deque = field(default_factory=lambda: deque(maxlen=60))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=60))
    
    current_state: PersonState = PersonState.NORMAL
    fall_detected_time: Optional[float] = None
    last_movement_time: Optional[float] = None
    immobility_alerted: bool = False


class FallAnalyzer:
    """
    Fall analyzer

    📐 Fall detection criteria:
    1. Body angle > 60° (from vertical)
    2. Aspect ratio > 1.3
    3. Vertical velocity > 12 pixels/frame
    4. Sudden angle change > 25°

    Each criterion has a score and sum > 0.55 = fall
    """
    
    def __init__(
        self, 
        fall_config: FallDetectionConfig = None,
        immobility_config: ImmobilityConfig = None
    ):
        self.fall_config = fall_config or FallDetectionConfig()
        self.immobility_config = immobility_config or ImmobilityConfig()

        # History of each person
        self.histories: Dict[int, PersonHistory] = {}
    
    def get_or_create_history(self, person_id: int) -> PersonHistory:
        """Get or create history for a person"""
        if person_id not in self.histories:
            self.histories[person_id] = PersonHistory()
            self.histories[person_id].last_movement_time = time.time()
        return self.histories[person_id]
    
    def analyze(
        self,
        person_id: int,
        keypoints,
        current_time: float = None
    ) -> AnalysisResult:
        """
        Analyze status of a person

        Args:
            person_id: Person ID
            keypoints: Keypoint array
            current_time: Current time

        Returns:
            Analysis result
        """
        if current_time is None:
            current_time = time.time()
        
        history = self.get_or_create_history(person_id)

        # ═══════════════════════════════════════════
        # Stage 1: Calculate current metrics
        # ═══════════════════════════════════════════
        
        center = calculate_center_of_mass(keypoints)
        angle = calculate_body_angle(keypoints)
        aspect_ratio = calculate_body_aspect_ratio(keypoints)

        # Save in history
        if center:
            history.positions.append(center)
        if angle is not None:
            history.angles.append(angle)
        history.timestamps.append(current_time)

        # If we don't have enough data
        if len(history.positions) < 5:
            return AnalysisResult(
                state=PersonState.NORMAL,
                confidence=0.0,
                body_angle=angle,
                aspect_ratio=aspect_ratio,
                vertical_velocity=None
            )
        
        # ═══════════════════════════════════════════
        # Stage 2: Calculate temporal metrics
        # ═══════════════════════════════════════════
        
        positions_list = list(history.positions)
        angles_list = list(history.angles)
        
        vertical_velocity = calculate_vertical_velocity(
            positions_list, 
            frame_count=10
        )
        
        angle_change = calculate_angle_change_rate(
            angles_list, 
            frame_count=10
        )
        
        movement = calculate_movement_amount(
            positions_list,
            frame_count=self.immobility_config.check_frames
        )
        
        # ═══════════════════════════════════════════
        # Stage 3: Calculate fall score
        # ═══════════════════════════════════════════
        
        confidence = 0.0

        # Criterion 1: Body angle 🔺
        if angle is not None:
            if angle > self.fall_config.angle_threshold_fall:
                confidence += 0.35
            elif angle > self.fall_config.angle_threshold_warning:
                confidence += 0.15
        
        # Criterion 2: Aspect ratio 📐
        if aspect_ratio is not None:
            if aspect_ratio > self.fall_config.aspect_ratio_threshold:
                confidence += 0.25
        
        # Criterion 3: Fall speed ⬇️
        if vertical_velocity is not None:
            if vertical_velocity > self.fall_config.vertical_velocity_threshold:
                confidence += 0.25
        
        # Criterion 4: Sudden angle change ⚡
        if angle_change is not None:
            if angle_change > self.fall_config.sudden_angle_change:
                confidence += 0.15
        
        # ═══════════════════════════════════════════
        # Stage 4: Determine status
        # ═══════════════════════════════════════════
        
        is_fall_event = False
        is_immobile_event = False

        # Detect new fall
        if confidence >= self.fall_config.fall_confidence_threshold:
            if history.current_state not in [PersonState.FALLEN, PersonState.IMMOBILE]:
                is_fall_event = True
                history.fall_detected_time = current_time
            
            history.current_state = PersonState.FALLEN
        
        # Check immobility after fall
        elif history.current_state == PersonState.FALLEN:
            if movement is not None:
                if movement < self.immobility_config.movement_threshold:
                    # Is immobile
                    time_since_fall = current_time - (history.fall_detected_time or current_time)
                    
                    if time_since_fall > self.immobility_config.threshold_seconds:
                        if not history.immobility_alerted:
                            is_immobile_event = True
                            history.immobility_alerted = True
                        history.current_state = PersonState.IMMOBILE
                else:
                    # Moved - reset
                    history.current_state = PersonState.NORMAL
                    history.fall_detected_time = None
                    history.immobility_alerted = False
        
        # Initial warning
        elif confidence > 0.3:
            history.current_state = PersonState.WARNING
        
        else:
            history.current_state = PersonState.NORMAL
            history.immobility_alerted = False
        
        return AnalysisResult(
            state=history.current_state,
            confidence=confidence,
            body_angle=angle,
            aspect_ratio=aspect_ratio,
            vertical_velocity=vertical_velocity,
            is_fall_event=is_fall_event,
            is_immobile_event=is_immobile_event
        )
    
    def reset_person(self, person_id: int):
        """Reset status of a person"""
        if person_id in self.histories:
            del self.histories[person_id]
