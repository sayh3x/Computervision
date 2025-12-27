"""
Fall Detection System - First Demo
Proof of Concept Demo

Run:
    python main.py                    # webcam
    python main.py --video path.mp4   # video file
"""

import cv2
import time
import argparse
from typing import Optional

from config import (
    CameraConfig, 
    PoseConfig, 
    FallDetectionConfig, 
    ImmobilityConfig
)
from core.pose_detector import PoseDetector
from core.fall_analyzer import FallAnalyzer, PersonState
from core.event_logger import EventLogger
from utils.visualization import (
    draw_skeleton, 
    draw_person_info, 
    draw_dashboard,
    get_state_color
)


class FallDetectionDemo:
    """
    Main class of fall detection demo
    """
    
    def __init__(
        self,
        video_source: Optional[str] = None,
        camera_config: CameraConfig = None,
        pose_config: PoseConfig = None,
        fall_config: FallDetectionConfig = None,
        immobility_config: ImmobilityConfig = None
    ):
        self.camera_config = camera_config or CameraConfig()
        
        if video_source:
            self.camera_config.source = video_source
        
        # System components
        self.pose_detector = PoseDetector(pose_config or PoseConfig())
        self.fall_analyzer = FallAnalyzer(
            fall_config or FallDetectionConfig(),
            immobility_config or ImmobilityConfig()
        )
        self.event_logger = EventLogger()

        # Statistics
        self.fall_count = 0
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
    
    def run(self):
        """Main demo execution"""
        
        print("\n" + "=" * 60)
        print("  🎥 FALL DETECTION DEMO - Phase 1")
        print("  Proof of Concept")
        print("=" * 60)
        
        # Open video source
        cap = cv2.VideoCapture(self.camera_config.source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_config.fps)
        
        print(f"✅ Video source opened: {self.camera_config.source}")
        print(f"📐 Resolution: {self.camera_config.width}x{self.camera_config.height}")
        print("\n🔄 Processing... Press 'Q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    # For video file, start from beginning
                    if isinstance(self.camera_config.source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)

                # Display
                cv2.imshow("Fall Detection Demo", processed_frame)

                # Key controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.reset()
                    print("🔄 System reset")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Show summary
            self.print_summary()
    
    def process_frame(self, frame) -> any:
        """Process one frame"""
        
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

        # Detect pose
        persons = self.pose_detector.detect(frame)

        # Analyze each person
        for person in persons:
            # Fall analysis
            result = self.fall_analyzer.analyze(
                person.person_id,
                person.keypoints,
                current_time
            )

            # Log fall event
            if result.is_fall_event:
                self.fall_count += 1
                self.event_logger.log_fall(
                    person.person_id,
                    result.confidence,
                    result.body_angle
                )
            
            # Log immobility event
            if result.is_immobile_event:
                history = self.fall_analyzer.histories.get(person.person_id)
                if history and history.fall_detected_time:
                    duration = current_time - history.fall_detected_time
                    self.event_logger.log_immobility(
                        person.person_id,
                        duration
                    )
            
            # Draw skeleton
            color = get_state_color(result.state)
            draw_skeleton(frame, person.keypoints, color)

            # Draw information
            draw_person_info(
                frame,
                person.bbox,
                result,
                person.person_id
            )

        # Draw dashboard
        draw_dashboard(
            frame,
            len(persons),
            self.fall_count,
            self.fps
        )
        
        return frame
    
    def reset(self):
        """Reset system"""
        self.fall_analyzer.histories.clear()
        self.fall_count = 0

    def print_summary(self):
        """Print final summary"""
        summary = self.event_logger.get_summary()
        
        print("\n" + "=" * 60)
        print("  📊 SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total Events: {summary['total_events']}")
        print(f"  Falls Detected: {summary['fall_count']}")
        print(f"  Immobility Alerts: {summary['immobility_count']}")
        print(f"\n  📁 Events saved to: fall_events.json")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection Demo - Phase 1"
    )
    parser.add_argument(
        "--video", 
        type=str, 
        default=None,
        help="Path to video file (default: webcam)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Frame width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Frame height"
    )
    
    args = parser.parse_args()
    
    camera_config = CameraConfig(
        source=args.video if args.video else 0,
        width=args.width,
        height=args.height
    )
    
    demo = FallDetectionDemo(
        camera_config=camera_config
    )
    
    demo.run()


if __name__ == "__main__":
    main()
