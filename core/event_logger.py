"""
Event logging module
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from core.fall_analyzer import PersonState


@dataclass
class Event:
    """A logged event"""
    timestamp: str
    event_type: str  # FALL, IMMOBILITY
    person_id: int
    confidence: float
    body_angle: float = None
    details: Dict[str, Any] = None


class EventLogger:
    """Logging and managing events"""
    
    def __init__(self, log_file: str = "fall_events.json"):
        self.log_file = Path(log_file)
        self.events: List[Event] = []

        # Load previous events if they exist
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.events = [Event(**e) for e in data]
            except:
                pass
    
    def log_fall(
        self,
        person_id: int,
        confidence: float,
        body_angle: float = None
    ):
        """Log fall event"""
        event = Event(
            timestamp=datetime.now().isoformat(),
            event_type="FALL",
            person_id=person_id,
            confidence=round(confidence, 3),
            body_angle=round(body_angle, 1) if body_angle else None
        )
        
        self.events.append(event)
        self._save()
        self._print_event(event, "🚨 FALL DETECTED")
    
    def log_immobility(
        self,
        person_id: int,
        duration_seconds: float
    ):
        """Log immobility event"""
        event = Event(
            timestamp=datetime.now().isoformat(),
            event_type="IMMOBILITY",
            person_id=person_id,
            confidence=0.9,
            details={"duration_seconds": round(duration_seconds, 1)}
        )
        
        self.events.append(event)
        self._save()
        self._print_event(event, "⚠️ IMMOBILITY ALERT")
    
    def _print_event(self, event: Event, prefix: str):
        """Print event in console"""
        print("\n" + "=" * 50)
        print(f"{prefix}")
        print(f"  Time: {event.timestamp}")
        print(f"  Person ID: {event.person_id}")
        print(f"  Confidence: {event.confidence}")
        if event.body_angle:
            print(f"  Body Angle: {event.body_angle}°")
        if event.details:
            print(f"  Details: {event.details}")
        print("=" * 50 + "\n")
    
    def _save(self):
        """Save events to file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(e) for e in self.events], 
                f, 
                indent=2, 
                ensure_ascii=False
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Statistical summary of events"""
        falls = [e for e in self.events if e.event_type == "FALL"]
        immobilities = [e for e in self.events if e.event_type == "IMMOBILITY"]
        
        return {
            "total_events": len(self.events),
            "fall_count": len(falls),
            "immobility_count": len(immobilities),
            "last_event": self.events[-1].timestamp if self.events else None
        }
