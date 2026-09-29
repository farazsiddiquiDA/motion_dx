"""
Base clinical assessment models and lifecycle state machines for MOTION DX.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class AssessmentState(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"


@dataclass
class AssessmentSession:
    """Encapsulates all recorded kinematic telemetry and clinical metadata for one session."""
    session_id: str = field(default_factory=lambda: f"MDX-{int(time.time())}-{uuid.uuid4().hex[:4].upper()}")
    patient_id: str = "ANON-001"
    assessment_type: str = "LIMB_ROM"  # 'LIMB_ROM' | 'TMD' | 'GAIT'
    target_joint: str = "KNEE"          # 'KNEE' | 'ELBOW' | 'SHOULDER' | 'JAW' | 'LOWER_LIMBS'
    target_side: str = "RIGHT"          # 'RIGHT' | 'LEFT' | 'BILATERAL'
    camera_type: str = "Generic"
    is_depth_enabled: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    state: AssessmentState = AssessmentState.IDLE
    tracking_quality: str = "GOOD"
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    time_series: List[Dict[str, Any]] = field(default_factory=list)
    clinical_notes: str = ""
    disclaimer: str = (
        "MOTION DX provides objective kinematic measurements and observations for clinical review. "
        "This software does NOT make medical diagnoses or formulate treatment plans."
    )

    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return round(self.end_time - self.start_time, 1)
        return round(time.time() - self.start_time, 1)
