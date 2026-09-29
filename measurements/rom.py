"""
Range of Motion (ROM) tracking and statistical aggregation engine.
Maintains continuous kinematic time-series data without diagnostic claims.
"""

import time
from typing import Dict, List, Optional, Any
import numpy as np


class ROMTracker:
    """
    Tracks joint angle kinematics over an active assessment session.
    Calculates current, minimum, maximum, and observed Range of Motion (ROM).
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.reset()

    def reset(self) -> None:
        """Clear all recorded telemetry and reset extrema."""
        self.is_active: bool = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.current_angle: float = 0.0
        self.min_angle: float = 999.0
        self.max_angle: float = -999.0
        self.samples: List[Dict[str, Any]] = []
        self.quality_history: List[str] = []

    def start(self) -> None:
        """Begin active recording."""
        self.reset()
        self.is_active = True
        self.start_time = time.time()

    def stop(self) -> None:
        """Halt active recording."""
        self.is_active = False
        self.end_time = time.time()

    def add_sample(
        self,
        angle: float,
        tracking_quality: str = "GOOD",
        mode: str = "2D",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a new angle observation.
        
        Args:
            angle: Observed joint angle in degrees.
            tracking_quality: 'GOOD', 'FAIR', or 'POOR'.
            mode: '3D' or '2D'.
            metadata: Optional dictionary of auxiliary values (e.g. side, landmarks).
        """
        if not self.is_active:
            # Still update current angle for real-time visual gauge even before start
            self.current_angle = round(angle, 1)
            return

        now = time.time()
        elapsed = round(now - (self.start_time or now), 2)
        self.current_angle = round(angle, 1)

        # Only update extrema if tracking quality is acceptable (not POOR)
        if tracking_quality in ("GOOD", "FAIR") and 0.0 <= angle <= 180.0:
            if angle < self.min_angle:
                self.min_angle = round(angle, 1)
            if angle > self.max_angle:
                self.max_angle = round(angle, 1)

        sample = {
            "elapsed_sec": elapsed,
            "angle": self.current_angle,
            "quality": tracking_quality,
            "mode": mode,
            "timestamp": now
        }
        if metadata:
            sample.update(metadata)

        self.samples.append(sample)
        self.quality_history.append(tracking_quality)

        if len(self.samples) > self.buffer_size:
            self.samples.pop(0)
            self.quality_history.pop(0)

    @property
    def observed_rom(self) -> float:
        """Calculated active range of motion: (max_angle - min_angle)."""
        if self.min_angle < 900.0 and self.max_angle > -900.0:
            return round(max(0.0, self.max_angle - self.min_angle), 1)
        return 0.0

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return round(end - self.start_time, 1)

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate summary kinematics for clinical review.
        Labels metrics clearly as observed measurements.
        """
        valid_samples = [s["angle"] for s in self.samples if s.get("quality") != "POOR"]
        
        has_extrema = (self.min_angle < 900.0 and self.max_angle > -900.0)
        min_val = self.min_angle if has_extrema else 0.0
        max_val = self.max_angle if has_extrema else 0.0
        rom_val = self.observed_rom

        overall_quality = "POOR"
        if self.quality_history:
            good_ratio = self.quality_history.count("GOOD") / len(self.quality_history)
            fair_ratio = self.quality_history.count("FAIR") / len(self.quality_history)
            if good_ratio >= 0.7:
                overall_quality = "GOOD"
            elif (good_ratio + fair_ratio) >= 0.6:
                overall_quality = "FAIR"

        return {
            "sample_count": len(self.samples),
            "valid_sample_count": len(valid_samples),
            "current_angle": self.current_angle,
            "minimum_angle": min_val,
            "maximum_angle": max_val,
            "observed_rom": rom_val,
            "mean_angle": round(float(np.mean(valid_samples)), 1) if valid_samples else 0.0,
            "std_angle": round(float(np.std(valid_samples)), 1) if valid_samples else 0.0,
            "duration_seconds": self.duration,
            "tracking_quality": overall_quality,
            "is_active": self.is_active
        }
