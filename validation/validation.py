"""
Clinical Validation Mode Engine for MOTION DX.
Compares automated system measurements against verified manual goniometer or optical reference values.
Calculates Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) strictly from recorded empirical trials.
Does NOT fabricate statistical accuracy percentages.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class ValidationTrial:
    trial_number: int
    system_value: float
    reference_value: float
    unit: str = "degrees"
    notes: str = ""

    @property
    def absolute_difference(self) -> float:
        return round(abs(self.system_value - self.reference_value), 2)

    @property
    def error(self) -> float:
        return round(self.system_value - self.reference_value, 2)


class ValidationSession:
    """Manages empirical validation trials and objective comparative error metrics."""

    def __init__(self, assessment_type: str = "KNEE_ROM"):
        self.assessment_type = assessment_type
        self.trials: List[ValidationTrial] = []

    def add_trial(self, system_val: float, ref_val: float, unit: str = "°", notes: str = "") -> ValidationTrial:
        trial_num = len(self.trials) + 1
        trial = ValidationTrial(
            trial_number=trial_num,
            system_value=round(float(system_val), 1),
            reference_value=round(float(ref_val), 1),
            unit=unit,
            notes=notes
        )
        self.trials.append(trial)
        return trial

    def clear(self) -> None:
        self.trials.clear()

    def remove_trial(self, index: int) -> bool:
        if 0 <= index < len(self.trials):
            self.trials.pop(index)
            # Re-index
            for i, t in enumerate(self.trials):
                t.trial_number = i + 1
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute comparative metrics from entered empirical trials.
        Strictly reports MAE, RMSE, and max error without unverified percentage claims.
        """
        if not self.trials:
            return {
                "trial_count": 0,
                "mean_absolute_error": 0.0,
                "rmse": 0.0,
                "max_absolute_error": 0.0,
                "mean_system": 0.0,
                "mean_reference": 0.0,
                "status": "No trials recorded yet."
            }

        abs_diffs = [t.absolute_difference for t in self.trials]
        errors = [t.error for t in self.trials]
        sys_vals = [t.system_value for t in self.trials]
        ref_vals = [t.reference_value for t in self.trials]

        mae = round(float(np.mean(abs_diffs)), 2)
        rmse = round(float(np.sqrt(np.mean(np.square(errors)))), 2)
        max_err = round(float(np.max(abs_diffs)), 2)

        return {
            "trial_count": len(self.trials),
            "mean_absolute_error": mae,
            "rmse": rmse,
            "max_absolute_error": max_err,
            "mean_system": round(float(np.mean(sys_vals)), 1),
            "mean_reference": round(float(np.mean(ref_vals)), 1),
            "unit": self.trials[0].unit if self.trials else "°",
            "status": f"Empirical validation across {len(self.trials)} trial(s)."
        }
