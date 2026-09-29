"""
Local JSON-based persistent storage for MOTION DX assessment sessions.
100% offline and privacy-preserving. No remote cloud or network transmission.
"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from assessments.base import AssessmentSession, AssessmentState


class SessionStore:
    """Local file-system storage for clinical prototype sessions."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(root, "data", "sessions")

        self.storage_dir = base_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_session(self, session: AssessmentSession) -> str:
        """Serialize and save session to a local JSON file."""
        filename = f"{session.session_id}.json"
        filepath = os.path.join(self.storage_dir, filename)

        data = {
            "session_id": session.session_id,
            "patient_id": session.patient_id,
            "assessment_type": session.assessment_type,
            "target_joint": session.target_joint,
            "target_side": session.target_side,
            "camera_type": session.camera_type,
            "is_depth_enabled": session.is_depth_enabled,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "duration_seconds": session.duration_seconds,
            "state": session.state.value if isinstance(session.state, AssessmentState) else str(session.state),
            "tracking_quality": session.tracking_quality,
            "metrics_summary": session.metrics_summary,
            "time_series_count": len(session.time_series),
            "time_series": session.time_series,
            "clinical_notes": session.clinical_notes,
            "disclaimer": session.disclaimer,
            "saved_at": time.time()
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return filepath

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific saved session by ID."""
        filepath = os.path.join(self.storage_dir, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all locally saved sessions sorted by most recent first."""
        sessions = []
        if not os.path.exists(self.storage_dir):
            return sessions

        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Return lightweight summary
                        sessions.append({
                            "session_id": data.get("session_id"),
                            "patient_id": data.get("patient_id"),
                            "assessment_type": data.get("assessment_type"),
                            "target_joint": data.get("target_joint"),
                            "start_time": data.get("start_time"),
                            "duration_seconds": data.get("duration_seconds", 0),
                            "tracking_quality": data.get("tracking_quality"),
                            "metrics_summary": data.get("metrics_summary", {}),
                            "filepath": fpath
                        })
                except Exception:
                    pass

        sessions.sort(key=lambda s: s.get("start_time", 0), reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        filepath = os.path.join(self.storage_dir, f"{session_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except OSError:
                return False
        return False
