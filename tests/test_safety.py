"""
Unit tests for clinical safety, robust depth handling, and missing landmark fault tolerance.
Verifies that MOTION DX never crashes on corrupt data or invalid depths, and avoids diagnostic labels.
"""

import unittest
import numpy as np
from camera.base import BaseCamera
from measurements.angles import calculate_joint_angle, calculate_angle_2d
from measurements.distance import calculate_distance_3d
from measurements.symmetry import calculate_symmetry_index


class MockCamera(BaseCamera):
    def connect(self): return True
    def disconnect(self): pass
    def read_frame(self): return None
    def is_connected(self): return True
    def get_status(self): return {}


class TestClinicalSafetyAndFaultTolerance(unittest.TestCase):

    def setUp(self):
        self.cam = MockCamera()

    def test_median_depth_sampling_robustness(self):
        # Create a depth map with noisy zeros and outliers
        depth_map = np.zeros((100, 100), dtype=np.float32)
        # Put realistic depth in the patch with some zeros and invalid negatives
        depth_map[48:53, 48:53] = np.array([
            [0.0, 0.82, 0.0, 0.81, 0.83],
            [0.80, -1.0, 0.82, 0.84, 0.0],
            [0.81, 0.82, 0.83, 0.0, 0.82],
            [0.0, 0.0, 0.82, 0.85, 0.81],
            [0.83, 0.82, 0.81, 0.82, 0.0]
        ], dtype=np.float32)

        # Sample at center (50, 50)
        sampled_depth = self.cam.sample_depth_median(depth_map, 50, 50, patch_size=5)
        # Should filter out zeros and -1.0, returning the stable median ~0.82
        self.assertGreater(sampled_depth, 0.79)
        self.assertLess(sampled_depth, 0.85)

    def test_out_of_bounds_depth_sampling(self):
        depth_map = np.ones((50, 50), dtype=np.float32)
        # Sample way out of bounds
        d1 = self.cam.sample_depth_median(depth_map, -10, 25)
        d2 = self.cam.sample_depth_median(depth_map, 100, 25)
        d3 = self.cam.sample_depth_median(None, 25, 25)
        self.assertEqual(d1, 0.0)
        self.assertEqual(d2, 0.0)
        self.assertEqual(d3, 0.0)

    def test_all_zero_or_nan_depth_patch(self):
        depth_map = np.zeros((50, 50), dtype=np.float32)
        sampled = self.cam.sample_depth_median(depth_map, 25, 25, patch_size=5)
        self.assertEqual(sampled, 0.0)

    def test_missing_landmark_angle_safety(self):
        # Passing None points must return 0.0 without throwing exceptions
        self.assertEqual(calculate_angle_2d(None, (0, 0), (1, 1)), 0.0)
        self.assertEqual(calculate_angle_2d((1, 1), None, (1, 1)), 0.0)
        self.assertEqual(calculate_angle_2d((1, 1), (0, 0), None), 0.0)

    def test_deprojection_with_zero_depth(self):
        pt = self.cam.deproject_pixel_to_3d(320, 240, 0.0)
        self.assertIsNone(pt)

    def test_safety_non_diagnostic_wording(self):
        # Verify symmetry output never contains diagnostic judgment words
        res = calculate_symmetry_index(120.0, 70.0)
        obs_text = res["observation"].lower()
        forbidden_words = ["disease", "diagnosis", "pathology", "abnormal", "unhealthy", "sick", "tmd detected"]
        for word in forbidden_words:
            self.assertNotIn(word, obs_text, f"Forbidden diagnostic word '{word}' found in clinical observation output.")


if __name__ == "__main__":
    unittest.main()
