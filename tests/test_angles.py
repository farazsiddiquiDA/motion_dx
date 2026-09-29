"""
Unit tests for 2D and 3D angle calculation algorithms.
"""

import unittest
import numpy as np
from measurements.angles import calculate_angle_2d, calculate_angle_3d, calculate_joint_angle


class TestAngleCalculations(unittest.TestCase):

    def test_right_angle_2d(self):
        # (0, 1) -> (0, 0) -> (1, 0) is exactly 90 degrees
        p1 = (0.0, 1.0)
        vertex = (0.0, 0.0)
        p3 = (1.0, 0.0)
        angle = calculate_angle_2d(p1, vertex, p3)
        self.assertAlmostEqual(angle, 90.0, places=2)

    def test_straight_line_2d(self):
        # (-1, 0) -> (0, 0) -> (1, 0) is exactly 180 degrees
        p1 = (-1.0, 0.0)
        vertex = (0.0, 0.0)
        p3 = (1.0, 0.0)
        angle = calculate_angle_2d(p1, vertex, p3)
        self.assertAlmostEqual(angle, 180.0, places=2)

    def test_acute_angle_2d(self):
        p1 = (1.0, 1.0)
        vertex = (0.0, 0.0)
        p3 = (1.0, 0.0)
        angle = calculate_angle_2d(p1, vertex, p3)
        self.assertAlmostEqual(angle, 45.0, places=2)

    def test_degenerate_zero_vector(self):
        # Coincident points should gracefully yield 0.0 without division by zero
        p1 = (0.0, 0.0)
        vertex = (0.0, 0.0)
        p3 = (1.0, 1.0)
        angle = calculate_angle_2d(p1, vertex, p3)
        self.assertEqual(angle, 0.0)

    def test_angle_3d(self):
        p1 = (0.0, 1.0, 0.0)
        vertex = (0.0, 0.0, 0.0)
        p3 = (0.0, 0.0, 1.0)
        angle = calculate_angle_3d(p1, vertex, p3)
        self.assertAlmostEqual(angle, 90.0, places=2)

    def test_joint_angle_fallback(self):
        # With missing or invalid 3D depth (z <= 0), should fallback to 2D
        p1_2d = (0.0, 1.0)
        v_2d = (0.0, 0.0)
        p3_2d = (1.0, 0.0)

        # Invalid depth
        p1_3d = (0.0, 1.0, 0.0)
        v_3d = (0.0, 0.0, -1.0)
        p3_3d = (1.0, 0.0, 0.0)

        angle, mode = calculate_joint_angle(p1_2d, v_2d, p3_2d, p1_3d, v_3d, p3_3d, prefer_3d=True)
        self.assertEqual(mode, "2D")
        self.assertAlmostEqual(angle, 90.0, places=1)


if __name__ == "__main__":
    unittest.main()
