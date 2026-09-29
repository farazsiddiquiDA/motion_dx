"""
Unit tests for distance and lateral midline deviation calculations.
"""

import unittest
from measurements.distance import calculate_distance_2d, calculate_distance_3d, point_to_line_distance_2d


class TestDistanceCalculations(unittest.TestCase):

    def test_distance_2d(self):
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        dist = calculate_distance_2d(p1, p2)
        self.assertAlmostEqual(dist, 5.0, places=3)

    def test_distance_3d(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (1.0, 2.0, 2.0)
        dist = calculate_distance_3d(p1, p2)
        self.assertAlmostEqual(dist, 3.0, places=3)

    def test_point_to_line_deviation(self):
        # Line from (0, 0) to (0, 10) (vertical line along y-axis)
        line_p1 = (0.0, 0.0)
        line_p2 = (0.0, 10.0)

        # Point (3, 5) is 3 units to the right
        point_right = (3.0, 5.0)
        abs_d, signed_d = point_to_line_distance_2d(point_right, line_p1, line_p2)
        self.assertAlmostEqual(abs_d, 3.0, places=2)

        # Point (-4, 5) is 4 units to the left
        point_left = (-4.0, 5.0)
        abs_d_l, signed_d_l = point_to_line_distance_2d(point_left, line_p1, line_p2)
        self.assertAlmostEqual(abs_d_l, 4.0, places=2)

    def test_none_handling(self):
        self.assertEqual(calculate_distance_2d(None, (1, 1)), 0.0)
        self.assertEqual(calculate_distance_3d(None, (1, 1, 1)), 0.0)
        d, s = point_to_line_distance_2d(None, (0, 0), (0, 1))
        self.assertEqual(d, 0.0)


if __name__ == "__main__":
    unittest.main()
