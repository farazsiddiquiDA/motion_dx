"""
Unit tests for bilateral symmetry and differential kinematic analysis.
"""

import unittest
from measurements.symmetry import calculate_symmetry_index


class TestSymmetryCalculations(unittest.TestCase):

    def test_perfect_symmetry(self):
        res = calculate_symmetry_index(120.0, 120.0)
        self.assertEqual(res["difference"], 0.0)
        self.assertEqual(res["absolute_difference"], 0.0)
        self.assertEqual(res["symmetry_index_pct"], 100.0)
        self.assertEqual(res["asymmetry_pct"], 0.0)

    def test_known_asymmetry(self):
        # Left = 120, Right = 100. Mean = 110.
        # Diff = 20. Asymmetry = 20 / 110 = 18.18%. Symmetry Index = 81.8%
        res = calculate_symmetry_index(120.0, 100.0)
        self.assertEqual(res["difference"], 20.0)
        self.assertEqual(res["absolute_difference"], 20.0)
        self.assertAlmostEqual(res["symmetry_index_pct"], 81.8, places=1)

    def test_zero_division_guard(self):
        res = calculate_symmetry_index(0.0, 0.0)
        self.assertEqual(res["symmetry_index_pct"], 100.0)
        self.assertEqual(res["absolute_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
