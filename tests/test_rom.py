"""
Unit tests for Range of Motion (ROM) dynamic tracking and statistical aggregation.
"""

import unittest
from measurements.rom import ROMTracker


class TestROMTracker(unittest.TestCase):

    def test_rom_tracking_cycle(self):
        tracker = ROMTracker()
        tracker.start()

        # Simulate knee flexion-extension cycle: 160 -> 40 -> 160
        angles = [160.0, 140.0, 110.0, 70.0, 40.0, 60.0, 100.0, 150.0, 165.0]
        for a in angles:
            tracker.add_sample(a, tracking_quality="GOOD")

        tracker.stop()

        self.assertAlmostEqual(tracker.min_angle, 40.0, places=1)
        self.assertAlmostEqual(tracker.max_angle, 165.0, places=1)
        self.assertAlmostEqual(tracker.observed_rom, 125.0, places=1)

    def test_rejection_of_poor_samples(self):
        tracker = ROMTracker()
        tracker.start()

        tracker.add_sample(90.0, tracking_quality="GOOD")
        tracker.add_sample(100.0, tracking_quality="GOOD")
        # Glitch sample during poor tracking quality
        tracker.add_sample(10.0, tracking_quality="POOR")
        tracker.add_sample(175.0, tracking_quality="POOR")

        summary = tracker.get_summary()
        # The POOR quality extrema should be excluded from valid clinical min/max
        self.assertEqual(tracker.min_angle, 90.0)
        self.assertEqual(tracker.max_angle, 100.0)
        self.assertEqual(tracker.observed_rom, 10.0)

    def test_reset(self):
        tracker = ROMTracker()
        tracker.start()
        tracker.add_sample(80.0, tracking_quality="GOOD")
        tracker.reset()

        self.assertFalse(tracker.is_active)
        self.assertEqual(tracker.observed_rom, 0.0)
        self.assertEqual(len(tracker.samples), 0)


if __name__ == "__main__":
    unittest.main()
