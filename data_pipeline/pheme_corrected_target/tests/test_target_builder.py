import unittest
from data_pipeline.pheme_corrected_target.target_builder import corrected_preventable_impact

class CorrectedTargetTest(unittest.TestCase):
    def test_multiple_roots_are_traversed(self):
        self.assertEqual(corrected_preventable_impact({"a": None, "b": "a", "c": None, "d": "c"}, {"a": 0., "b": 2000., "c": 0., "d": 2000.}, {"a", "c"}), 2)
    def test_observed_root_blocks_future_descendants(self):
        self.assertEqual(corrected_preventable_impact({"a": None, "b": "a", "c": "b"}, {"a": 0., "b": 1900., "c": 2000.}, {"a"}), 2)
    def test_cycle_without_root_is_rejected(self):
        with self.assertRaises(ValueError): corrected_preventable_impact({"a": "b", "b": "a"}, {"a": 0., "b": 1.}, set())

if __name__ == "__main__": unittest.main()
