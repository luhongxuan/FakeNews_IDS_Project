import unittest
from benchmarks.pheme_fixed30_unified.evaluation.common import load_validate, evaluate
class BenchmarkTest(unittest.TestCase):
    def test_frozen_contract(self):
        allow,scores,eligible=load_validate(); self.assertEqual(len(allow),2373); self.assertEqual(len(eligible),7); self.assertEqual(scores.groupby("model").size().nunique(),1)
    def test_macro_is_bounded(self):
        _,scores,eligible=load_validate(); _,macro=evaluate(scores,eligible[:1]); self.assertTrue(macro.macro_corrected_impact_capture.between(0,1).all())
if __name__=="__main__": unittest.main()
