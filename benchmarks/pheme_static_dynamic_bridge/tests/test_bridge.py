import unittest
from benchmarks.pheme_static_dynamic_bridge.evaluation.common import load_validate,evaluate
class BridgeTest(unittest.TestCase):
 def test_contract_and_metrics(self):
  fixed,dynamic,events=load_validate();self.assertEqual(len(events),7);per,macro,selected=evaluate(fixed,dynamic,events[:1]);self.assertTrue(macro.macro_unified_horizon_capture_at_budget50.between(0,1).all());self.assertTrue((per.interventions==50).all());self.assertIn("random_expected_static",set(macro.model))
if __name__=="__main__":unittest.main()
