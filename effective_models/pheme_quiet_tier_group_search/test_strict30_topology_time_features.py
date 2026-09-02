from __future__ import annotations

import unittest

import pandas as pd

from materialize_strict30_topology_time_features import topology_time_block


class TopologyTimeTests(unittest.TestCase):
    def test_safe_zero_reply_row_is_finite(self) -> None:
        frame = pd.DataFrame([{ "thread_id":"x", "activity_observed_replies":0., "temporal_active_minute_count":0., "temporal_first10_fraction":1., "temporal_mid10_fraction":0., "temporal_last10_fraction":0., "temporal_hist_1m_entropy":0., "temporal_hist_1m_gini":0., "topology_max_depth":0., "topology_leaf_fraction":1., "topology_root_children":0., "topology_width_to_depth":1., "topology_depth_entropy":0., "topology_outdegree_entropy":0., "topology_outdegree_gini":0., "topology_outdegree_hhi":0. }])
        result = topology_time_block(frame)
        self.assertEqual(result.shape[1], 16)
        self.assertTrue(result.drop(columns="thread_id").notna().all().all())


if __name__ == "__main__":
    unittest.main()
