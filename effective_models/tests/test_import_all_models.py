from __future__ import annotations

import importlib
from pathlib import Path
import unittest


MODEL_MODULES = (
    "effective_models.twitter_graph_only_rf.training.common",
    "effective_models.twitter_weak_order_scalar_mlp.training.common",
    "effective_models.pheme_graph7_author_aware_rf.training.common",
    "effective_models.pheme_graph7_discourse_rf.training.common",
    "effective_models.pheme_v5_text_rf.training.common",
    "effective_models.pheme_active_quiet_calibrated_rf.training.common",
    "effective_models.pheme_multicheckpoint_rf.training.common",
    "effective_models.pheme_hawkes_multicheckpoint_rf.training.common",
)


class AllModelImportTest(unittest.TestCase):
    def test_all_eight_models_import_in_one_process(self) -> None:
        imported = [importlib.import_module(name) for name in MODEL_MODULES]
        self.assertEqual(len(imported), 8)
        self.assertEqual({module.__name__ for module in imported}, set(MODEL_MODULES))

    def test_local_artifact_contracts_stay_inside_each_model(self) -> None:
        from effective_models.twitter_graph_only_rf import config as twitter
        from effective_models.pheme_v5_text_rf import config as v5
        from effective_models.pheme_active_quiet_calibrated_rf import config as active_quiet

        contracts = (
            (twitter.MODEL_ROOT, twitter.DATA_ROOT, twitter.SPLIT_ROOT),
            (v5.MODEL_ROOT, v5.ASSET_DIR),
            (active_quiet.MODEL_ROOT, active_quiet.ASSET_DIR),
        )
        for model_root, *paths in contracts:
            for path in paths:
                self.assertTrue(Path(path).is_relative_to(model_root), f"{path} escapes {model_root}")


if __name__ == "__main__":
    unittest.main()
