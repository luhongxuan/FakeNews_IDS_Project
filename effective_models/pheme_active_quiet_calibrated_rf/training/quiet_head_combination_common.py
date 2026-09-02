"""Strict-30 safe utility head built from predeclared quiet feature groups."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
TIER_ROOT = ROOT / "effective_models" / "pheme_quiet_tier_group_search"
sys.path.insert(0, str(TIER_ROOT))
from strict30_protected_feature_registry import FEATURE_GROUPS, feature_group  # noqa: E402

GROUPS = FEATURE_GROUPS + ("source_reply_embedding",)
SMOKE_COMBOS = (("source_reply_embedding",), ("activity_level", "temporal_dynamics", "topology"), ("source_reply_embedding", "activity_level", "temporal_dynamics", "topology"), GROUPS)


def combinations_for(mode: str) -> list[tuple[str, ...]]:
    if mode == "smoke": return list(SMOKE_COMBOS)
    return [combo for size in range(1, len(GROUPS)+1) for combo in combinations(GROUPS, size)]


def columns_by_group(frame: pd.DataFrame) -> dict[str, list[str]]:
    groups = {group: [column for column in frame if feature_group(column) == group] for group in FEATURE_GROUPS}
    if any(not columns for columns in groups.values()): raise ValueError("missing legal head feature group")
    return groups


def fit_score(train: pd.DataFrame, test: pd.DataFrame, groups: tuple[str, ...], columns: dict[str, list[str]], embeddings: dict[str, np.ndarray], seed: int, trees: int) -> np.ndarray:
    scalar = [column for group in groups if group != "source_reply_embedding" for column in columns[group]]
    x_train = train[scalar].to_numpy(np.float32) if scalar else np.empty((len(train), 0), np.float32)
    x_test = test[scalar].to_numpy(np.float32) if scalar else np.empty((len(test), 0), np.float32)
    if "source_reply_embedding" in groups:
        raw_train = np.vstack([embeddings[value] for value in train.thread_id]); raw_test = np.vstack([embeddings[value] for value in test.thread_id])
        scaler = StandardScaler().fit(raw_train); pca = PCA(n_components=min(32, len(train)-1, raw_train.shape[1]), random_state=seed).fit(scaler.transform(raw_train))
        x_train = np.hstack([x_train, pca.transform(scaler.transform(raw_train))]); x_test = np.hstack([x_test, pca.transform(scaler.transform(raw_test))])
    model = RandomForestRegressor(n_estimators=trees, max_depth=8, min_samples_leaf=3, max_features=.8, n_jobs=1, random_state=seed)
    return model.fit(x_train, train.preventable_y.to_numpy(np.float32)).predict(x_test).astype(np.float32)
