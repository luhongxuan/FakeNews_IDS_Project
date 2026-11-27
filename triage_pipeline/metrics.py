"""Tie-aware ranking metrics and paired author-cluster uncertainty."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def expected_topk_sum(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    if not len(scores):
        return 0.0
    order = np.argsort(-scores, kind="stable")
    scores, targets = scores[order], targets[order]
    total = 0.0
    index = used = 0
    while used < min(k, len(scores)):
        end = index + 1
        while end < len(scores) and scores[end] == scores[index]:
            end += 1
        take = min(end - index, k - used)
        total += float(targets[index:end].mean()) * take
        used += take
        index = end
    return total


def grouped_metric(rows: list[dict], method: str, fraction: float, fixed_k: int | None = None) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    ratios, correlations, zero_oracle = [], [], []
    for group, items in sorted(groups.items()):
        score = np.asarray([row["scores"][method] for row in items], dtype=float)
        target = np.asarray([row["target"] for row in items], dtype=float)
        if not np.isfinite(score).all() or not np.isfinite(target).all():
            raise ValueError("nonfinite metric input")
        k = min(len(items), fixed_k if fixed_k is not None else math.ceil(fraction * len(items)))
        oracle = expected_topk_sum(target, target, k)
        if oracle:
            ratios.append(expected_topk_sum(score, target, k) / oracle)
        else:
            zero_oracle.append(group)
        if len(items) > 1 and np.unique(score).size > 1 and np.unique(target).size > 1:
            value = float(spearmanr(score, target).statistic)
            if math.isfinite(value):
                correlations.append(value)
    return {"mean_capture": float(np.mean(ratios)) if ratios else None,
            "valid_groups": len(ratios), "zero_oracle_groups": zero_oracle,
            "mean_spearman": float(np.mean(correlations)) if correlations else None,
            "spearman_groups": len(correlations)}


def paired_cluster_ci(rows: list[dict], left: str, right: str, fraction: float,
                      draws: int = 2000, seed: int = 42, progress=None) -> dict:
    clusters = sorted({row["cluster_id"] for row in rows})
    members = {cluster: [row for row in rows if row["cluster_id"] == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)
    differences = []
    for completed in range(1, draws + 1):
        sampled = [row for cluster in rng.choice(clusters, size=len(clusters), replace=True)
                   for row in members[cluster]]
        lhs = grouped_metric(sampled, left, fraction)["mean_capture"]
        rhs = grouped_metric(sampled, right, fraction)["mean_capture"]
        if lhs is not None and rhs is not None:
            differences.append(lhs - rhs)
        if progress and (completed % 100 == 0 or completed == draws):
            progress(completed, draws)
    return {"lower": float(np.quantile(differences, .025)) if differences else None,
            "upper": float(np.quantile(differences, .975)) if differences else None,
            "draws": draws, "usable_draws": len(differences), "seed": seed}


def method_intervals(rows: list[dict], methods: tuple[str, ...], fraction: float,
                     draws: int = 2000, seed: int = 42, progress=None) -> dict:
    clusters = sorted({row["cluster_id"] for row in rows})
    members = {cluster: [row for row in rows if row["cluster_id"] == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)
    values = {method: [] for method in methods}
    for completed in range(1, draws + 1):
        sampled = [row for cluster in rng.choice(clusters, size=len(clusters), replace=True)
                   for row in members[cluster]]
        for method in methods:
            value = grouped_metric(sampled, method, fraction)["mean_capture"]
            if value is not None:
                values[method].append(value)
        if progress and (completed % 100 == 0 or completed == draws):
            progress(completed, draws)
    return {method: {"lower": float(np.quantile(samples, .025)) if samples else None,
                     "upper": float(np.quantile(samples, .975)) if samples else None,
                     "usable_draws": len(samples), "draws": draws, "seed": seed}
            for method, samples in values.items()}
