"""Canonical intervention-ranking metrics for v5 fixed-30 evaluation."""
from __future__ import annotations

import random

import numpy as np
import torch

from effective_models.pheme_v5_text_rf import config


def _top_ids(rows: list[dict[str, object]], field: str, budget: int) -> set[str]:
    ordered = sorted(rows, key=lambda row: float(row[field]), reverse=True)
    return {str(row["thread_id"]) for row in ordered[:budget]}


def selection_metrics(graphs: list[object], predictions: np.ndarray) -> dict[str, float | int]:
    rows = [
        {
            "thread_id": str(graph.thread_id),
            "prediction": float(prediction),
            "impact": int(graph.preventable_impact.item()),
            "future": int(round(float(torch.expm1(graph.y).item()))),
            "size": int(graph.num_nodes),
        }
        for graph, prediction in zip(graphs, predictions, strict=True)
    ]
    budget = min(config.TOP_K, len(rows))
    selected = _top_ids(rows, "prediction", budget)
    size_selected = _top_ids(rows, "size", budget)
    ids = [str(row["thread_id"]) for row in rows]
    random_selected = set(random.Random(config.SEED).sample(ids, budget))
    by_id = {str(row["thread_id"]): row for row in rows}
    total_future = sum(int(row["future"]) for row in rows)
    total_impact = sum(int(row["impact"]) for row in rows)

    def score(chosen: set[str]) -> tuple[int, float, float]:
        blocked = sum(int(by_id[thread_id]["impact"]) for thread_id in chosen)
        crr = blocked / total_future if total_future else 0.0
        capture = blocked / total_impact if total_impact else 0.0
        return blocked, crr, capture

    model = score(selected)
    size = score(size_selected)
    random_result = score(random_selected)
    return {
        "n_test": len(rows),
        "blocked_model": model[0],
        "crr_model": model[1],
        "preventable_recall_model": model[2],
        "blocked_size": size[0],
        "crr_size": size[1],
        "preventable_recall_size": size[2],
        "blocked_random": random_result[0],
        "crr_random": random_result[1],
        "preventable_recall_random": random_result[2],
    }
