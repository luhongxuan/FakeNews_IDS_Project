"""Snapshot-only source and observed-reply text representation."""

from __future__ import annotations

import numpy as np
import torch

from effective_models.pheme_active_quiet_calibrated_rf import config


def text_matrix(graphs) -> np.ndarray:
    """Return source plus observed-reply centroid embeddings per snapshot."""
    rows = []
    for graph in graphs:
        source_indices = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
        if source_indices.numel() != 1:
            raise ValueError(f"Expected one source node for {graph.thread_id}")
        embedding = graph.x[:, config.EMBEDDING_START:].detach().cpu().numpy().astype(np.float32)
        if embedding.shape[1] != 768 or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid raw text embedding for {graph.thread_id}")
        source_index = int(source_indices.item())
        source = embedding[source_index]
        replies = np.delete(embedding, source_index, axis=0)
        centroid = replies.mean(axis=0) if len(replies) else np.zeros_like(source)
        rows.append(np.concatenate([source, centroid]))
    return np.vstack(rows)
