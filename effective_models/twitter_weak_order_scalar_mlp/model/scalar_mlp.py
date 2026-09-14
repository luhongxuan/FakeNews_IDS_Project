from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


class ScalarNet(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimensions, 32), nn.ReLU(), nn.Dropout(0.05), nn.Linear(32, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def head_pairs(target: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-target, kind="stable")
    head, mid, rest = order[:min(10, len(order))], order[10:min(40, len(order))], order[40:]
    rng, high, low = np.random.default_rng(seed), [], []
    for item in head:
        sampled = rng.choice(rest, size=min(16, len(rest)), replace=False) if len(rest) else np.array([], dtype=int)
        candidates = np.concatenate([mid, sampled])
        candidates = candidates[target[item] > target[candidates]]
        high.extend([item] * len(candidates)); low.extend(candidates.tolist())
    return np.asarray(high, dtype=int), np.asarray(low, dtype=int)


def train_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int,
                  epochs: int, order_lambda: float) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train[features])
    x = torch.tensor(scaler.transform(train[features]), dtype=torch.float32)
    z = torch.tensor(scaler.transform(test[features]), dtype=torch.float32)
    raw = train.preventable_impact.to_numpy(dtype=float)
    target = torch.tensor(np.log1p(raw), dtype=torch.float32)
    quantile = max(float(np.quantile(raw, 0.9)), 1.0)
    weights = torch.tensor(1 + 2 * np.clip(raw / quantile, 0, 1), dtype=torch.float32)
    high_np, low_np = head_pairs(raw, seed)
    high, low = torch.tensor(high_np), torch.tensor(low_np)
    pair_weight = torch.tensor(np.clip(np.abs(raw[high_np] - raw[low_np]) / quantile, 0.25, 3), dtype=torch.float32) if len(high_np) else torch.ones(1)
    model = ScalarNet(len(features)); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(); score = model(x)
        utility = (F.smooth_l1_loss(score, target, reduction="none") * weights).mean()
        order_loss = (F.softplus(-(score[high] - score[low])) * pair_weight).mean() if len(high) else score.sum() * 0
        loss = utility + order_lambda * order_loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss")
        loss.backward(); optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % 40 == 0:
            print(f"      epoch {epoch}/{epochs}: utility={utility.item():.4f}, order={order_loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        prediction = model(z).numpy()
    return prediction, {"utility": float(utility.item()), "order": float(order_loss.item()), "head_pairs": len(high_np)}

