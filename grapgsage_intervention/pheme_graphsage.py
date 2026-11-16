"""
GraphSAGE Dual-Output Head
===========================
架構：
  - 共用 GraphSAGE backbone（3層 SAGEConv）
  - Head A：graph-level rumour 分類（is_rumour 0/1）
  - Head B：node-level coverage_score 預測（回歸，0~1）

Head A 的流程：
  節點特徵 x (397維)
    → SAGEConv layer 1 → BN → ReLU → Dropout
    → SAGEConv layer 2 → BN → ReLU → Dropout
    → SAGEConv layer 3 → BN → ReLU
    → global_mean_pool（所有節點平均，得到圖向量）
    → MLP → sigmoid → is_rumour 預測

Head B 的流程：
  同一個 backbone 的節點嵌入
    → MLP → sigmoid → coverage_score 預測（每個節點一個值）

Loss：
  total_loss = alpha * CE_loss(Head A) + (1 - alpha) * MSE_loss(Head B)
  預設 alpha=0.7（分類任務為主，回歸任務為輔）

設計原則：
- has_replies=0 的 thread（只有 source tweet）：
  Head A 照常跑（單節點圖，global_mean_pool 就是那個節點的嵌入）
  Head B 的 coverage_score=0，模型會學到「單節點 → coverage=0」
- 邊方向：parent → child（單向）
  之後可以加 add_self_loops 或 bi-directional，目前先單向
- LOEO 分割：train/test 按 event_id 切，不做 random split
  StandardScaler 在 train fold 上 fit，test fold 用同一個 scaler transform
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ------------------------------------------------------------------
# 0. 常數
# ------------------------------------------------------------------

NUMERIC_FEAT_COLS = [
    "depth", "offset_sec_log", "reply_latency_sec_log", "is_source",
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "followers_log", "friends_log", "statuses_log",
    "verified", "account_age_days_log",
]
NUM_NUMERIC = len(NUMERIC_FEAT_COLS)   # 13
NUM_EMBED   = 768
IN_CHANNELS = NUM_NUMERIC + NUM_EMBED  # 397


# ------------------------------------------------------------------
# 1. 模型架構
# ------------------------------------------------------------------

class GraphSAGEDual(nn.Module):
    """
    GraphSAGE with dual output heads.

    參數：
        in_channels:   節點特徵維度（預設 397）
        hidden:        隱藏層維度
        num_layers:    SAGEConv 層數
        dropout:       dropout 比率
        alpha:         分類 loss 的權重（1-alpha = 回歸 loss 權重）
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        alpha: float = 0.7,
    ):
        super().__init__()
        self.alpha = alpha
        self.dropout = dropout

        # --- Backbone: SAGEConv 層 ---
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            in_ch  = in_channels if i == 0 else hidden
            self.convs.append(SAGEConv(in_ch, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        # --- Head A: graph-level 分類（rumour / non-rumour）---
        self.head_a = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        # --- Head B: node-level coverage_score 回歸 ---
        self.head_b = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),   # coverage_score ∈ [0, 1]
        )

    def forward(self, x, edge_index, batch):
        """
        x:          節點特徵 [N, in_channels]
        edge_index: 邊 [2, E]
        batch:      每個節點屬於哪個圖 [N]

        回傳：
            logit_a:  graph-level logit [B, 1]（未過 sigmoid）
            pred_b:   node-level coverage_score [N, 1]（已過 sigmoid）
            node_emb: 節點嵌入 [N, hidden]（方便 debug）
        """
        # Backbone
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = x  # [N, hidden]

        # Head A: global pooling → 分類
        graph_emb = global_mean_pool(node_emb, batch)  # [B, hidden]
        logit_a   = self.head_a(graph_emb)              # [B, 1]

        # Head B: node-level 回歸
        pred_b = self.head_b(node_emb)  # [N, 1]

        return logit_a, pred_b, node_emb

    def compute_loss(self, logit_a, pred_b, y, node_y):
        """
        logit_a: [B, 1]，graph-level logit
        pred_b:  [N, 1]，node-level coverage_score 預測
        y:       [B]，graph-level label（0/1）
        node_y:  [N]，node-level coverage_score（0~1）
        """

        
        loss_a = F.binary_cross_entropy_with_logits(
            logit_a.squeeze(-1), y.float()
        )
        loss_b = F.mse_loss(
            pred_b.squeeze(-1), node_y
        )
        return self.alpha * loss_a + (1 - self.alpha) * loss_b, loss_a, loss_b


# ------------------------------------------------------------------
# 2. 資料前處理（Scaler 套用）
# ------------------------------------------------------------------

def apply_scaler_to_graphs(
    graphs: list[Data],
    scaler: StandardScaler,
    fit: bool = False,
) -> list[Data]:
    """
    對圖列表的節點特徵做 StandardScaler（只對前 13 維數值特徵）。
    fit=True：先 fit 再 transform（用在 train fold）
    fit=False：只 transform（用在 test fold）
    """
    # 收集所有節點的前 13 維
    all_numeric = np.vstack([
        g.x[:, :NUM_NUMERIC].numpy() for g in graphs
    ])

    if fit:
        scaler.fit(all_numeric)

    new_graphs = []
    for g in graphs:
        x = g.x.numpy().copy()
        x[:, :NUM_NUMERIC] = scaler.transform(x[:, :NUM_NUMERIC])
        new_g = Data(
            x          = torch.tensor(x, dtype=torch.float),
            edge_index = g.edge_index,
            y          = g.y,
            node_y     = g.node_y,
            thread_id  = g.thread_id,
            event_id   = g.event_id,
            num_nodes  = g.num_nodes,
        )
        new_graphs.append(new_g)
    return new_graphs


# ------------------------------------------------------------------
# 3. 訓練 / 評估
# ------------------------------------------------------------------

def train_epoch(
    model: GraphSAGEDual,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = total_a = total_b = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logit_a, pred_b, _ = model(batch.x, batch.edge_index, batch.batch)
        loss, la, lb = model.compute_loss(
            logit_a, pred_b, batch.y, batch.node_y
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_a    += la.item()
        total_b    += lb.item()
    n = len(loader)
    return {"loss": total_loss/n, "loss_a": total_a/n, "loss_b": total_b/n}


@torch.no_grad()
def evaluate(
    model: GraphSAGEDual,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    y_true, y_prob = [], []
    node_y_true, node_y_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        logit_a, pred_b, _ = model(batch.x, batch.edge_index, batch.batch)

        y_true.extend(batch.y.cpu().numpy())
        y_prob.extend(torch.sigmoid(logit_a).squeeze(-1).cpu().numpy())
        node_y_true.extend(batch.node_y.cpu().numpy())
        node_y_pred.extend(pred_b.squeeze(-1).cpu().numpy())

    y_true  = np.array(y_true)
    y_prob  = np.array(y_prob)
    y_pred  = (y_prob >= 0.5).astype(int)
    node_yt = np.array(node_y_true)
    node_yp = np.array(node_y_pred)

    return {
        # Head A 指標
        "acc":      accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_rumour":f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc":      roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        # Head B 指標
        "node_mse": mean_squared_error(node_yt, node_yp),
        "node_mae": mean_absolute_error(node_yt, node_yp),
    }


# ------------------------------------------------------------------
# 4. LOEO 訓練迴圈
# ------------------------------------------------------------------

def run_loeo(
    graphs: list[Data],
    hidden: int = 256,
    num_layers: int = 3,
    dropout: float = 0.3,
    alpha: float = 0.7,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    device_str: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    Leave-One-Event-Out 交叉驗證。

    回傳：
        results: {event_id: {指標dict}} 的字典
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"使用裝置：{device}")

    events = sorted(set(g.event_id for g in graphs))
    results = {}

    for test_event in events:
        train_graphs = [g for g in graphs if g.event_id != test_event]
        test_graphs  = [g for g in graphs if g.event_id == test_event]

        if verbose:
            print(f"\n{'='*55}")
            print(f"LOEO fold：test_event = {test_event}")
            print(f"  train={len(train_graphs)}  test={len(test_graphs)}")

        # Scaler：只在 train fold 上 fit
        scaler = StandardScaler()
        train_graphs = apply_scaler_to_graphs(train_graphs, scaler, fit=True)
        test_graphs  = apply_scaler_to_graphs(test_graphs,  scaler, fit=False)

        from sklearn.model_selection import train_test_split
        train_g, val_g = train_test_split(train_graphs, test_size=0.1, random_state=42)
        train_loader = DataLoader(train_g,    batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_g,      batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        # 模型
        model = GraphSAGEDual(
            in_channels=IN_CHANNELS,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            alpha=alpha,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best_f1 = 0.0
        best_metrics = {}
        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            # 每個 epoch 都算 val loss
            val_metrics = evaluate(model, val_loader, device)
            val_loss = 1 - val_metrics["f1_macro"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0 or epoch == epochs:
                metrics = evaluate(model, test_loader, device)
                if metrics["f1_macro"] > best_f1:
                    best_f1 = metrics["f1_macro"]
                    best_metrics = metrics.copy()
                if verbose:
                    print(
                        f"  epoch {epoch:3d} | "
                        f"loss={train_loss['loss']:.4f} "
                        f"(cls={train_loss['loss_a']:.4f} reg={train_loss['loss_b']:.4f}) | "
                        f"val_loss={val_loss:.4f} patience={patience_counter} | "
                        f"acc={metrics['acc']:.3f} "
                        f"f1_macro={metrics['f1_macro']:.3f} "
                        f"auc={metrics['auc']:.3f} "
                        f"node_mae={metrics['node_mae']:.4f}"
                    )

        # early stopping 後，用最佳 val 狀態重新評估 test
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        metrics = evaluate(model, test_loader, device)
        best_metrics = metrics
        results[test_event] = best_metrics
        if verbose:
            print(f"  最佳 f1_macro={metrics['f1_macro']:.3f}")

    # 彙總
    if verbose:
        print(f"\n{'='*55}")
        print("LOEO 結果彙總：")
        metrics_keys = ["acc", "f1_macro", "f1_rumour", "auc", "node_mse", "node_mae"]
        header = f"{'event':<22}" + "".join(f"{k:>12}" for k in metrics_keys)
        print(header)
        for ev, m in results.items():
            row = f"{ev:<22}" + "".join(f"{m.get(k,0):>12.4f}" for k in metrics_keys)
            print(row)
        means = {k: np.mean([m.get(k,0) for m in results.values()]) for k in metrics_keys}
        print("-" * (22 + 12 * len(metrics_keys)))
        mean_row = f"{'mean':<22}" + "".join(f"{means[k]:>12.4f}" for k in metrics_keys)
        print(mean_row)

    return results


# ------------------------------------------------------------------
# 5. 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    PT_PATH = os.getenv("GRAPHS_PT", "pheme_graphs_roberta_30min.pt")

    print(f"載入圖資料集：{PT_PATH}")
    graphs = torch.load(PT_PATH, weights_only=False)
    print(f"共 {len(graphs)} 張圖")

    results = run_loeo(
        graphs,
        hidden     = 256,
        num_layers = 3,
        dropout    = 0.3,
        alpha      = 0.7,
        lr         = 1e-3,
        epochs     = 50,
        batch_size = 32,
    )