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

------------------------------------------------------------------
本次修改（解決 Head B 訓練出來卻輸給「猜訓練集平均值」這個笨 baseline
的問題，根源是模型選擇標準只看 Head A）：

1. early stopping / 選最佳權重的標準，從「只看 val f1_macro」改成
   「同時看 val f1_macro 跟 val node_mae，比例比照訓練時的 alpha」。
   node_mae 會先除以「猜訓練集平均值」這個 baseline 的 MAE 做標準化
   （node_mae_ratio），避免 MAE 原始數值跟 f1_macro 尺度對不上，
   直接加減會被其中一個牽著走。
       val_score = alpha * f1_macro - (1 - alpha) * node_mae_ratio
   選 val_score 最大的那個 epoch 存起來，不是選 f1_macro 最高的。

2. 每個 fold 自動算「用 train fold 的 coverage_score 平均值，對
   test fold 每個節點都猜同一個值」這個 baseline 的 MAE，跟模型
   實際的 node_mae 一起印出來、一起存進 results，並標記
   beats_baseline_B（True/False）。以後每次跑完不用再另外寫腳本
   確認 Head B 有沒有學到東西。

3. test 集合太小的 fold（預設 < 50 張圖，像先前 ebola-essien 那種
   只有 14 張的）在彙總平均時自動排除，避免這種統計上不具意義的
   fold 把整體平均拉偏，但仍然逐一列出來，不會憑空消失不見。
------------------------------------------------------------------
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

MIN_TEST_GRAPHS_FOR_SUMMARY = 50  # test 圖數少於這個門檻的 fold，不計入彙總平均


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


def constant_baseline_mae(loader: DataLoader, constant_value: float) -> float:
    """
    算「不管輸入是什麼，通通猜 constant_value」這個最笨的 baseline，
    對這個 loader 裡所有節點的 node_y 的 MAE。用來檢驗 Head B 訓練出來
    的權重，是不是真的比什麼都不做還學到東西。
    """
    all_node_y = []
    for batch in loader:
        all_node_y.extend(batch.node_y.numpy())
    all_node_y = np.array(all_node_y)
    if len(all_node_y) == 0:
        return 0.0
    return float(np.mean(np.abs(all_node_y - constant_value)))


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
    min_test_graphs_for_summary: int = MIN_TEST_GRAPHS_FOR_SUMMARY,
    verbose: bool = True,
) -> dict:
    """
    Leave-One-Event-Out 交叉驗證。

    回傳：
        results: {event_id: {指標dict}} 的字典，每個指標 dict 額外含
        node_mae_baseline（猜訓練集平均值的 baseline MAE）、
        beats_baseline_B（模型 node_mae 是否小於這個 baseline）、
        n_test（這個 fold 的 test 圖數，判斷是不是太小不具統計意義）。
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

        # --- Head B 的 baseline：用 train fold 的 coverage_score 平均值，
        #     對 val / test 通通猜這個值，看 MAE 是多少。這是「不訓練
        #     Head B 也能做到的最低標準」，模型的 node_mae 沒有明顯優於
        #     這個數字，就代表 Head B 沒有學到有用的東西。 ---
        train_node_y_mean = float(np.mean(
            np.concatenate([g.node_y.numpy() for g in train_g])
        ))
        baseline_val_mae  = constant_baseline_mae(val_loader,  train_node_y_mean)
        baseline_test_mae = constant_baseline_mae(test_loader, train_node_y_mean)
        if verbose:
            print(f"  Head B baseline（猜 train 平均值={train_node_y_mean:.4f}）："
                  f"val_mae={baseline_val_mae:.4f}  test_mae={baseline_test_mae:.4f}")

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
        best_val_score = -float("inf")  # 越大越好（跟舊版 best_val_loss 相反方向）
        patience = 15
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            # 每個 epoch 都算 val 指標
            val_metrics = evaluate(model, val_loader, device)

            # Head B 的誤差先用 baseline 標準化，避免跟 f1_macro 尺度對不上：
            # ratio < 1 代表贏過 baseline，ratio >= 1 代表沒學到東西甚至更差。
            node_mae_ratio = val_metrics["node_mae"] / max(baseline_val_mae, 1e-6)

            # 選權重的標準改成同時看兩個 head，比例比照訓練時的 alpha，
            # 不再是只看 f1_macro 的 (1 - f1_macro)。
            val_score = alpha * val_metrics["f1_macro"] - (1 - alpha) * node_mae_ratio

            if val_score > best_val_score:
                best_val_score = val_score
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
                if verbose:
                    print(
                        f"  epoch {epoch:3d} | "
                        f"loss={train_loss['loss']:.4f} "
                        f"(cls={train_loss['loss_a']:.4f} reg={train_loss['loss_b']:.4f}) | "
                        f"val_score={val_score:+.4f} "
                        f"(f1={val_metrics['f1_macro']:.3f} "
                        f"mae_ratio={node_mae_ratio:.3f}) "
                        f"patience={patience_counter} | "
                        f"test: acc={metrics['acc']:.3f} "
                        f"f1_macro={metrics['f1_macro']:.3f} "
                        f"auc={metrics['auc']:.3f} "
                        f"node_mae={metrics['node_mae']:.4f}"
                    )

        # early stopping 後，用綜合分數最佳的 val 狀態重新評估 test
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        metrics = evaluate(model, test_loader, device)
        metrics["node_mae_baseline"] = baseline_test_mae
        metrics["beats_baseline_B"] = 1.0 if metrics["node_mae"] < baseline_test_mae else 0.0
        metrics["n_test"] = len(test_graphs)
        best_metrics = metrics
        results[test_event] = best_metrics
        if verbose:
            beat_str = "贏過" if metrics["beats_baseline_B"] else "沒贏過"
            print(
                f"  最終 f1_macro={metrics['f1_macro']:.3f}  "
                f"node_mae={metrics['node_mae']:.4f}（baseline={baseline_test_mae:.4f}，"
                f"{beat_str} baseline）"
            )

    # 彙總
    if verbose:
        print(f"\n{'='*55}")
        print("LOEO 結果彙總：")
        metrics_keys = ["n_test", "acc", "f1_macro", "f1_rumour", "auc",
                         "node_mse", "node_mae", "node_mae_baseline", "beats_baseline_B"]
        header = f"{'event':<22}" + "".join(f"{k:>14}" for k in metrics_keys)
        print(header)

        small_folds = []
        for ev, m in results.items():
            row = f"{ev:<22}" + "".join(f"{m.get(k, 0):>14.4f}" for k in metrics_keys)
            if m.get("n_test", 0) < min_test_graphs_for_summary:
                row += "   <- test 圖數太少，不計入下面的彙總平均"
                small_folds.append(ev)
            print(row)

        included = {ev: m for ev, m in results.items() if ev not in small_folds}
        if included:
            means = {k: np.mean([m.get(k, 0) for m in included.values()]) for k in metrics_keys}
            print("-" * (22 + 14 * len(metrics_keys)))
            mean_row = f"{'mean (排除小fold)':<22}" + "".join(
                f"{means[k]:>14.4f}" for k in metrics_keys
            )
            print(mean_row)
        if small_folds:
            print(f"\n排除的小 fold（test < {min_test_graphs_for_summary} 張圖，"
                  f"統計上不具代表性）: {small_folds}")

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