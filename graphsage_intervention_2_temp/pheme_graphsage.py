"""
GraphSAGE Dual-Output Head
===========================
架構：
  - 共用 GraphSAGE backbone（3層 SAGEConv）
  - Head A：graph-level 迴歸，預測這個 thread 在早期窗口之後總共還會
            長出幾個節點（future_growth，log1p 轉換），拿來跨 thread
            排序、在預算限制下決定優先介入哪些 thread
  - Head B：node-level 迴歸，預測每個節點的 coverage_score（0~1），
            給已經被 Head A 選中的高風險 thread，決定 thread 內部
            哪個節點值得介入

------------------------------------------------------------------
本次修改（重點）：

1. **Head A 從分類換成迴歸**。is_rumour 現在是已知的篩選條件（PHEME
   本身就有這個標籤），不是要預測的東西——你們的研究貢獻不在「判斷
   是不是謠言」，是「已知是謠言事件後，怎麼決定優先介入哪個 thread、
   thread 裡哪個節點」。舊版 Head A 做 is_rumour 分類，跟這個研究
   問題已經對不上了，改成預測 future_growth（log1p 轉換過，跟
   Head B 的 coverage_score 一樣是零膨脹分布）。

2. **選模型權重的標準統一化**。兩個 head 現在都是迴歸，都用「模型
   MAE ÷ 猜訓練集平均值的 baseline MAE」這個比例（越小越好），
   再用訓練時同樣的 alpha 比例合併：
       val_score = -(alpha * mae_ratio_a + (1 - alpha) * mae_ratio_b)
   （取負號是因為 ratio 越小越好，val_score 維持「越大越好」的慣例，
   跟 early stopping 邏輯一致）
   這比舊版「一個用 f1_macro、一個用 MAE，尺度對不上」乾淨很多。

3. **新增 Spearman 排序相關係數**（Head A、Head B 各自算一次）。
   MAE/MSE 在零膨脹分布下容易被大量真實 0 值撐出好看的數字，掩蓋
   模型有沒有真的學會「排序」——而你們最終要用這個分數去跨 thread/
   跨節點排序決定介入優先序，排序品質才是真正該看的指標，MAE 只是
   輔助參考。

4. 兩個 head 都算 baseline、beats_baseline，彙總表小 fold（test 圖數
   太少）自動排除但保留列出，邏輯跟之前修 Head B 時一樣，這次
   Head A 也套用同一套。
------------------------------------------------------------------
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
IN_CHANNELS = NUM_NUMERIC + NUM_EMBED  # 781

MIN_TEST_GRAPHS_FOR_SUMMARY = 50  # test 圖數少於這個門檻的 fold，不計入彙總平均


# ------------------------------------------------------------------
# 1. 模型架構
# ------------------------------------------------------------------

class GraphSAGEDual(nn.Module):
    """
    GraphSAGE with dual regression heads.

    參數：
        in_channels:   節點特徵維度（781）
        hidden:        隱藏層維度
        num_layers:    SAGEConv 層數
        dropout:       dropout 比率
        alpha:         Head A loss 的權重（1-alpha = Head B loss 權重）
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

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            in_ch  = in_channels if i == 0 else hidden
            self.convs.append(SAGEConv(in_ch, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        # --- Head A: graph-level 迴歸（future_growth，log1p 轉換後）---
        # 用 Softplus 而不是不加限制的線性輸出：log1p(future_growth) 一定
        # >= 0，讓輸出結構性地保證非負，比讓模型自己學「不要輸出負的」更穩。
        self.head_a = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
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
        回傳：
            pred_a:   graph-level future_growth 預測 [B, 1]（已過 Softplus，非負）
            pred_b:   node-level coverage_score [N, 1]（已過 sigmoid，[0,1]）
            node_emb: 節點嵌入 [N, hidden]（方便 debug）
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = x  # [N, hidden]

        graph_emb = global_mean_pool(node_emb, batch)  # [B, hidden]
        pred_a    = self.head_a(graph_emb)              # [B, 1]

        pred_b = self.head_b(node_emb)  # [N, 1]

        return pred_a, pred_b, node_emb

    def compute_loss(self, pred_a, pred_b, y, node_y):
        """
        pred_a: [B, 1]，graph-level future_growth 預測
        pred_b: [N, 1]，node-level coverage_score 預測
        y:      [B]，graph-level future_growth（log1p 轉換後的迴歸目標）
        node_y: [N]，node-level coverage_score（0~1）
        """
        loss_a = F.mse_loss(pred_a.squeeze(-1), y.float())
        loss_b = F.mse_loss(pred_b.squeeze(-1), node_y)
        return self.alpha * loss_a + (1 - self.alpha) * loss_b, loss_a, loss_b


# ------------------------------------------------------------------
# 2. 資料前處理（Scaler 套用）
# ------------------------------------------------------------------

def apply_scaler_to_graphs(
    graphs: list[Data],
    scaler: StandardScaler,
    fit: bool = False,
) -> list[Data]:
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
        pred_a, pred_b, _ = model(batch.x, batch.edge_index, batch.batch)
        loss, la, lb = model.compute_loss(
            pred_a, pred_b, batch.y, batch.node_y
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
    y_true, y_pred = [], []
    node_y_true, node_y_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        pred_a, pred_b, _ = model(batch.x, batch.edge_index, batch.batch)

        y_true.extend(batch.y.cpu().numpy().ravel())
        y_pred.extend(pred_a.squeeze(-1).cpu().numpy())
        node_y_true.extend(batch.node_y.cpu().numpy())
        node_y_pred.extend(pred_b.squeeze(-1).cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    node_yt = np.array(node_y_true)
    node_yp = np.array(node_y_pred)

    # Spearman 在常數輸入（例如全部節點預測值一樣）時會回傳 NaN，
    # 用 0.0 頂著，避免彙總計算時整個炸掉。
    def safe_spearman(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        r, _ = spearmanr(a, b)
        return 0.0 if np.isnan(r) else r

    return {
        # Head A 指標（graph-level future_growth 迴歸）
        "mse_a": mean_squared_error(y_true, y_pred),
        "mae_a": mean_absolute_error(y_true, y_pred),
        "spearman_a": safe_spearman(y_true, y_pred),
        # Head B 指標（node-level coverage_score 迴歸）
        "node_mse": mean_squared_error(node_yt, node_yp),
        "node_mae": mean_absolute_error(node_yt, node_yp),
        "spearman_b": safe_spearman(node_yt, node_yp),
    }


def constant_baseline_mae(loader: DataLoader, constant_value: float, field: str) -> float:
    """
    算「不管輸入是什麼，通通猜 constant_value」這個最笨的 baseline，
    對這個 loader 裡所有 `field`（'y' 或 'node_y'）的 MAE。
    """
    all_vals = []
    for batch in loader:
        vals = batch.y.numpy().ravel() if field == "y" else batch.node_y.numpy()
        all_vals.extend(vals)
    all_vals = np.array(all_vals)
    if len(all_vals) == 0:
        return 0.0
    return float(np.mean(np.abs(all_vals - constant_value)))


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
    return_models: bool = False,
    verbose: bool = True,
):
    """
    Leave-One-Event-Out 交叉驗證。

    return_models=False（預設，跟舊版行為一致）：
        回傳 results，{event_id: {指標dict}}。
    return_models=True：
        回傳 (results, models)，models 是 {event_id: 已載入該 fold
        最佳權重、且已切到 eval() 模式的模型}，連同對應的 scaler 一起
        存在 model.fitted_scaler_ 屬性上，供 pheme_intervention_sim.py
        直接拿去對這個 event 的 test threads 做推論，不用重新訓練
        （避免訓練邏輯在兩個檔案裡各自維護、容易兜不起來）。
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"使用裝置：{device}")

    events = sorted(set(g.event_id for g in graphs))
    results = {}
    models = {} if return_models else None
    run_start_time = time.time()

    for fold_i, test_event in enumerate(events, 1):
        train_graphs = [g for g in graphs if g.event_id != test_event]
        test_graphs  = [g for g in graphs if g.event_id == test_event]

        if verbose:
            print(f"\n{'='*55}")
            print(f"LOEO fold {fold_i}/{len(events)}：test_event = {test_event}"
                  f"（目前累計耗時 {(time.time() - run_start_time)/60:.1f} 分鐘）")
            print(f"  train={len(train_graphs)}  test={len(test_graphs)}")

        scaler = StandardScaler()
        train_graphs = apply_scaler_to_graphs(train_graphs, scaler, fit=True)
        test_graphs  = apply_scaler_to_graphs(test_graphs,  scaler, fit=False)

        from sklearn.model_selection import train_test_split
        train_g, val_g = train_test_split(train_graphs, test_size=0.1, random_state=42)
        train_loader = DataLoader(train_g,    batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_g,      batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

        # --- 兩個 head 各自的 baseline：用 train fold 的平均值，
        #     對 val / test 通通猜這個值。---
        train_y_mean      = float(np.mean([g.y.item() for g in train_g]))
        train_node_y_mean = float(np.mean(
            np.concatenate([g.node_y.numpy() for g in train_g])
        ))
        baseline_val_mae_a  = constant_baseline_mae(val_loader,  train_y_mean, "y")
        baseline_val_mae_b  = constant_baseline_mae(val_loader,  train_node_y_mean, "node_y")
        baseline_test_mae_a = constant_baseline_mae(test_loader, train_y_mean, "y")
        baseline_test_mae_b = constant_baseline_mae(test_loader, train_node_y_mean, "node_y")
        if verbose:
            print(f"  baseline（猜 train 平均值）："
                  f"A: train_y_mean={train_y_mean:.4f} val_mae={baseline_val_mae_a:.4f} | "
                  f"B: train_node_y_mean={train_node_y_mean:.4f} val_mae={baseline_val_mae_b:.4f}")

        model = GraphSAGEDual(
            in_channels=IN_CHANNELS,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            alpha=alpha,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best_val_score = -float("inf")
        patience = 15
        patience_counter = 0
        best_state = None

        fold_start_time = time.time()
        if verbose:
            print(f"  開始訓練（最多 {epochs} epoch，patience={patience}）...")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            val_metrics = evaluate(model, val_loader, device)

            mae_ratio_a = val_metrics["mae_a"] / max(baseline_val_mae_a, 1e-6)
            mae_ratio_b = val_metrics["node_mae"] / max(baseline_val_mae_b, 1e-6)
            # 兩個都是「比例越小越好」，val_score 維持「越大越好」的慣例。
            val_score = -(alpha * mae_ratio_a + (1 - alpha) * mae_ratio_b)

            improved = val_score > best_val_score
            if improved:
                best_val_score = val_score
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            epoch_sec = time.time() - epoch_start
            elapsed_min = (time.time() - fold_start_time) / 60

            # 每個 epoch 都印一行精簡狀態，避免長時間沒輸出讓人以為卡住；
            # 每 10 個 epoch（或有變好、或最後一個 epoch）額外印一行完整的
            # test 集指標，不用每個 epoch 都花時間跑 test 評估。
            if verbose:
                mark = "*" if improved else " "
                print(
                    f"  epoch {epoch:3d}/{epochs} {mark}| "
                    f"loss={train_loss['loss']:.4f} | "
                    f"val_score={val_score:+.4f} "
                    f"(ratio_A={mae_ratio_a:.3f} ratio_B={mae_ratio_b:.3f}) | "
                    f"patience={patience_counter}/{patience} | "
                    f"{epoch_sec:.1f}s/epoch，累計 {elapsed_min:.1f} 分鐘"
                )

            if patience_counter >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0 or epoch == epochs:
                metrics = evaluate(model, test_loader, device)
                if verbose:
                    print(
                        f"    [第 {epoch} epoch，test 集參考數字] "
                        f"mae_a={metrics['mae_a']:.4f} "
                        f"spear_a={metrics['spearman_a']:+.3f} "
                        f"node_mae={metrics['node_mae']:.4f} "
                        f"spear_b={metrics['spearman_b']:+.3f}"
                    )

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        if return_models:
            model.fitted_scaler_ = scaler  # 這個 fold 的 test 圖也要用同一個 scaler transform
            models[test_event] = model

        metrics = evaluate(model, test_loader, device)
        metrics["mae_a_baseline"] = baseline_test_mae_a
        metrics["mae_b_baseline"] = baseline_test_mae_b
        metrics["beats_baseline_A"] = 1.0 if metrics["mae_a"] < baseline_test_mae_a else 0.0
        metrics["beats_baseline_B"] = 1.0 if metrics["node_mae"] < baseline_test_mae_b else 0.0
        metrics["n_test"] = len(test_graphs)
        results[test_event] = metrics
        if verbose:
            print(f"  這個 fold 花了 {(time.time() - fold_start_time)/60:.1f} 分鐘")
            print(
                f"  最終 mae_a={metrics['mae_a']:.4f}（baseline={baseline_test_mae_a:.4f}，"
                f"{'贏' if metrics['beats_baseline_A'] else '沒贏'}）"
                f"  spear_a={metrics['spearman_a']:+.3f} | "
                f"node_mae={metrics['node_mae']:.4f}（baseline={baseline_test_mae_b:.4f}，"
                f"{'贏' if metrics['beats_baseline_B'] else '沒贏'}）"
                f"  spear_b={metrics['spearman_b']:+.3f}"
            )

    # 彙總
    if verbose:
        print(f"\n{'='*55}")
        print("LOEO 結果彙總：")
        metrics_keys = [
            "n_test",
            "mae_a", "mae_a_baseline", "beats_baseline_A", "spearman_a",
            "node_mae", "mae_b_baseline", "beats_baseline_B", "spearman_b",
        ]
        header = f"{'event':<22}" + "".join(f"{k:>16}" for k in metrics_keys)
        print(header)

        small_folds = []
        for ev, m in results.items():
            row = f"{ev:<22}" + "".join(f"{m.get(k, 0):>16.4f}" for k in metrics_keys)
            if m.get("n_test", 0) < min_test_graphs_for_summary:
                row += "   <- test 圖數太少，不計入下面的彙總平均"
                small_folds.append(ev)
            print(row)

        included = {ev: m for ev, m in results.items() if ev not in small_folds}
        if included:
            means = {k: np.mean([m.get(k, 0) for m in included.values()]) for k in metrics_keys}
            print("-" * (22 + 16 * len(metrics_keys)))
            mean_row = f"{'mean (排除小fold)':<22}" + "".join(
                f"{means[k]:>16.4f}" for k in metrics_keys
            )
            print(mean_row)
        if small_folds:
            print(f"\n排除的小 fold（test < {min_test_graphs_for_summary} 張圖）: {small_folds}")

    if return_models:
        return results, models
    return results


# ------------------------------------------------------------------
# 5. 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    PT_PATH = os.getenv("GRAPHS_PT", "pheme_graphs_roberta_30min_v2.pt")

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