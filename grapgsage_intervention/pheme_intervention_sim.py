"""
介入模擬（Intervention Simulation）
=====================================
流程：
  1. 載入訓練好的 GraphSAGE 模型
  2. 對每個 test thread 的前 30 分鐘傳播樹做推論
  3. Head B 預測每個節點的 coverage_score
  4. 選出預測分數最高的前 K 個節點作為介入目標
  5. 用完整傳播樹驗證：移除這 K 個節點後，實際能阻斷多少後續傳播
  6. 與兩個 baseline 比較：隨機選節點、選最淺深度的節點

評估指標：
  Cascade Reduction Rate (CRR) =
      被阻斷的節點數 / 30 分鐘後的總後代節點數

  CRR 範圍 [0, 1]，越高代表介入效果越好。
  CRR = 1.0 表示完全阻斷所有後續傳播。
  CRR = 0.0 表示介入完全沒有效果。

設計原則：
  - 只對「確實有後續傳播」的 thread 做模擬
    （30 分鐘後沒有任何新節點的 thread，介入沒有意義，排除）
  - 只對 Head A 正確分類為謠言的 thread 做模擬
    （系統實際上只會對被判定為謠言的 thread 啟動介入）
  - K 個節點的選取不重複，若 thread 節點數 < K，則取全部節點
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# 0. 常數
# ------------------------------------------------------------------

K_INTERVENE = 5      # 每次介入的節點數
WINDOW_SEC  = 1800   # 30 分鐘


# ------------------------------------------------------------------
# 1. 計算「移除節點集合後能阻斷的後代數量」
# ------------------------------------------------------------------

def compute_blocked_descendants(
    parent_map: dict[str, Optional[str]],
    offset_map: dict[str, float],
    intervene_ids: set[str],
    window_sec: float = WINDOW_SEC,
) -> int:
    """
    模擬移除 intervene_ids 裡的節點後，window_sec 之後無法被觸達的節點數量。

    邏輯：
      從 root 出發做 DFS，遇到被介入的節點就停止（它的整個子樹都被阻斷）。
      只計算 offset_sec > window_sec 的後代（窗口後才出現的節點才算被阻斷）。
    """
    children_map: dict[str, list[str]] = defaultdict(list)
    root_id = None
    for tid, pid in parent_map.items():
        if pid is None:
            root_id = tid
        else:
            children_map[pid].append(tid)

    if root_id is None:
        return 0

    blocked = 0

    def dfs(node_id: str, is_blocked: bool) -> None:
        nonlocal blocked
        node_offset = offset_map.get(node_id, 0.0)
        if np.isnan(node_offset):
            node_offset = 0.0

        # 如果這個節點本身被介入，它的所有後代都被阻斷
        if node_id in intervene_ids:
            is_blocked = True

        # 計算這個節點本身是否是「被阻斷的窗口後節點」
        if is_blocked and node_offset > window_sec and node_id != root_id:
            blocked += 1

        for child in children_map.get(node_id, []):
            dfs(child, is_blocked)

    dfs(root_id, False)
    return blocked


def compute_total_future_nodes(
    parent_map: dict[str, Optional[str]],
    offset_map: dict[str, float],
    window_sec: float = WINDOW_SEC,
) -> int:
    """計算 window_sec 之後出現的總節點數（介入前的基準）"""
    return sum(
        1 for tid, offset in offset_map.items()
        if not np.isnan(offset) and offset > window_sec
    )


# ------------------------------------------------------------------
# 2. 三種介入策略
# ------------------------------------------------------------------

def strategy_gnn(node_scores: dict[str, float], k: int) -> set[str]:
    """GNN Head B：選預測 coverage_score 最高的 K 個節點"""
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    return {tid for tid, _ in sorted_nodes[:k]}


def strategy_random(node_ids: list[str], k: int, seed: int = 42) -> set[str]:
    """Baseline 1：隨機選 K 個節點"""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(node_ids, size=min(k, len(node_ids)), replace=False)
    return set(chosen.tolist())


def strategy_shallowest(
    node_ids: list[str],
    depth_map: dict[str, int],
    k: int,
) -> set[str]:
    """Baseline 2：選深度最淺（最靠近 root）的 K 個節點，不選 root 本身"""
    candidates = [tid for tid in node_ids if depth_map.get(tid, 0) > 0]
    sorted_nodes = sorted(candidates, key=lambda x: depth_map.get(x, 999))
    return set(sorted_nodes[:k])


# ------------------------------------------------------------------
# 3. 對單一 thread 做介入模擬
# ------------------------------------------------------------------

def simulate_thread(
    thread_id: str,
    full_parent_map: dict[str, Optional[str]],
    full_offset_map: dict[str, float],
    depth_map: dict[str, int],
    early_node_ids: list[str],    # 30 分鐘內的節點（GNN 看到的）
    gnn_scores: dict[str, float], # Head B 預測的 coverage_score
    k: int = K_INTERVENE,
    window_sec: float = WINDOW_SEC,
    n_random_trials: int = 10,    # 隨機 baseline 跑幾次取平均
) -> Optional[dict]:
    """
    對單一 thread 模擬三種介入策略，回傳 CRR 比較結果。
    若 thread 在窗口後沒有任何新節點，回傳 None（無意義模擬）。
    """
    total_future = compute_total_future_nodes(full_parent_map, full_offset_map, window_sec)
    if total_future == 0:
        return None  # 30 分鐘後沒有任何新節點，不做模擬

    # GNN 策略
    intervene_gnn = strategy_gnn(gnn_scores, k)
    blocked_gnn = compute_blocked_descendants(
        full_parent_map, full_offset_map, intervene_gnn, window_sec
    )
    crr_gnn = blocked_gnn / total_future

    # 最淺節點策略
    intervene_shallow = strategy_shallowest(early_node_ids, depth_map, k)
    blocked_shallow = compute_blocked_descendants(
        full_parent_map, full_offset_map, intervene_shallow, window_sec
    )
    crr_shallow = blocked_shallow / total_future

    # 隨機策略（多次取平均）
    crr_random_list = []
    for trial in range(n_random_trials):
        intervene_random = strategy_random(early_node_ids, k, seed=trial)
        blocked_random = compute_blocked_descendants(
            full_parent_map, full_offset_map, intervene_random, window_sec
        )
        crr_random_list.append(blocked_random / total_future)
    crr_random = float(np.mean(crr_random_list))

    return {
        "thread_id":    thread_id,
        "total_future": total_future,
        "crr_gnn":      crr_gnn,
        "crr_shallow":  crr_shallow,
        "crr_random":   crr_random,
        "blocked_gnn":     blocked_gnn,
        "blocked_shallow": blocked_shallow,
    }


# ------------------------------------------------------------------
# 4. 從 GNN 模型取得 Head B 預測值
# ------------------------------------------------------------------

@torch.no_grad()
def get_node_predictions(
    model,
    graphs: list[Data],
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, dict[str, float]]:
    """
    對 graphs 裡的每張圖做推論，回傳：
      {thread_id: {tweet_id_idx: predicted_coverage_score}}

    注意：這裡用節點的 index（0, 1, 2...）作為 key，
    因為 graphs 裡沒有直接存 tweet_id 到節點 index 的對應。
    呼叫端需要自己對應 tweet_id。
    """
    from torch_geometric.loader import DataLoader as PygDataLoader
    model.eval()
    loader = PygDataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_preds = {}
    for batch in loader:
        batch = batch.to(device)
        _, pred_b, _ = model(batch.x, batch.edge_index, batch.batch)
        pred_b = pred_b.squeeze(-1).cpu().numpy()

        # 拆回每張圖
        node_counts = batch.ptr[1:] - batch.ptr[:-1]
        offset = 0
        for i, count in enumerate(node_counts.tolist()):
            tid = batch.thread_id[i] if hasattr(batch, 'thread_id') else str(i)
            all_preds[tid] = pred_b[offset:offset + count]
            offset += count

    return all_preds


# ------------------------------------------------------------------
# 5. 主流程：LOEO 介入模擬
# ------------------------------------------------------------------

def run_intervention_simulation(
    graphs_30min: list[Data],       # 30 分鐘版的圖（GNN 輸入）
    graphs_full: list[Data],        # 完整版的圖（用來驗證真實阻斷效果）
    reply_df,                       # pheme_reply_level_v2.csv（取 parent_map 和 offset_map）
    model_class,                    # GraphSAGEDual class
    model_kwargs: dict,             # 建模型用的參數
    k: int = K_INTERVENE,
    batch_size: int = 32,
    device_str: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    LOEO 介入模擬：每個 fold 訓練一個模型，然後對 test 事件做介入模擬。

    回傳：{event_id: {crr_gnn, crr_shallow, crr_random, ...}}
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from pheme_graphsage import apply_scaler_to_graphs, train_epoch, evaluate

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"介入模擬裝置：{device}，K={k}")

    # 建立 thread_id → full 圖的對應
    full_graph_map = {g.thread_id: g for g in graphs_full}

    # 建立 thread_id → parent_map, offset_map, depth_map
    thread_maps = {}
    for thread_id, grp in reply_df.groupby("thread_id"):
        thread_id = str(thread_id)
        parent_map = {
            str(row["tweet_id"]): (str(row["parent_id"])
                                   if isinstance(row["parent_id"], str)
                                   and row["parent_id"] != "nan"
                                   else None)
            for _, row in grp.iterrows()
        }
        offset_map = {
            str(row["tweet_id"]): float(row["offset_sec"])
            if not pd.isna(row["offset_sec"]) else np.nan
            for _, row in grp.iterrows()
        }
        depth_map = {
            str(row["tweet_id"]): int(row["depth"])
            for _, row in grp.iterrows()
        }
        thread_maps[thread_id] = (parent_map, offset_map, depth_map)

    events = sorted(set(g.event_id for g in graphs_30min))
    all_results = {}
    event_summary = {}

    for test_event in events:
        train_graphs = [g for g in graphs_30min if g.event_id != test_event]
        test_graphs  = [g for g in graphs_30min if g.event_id == test_event]

        if verbose:
            print(f"\n{'='*55}")
            print(f"LOEO fold：test_event = {test_event}  (test={len(test_graphs)} threads)")

        # 訓練模型（沿用 pheme_graphsage.py 的邏輯）
        scaler = StandardScaler()
        train_graphs_scaled = apply_scaler_to_graphs(train_graphs, scaler, fit=True)
        test_graphs_scaled  = apply_scaler_to_graphs(test_graphs,  scaler, fit=False)

        train_g, val_g = train_test_split(train_graphs_scaled, test_size=0.1, random_state=42)
        train_loader = DataLoader(train_g,  batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_g,    batch_size=batch_size)

        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, 51):
            train_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            val_metrics = evaluate(model, val_loader, device)
            val_loss = 1 - val_metrics["f1_macro"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= 15:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # 取得 Head B 預測值
        all_node_preds = get_node_predictions(model, test_graphs_scaled, device, batch_size)

        # 對每個 test thread 做介入模擬
        fold_results = []
        for g, g_scaled in zip(test_graphs, test_graphs_scaled):
            thread_id = g.thread_id

            # 取得這個 thread 的 parent_map, offset_map, depth_map
            if thread_id not in thread_maps:
                continue
            parent_map, offset_map, depth_map = thread_maps[thread_id]

            # 30 分鐘內的節點 tweet_id 列表（跟圖裡節點的順序一致）
            # 從 reply_df 取，按照 feat_df 建圖時的順序
            thread_reply = reply_df[reply_df["thread_id"] == thread_id]
            # 只取 30 分鐘內的節點
            early_mask = (thread_reply["offset_sec"].isna()) | \
                         (thread_reply["offset_sec"] <= WINDOW_SEC)
            early_tweets = thread_reply[early_mask]["tweet_id"].astype(str).tolist()

            # GNN 預測的 coverage_score（index 對應到 early_tweets 的順序）
            pred_scores = all_node_preds.get(thread_id, np.array([]))
            if len(pred_scores) == 0:
                continue

            # 建立 {tweet_id: predicted_score} 的對應
            gnn_scores = {}
            for i, tid in enumerate(early_tweets[:len(pred_scores)]):
                gnn_scores[tid] = float(pred_scores[i])

            # 只對 Head A 正確判定為謠言的 thread 做模擬
            with torch.no_grad():
                g_dev = g_scaled.to(device)
                from torch_geometric.data import Batch
                batch_obj = Batch.from_data_list([g_dev])
                logit_a, _, _ = model(batch_obj.x, batch_obj.edge_index, batch_obj.batch)
                pred_label = int(torch.sigmoid(logit_a).item() >= 0.5)

            true_label = int(g.y.item())
            if pred_label != 1 or true_label != 1:
                continue  # 只模擬正確偵測到的謠言 thread

            result = simulate_thread(
                thread_id=thread_id,
                full_parent_map=parent_map,
                full_offset_map=offset_map,
                depth_map=depth_map,
                early_node_ids=early_tweets,
                gnn_scores=gnn_scores,
                k=k,
            )
            if result is not None:
                fold_results.append(result)

        if fold_results:
            crr_gnn     = np.mean([r["crr_gnn"]     for r in fold_results])
            crr_shallow = np.mean([r["crr_shallow"]  for r in fold_results])
            crr_random  = np.mean([r["crr_random"]   for r in fold_results])
            n_simulated = len(fold_results)
        else:
            crr_gnn = crr_shallow = crr_random = 0.0
            n_simulated = 0

        event_summary[test_event] = {
            "n_simulated": n_simulated,
            "crr_gnn":     crr_gnn,
            "crr_shallow": crr_shallow,
            "crr_random":  crr_random,
        }
        all_results[test_event] = fold_results

        if verbose:
            print(f"  模擬 thread 數：{n_simulated}")
            print(f"  CRR（GNN）   ：{crr_gnn:.4f}")
            print(f"  CRR（最淺）  ：{crr_shallow:.4f}")
            print(f"  CRR（隨機）  ：{crr_random:.4f}")

    # 彙總
    if verbose:
        print(f"\n{'='*55}")
        print(f"介入模擬彙總（K={k}）")
        print(f"{'event':<22} {'n':>6} {'CRR_GNN':>10} {'CRR_shallow':>12} {'CRR_random':>11}")
        total_n = 0
        crr_gnns, crr_shallows, crr_randoms = [], [], []
        for ev, s in event_summary.items():
            print(f"{ev:<22} {s['n_simulated']:>6} "
                  f"{s['crr_gnn']:>10.4f} "
                  f"{s['crr_shallow']:>12.4f} "
                  f"{s['crr_random']:>11.4f}")
            if s['n_simulated'] > 0:
                crr_gnns.append(s['crr_gnn'])
                crr_shallows.append(s['crr_shallow'])
                crr_randoms.append(s['crr_random'])
                total_n += s['n_simulated']
        print("-" * 63)
        print(f"{'mean':<22} {total_n:>6} "
              f"{np.mean(crr_gnns):>10.4f} "
              f"{np.mean(crr_shallows):>12.4f} "
              f"{np.mean(crr_randoms):>11.4f}")

    return {"event_summary": event_summary, "all_results": all_results}


# ------------------------------------------------------------------
# 6. 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from pheme_graphsage import GraphSAGEDual, IN_CHANNELS

    GRAPHS_30MIN = os.getenv("GRAPHS_30MIN", "pheme_graphs_roberta_30min.pt")
    GRAPHS_FULL  = os.getenv("GRAPHS_FULL",  "pheme_graphs_roberta_full.pt")
    REPLY_CSV    = os.getenv("REPLY_CSV",    "pheme_reply_level_v2.csv")

    print("載入圖資料集...")
    graphs_30min = torch.load(GRAPHS_30MIN, weights_only=False)
    graphs_full  = torch.load(GRAPHS_FULL,  weights_only=False)
    print(f"30min：{len(graphs_30min)} 張圖，full：{len(graphs_full)} 張圖")

    print("載入 reply-level 表...")
    reply_df = pd.read_csv(
        REPLY_CSV,
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str},
        low_memory=False,
    )
    print(f"reply_df：{len(reply_df)} 列")

    model_kwargs = dict(
        in_channels=IN_CHANNELS,
        hidden=256,
        num_layers=3,
        dropout=0.3,
        alpha=0.7,
    )

    results = run_intervention_simulation(
        graphs_30min=graphs_30min,
        graphs_full=graphs_full,
        reply_df=reply_df,
        model_class=GraphSAGEDual,
        model_kwargs=model_kwargs,
        k=K_INTERVENE,
    )