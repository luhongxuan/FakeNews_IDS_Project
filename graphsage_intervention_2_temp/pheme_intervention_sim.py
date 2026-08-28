"""
介入模擬（Intervention Simulation）—— 兩階段版本
=====================================
流程（對應你們的研究框架，兩個階段都要跑）：

  Stage 1（事件層級，跨 thread 排序 + 預算限制）：
    1. 對一個事件底下所有 test thread，用 Head A 預測的 future_growth
       排序
    2. 在預算限制下（BUDGET_THREADS，一個事件最多能介入幾個 thread），
       選出預測風險最高的前 N 個 thread
    3. 沒被選中的 thread 完全不介入，對事件的 CRR 貢獻是 0

  Stage 2（thread 內部，節點層級介入，沿用你原本已經驗證過的邏輯）：
    4. 對 Stage 1 選中的每個 thread，Head B 預測每個節點的 coverage_score
    5. 選出預測分數最高的前 K 個節點介入
    6. 用完整傳播樹驗證：移除這些節點後，實際能阻斷多少後續傳播

  評估（事件層級聚合，不是逐 thread 平均）：
    Event-level CRR =
        這個事件裡「被阻斷的節點數」加總
        ÷ 這個事件裡「30 分鐘後的總後代節點數」加總

    用加總而不是逐 thread 平均，是因為大 thread 本來就該佔比較重的
    權重——一個 thread 平均阻斷率算得再高，如果它本身很小，對整個
    事件的實際曝光減少貢獻也小；事件層級聚合才反映「這個事件整體
    的假訊息曝光減少了多少」，這是你們真正要交付的指標。

比較的策略（用來拆解「Head A 的排序有沒有貢獻」跟「Head B 的節點選擇
有沒有貢獻」，不要混在一起看）：

  Thread 選擇策略（Stage 1，決定要不要介入這個 thread）：
    - head_a_rank      ：用 Head A 預測值排序，選前 N 個（主策略）
    - observed_size_rank：用 30 分鐘內已經觀察到的節點數排序，選前 N 個
                          （模擬「不用 ML，單純看目前討論量」的土法煉鋼）
    - random_threads   ：隨機選 N 個（下限 baseline）

  節點選擇策略（Stage 2，決定 thread 內部要介入哪些節點）：
    固定用 Head B 的預測值（strategy_gnn），跟你原本的設計一致——
    這裡的重點是拆解 Stage 1，不是重新驗證 Stage 2（那個已經在
    pheme_graphsage.py 的 LOEO 結果裡驗證過了）。

------------------------------------------------------------------
本次修改：
1. 不再自己重新訓練模型——舊版這裡內建的訓練迴圈是分類任務時代的
   邏輯（torch.sigmoid(logit_a) 那段），跟今天改成迴歸的 Head A 已經
   對不上，會直接壞掉。改成呼叫 pheme_graphsage.run_loeo(...,
   return_models=True)，直接複用已經驗證過、邏輯正確的訓練流程，
   兩邊不用各自維護一份容易兜不起來的訓練邏輯。
2. 拿掉「只模擬 Head A 判定為謠言的 thread」這個舊邏輯——is_rumour
   現在是已知的篩選條件（PHEME 本身就有），不是 Head A 要預測的東西，
   這個判斷分支已經沒有意義。
3. 新增 Stage 1（事件層級、預算限制下的跨 thread 排序），這是你們
   研究真正要驗證的部分，舊版完全沒有做。
4. CRR 從逐 thread 平均改成事件層級加總，理由見上面說明。
------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# ------------------------------------------------------------------
# 0. 常數
# ------------------------------------------------------------------

K_INTERVENE     = 5      # Stage 2：每個被選中的 thread，介入幾個節點
BUDGET_THREADS  = 10     # Stage 1：每個事件最多介入幾個 thread
WINDOW_SEC      = 1800   # 30 分鐘


# ------------------------------------------------------------------
# 1. 計算「移除節點集合後能阻斷的後代數量」（沿用原本邏輯，沒有變）
# ------------------------------------------------------------------

def compute_blocked_descendants(
    parent_map: dict[str, Optional[str]],
    offset_map: dict[str, float],
    intervene_ids: set[str],
    window_sec: float = WINDOW_SEC,
) -> int:
    """
    模擬移除 intervene_ids 裡的節點後，window_sec 之後無法被觸達的節點數量。
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

        if node_id in intervene_ids:
            is_blocked = True

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
# 2. Stage 2：thread 內部的節點選擇策略（沿用原本邏輯，沒有變）
# ------------------------------------------------------------------

def strategy_gnn_nodes(node_scores: dict[str, float], k: int) -> set[str]:
    """Head B：選預測 coverage_score 最高的 K 個節點"""
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    return {tid for tid, _ in sorted_nodes[:k]}


def strategy_random_nodes(node_ids: list[str], k: int, seed: int = 42) -> set[str]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(node_ids, size=min(k, len(node_ids)), replace=False)
    return set(chosen.tolist())


# ------------------------------------------------------------------
# 3. Stage 1：事件層級的 thread 選擇策略（新增）
# ------------------------------------------------------------------

def select_threads_by_head_a(
    thread_scores: dict[str, float],  # thread_id -> Head A 預測值
    budget: int,
) -> set[str]:
    """用 Head A 預測值排序，選預算內風險最高的 thread。"""
    sorted_threads = sorted(thread_scores.items(), key=lambda x: x[1], reverse=True)
    return {tid for tid, _ in sorted_threads[:budget]}


def select_threads_by_observed_size(
    thread_node_counts: dict[str, int],  # thread_id -> 30分鐘內節點數
    budget: int,
) -> set[str]:
    """用目前已觀察到的討論量排序（不用 ML 的土法煉鋼 baseline）。"""
    sorted_threads = sorted(thread_node_counts.items(), key=lambda x: x[1], reverse=True)
    return {tid for tid, _ in sorted_threads[:budget]}


def select_threads_random(
    thread_ids: list[str],
    budget: int,
    seed: int = 42,
) -> set[str]:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(thread_ids, size=min(budget, len(thread_ids)), replace=False)
    return set(chosen.tolist())


# ------------------------------------------------------------------
# 4. 對單一 thread 做 Stage 2 節點介入模擬（沿用原本邏輯）
# ------------------------------------------------------------------

def simulate_thread_nodes(
    full_parent_map: dict[str, Optional[str]],
    full_offset_map: dict[str, float],
    early_node_ids: list[str],
    gnn_scores: dict[str, float],
    k: int = K_INTERVENE,
    window_sec: float = WINDOW_SEC,
) -> tuple[int, int]:
    """回傳 (blocked_by_gnn_nodes, blocked_by_random_nodes)。"""
    intervene_gnn = strategy_gnn_nodes(gnn_scores, k)
    blocked_gnn = compute_blocked_descendants(
        full_parent_map, full_offset_map, intervene_gnn, window_sec
    )

    intervene_random = strategy_random_nodes(early_node_ids, k)
    blocked_random = compute_blocked_descendants(
        full_parent_map, full_offset_map, intervene_random, window_sec
    )

    return blocked_gnn, blocked_random


# ------------------------------------------------------------------
# 5. 從模型取得 Head A（thread 層級）與 Head B（節點層級）預測值
# ------------------------------------------------------------------

@torch.no_grad()
def get_predictions_for_event(
    model,
    test_graphs: list[Data],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    回傳：
        thread_scores: {thread_id: Head A 預測值}（越大代表預測未來
                        成長越多，排序用，不用還原 log1p，因為只要
                        排序正確，單調轉換不影響排序結果）
        node_scores:   {thread_id: 該 thread 每個節點的 Head B 預測值
                        陣列，順序跟圖裡節點的順序一致}
    """
    model.eval()
    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    thread_scores = {}
    node_scores = {}

    for batch in loader:
        batch = batch.to(device)
        pred_a, pred_b, _ = model(batch.x, batch.edge_index, batch.batch)
        pred_a = pred_a.squeeze(-1).cpu().numpy()
        pred_b = pred_b.squeeze(-1).cpu().numpy()

        node_counts = (batch.ptr[1:] - batch.ptr[:-1]).cpu().numpy()
        offset = 0
        for i, count in enumerate(node_counts):
            tid = batch.thread_id[i]
            thread_scores[tid] = float(pred_a[i])
            node_scores[tid] = pred_b[offset:offset + count]
            offset += count

    return thread_scores, node_scores


# ------------------------------------------------------------------
# 6. 主流程：LOEO 兩階段介入模擬
# ------------------------------------------------------------------

def run_intervention_simulation(
    graphs_30min: list[Data],
    reply_df,
    budget_threads: int = BUDGET_THREADS,
    k_nodes: int = K_INTERVENE,
    loeo_kwargs: Optional[dict] = None,
    device_str: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    兩階段 LOEO 介入模擬。訓練這步直接呼叫 pheme_graphsage.run_loeo(
    ..., return_models=True)，不在這裡重新寫一份訓練邏輯。

    回傳：{event_id: {各策略的 event-level CRR、n_threads_total、
                       n_threads_intervened}}
    """
    import time
    import pandas as pd
    from pheme_graphsage import run_loeo

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"介入模擬裝置：{device}，budget_threads={budget_threads}，k_nodes={k_nodes}")
        print("\n先跑 LOEO 訓練（複用 pheme_graphsage.run_loeo），拿每個 fold 的模型...")

    loeo_kwargs = loeo_kwargs or {}
    train_results, models = run_loeo(
        graphs_30min, return_models=True, device_str=device_str,
        verbose=verbose, **loeo_kwargs,
    )

    if verbose:
        print("\n訓練完成，開始跑介入模擬 ...")

    # thread_id -> (parent_map, offset_map)，從完整傳播樹的角度算真正的阻斷效果
    if verbose:
        print("建立 thread 的 parent_map / offset_map ...")
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
        thread_maps[thread_id] = (parent_map, offset_map)

    events = sorted(set(g.event_id for g in graphs_30min))
    event_summary = {}
    run_start = time.time()

    for event_i, test_event in enumerate(events, 1):
        if test_event not in models:
            if verbose:
                print(f"\n[{event_i}/{len(events)}] {test_event}：沒有對應的訓練模型，跳過"
                      f"（多半是 run_loeo 裡這個 fold 的資料量太小）")
            continue

        model = models[test_event]
        test_graphs = [g for g in graphs_30min if g.event_id == test_event]
        scaler = model.fitted_scaler_

        from pheme_graphsage import apply_scaler_to_graphs
        test_graphs_scaled = apply_scaler_to_graphs(test_graphs, scaler, fit=False)

        thread_scores, node_scores = get_predictions_for_event(
            model, test_graphs_scaled, device
        )

        # Stage 1 用的輔助資訊：每個 thread 30 分鐘內已觀察到的節點數
        thread_node_counts = {g.thread_id: g.num_nodes for g in test_graphs}
        all_thread_ids = list(thread_node_counts.keys())
        budget = min(budget_threads, len(all_thread_ids))

        selected_by_strategy = {
            "head_a_rank": select_threads_by_head_a(thread_scores, budget),
            "observed_size_rank": select_threads_by_observed_size(thread_node_counts, budget),
            "random_threads": select_threads_random(all_thread_ids, budget),
        }

        # 對每個策略，累加事件層級的 blocked / total_future
        blocked_sum = {s: 0 for s in selected_by_strategy}
        blocked_sum["random_threads_random_nodes"] = 0  # 完全沒有 ML 的下限
        total_future_sum = 0
        n_threads_with_future = 0

        for g in test_graphs:
            thread_id = g.thread_id
            if thread_id not in thread_maps:
                continue
            parent_map, offset_map = thread_maps[thread_id]

            total_future = compute_total_future_nodes(parent_map, offset_map, WINDOW_SEC)
            if total_future == 0:
                continue  # 這個 thread 30 分鐘後沒有任何新節點，不算進分母

            total_future_sum += total_future
            n_threads_with_future += 1

            thread_reply = reply_df[reply_df["thread_id"] == thread_id]
            early_mask = (thread_reply["offset_sec"].isna()) | \
                         (thread_reply["offset_sec"] <= WINDOW_SEC)
            early_tweets = thread_reply[early_mask]["tweet_id"].astype(str).tolist()

            preds = node_scores.get(thread_id, np.array([]))
            gnn_scores = {
                tid: float(preds[i])
                for i, tid in enumerate(early_tweets[:len(preds)])
            }

            for strategy_name, selected_threads in selected_by_strategy.items():
                if thread_id not in selected_threads:
                    continue  # 這個策略沒選中這個 thread，不介入，貢獻 0
                intervene = strategy_gnn_nodes(gnn_scores, k_nodes)
                blocked = compute_blocked_descendants(
                    parent_map, offset_map, intervene, WINDOW_SEC
                )
                blocked_sum[strategy_name] += blocked

            # 完全沒有 ML 的下限：thread 也隨機選，thread 內節點也隨機選
            if thread_id in selected_by_strategy["random_threads"]:
                # 复用同一批被隨機選中的 thread，節點選擇改成隨機
                intervene_r = strategy_random_nodes(early_tweets, k_nodes)
                blocked_r = compute_blocked_descendants(
                    parent_map, offset_map, intervene_r, WINDOW_SEC
                )
                blocked_sum["random_threads_random_nodes"] += blocked_r

        event_crr = {
            f"crr_{s}": (blocked_sum[s] / total_future_sum if total_future_sum > 0 else 0.0)
            for s in blocked_sum
        }
        event_summary[test_event] = {
            "n_threads_total": len(all_thread_ids),
            "n_threads_with_future": n_threads_with_future,
            "n_threads_intervened": budget,
            "total_future_sum": total_future_sum,
            **event_crr,
        }

        if verbose:
            print(f"\n[{event_i}/{len(events)}] {test_event}"
                  f"（累計耗時 {(time.time()-run_start)/60:.1f} 分鐘）")
            print(f"  總 thread 數={len(all_thread_ids)}，"
                  f"有未來活動的={n_threads_with_future}，"
                  f"預算內介入={budget}")
            for s in selected_by_strategy:
                print(f"  CRR [{s:<20}] = {event_crr[f'crr_{s}']:.4f}")
            print(f"  CRR [random_threads_random_nodes] = "
                  f"{event_crr['crr_random_threads_random_nodes']:.4f}（完全沒有 ML 的下限）")

    # 彙總
    if verbose:
        print(f"\n{'='*70}")
        print(f"介入模擬彙總（budget_threads={budget_threads}, k_nodes={k_nodes}）")
        strategy_keys = [
            "crr_head_a_rank", "crr_observed_size_rank",
            "crr_random_threads", "crr_random_threads_random_nodes",
        ]
        header = f"{'event':<22}" + "".join(f"{k.replace('crr_',''):>22}" for k in strategy_keys)
        print(header)
        for ev, s in event_summary.items():
            row = f"{ev:<22}" + "".join(f"{s.get(k,0):>22.4f}" for k in strategy_keys)
            print(row)
        if event_summary:
            means = {k: np.mean([s.get(k,0) for s in event_summary.values()]) for k in strategy_keys}
            print("-" * (22 + 22*len(strategy_keys)))
            print(f"{'mean':<22}" + "".join(f"{means[k]:>22.4f}" for k in strategy_keys))
            print(f"\n（這是逐事件的 CRR 再平均，不是把所有事件的 blocked/total_future"
                  f"直接加總——逐事件平均讓每個事件的權重一樣，方便看『在多少事件上"
                  f"head_a_rank 贏過其他策略』；如果要看『整體實際減少的曝光總量』，"
                  f"要另外把所有事件的 blocked 總和除以 total_future 總和，我可以另外算）")

    return {"event_summary": event_summary, "train_results": train_results}


# ------------------------------------------------------------------
# 7. 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    GRAPHS_30MIN = os.getenv("GRAPHS_30MIN", "pheme_graphs_roberta_30min_v2.pt")
    REPLY_CSV    = os.getenv("REPLY_CSV",    "pheme_reply_level_v3.csv")

    print("載入圖資料集...")
    graphs_30min = torch.load(GRAPHS_30MIN, weights_only=False)
    print(f"30min：{len(graphs_30min)} 張圖")

    print("載入 reply-level 表...")
    reply_df = pd.read_csv(
        REPLY_CSV,
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str},
        low_memory=False,
    )
    print(f"reply_df：{len(reply_df)} 列")

    results = run_intervention_simulation(
        graphs_30min=graphs_30min,
        reply_df=reply_df,
        budget_threads=BUDGET_THREADS,
        k_nodes=K_INTERVENE,
    )