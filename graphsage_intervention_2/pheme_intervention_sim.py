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

    關鍵修正：一個節點被介入，只能阻斷「還沒發生、且是透過一連串同樣
    還沒發生的節點才會出現」的後代——不能阻斷「窗口內已經存在的子節點」
    自己的未來。舉例：來源推文在窗口內已經有一個回覆 C（C 已經是真實
    存在的推文），這時候介入來源推文，C 並不會消失，C 之後會不會繼續
    被回覆是 C 自己的事，跟來源推文還在不在無關。

    做法：DFS 時區分每個節點「在決策當下（window_sec）是否已經存在」。
    已經存在的節點，它自己的命運只取決於它本身有沒有被選進
    intervene_ids，不會被祖先的介入狀態影響；只有「還沒發生的節點」
    才會因為祖先鏈上有節點被阻斷而跟著被阻斷（因為它的出現本來就
    是被那個還沒發生的祖先卡住的）。
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
        """
        is_blocked：這個節點的出現有沒有被祖先鏈上的介入卡住——只對
        「還沒發生的節點」有意義，已經存在的節點不受這個影響。
        """
        nonlocal blocked
        node_offset = offset_map.get(node_id, 0.0)
        if np.isnan(node_offset):
            node_offset = 0.0

        exists_at_decision_time = node_offset <= window_sec

        if exists_at_decision_time:
            # 已經存在，命運只看自己有沒有被選中介入，不繼承祖先的
            # 阻斷狀態——祖先被移除，不會讓已經存在的節點消失。
            child_is_blocked = node_id in intervene_ids
        else:
            # 還沒發生，是否被阻斷看祖先鏈：只要往上追溯到某個還沒
            # 發生、且被介入的節點，這整條鏈上還沒發生的節點都算
            # 被阻斷（因為它們的出現本來就是被那個節點卡住的）。
            if is_blocked and node_id != root_id:
                blocked += 1
            child_is_blocked = is_blocked

        for child in children_map.get(node_id, []):
            dfs(child, child_is_blocked)

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

def strategy_gnn_nodes(node_scores: dict[str, float], k: int,
                        always_include: Optional[set] = None) -> set[str]:
    """
    Head B：選預測 coverage_score 最高的 K 個節點。

    always_include（通常是來源推文 id）：一定會被納入介入集合，不佔
    Head B 排序的名額——來源推文幾乎必然是分數最高、也應該永遠被
    介入的節點，不需要模型判斷。node_scores 傳進來時應該已經不含
    always_include 裡的節點（呼叫端負責，這裡只是保險再濾一次）。
    """
    always_include = always_include or set()
    remaining_k = max(k - len(always_include), 0)
    candidates = {tid: s for tid, s in node_scores.items() if tid not in always_include}
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    picked = {tid for tid, _ in sorted_nodes[:remaining_k]}
    return always_include | picked


def strategy_oracle_nodes(true_node_scores: dict[str, float], k: int,
                           always_include: Optional[set] = None) -> set[str]:
    """
    Oracle Node：用真實 coverage_score_30min（不是 Head B 預測的）選
    前 K 個節點（扣掉 always_include 之後的剩餘名額）。這是「拿標準
    答案的分數排序」，不是真正窮舉所有 K 個節點組合去找最大化
    compute_blocked_descendants() 的最優解（那是組合優化問題，跟你們
    之前放棄 CELF 的理由一樣，真正求解太慢）。當作 Stage 2 表現的
    診斷上限參考即可，不是嚴格意義的最優解。
    """
    always_include = always_include or set()
    remaining_k = max(k - len(always_include), 0)
    candidates = {tid: s for tid, s in true_node_scores.items() if tid not in always_include}
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    picked = {tid for tid, _ in sorted_nodes[:remaining_k]}
    return always_include | picked


def strategy_random_nodes(node_ids: list[str], k: int, seed: int = 42,
                           always_include: Optional[set] = None) -> set[str]:
    always_include = always_include or set()
    remaining_k = max(k - len(always_include), 0)
    candidates = [tid for tid in node_ids if tid not in always_include]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=min(remaining_k, len(candidates)), replace=False)
    return always_include | set(chosen.tolist())


# ------------------------------------------------------------------
# 3. Stage 1：事件層級的 thread 選擇策略（新增）
# ------------------------------------------------------------------

def select_threads_by_score(
    thread_scores: dict[str, float],  # thread_id -> 排序用分數（可以是 Head A 預測值，
                                        # 也可以是真實 future_growth，看呼叫端傳什麼進來）
    budget: int,
) -> set[str]:
    """通用的排序選 thread 函式，分數來源不限——Head A 預測值跟
    Oracle 用的真實 future_growth 都用這個函式選，只是輸入的分數
    字典不一樣，避免同一段邏輯寫兩次。"""
    sorted_threads = sorted(thread_scores.items(), key=lambda x: x[1], reverse=True)
    return {tid for tid, _ in sorted_threads[:budget]}


def select_threads_by_observed_size(
    thread_node_counts: dict[str, int],  # thread_id -> 30分鐘內節點數
    budget: int,
) -> set[str]:
    """用目前已觀察到的討論量排序（不用 ML 的土法煉鋼 baseline）。"""
    return select_threads_by_score(thread_node_counts, budget)


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
    min_threads_with_future_for_summary: int = 30,
    device_str: str = "auto",
    verbose: bool = True,
) -> dict:
    """
    兩階段 LOEO 介入模擬。訓練這步直接呼叫 pheme_graphsage.run_loeo(
    ..., return_models=True)，不在這裡重新寫一份訓練邏輯。

    min_threads_with_future_for_summary：一個事件裡「有未來活動的
    thread 數」少於這個門檻，CRR 彙總平均時排除（但仍會逐一列出）。
    跟 run_loeo 排除小 fold 的邏輯是同一個道理——像 gurlitt 這種
    只有 5 個目標可以命中的事件，CRR 天生就只會落在 0、0.2、0.4...
    這種粗顆粒的值上，單一模型初始化的隨機性就能讓它從全對擺盪到
    全錯，直接把這種事件跟其他有幾百個目標的事件用同一個權重平均，
    會讓整體結論被這種高變異雜訊主導。

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

        # ---- 第一輪：收集每個 thread 的真實資訊（Oracle 用），
        #      跟每個節點的預測/真實分數，避免後面重複算 ----
        true_future_map = {}       # thread_id -> 真實 future_growth（Oracle Growth 排序用）
        early_tweets_map = {}      # thread_id -> 30分鐘內節點 id 列表
        gnn_scores_map = {}        # thread_id -> {tweet_id: Head B 預測分數}
        true_node_scores_map = {}  # thread_id -> {tweet_id: 真實 coverage_score_30min}（Oracle Node 用）

        for g in test_graphs:
            thread_id = g.thread_id
            if thread_id not in thread_maps:
                continue
            parent_map, offset_map = thread_maps[thread_id]

            total_future = compute_total_future_nodes(parent_map, offset_map, WINDOW_SEC)
            true_future_map[thread_id] = total_future
            if total_future == 0:
                continue

            thread_reply = reply_df[reply_df["thread_id"] == thread_id]
            early_mask = (thread_reply["offset_sec"].isna()) | \
                         (thread_reply["offset_sec"] <= WINDOW_SEC)
            early_tweets = thread_reply[early_mask]["tweet_id"].astype(str).tolist()
            early_tweets_map[thread_id] = early_tweets

            preds = node_scores.get(thread_id, np.array([]))
            gnn_scores_map[thread_id] = {
                tid: float(preds[i]) for i, tid in enumerate(early_tweets[:len(preds)])
            }
            # g.node_y 存的就是這個 thread（窗口內節點）的真實 coverage_score_30min，
            # 順序跟建圖時的節點順序一致——這裡沿用跟 gnn_scores 同樣的「早期節點
            # 順序對應」假設，是既有程式碼就有的假設，不是這次新引入的風險。
            true_scores = g.node_y.cpu().numpy()
            true_node_scores_map[thread_id] = {
                tid: float(true_scores[i]) for i, tid in enumerate(early_tweets[:len(true_scores)])
            }

        # ---- Stage 1：thread 選擇策略，新增 oracle_growth_rank（用真實
        #      future_growth 排序，不是 Head A 預測值——這是 Stage 1 的
        #      診斷上限：如果 Head A 完美預測，最多能選多好）----
        selected_by_strategy = {
            "head_a_rank": select_threads_by_score(thread_scores, budget),
            "oracle_growth_rank": select_threads_by_score(true_future_map, budget),
            "observed_size_rank": select_threads_by_observed_size(thread_node_counts, budget),
            "random_threads": select_threads_random(all_thread_ids, budget),
        }

        # ---- 診斷：node 選擇策略比較有沒有意義，取決於被選中的 thread
        #      是不是真的有超過 k_nodes 個節點可以挑——如果窗口內節點數
        #      <= k_nodes，不管用哪種排序方法都是「全部都選」，Head B
        #      跟 Oracle Node 的差異會被稀釋成 0，讓這兩欄的比較失去意義。----
        threads_to_check = selected_by_strategy["head_a_rank"] | selected_by_strategy["oracle_growth_rank"]
        trivial_count = sum(
            1 for tid in threads_to_check
            if thread_node_counts.get(tid, 0) <= k_nodes
        )
        # ---- 進一步診斷：不再只用「節點數不夠」這個間接推論，直接印出
        #      一個實際案例的 Head B 預測分數 vs 真實分數本身，看兩組
        #      數字是不是真的不一樣、選出來的節點集合是不是真的不一樣 ----
        if verbose and threads_to_check:
            sample_candidates = [tid for tid in threads_to_check
                                  if thread_node_counts.get(tid, 0) > k_nodes]
            sample_tid = sample_candidates[0] if sample_candidates else next(iter(threads_to_check))
            sample_gnn = gnn_scores_map.get(sample_tid, {})
            sample_true = true_node_scores_map.get(sample_tid, {})
            print(f"  [診斷] 抽樣 thread={sample_tid}（節點數={thread_node_counts.get(sample_tid)}，"
                  f"優先挑節點數 > k_nodes 的樣本，這樣排序方法不同才「應該」選出不同節點）：")
            print(f"    Head B 預測分數: {[round(v, 4) for v in sample_gnn.values()]}")
            print(f"    真實分數        : {[round(v, 4) for v in sample_true.values()]}")
            # sample_tid 本身就是這個 thread 的來源推文 id（thread_id 的定義就是
            # source_id），always_include 讓它自動介入、不佔排序名額。
            gnn_top5 = strategy_gnn_nodes(sample_gnn, k_nodes, always_include={sample_tid})
            oracle_top5 = strategy_oracle_nodes(sample_true, k_nodes, always_include={sample_tid})
            print(f"    Head B 選的節點（來源推文自動包含）: {gnn_top5}")
            print(f"    Oracle 選的節點（來源推文自動包含）: {oracle_top5}")
            print(f"    扣掉來源推文之後，兩者選的節點是否相同: "
                  f"{(gnn_top5 - {sample_tid}) == (oracle_top5 - {sample_tid})}")

        # ---- Stage 2：node 選擇策略只在 head_a_rank / oracle_growth_rank
        #      這兩個 thread 策略上，各自搭配 Head B 預測跟 Oracle Node
        #      真實分數，湊成 2x2，才能拆出問題在 Stage 1 還是 Stage 2。
        #      三種策略都會自動包含來源推文（thread_id 本身），只在
        #      剩餘名額裡用排序方法挑非來源節點。----
        combo_names = [
            ("head_a_rank", "head_b_nodes"),
            ("oracle_growth_rank", "head_b_nodes"),
            ("head_a_rank", "oracle_nodes"),
            ("oracle_growth_rank", "oracle_nodes"),
            ("observed_size_rank", "head_b_nodes"),
            ("random_threads", "head_b_nodes"),
        ]
        blocked_sum = {f"{t}__{n}": 0 for t, n in combo_names}
        blocked_sum["random_threads__random_nodes"] = 0  # 完全沒有 ML 的下限
        total_future_sum = 0
        n_threads_with_future = 0

        for thread_id, total_future in true_future_map.items():
            if total_future == 0:
                continue
            parent_map, offset_map = thread_maps[thread_id]
            total_future_sum += total_future
            n_threads_with_future += 1

            early_tweets = early_tweets_map[thread_id]
            gnn_scores = gnn_scores_map[thread_id]
            true_scores = true_node_scores_map[thread_id]
            source_set = {thread_id}  # thread_id 本身就是這個 thread 的來源推文 id

            for thread_strategy, node_strategy in combo_names:
                if thread_id not in selected_by_strategy[thread_strategy]:
                    continue  # 這個策略沒選中這個 thread，不介入，貢獻 0
                if node_strategy == "head_b_nodes":
                    intervene = strategy_gnn_nodes(gnn_scores, k_nodes, always_include=source_set)
                else:  # oracle_nodes
                    intervene = strategy_oracle_nodes(true_scores, k_nodes, always_include=source_set)
                blocked = compute_blocked_descendants(
                    parent_map, offset_map, intervene, WINDOW_SEC
                )
                blocked_sum[f"{thread_strategy}__{node_strategy}"] += blocked

            # 完全沒有 ML 的下限：thread 也隨機選，thread 內節點也隨機選
            # （這裡也讓來源推文自動介入，才是跟其他策略公平的下限比較——
            # 不然這個 baseline 會顯得比實際情況更弱，因為它連「一定要
            # 擋來源推文」這種不需要 ML 就懂的常識都沒有）
            if thread_id in selected_by_strategy["random_threads"]:
                intervene_r = strategy_random_nodes(early_tweets, k_nodes, always_include=source_set)
                blocked_r = compute_blocked_descendants(
                    parent_map, offset_map, intervene_r, WINDOW_SEC
                )
                blocked_sum["random_threads__random_nodes"] += blocked_r

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
            print(f"  --- Stage 1 拆解（node 選擇都固定用 Head B）---")
            print(f"  CRR [head_a_rank            + head_b_nodes] = {event_crr['crr_head_a_rank__head_b_nodes']:.4f}")
            print(f"  CRR [oracle_growth_rank      + head_b_nodes] = {event_crr['crr_oracle_growth_rank__head_b_nodes']:.4f}"
                  f"  <- Stage 1 診斷上限")
            print(f"  --- Stage 2 拆解（thread 選擇都固定用 Head A）---")
            print(f"  CRR [head_a_rank            + oracle_nodes ] = {event_crr['crr_head_a_rank__oracle_nodes']:.4f}"
                  f"  <- Stage 2 診斷上限")
            print(f"  CRR [oracle_growth_rank      + oracle_nodes ] = {event_crr['crr_oracle_growth_rank__oracle_nodes']:.4f}"
                  f"  <- 整套方法理論上限")
            print(f"  --- 其他對照 ---")
            print(f"  CRR [observed_size_rank      + head_b_nodes] = {event_crr['crr_observed_size_rank__head_b_nodes']:.4f}")
            print(f"  CRR [random_threads          + head_b_nodes] = {event_crr['crr_random_threads__head_b_nodes']:.4f}")
            print(f"  CRR [random_threads          + random_nodes] = "
                  f"{event_crr['crr_random_threads__random_nodes']:.4f}（完全沒有 ML 的下限）")

    # 彙總
    if verbose:
        print(f"\n{'='*70}")
        print(f"介入模擬彙總（budget_threads={budget_threads}, k_nodes={k_nodes}）")
        strategy_keys = [
            "crr_head_a_rank__head_b_nodes",
            "crr_oracle_growth_rank__head_b_nodes",
            "crr_head_a_rank__oracle_nodes",
            "crr_oracle_growth_rank__oracle_nodes",
            "crr_observed_size_rank__head_b_nodes",
            "crr_random_threads__head_b_nodes",
            "crr_random_threads__random_nodes",
        ]
        short_names = [
            "A+B(現況)", "OracleG+B", "A+OracleN", "OracleG+OracleN",
            "size+B", "rand+B", "rand+rand",
        ]
        header = f"{'event':<22}" + "".join(f"{n:>17}" for n in short_names)
        print(header)

        small_events = []
        for ev, s in event_summary.items():
            row = f"{ev:<22}" + "".join(f"{s.get(k,0):>17.4f}" for k in strategy_keys)
            if s.get("n_threads_with_future", 0) < min_threads_with_future_for_summary:
                row += (f"   <- 有未來活動的 thread 只有 "
                        f"{s.get('n_threads_with_future', 0)} 個，不計入下面的彙總平均")
                small_events.append(ev)
            print(row)

        included = {ev: s for ev, s in event_summary.items() if ev not in small_events}
        if included:
            means = {k: np.mean([s.get(k,0) for s in included.values()]) for k in strategy_keys}
            print("-" * (22 + 17*len(strategy_keys)))
            print(f"{'mean (排除小事件)':<22}" + "".join(f"{means[k]:>17.4f}" for k in strategy_keys))
            if small_events:
                print(f"\n排除的小事件（有未來活動的 thread < {min_threads_with_future_for_summary} 個，"
                      f"CRR 波動天生較大，不適合跟其他事件用同一個權重平均）: {small_events}")
            print(f"\n判讀方式：")
            print(f"  A+B(現況) vs OracleG+B   的差距 = Stage 1（Head A 排序）還有多少改善空間")
            print(f"  A+B(現況) vs A+OracleN   的差距 = Stage 2（Head B 選節點）還有多少改善空間")
            print(f"  哪個差距大，就代表問題主要在那一半，優先改那邊")
            print(f"  OracleG+OracleN = 兩階段都完美時的理論上限，A+B(現況) 離這個上限多遠，")
            print(f"  就是整套系統目前實際上還有多少（不切分 Stage 1/2 的）總改善空間")
            print(f"\n（以上是逐事件的 CRR 再平均，不是把所有事件的 blocked/total_future"
                  f"直接加總——逐事件平均讓每個事件的權重一樣；如果要看『整體實際減少的"
                  f"曝光總量』，要另外把所有事件的 blocked 總和除以 total_future 總和，"
                  f"我可以另外算）")

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