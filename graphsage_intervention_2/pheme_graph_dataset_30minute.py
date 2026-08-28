from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# ------------------------------------------------------------------
# 常數
# ------------------------------------------------------------------

NUMERIC_FEAT_COLS = [
    "depth", "offset_sec_log", "reply_latency_sec_log", "is_source",
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "followers_log", "friends_log", "statuses_log",
    "verified", "account_age_days_log",
]
EMBED_COLS    = [f"emb_{i}" for i in range(768)]
ALL_FEAT_COLS = NUMERIC_FEAT_COLS + EMBED_COLS  # 781 維

# 時間窗口對應的 coverage_score 欄位（節點層級，Head B 用）
WINDOW_LABEL_MAP = {
    None:   "coverage_score",          # 完整傳播樹
    1800:   "coverage_score_30min",    # 30 分鐘
    3600:   "coverage_score_60min",    # 60 分鐘
}

# 時間窗口對應的 future_growth 欄位（thread 層級，新版 Head A 用）
# window_sec=None（完整傳播樹）沒有「未來成長」這個概念可言——完整樹本身
# 已經是全部資訊，沒有窗口截斷，沒有「窗口之後」這件事。這個版本目前
# 不是主要實驗用的，thread_to_data() 遇到 window_sec=None 會退回舊的
# is_rumour 分類，只是為了不讓這個分支壞掉，不是建議拿來訓練用。
WINDOW_GROWTH_MAP = {
    None:   None,
    1800:   "future_growth_30min",
    3600:   "future_growth_60min",
}


# ------------------------------------------------------------------
# 1. 載入資料
# ------------------------------------------------------------------

def load_dataframes(
    node_feat_csv: str,
    reply_level_csv: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if verbose:
        print("載入節點特徵表...")
    feat_df = pd.read_csv(
        node_feat_csv,
        dtype={"thread_id": str, "tweet_id": str},
        low_memory=False,
    )
    if verbose:
        print("載入 reply-level 表...")
    reply_df = pd.read_csv(
        reply_level_csv,
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str},
        low_memory=False,
    )
    if verbose:
        print(f"節點特徵表：{len(feat_df)} 列，{len(feat_df.columns)} 欄")
        print(f"reply-level 表：{len(reply_df)} 列")
    return feat_df, reply_df


# ------------------------------------------------------------------
# 2. 合併 reply_df 裡的時間、windowed coverage、future_growth 到 feat_df
# ------------------------------------------------------------------

def merge_window_columns(
    feat_df: pd.DataFrame,
    reply_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    把 reply_df 裡的 offset_sec、coverage_score_30min/60min（節點層級）、
    future_growth_30min/60min（thread 層級）合併進 feat_df，供建圖時使用。
    """
    cols_to_merge = [
        "tweet_id", "offset_sec",
        "coverage_score_30min", "coverage_score_60min",
        "future_growth_30min", "future_growth_60min",
    ]
    missing = [c for c in cols_to_merge if c not in feat_df.columns]
    if missing:
        merge_cols = ["tweet_id"] + [c for c in missing if c != "tweet_id"]
        # 有些欄位（例如 future_growth）可能不在 reply_df 裡（舊版 CSV），
        # 這裡只合併真的存在的欄位，缺的之後在 thread_to_data() 會 fallback。
        merge_cols = [c for c in merge_cols if c == "tweet_id" or c in reply_df.columns]
        feat_df = feat_df.merge(
            reply_df[merge_cols],
            on="tweet_id",
            how="left",
        )
    return feat_df


# ------------------------------------------------------------------
# 3. StandardScaler
# ------------------------------------------------------------------

def fit_scaler(feat_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(feat_df[NUMERIC_FEAT_COLS].values)
    return scaler


# ------------------------------------------------------------------
# 4. 單一 thread → PyG Data
# ------------------------------------------------------------------

def thread_to_data(
    thread_id: str,
    feat_df: pd.DataFrame,
    reply_df: pd.DataFrame,
    scaler: StandardScaler,
    window_sec: Optional[float] = None,
) -> Optional[Data]:
    """
    window_sec：
      None  → 完整傳播樹（原始行為，y 退回 is_rumour 分類，不是主要實驗用）
      1800  → 只保留 30 分鐘內的節點和邊
              node_y 用 coverage_score_30min（節點層級，Head B）
              y      用 future_growth_30min（thread 層級，新版 Head A）
      3600  → 同上，60 分鐘版本

    語意（window_sec=1800 為例）：
      - GNN 輸入：前 30 分鐘出現的節點和邊（早期資訊）
      - y（新版 Head A）：這個 thread 在 30 分鐘之後總共還會長出幾個
        節點——早期窗口預測未來規模，拿來跨 thread 排序、決定預算限制
        下優先介入哪些
      - node_y（Head B）：這個節點在 30 分鐘之後能影響多少後代——
        給被選中的高風險 thread，決定 thread 內部哪個節點值得介入
      - 兩者都是用未來資訊當訓練目標，這是合法的監督式學習，不是
        temporal leakage（leakage 指的是「輸入」洩漏未來資訊，不是
        「目標」本身來自未來）
    """
    nodes = feat_df[feat_df["thread_id"] == thread_id].copy()
    if len(nodes) == 0:
        return None

    # 時間截斷：只保留窗口內的節點
    if window_sec is not None:
        offset_col = "offset_sec"
        if offset_col in nodes.columns:
            # NaN 的情況（已刪除推文）：保留在窗口內（offset 未知，保守處理）
            mask = (nodes[offset_col].isna()) | (nodes[offset_col] <= window_sec)
            nodes = nodes[mask].copy()
        if len(nodes) == 0:
            return None

    tweet_ids = nodes["tweet_id"].tolist()
    id_to_idx = {tid: i for i, tid in enumerate(tweet_ids)}

    # 節點特徵矩陣
    numeric_vals = scaler.transform(nodes[NUMERIC_FEAT_COLS].values)
    embed_vals   = nodes[EMBED_COLS].values.astype(np.float32)
    x_np = np.concatenate([numeric_vals, embed_vals], axis=1).astype(np.float32)
    if np.isnan(x_np).any():
        return None
    x = torch.tensor(x_np, dtype=torch.float)

    # 邊：只保留兩端節點都在窗口內的邊。補上反向邊（child → parent），
    # 讓每個節點的 embedding 也能吸收到自己既有子節點（窗口內已出現的
    # 回覆/轉推）的資訊——這是預測 future_growth / coverage_score 很
    # 自然的早期訊號，單向邊會讓節點完全吸收不到這個資訊。
    thread_replies = reply_df[reply_df["thread_id"] == thread_id]
    src_list, dst_list = [], []
    for _, row in thread_replies.iterrows():
        pid = row["parent_id"]
        tid = str(row["tweet_id"])
        if not isinstance(pid, str) or pid == "nan" or pd.isna(pid):
            continue
        if pid in id_to_idx and tid in id_to_idx:
            src_list.append(id_to_idx[pid])
            dst_list.append(id_to_idx[tid])

    if src_list:
        edge_index_fwd = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_index_bwd = torch.tensor([dst_list, src_list], dtype=torch.long)
        edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ---- Graph-level label y：新版 Head A ----
    growth_col = WINDOW_GROWTH_MAP.get(window_sec)
    if growth_col is not None and growth_col in nodes.columns:
        future_growth = float(nodes[growth_col].iloc[0])
        # future_growth 是計數值，零膨脹分布（很多 thread 窗口後就沒動靜），
        # 跟 coverage_score 遇到的狀況一樣，先做 log1p 轉換再當迴歸目標。
        y = torch.tensor([math.log1p(max(future_growth, 0.0))], dtype=torch.float)
    else:
        # window_sec=None，或 CSV 裡還沒有 future_growth 欄位（舊版）：
        # 退回 is_rumour 分類，只是為了不讓這個分支壞掉，不建議拿來訓練。
        is_rumour = int(nodes["is_rumour"].iloc[0])
        y = torch.tensor([is_rumour], dtype=torch.long)

    # Node-level label（Head B，沒變）
    label_col = WINDOW_LABEL_MAP.get(window_sec, "coverage_score")
    if label_col not in nodes.columns:
        label_col = "coverage_score"
    node_y = torch.tensor(
        nodes[label_col].fillna(0.0).values.astype(np.float32),
        dtype=torch.float,
    )

    event_id = nodes["event_id"].iloc[0]

    # is_source 遮罩：用來源欄位本身（未經 scaler 轉換）判斷，不從 x 裡
    # 反推——因為 x 的 is_source 那一欄會被 StandardScaler 轉換過，
    # 已經不是乾淨的 0/1，這裡另外存一份明確的布林遮罩，供之後訓練/
    # 模擬時判斷「這是不是來源推文」用，不會因為特徵被標準化而混淆。
    is_source_mask = torch.tensor(
        nodes["is_source"].values.astype(bool)
    )

    # 加速度特徵：窗口內節點裡，有多少比例是在「後半段」才出現的
    # （而不是前半段）——同一個節點數，前段湧入還沒停 vs 後段才慢慢
    # 冒出來，未來走向可能完全不同，現有特徵沒有捕捉到這個時序差異。
    # 用未經 scaler 轉換的原始 offset_sec 算，跟 size 特徵同一個邏輯：
    # 直接算出來、不用讓模型自己從節點特徵裡反推。
    half_window = (window_sec / 2.0) if window_sec is not None else None
    if half_window is not None and offset_col in nodes.columns:
        offsets = nodes[offset_col]
        # NaN（已刪除推文，時間未知）當作前半段，跟其他截斷邏輯一致的
        # 保守處理，不讓未知時間的節點被算進「加速」訊號裡。
        late_count = int(((offsets.notna()) & (offsets > half_window)).sum())
        total_count = len(nodes)
        late_frac = late_count / total_count if total_count > 0 else 0.0
    else:
        late_frac = 0.0
    late_activity_frac = torch.tensor([late_frac], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        node_y=node_y,
        thread_id=thread_id,
        event_id=event_id,
        num_nodes=len(tweet_ids),
        is_source_mask=is_source_mask,
        late_activity_frac=late_activity_frac,
    )


# ------------------------------------------------------------------
# 5. 建立完整資料集
# ------------------------------------------------------------------

def build_graph_dataset(
    feat_df: pd.DataFrame,
    reply_df: pd.DataFrame,
    scaler: StandardScaler,
    window_sec: Optional[float] = None,
    only_rumour: bool = True,
    verbose: bool = True,
) -> list[Data]:
    """
    only_rumour=True（預設）：只用 is_rumour==1 的 thread 建圖。

    這是這次稽核發現的落差：is_rumour 一路以來在註解裡都寫著「已知
    的篩選條件」，但整條 pipeline 從來沒有真的拿它去篩選過資料——
    Head A 訓練、介入模擬，一直都是在「謠言+一般新聞混在一起」的
    整體上跑。以 charliehebdo 為例，2079 個 thread 裡有 78%（1621
    個）其實是非謠言的一般新聞討論，不該出現在候選池裡——現實中
    你不會對已證實的真實新聞做「介入」，這些 thread 一直在跟真正
    的謠言 thread 搶預算名額，也讓 Head A 訓練學到的是「不管是不是
    謠言，什麼樣的推文串會長大」，稀釋了任務原本該聚焦的訊號。

    only_rumour=False 保留給需要對照組（例如想確認「有沒有篩選」
    這件事本身的影響有多大）時使用。
    """
    import time
    feat_df = merge_window_columns(feat_df, reply_df)

    if only_rumour:
        before = feat_df["thread_id"].nunique()
        feat_df = feat_df[feat_df["is_rumour"] == 1].copy()
        after = feat_df["thread_id"].nunique()
        if verbose:
            print(f"only_rumour=True：只保留 is_rumour==1 的 thread，"
                  f"{before} -> {after} 個（排除掉 {before - after} 個非謠言 thread）")

    thread_ids = feat_df["thread_id"].unique()
    window_str = f"{int(window_sec//60)}min" if window_sec else "full"
    start_time = time.time()
    if verbose:
        print(f"開始建圖（window={window_str}），共 {len(thread_ids)} 個 thread...")

    graphs = []
    skipped = 0
    for i, tid in enumerate(thread_ids):
        g = thread_to_data(tid, feat_df, reply_df, scaler, window_sec=window_sec)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1
        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_sec = (len(thread_ids) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{window_str}] 進度：{i+1}/{len(thread_ids)}，已建圖 {len(graphs)}，"
                  f"耗時 {elapsed:.0f} 秒，預估剩餘 {eta_sec:.0f} 秒")

    if verbose:
        print(f"\n[{window_str}] 完成：{len(graphs)} 張圖，跳過 {skipped} 個，"
              f"總耗時 {(time.time() - start_time):.0f} 秒")
        node_counts = [g.num_nodes for g in graphs]
        edge_counts = [g.edge_index.shape[1] for g in graphs]
        print(f"節點數：min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
        print(f"邊數：min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}"
              f"（含正反雙向）")
        if graphs[0].y.dtype == torch.float:
            y_vals = [g.y.item() for g in graphs]
            print(f"y（future_growth，log1p 轉換後）：min={min(y_vals):.3f}, "
                  f"max={max(y_vals):.3f}, mean={np.mean(y_vals):.3f}")
        else:
            rumour_count = sum(1 for g in graphs if g.y.item() == 1)
            print(f"（window=None，y 是 is_rumour 分類）rumour：{rumour_count}，"
                  f"non-rumour：{len(graphs)-rumour_count}")

    return graphs


# ------------------------------------------------------------------
# 6. LOEO 分割 / 儲存載入
# ------------------------------------------------------------------

def split_by_event(graphs, test_event):
    train = [g for g in graphs if g.event_id != test_event]
    test  = [g for g in graphs if g.event_id == test_event]
    return train, test

def get_all_events(graphs):
    return sorted(set(g.event_id for g in graphs))

def save_dataset(graphs, path):
    torch.save(graphs, path)
    print(f"已儲存 {len(graphs)} 張圖至 {path}")

def load_dataset(path):
    graphs = torch.load(path, weights_only=False)
    print(f"載入 {len(graphs)} 張圖從 {path}")
    return graphs


# ------------------------------------------------------------------
# 7. 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    NODE_FEAT_CSV   = os.getenv("NODE_FEAT_CSV",   "pheme_node_features_roberta.csv")
    REPLY_LEVEL_CSV = os.getenv("REPLY_LEVEL_CSV", "pheme_reply_level_v3.csv")
    ONLY_RUMOUR     = os.getenv("ONLY_RUMOUR", "1") == "1"

    feat_df, reply_df = load_dataframes(NODE_FEAT_CSV, REPLY_LEVEL_CSV)

    # scaler 的統計基準要跟實際會拿去訓練的資料一致——如果只用謠言
    # thread 訓練，scaler 也該只用謠言 thread 的數值分布去 fit，不然
    # 標準化的基準會被根本不會出現在訓練資料裡的非謠言 thread 影響。
    scaler_fit_df = feat_df[feat_df["is_rumour"] == 1] if ONLY_RUMOUR else feat_df
    print(f"scaler 用 {'僅謠言' if ONLY_RUMOUR else '全部'} thread 的 "
          f"{scaler_fit_df['thread_id'].nunique()} 個 thread 的數值特徵 fit")
    scaler = fit_scaler(scaler_fit_df)

    # 建三個版本的圖（window=None 的完整版目前只是保留相容性，不是主要實驗用）
    for window, suffix in [(None, "full"), (1800, "30min"), (3600, "60min")]:
        graphs = build_graph_dataset(feat_df, reply_df, scaler, window_sec=window,
                                       only_rumour=ONLY_RUMOUR)
        save_dataset(graphs, f"pheme_graphs_roberta_{suffix}_v2.pt")
        g = graphs[0]
        print(f"\n[{suffix}] 第一張圖：x={g.x.shape}, edges={g.edge_index.shape[1]}, "
              f"y={g.y}, node_y_mean={g.node_y.mean():.4f}")