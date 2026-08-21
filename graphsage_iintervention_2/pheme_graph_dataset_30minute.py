from __future__ import annotations

import os
os.chdir(r"C:\FakeNews_IDS_Project")
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

# 時間窗口對應的 coverage_score 欄位
WINDOW_LABEL_MAP = {
    None:   "coverage_score",          # 完整傳播樹
    1800:   "coverage_score_30min",    # 30 分鐘
    3600:   "coverage_score_60min",    # 60 分鐘
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
# 2. 合併 reply_df 裡的時間和 windowed coverage 到 feat_df
# ------------------------------------------------------------------

def merge_window_columns(
    feat_df: pd.DataFrame,
    reply_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    把 reply_df 裡的 offset_sec、coverage_score_30min、coverage_score_60min
    合併進 feat_df，供建圖時使用。
    """
    cols_to_merge = ["tweet_id", "offset_sec", "coverage_score_30min", "coverage_score_60min"]
    # 只合併還沒有的欄位
    missing = [c for c in cols_to_merge if c not in feat_df.columns]
    if missing:
        merge_cols = ["tweet_id"] + [c for c in missing if c != "tweet_id"]
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
      None  → 完整傳播樹（原始行為）
      1800  → 只保留 30 分鐘內的節點和邊，label 用 coverage_score_30min
      3600  → 只保留 60 分鐘內的節點和邊，label 用 coverage_score_60min

    coverage_score 語意（window_sec=1800 為例）：
      - GNN 輸入：前 30 分鐘出現的節點和邊（早期資訊）
      - node_y label：這個節點在 30 分鐘之後能影響多少後代（未來資訊）
      - 模型學的是：從早期傳播結構預測未來影響力，不是從結構數後代
    """
    nodes = feat_df[feat_df["thread_id"] == thread_id].copy()
    if len(nodes) == 0:
        return None

    # 時間截斷：只保留窗口內的節點
    if window_sec is not None:
        # offset_sec 欄位在 feat_df 裡（已由 merge_window_columns 合併進來）
        # source tweet 的 offset_sec = 0，一定在窗口內
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

    # 邊：只保留兩端節點都在窗口內的邊
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
        # 原本這裡只有單向邊（parent → child），代表 SAGEConv 訊息傳遞時
        # 每個節點只吸收得到「祖先鏈」的資訊，完全吸收不到自己既有子節點
        # （已經出現的回覆/轉推）的資訊——但這正是預測 coverage_score
        # （這個節點未來會不會繼續擴散）最自然、也不算作弊的早期訊號
        # （「窗口內已經有幾個回覆」不是未來資訊，是當下就看得到的）。
        # 跟完整版 pheme_graph_dataset.py 的做法對齊，補上反向邊
        # （child → parent），讓這個訊號能傳到節點自己的 embedding 裡。
        edge_index_fwd = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_index_bwd = torch.tensor([dst_list, src_list], dtype=torch.long)
        edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Graph-level label
    is_rumour = int(nodes["is_rumour"].iloc[0])
    y = torch.tensor([is_rumour], dtype=torch.long)

    # Node-level label：根據 window_sec 選對應的 coverage_score 欄位
    label_col = WINDOW_LABEL_MAP.get(window_sec, "coverage_score")
    if label_col not in nodes.columns:
        label_col = "coverage_score"  # fallback
    node_y = torch.tensor(
        nodes[label_col].fillna(0.0).values.astype(np.float32),
        dtype=torch.float,
    )

    event_id = nodes["event_id"].iloc[0]

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        node_y=node_y,
        thread_id=thread_id,
        event_id=event_id,
        num_nodes=len(tweet_ids),
    )


# ------------------------------------------------------------------
# 5. 建立完整資料集
# ------------------------------------------------------------------

def build_graph_dataset(
    feat_df: pd.DataFrame,
    reply_df: pd.DataFrame,
    scaler: StandardScaler,
    window_sec: Optional[float] = None,
    verbose: bool = True,
) -> list[Data]:
    """
    window_sec：None = 完整傳播樹，1800 = 30分鐘，3600 = 60分鐘
    """
    # 確保 feat_df 有 offset_sec 和 windowed coverage 欄位
    feat_df = merge_window_columns(feat_df, reply_df)

    thread_ids = feat_df["thread_id"].unique()
    window_str = f"{int(window_sec//60)}min" if window_sec else "full"
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
        if verbose and (i + 1) % 500 == 0:
            print(f"  進度：{i+1}/{len(thread_ids)}，已建圖 {len(graphs)}")

    if verbose:
        print(f"\n完成：{len(graphs)} 張圖，跳過 {skipped} 個")
        node_counts = [g.num_nodes for g in graphs]
        edge_counts = [g.edge_index.shape[1] for g in graphs]
        rumour_count = sum(1 for g in graphs if g.y.item() == 1)
        print(f"rumour：{rumour_count}，non-rumour：{len(graphs)-rumour_count}")
        print(f"節點數：min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
        print(f"邊數：min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}"
              f"（含正反雙向，跟原本單向版比會剛好變兩倍，是預期中的變化，不是重複建邊）")

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
    REPLY_LEVEL_CSV = os.getenv("REPLY_LEVEL_CSV", "pheme_reply_level_v2.csv")

    feat_df, reply_df = load_dataframes(NODE_FEAT_CSV, REPLY_LEVEL_CSV)
    scaler = fit_scaler(feat_df)

    # 建三個版本的圖
    for window, suffix in [(None, "full"), (1800, "30min"), (3600, "60min")]:
        graphs = build_graph_dataset(feat_df, reply_df, scaler, window_sec=window)
        save_dataset(graphs, f"pheme_graphs_roberta_{suffix}.pt")
        g = graphs[0]
        print(f"\n[{suffix}] 第一張圖：x={g.x.shape}, edges={g.edge_index.shape[1]}, "
              f"y={g.y.item()}, node_y_mean={g.node_y.mean():.4f}")