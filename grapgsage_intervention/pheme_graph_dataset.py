"""
PHEME PyTorch Geometric 圖資料集建構
=====================================
輸入：
  - pheme_node_features.csv：節點特徵（397維）+ coverage_score
  - pheme_reply_level.csv：邊資訊（parent_id → tweet_id）

輸出：
  - List[torch_geometric.data.Data]：每個 thread 一張圖
  - 同時存成 .pt 檔案供後續直接載入

每張圖的結構：
  Data(
      x          : FloatTensor [N, 397]   節點特徵矩陣
      edge_index : LongTensor  [2, E]     邊（parent→child，COO格式）
      y          : LongTensor  [1]        graph-level label（is_rumour）
      node_y     : FloatTensor [N]        node-level label（coverage_score）
      thread_id  : str                    用於追蹤和 LOEO 分割
      event_id   : str                    用於 LOEO 分割
  )

設計原則：
- 邊的方向：parent → child（訊息從上往下傳）
  之後 GraphSAGE 可以選擇加上反向邊（child → parent）做 bi-directional
  目前先只用單向，讓模型架構決定要不要加反向
- 節點編號：在每張圖內部從 0 重新編號
  （PyG 的 edge_index 要求節點編號從 0 開始）
- 特徵標準化（StandardScaler）：
  只對前 13 維的數值特徵做，sentence-transformer 嵌入不做
  （嵌入本身已經是 L2 normalized，再 scale 反而會破壞語意距離）
  StandardScaler 在這裡 fit，但你在訓練時要注意：
  正式實驗應該在 train fold 上 fit、用同樣的 scaler transform test fold
  這裡提供一個 get_scaler() 函式讓訓練迴圈取用
- LOEO 分割：提供 split_by_event() 函式，回傳 train/test 的圖列表
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# ------------------------------------------------------------------
# 0. 常數
# ------------------------------------------------------------------

# 前 13 維數值特徵的欄位名稱（需要 StandardScaler）
NUMERIC_FEAT_COLS = [
    "depth", "offset_sec_log", "reply_latency_sec_log", "is_source",
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "followers_log", "friends_log", "statuses_log",
    "verified", "account_age_days_log",
]

# sentence-transformer 嵌入欄位（不做 StandardScaler）
EMBED_COLS = [f"emb_{i}" for i in range(768)]

ALL_FEAT_COLS = NUMERIC_FEAT_COLS + EMBED_COLS  # 397 維


# ------------------------------------------------------------------
# 1. 載入資料
# ------------------------------------------------------------------

def load_dataframes(
    node_feat_csv: str,
    reply_level_csv: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    載入節點特徵表和 reply-level 表，做基本的型別對齊。
    """
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
# 2. StandardScaler（只對數值特徵 fit）
# ------------------------------------------------------------------

def fit_scaler(feat_df: pd.DataFrame) -> StandardScaler:
    """
    對全資料集的前 13 維數值特徵 fit StandardScaler。
    
    注意：正式實驗裡應該只在 train fold 上 fit，
    這裡的全資料集版本用於快速驗證 pipeline 是否正常。
    訓練迴圈裡請用 fit_scaler(train_graphs_feat_df) 取得 scaler，
    再用同一個 scaler transform test fold。
    """
    scaler = StandardScaler()
    scaler.fit(feat_df[NUMERIC_FEAT_COLS].values)
    return scaler


# ------------------------------------------------------------------
# 3. 單一 thread → PyG Data 物件
# ------------------------------------------------------------------

def thread_to_data(
    thread_id: str,
    feat_df: pd.DataFrame,
    reply_df: pd.DataFrame,
    scaler: StandardScaler,
) -> Optional[Data]:
    """
    把單一 thread 的節點特徵和邊資訊轉成 PyG Data 物件。
    
    回傳 None 的情況：
    - 節點數為 0（資料問題）
    - 特徵欄位有 NaN（不應該發生，但加防護）
    """
    # 取出這個 thread 的節點
    nodes = feat_df[feat_df["thread_id"] == thread_id].copy()
    if len(nodes) == 0:
        return None

    # 節點編號：在圖內從 0 重新編號
    tweet_ids = nodes["tweet_id"].tolist()
    id_to_idx = {tid: i for i, tid in enumerate(tweet_ids)}

    # 節點特徵矩陣 x
    numeric_vals = scaler.transform(nodes[NUMERIC_FEAT_COLS].values)
    embed_vals = nodes[EMBED_COLS].values.astype(np.float32)
    x_np = np.concatenate([numeric_vals, embed_vals], axis=1).astype(np.float32)

    if np.isnan(x_np).any():
        return None  # 不應該發生，但加防護

    x = torch.tensor(x_np, dtype=torch.float)

    # 邊：從 reply-level 表取 parent_id → tweet_id
    thread_replies = reply_df[reply_df["thread_id"] == thread_id]
    src_list, dst_list = [], []
    for _, row in thread_replies.iterrows():
        pid = row["parent_id"]
        tid = str(row["tweet_id"])
        # source tweet 沒有 parent，跳過
        if not isinstance(pid, str) or pid == "nan" or pd.isna(pid):
            continue
        if pid in id_to_idx and tid in id_to_idx:
            src_list.append(id_to_idx[pid])   # parent
            dst_list.append(id_to_idx[tid])   # child

    if src_list:
        # 原本的單向邊（parent → child）
        edge_index_fwd = torch.tensor([src_list, dst_list], dtype=torch.long)
        # 加上反向邊（child → parent）
        edge_index_bwd = torch.tensor([dst_list, src_list], dtype=torch.long)
        edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Graph-level label
    is_rumour = int(nodes["is_rumour"].iloc[0])
    y = torch.tensor([is_rumour], dtype=torch.long)

    # Node-level label（coverage_score）
    node_y = torch.tensor(
        nodes["coverage_score"].values.astype(np.float32),
        dtype=torch.float,
    )

    # event_id（用於 LOEO 分割）
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
# 4. 建立完整資料集
# ------------------------------------------------------------------

def build_graph_dataset(
    feat_df: pd.DataFrame,
    reply_df: pd.DataFrame,
    scaler: StandardScaler,
    verbose: bool = True,
) -> list[Data]:
    """
    對所有 thread 建立 PyG Data 物件列表。
    """
    thread_ids = feat_df["thread_id"].unique()
    if verbose:
        print(f"開始建圖，共 {len(thread_ids)} 個 thread...")

    graphs = []
    skipped = 0
    for i, tid in enumerate(thread_ids):
        g = thread_to_data(tid, feat_df, reply_df, scaler)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1

        if verbose and (i + 1) % 500 == 0:
            print(f"  進度：{i+1}/{len(thread_ids)}，已建圖 {len(graphs)}")

    if verbose:
        print(f"\n完成：{len(graphs)} 張圖，跳過 {skipped} 個")
        # 統計
        node_counts = [g.num_nodes for g in graphs]
        edge_counts = [g.edge_index.shape[1] for g in graphs]
        rumour_count = sum(1 for g in graphs if g.y.item() == 1)
        print(f"rumour：{rumour_count}，non-rumour：{len(graphs)-rumour_count}")
        print(f"節點數：min={min(node_counts)}, max={max(node_counts)}, "
              f"mean={np.mean(node_counts):.1f}")
        print(f"邊數：min={min(edge_counts)}, max={max(edge_counts)}, "
              f"mean={np.mean(edge_counts):.1f}")

    return graphs


# ------------------------------------------------------------------
# 5. LOEO 分割
# ------------------------------------------------------------------

def split_by_event(
    graphs: list[Data],
    test_event: str,
) -> tuple[list[Data], list[Data]]:
    """
    Leave-One-Event-Out 分割。
    
    回傳：(train_graphs, test_graphs)
    """
    train = [g for g in graphs if g.event_id != test_event]
    test  = [g for g in graphs if g.event_id == test_event]
    return train, test


def get_all_events(graphs: list[Data]) -> list[str]:
    return sorted(set(g.event_id for g in graphs))


# ------------------------------------------------------------------
# 6. 儲存 / 載入
# ------------------------------------------------------------------

def save_dataset(graphs: list[Data], path: str) -> None:
    torch.save(graphs, path)
    print(f"已儲存 {len(graphs)} 張圖至 {path}")


def load_dataset(path: str) -> list[Data]:
    graphs = torch.load(path, weights_only=False)
    print(f"載入 {len(graphs)} 張圖從 {path}")
    return graphs


# ------------------------------------------------------------------
# 7. 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    NODE_FEAT_CSV  = os.getenv("NODE_FEAT_CSV",  'pheme_node_features_loeo_finetuned.csv')
    REPLY_LEVEL_CSV = os.getenv("REPLY_LEVEL_CSV", "pheme_reply_level.csv")
    OUTPUT_PT      = os.getenv("OUTPUT_PT",       'pheme_graphs_loeo_finetuned.pt')

    # 載入
    feat_df, reply_df = load_dataframes(NODE_FEAT_CSV, REPLY_LEVEL_CSV)

    # Scaler（全資料集 fit，正式訓練時請改成只對 train fold fit）
    print("Fit StandardScaler...")
    scaler = fit_scaler(feat_df)

    # 建圖
    graphs = build_graph_dataset(feat_df, reply_df, scaler)

    # 簡單驗證第一張圖
    g = graphs[0]
    print(f"\n第一張圖：")
    print(f"  thread_id : {g.thread_id}")
    print(f"  event_id  : {g.event_id}")
    print(f"  x.shape   : {g.x.shape}")
    print(f"  edge_index: {g.edge_index.shape}")
    print(f"  y         : {g.y.item()}")
    print(f"  node_y    : {g.node_y[:5]}")

    # LOEO 驗證
    events = get_all_events(graphs)
    print(f"\n事件列表：{events}")
    train, test = split_by_event(graphs, test_event=events[0])
    print(f"LOEO test_event={events[0]}：train={len(train)}, test={len(test)}")

    # 儲存
    save_dataset(graphs, OUTPUT_PT)