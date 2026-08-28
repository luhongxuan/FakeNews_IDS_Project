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
import random
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
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

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
FOLLOWERS_LOG_IDX = NUMERIC_FEAT_COLS.index("followers_log")  # 用來抓「這棵樹裡粉絲數最高的參與者」
DEPTH_IDX = NUMERIC_FEAT_COLS.index("depth")  # 用來抓樹的「形狀」（結構性病毒度），不是單純大小

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
        rank_weight: float = 0.5,
        rank_margin: float = 0.1,
    ):
        """
        rank_weight：Head A 的 loss 裡，排序損失佔的比例（1-rank_weight
        給 MSE）。設 0 就是退回純 MSE（舊行為），方便你之後想做消融
        實驗比較「加不加排序損失」的差異。
        rank_margin：排序損失的 margin，兩個預測值的差距要超過這個
        margin 才算「排序排對了、不用再懲罰」。
        """
        super().__init__()
        self.alpha = alpha
        self.dropout = dropout
        self.rank_weight = rank_weight
        self.rank_margin = rank_margin

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for i in range(num_layers):
            in_ch  = in_channels if i == 0 else hidden
            self.convs.append(SAGEConv(in_ch, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        # --- Head A: graph-level 迴歸（future_growth，log1p 轉換後）---
        # 輸入是 mean_pool + max_pool（各 hidden 維）+ 5 個明確特徵
        # （log1p(節點數)、最高粉絲數、平均深度、最大深度、加速度）。
        #
        # 加 max_pool 的理由：mean pooling 會把「一個特別突出的節點」
        # 稀釋掉——跟我們發現 size 訊號被稀釋是同一個問題的不同面向。
        # max pooling 保留「這棵樹裡最極端的訊號是什麼」，跟 mean
        # pooling（整體平均樣貌）互補，不是取代。
        #
        # 加「最高粉絲數」的理由：跟 size 特徵同一個邏輯——一個關鍵的
        # 高影響力回覆者，他的粉絲數經過 mean pooling 會被其他普通
        # 用戶平均掉。直接把「這群參與者裡最有影響力的是誰」抽出來，
        # 不用讓模型自己從被稀釋過的平均訊號裡反推。
        #
        # 加「平均深度」「最大深度」的理由：這是一個全新類別的訊號，
        # 不是前面幾個特徵（size、影響力、加速度）的延伸——傳播學文獻
        # 裡「結構性病毒度」（structural virality，用樹的深度衡量）
        # 常常比單純的節點數量更能預測最終規模。同樣是 20 個節點，
        # 一棵「一個人發、20 個人各自回」的扁平樹，跟一棵「一路接龍
        # 傳下去」的深樹，未來走向可能完全不同，現有的 size/影響力
        # 特徵完全沒有捕捉這個「形狀」上的差異。depth 本來就是 x 裡
        # 的原始特徵，直接用 pooling 抓出來，不用重建圖。
        #
        # 用 Softplus 而不是不加限制的線性輸出：log1p(future_growth) 一定
        # >= 0，讓輸出結構性地保證非負，比讓模型自己學「不要輸出負的」更穩。
        self.head_a = nn.Sequential(
            nn.Linear(hidden * 2 + 5, hidden // 2),
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

    def forward(self, x, edge_index, batch, late_activity_frac=None):
        """
        回傳：
            pred_a:   graph-level future_growth 預測 [B, 1]（已過 Softplus，非負）
            pred_b:   node-level coverage_score [N, 1]（已過 sigmoid，[0,1]）
            node_emb: 節點嵌入 [N, hidden]（方便 debug）

        late_activity_frac: [B] 或 [B,1]，窗口內節點裡「後半段才出現」
        的比例（建圖時算好、存在 Data 物件裡）。同一個 node_counts，
        前段湧入還沒停 vs 後段才慢慢冒出來，未來走向可能完全不同，
        這個特徵補上 size 特徵沒有的時序資訊。
        """
        # 在 GNN 層之前，先從原始輸入特徵抓「這棵樹裡粉絲數最高的
        # 參與者」跟「樹的深度」——用原始輸入而不是學到的嵌入，這些
        # 訊號才不會被 GNN 訊息傳遞過程中的平均/混合再次稀釋掉。
        max_followers_feat = global_max_pool(
            x[:, FOLLOWERS_LOG_IDX:FOLLOWERS_LOG_IDX + 1], batch
        )  # [B, 1]
        mean_depth_feat = global_mean_pool(
            x[:, DEPTH_IDX:DEPTH_IDX + 1], batch
        )  # [B, 1]，樹的「形狀」——扁平 vs 一路接龍傳下去
        max_depth_feat = global_max_pool(
            x[:, DEPTH_IDX:DEPTH_IDX + 1], batch
        )  # [B, 1]，這棵樹目前傳了幾層

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_emb = x  # [N, hidden]

        mean_emb = global_mean_pool(node_emb, batch)  # [B, hidden]
        max_emb  = global_max_pool(node_emb, batch)   # [B, hidden]

        # 每張圖目前的節點數（= 30 分鐘窗口內已經觀察到的節點數，
        # 跟 observed_size_rank baseline 用的是同一個數字），從 batch
        # 索引張量直接數出來，不需要額外傳參數、也不用改 Data schema。
        node_counts = torch.bincount(batch, minlength=mean_emb.size(0)).float()
        size_feat = torch.log1p(node_counts).unsqueeze(-1)  # [B, 1]

        if late_activity_frac is None:
            # 沒傳進來（例如舊版 Data 物件、還沒重建圖）補 0，不讓
            # forward 直接壞掉，但這種情況下這個特徵沒有作用。
            late_frac = torch.zeros(mean_emb.size(0), 1, device=mean_emb.device)
        else:
            late_frac = late_activity_frac
            if late_frac.dim() == 1:
                late_frac = late_frac.unsqueeze(-1)

        graph_feat = torch.cat(
            [mean_emb, max_emb, size_feat, max_followers_feat,
             late_frac, mean_depth_feat, max_depth_feat], dim=1
        )  # [B, hidden*2+5]
        pred_a = self.head_a(graph_feat)  # [B, 1]

        pred_b = self.head_b(node_emb)  # [N, 1]

        return pred_a, pred_b, node_emb

    def compute_loss(self, pred_a, pred_b, y, node_y, is_source_mask, event_ids=None):
        """
        pred_a: [B, 1]，graph-level future_growth 預測
        pred_b: [N, 1]，node-level coverage_score 預測
        y:      [B]，graph-level future_growth（log1p 轉換後的迴歸目標）
        node_y: [N]，node-level coverage_score（0~1）
        is_source_mask: [N]，True 代表這個節點是來源推文
        event_ids: 長度 B 的 list，每個 thread 屬於哪個事件——排序損失
        只在「同一個事件內」的 thread 之間兩兩比較，因為 Head A 真正
        要做的決策就是同一事件內排序，跨事件比較預測值沒有意義（不同
        事件的 thread 不會被放在同一個預算池裡競爭）。

        Head A 的 loss 現在是 MSE + 排序損失的加權組合：
        MSE 讓預測值的絕對尺度大致合理，排序損失直接針對「同事件內，
        誰該排前面」給梯度——這是 Stage 1（跨 thread 排序）真正被
        評估的方式，MSE 對這件事沒有直接的獎勵訊號，加上排序損失
        才會讓模型的優化目標跟實際被使用的方式對齊。

        Head B 的 loss 刻意排除來源節點：來源推文幾乎必然是分數最高、
        也應該永遠被介入的節點，不需要模型學——讓它留在 loss 裡，
        模型會學到「猜是不是來源推文」這個捷徑就能大幅壓低 MSE，
        代價是放棄學習真正需要判斷的部分。排除來源節點後，梯度訊號
        全部用在這個真正的任務上。
        """
        loss_a_mse = F.mse_loss(pred_a.squeeze(-1), y.float())

        loss_a_rank = torch.tensor(0.0, device=pred_a.device)
        if event_ids is not None and self.rank_weight > 0:
            pred_flat = pred_a.squeeze(-1)
            y_flat = y.float()

            groups: dict = {}
            for idx, eid in enumerate(event_ids):
                groups.setdefault(eid, []).append(idx)

            pair_losses = []
            for eid, idxs in groups.items():
                if len(idxs) < 2:
                    continue  # 這個 batch 裡這個事件只有一個 thread，沒有可比較的對象
                idxs_t = torch.tensor(idxs, device=pred_a.device)
                p = pred_flat[idxs_t]
                yv = y_flat[idxs_t]
                n = len(idxs)
                pi = p.unsqueeze(1).expand(n, n)
                pj = p.unsqueeze(0).expand(n, n)
                yi = yv.unsqueeze(1).expand(n, n)
                yj = yv.unsqueeze(0).expand(n, n)
                # 只看真實值 y_i > y_j 的組合：這種情況下希望預測值
                # pred_i 也比 pred_j 大（差距至少要有 rank_margin），
                # 不然就懲罰。y_i == y_j 的組合沒有明確的排序方向，跳過。
                mask = yi > yj
                if mask.sum() == 0:
                    continue
                margin_loss = F.relu(self.rank_margin - (pi - pj))[mask]
                pair_losses.append(margin_loss.mean())

            if pair_losses:
                loss_a_rank = torch.stack(pair_losses).mean()

        loss_a = (1 - self.rank_weight) * loss_a_mse + self.rank_weight * loss_a_rank

        non_source = ~is_source_mask
        if non_source.sum() > 0:
            loss_b = F.mse_loss(
                pred_b.squeeze(-1)[non_source], node_y[non_source]
            )
        else:
            # 極端情況：這個 batch 全部都是來源節點（單節點圖），
            # 沒有非來源節點可以算 loss_b，給 0 避免除零/空張量報錯。
            loss_b = torch.tensor(0.0, device=pred_b.device)

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
            is_source_mask = g.is_source_mask,
            late_activity_frac = g.late_activity_frac,
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
        pred_a, pred_b, _ = model(batch.x, batch.edge_index, batch.batch,
                                    late_activity_frac=batch.late_activity_frac)
        loss, la, lb = model.compute_loss(
            pred_a, pred_b, batch.y, batch.node_y, batch.is_source_mask,
            event_ids=batch.event_id,
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
    node_y_true, node_y_pred, node_is_source = [], [], []

    for batch in loader:
        batch = batch.to(device)
        pred_a, pred_b, _ = model(batch.x, batch.edge_index, batch.batch,
                                    late_activity_frac=batch.late_activity_frac)
        y_true.extend(batch.y.cpu().numpy().ravel())
        y_pred.extend(pred_a.squeeze(-1).cpu().numpy())
        node_y_true.extend(batch.node_y.cpu().numpy())
        node_y_pred.extend(pred_b.squeeze(-1).cpu().numpy())
        node_is_source.extend(batch.is_source_mask.cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    node_yt = np.array(node_y_true)
    node_yp = np.array(node_y_pred)
    is_src  = np.array(node_is_source)

    # Spearman 在常數輸入（例如全部節點預測值一樣）時會回傳 NaN，
    # 用 0.0 頂著，避免彙總計算時整個炸掉。
    def safe_spearman(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        r, _ = spearmanr(a, b)
        return 0.0 if np.isnan(r) else r

    non_source = ~is_src
    # node_mae / spearman_b 現在只看非來源節點——這才是 Head B 實際被
    # 訓練、也實際被拿去做決策的部分（來源推文永遠自動介入，不需要
    # 模型判斷）。另外保留 all-node 版本（後綴 _all）方便跟舊結果比較，
    # 但判斷模型好不好，看不加 _all 的這組才是誠實的。
    return {
        "mse_a": mean_squared_error(y_true, y_pred),
        "mae_a": mean_absolute_error(y_true, y_pred),
        "spearman_a": safe_spearman(y_true, y_pred),

        "node_mse": mean_squared_error(node_yt[non_source], node_yp[non_source])
                    if non_source.sum() > 0 else 0.0,
        "node_mae": mean_absolute_error(node_yt[non_source], node_yp[non_source])
                    if non_source.sum() > 0 else 0.0,
        "spearman_b": safe_spearman(node_yt[non_source], node_yp[non_source])
                      if non_source.sum() > 0 else 0.0,

        "node_mse_all": mean_squared_error(node_yt, node_yp),
        "node_mae_all": mean_absolute_error(node_yt, node_yp),
        "spearman_b_all": safe_spearman(node_yt, node_yp),
    }


def constant_baseline_mae(
    loader: DataLoader, constant_value: float, field: str,
    non_source_only: bool = True,
) -> float:
    """
    算「不管輸入是什麼，通通猜 constant_value」這個最笨的 baseline，
    對這個 loader 裡所有 `field`（'y' 或 'node_y'）的 MAE。
    non_source_only=True（預設）時，node_y 只看非來源節點——跟 Head B
    現在實際訓練/評估的範圍對齊，baseline 才是公平的對照組。
    """
    all_vals = []
    for batch in loader:
        if field == "y":
            vals = batch.y.numpy().ravel()
        else:
            vals = batch.node_y.numpy()
            if non_source_only:
                vals = vals[~batch.is_source_mask.numpy()]
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
    rank_weight: float = 0.5,
    rank_margin: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    device_str: str = "auto",
    min_test_graphs_for_summary: int = MIN_TEST_GRAPHS_FOR_SUMMARY,
    return_models: bool = False,
    seed: Optional[int] = 42,
    verbose: bool = True,
):
    """
    Leave-One-Event-Out 交叉驗證。

    seed：固定隨機種子（模型初始化、dropout、train/val 切分）。設固定
    值才能讓「改超參數前後」的比較公平——不然這次結果的差異，沒辦法
    分清楚是超參數真的有效果、還是剛好抽到不同的隨機初始化。想看
    多次隨機重跑的變異程度，才設 None 或每次換不同的值。

    rank_weight：Head A 排序損失的權重（0 = 退回純 MSE，方便做消融
    實驗比較）。

    return_models=False（預設，跟舊版行為一致）：
        回傳 results，{event_id: {指標dict}}。
    return_models=True：
        回傳 (results, models)，models 是 {event_id: 已載入該 fold
        最佳權重、且已切到 eval() 模式的模型}，連同對應的 scaler 一起
        存在 model.fitted_scaler_ 屬性上，供 pheme_intervention_sim.py
        直接拿去對這個 event 的 test threads 做推論，不用重新訓練
        （避免訓練邏輯在兩個檔案裡各自維護、容易兜不起來）。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 光是固定 seed 不夠——CUDA 上很多平行運算（尤其 GNN 訊息傳遞
        # 對鄰居做加總這類 scatter/reduce 操作）浮點數加總順序不保證
        # 每次一樣，微小數值誤差在梯度下降過程裡會被放大，導致「同樣
        # 的 seed」在 GPU 上訓練出來的結果還是每次不同。這裡把確定性
        # 運算模式打開，才是真正讓同一個 seed 可重現的必要條件。
        # warn_only=True：遇到沒有確定性版本的運算只警告、不報錯中斷，
        # 盡量做到確定性，不會因為某個運算沒有確定性核心就整個跑不動。
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"使用裝置：{device}" + (f"，固定隨機種子={seed}" if seed is not None else "，隨機種子未固定"))

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
        #     對 val / test 通通猜這個值。Head B 的平均值只算非來源
        #     節點——來源節點已經不在訓練/評估範圍內，baseline 要
        #     對齊，不然會拿一個包含來源節點、偏高的平均值去跟只
        #     算非來源節點的模型表現比，比較本身就不公平。---
        train_y_mean = float(np.mean([g.y.item() for g in train_g]))
        train_node_y_non_source = np.concatenate([
            g.node_y.numpy()[~g.is_source_mask.numpy()] for g in train_g
        ])
        train_node_y_mean = (
            float(train_node_y_non_source.mean())
            if len(train_node_y_non_source) > 0 else 0.0
        )
        baseline_val_mae_a  = constant_baseline_mae(val_loader,  train_y_mean, "y")
        baseline_val_mae_b  = constant_baseline_mae(val_loader,  train_node_y_mean, "node_y")
        baseline_test_mae_a = constant_baseline_mae(test_loader, train_y_mean, "y")
        baseline_test_mae_b = constant_baseline_mae(test_loader, train_node_y_mean, "node_y")
        if verbose:
            print(f"  baseline（猜 train 平均值，Head B 只算非來源節點）："
                  f"A: train_y_mean={train_y_mean:.4f} val_mae={baseline_val_mae_a:.4f} | "
                  f"B: train_node_y_mean={train_node_y_mean:.4f} val_mae={baseline_val_mae_b:.4f}")

        model = GraphSAGEDual(
            in_channels=IN_CHANNELS,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            alpha=alpha,
            rank_weight=rank_weight,
            rank_margin=rank_margin,
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
# 4b. 兩階段訓練：全部資料打底 → 謠言子集微調
# ------------------------------------------------------------------

def _run_training_stage(
    model, train_g, val_g, alpha, lr, epochs, patience,
    batch_size, device, stage_label, verbose,
):
    """
    共用的訓練迴圈：給一個已經建好的 model（可能全新初始化，也可能
    已經在別的階段訓練過），在 train_g/val_g 上訓練到 early stopping，
    回傳訓練好、已經載入最佳權重、且切到 eval() 模式的 model。

    抽出來當共用函式，是因為打底階段跟微調階段要跑的是同一套邏輯
    （train_epoch → 算 val_score → early stopping），只有資料、
    epoch 數、patience 不一樣，不用寫兩次容易兜不起來的重複邏輯。
    """
    train_loader = DataLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_g,   batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_y_mean = float(np.mean([g.y.item() for g in train_g]))
    train_node_y_non_source = np.concatenate([
        g.node_y.numpy()[~g.is_source_mask.numpy()] for g in train_g
    ])
    train_node_y_mean = (
        float(train_node_y_non_source.mean())
        if len(train_node_y_non_source) > 0 else 0.0
    )
    baseline_val_mae_a = constant_baseline_mae(val_loader, train_y_mean, "y")
    baseline_val_mae_b = constant_baseline_mae(val_loader, train_node_y_mean, "node_y")

    best_val_score = -float("inf")
    patience_counter = 0
    best_state = None
    stage_start = time.time()

    if verbose:
        print(f"  [{stage_label}] 開始訓練（最多 {epochs} epoch，patience={patience}，"
              f"train={len(train_g)} val={len(val_g)}）...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        mae_ratio_a = val_metrics["mae_a"] / max(baseline_val_mae_a, 1e-6)
        mae_ratio_b = val_metrics["node_mae"] / max(baseline_val_mae_b, 1e-6)
        val_score = -(alpha * mae_ratio_a + (1 - alpha) * mae_ratio_b)

        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if verbose:
            mark = "*" if improved else " "
            print(
                f"  [{stage_label}] epoch {epoch:3d}/{epochs} {mark}| "
                f"loss={train_loss['loss']:.4f} | "
                f"val_score={val_score:+.4f} "
                f"(ratio_A={mae_ratio_a:.3f} ratio_B={mae_ratio_b:.3f}) | "
                f"patience={patience_counter}/{patience} | "
                f"{time.time()-epoch_start:.1f}s/epoch，累計 "
                f"{(time.time()-stage_start)/60:.1f} 分鐘"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"  [{stage_label}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


def run_loeo_pretrain_finetune(
    rumour_graphs: list[Data],
    full_graphs: list[Data],
    hidden: int = 256,
    num_layers: int = 3,
    dropout: float = 0.3,
    alpha: float = 0.7,
    rank_weight: float = 0.0,
    rank_margin: float = 0.1,
    lr: float = 1e-3,
    pretrain_epochs: int = 15,
    pretrain_patience: int = 5,
    finetune_epochs: int = 50,
    finetune_patience: int = 15,
    batch_size: int = 32,
    device_str: str = "auto",
    min_test_graphs_for_summary: int = MIN_TEST_GRAPHS_FOR_SUMMARY,
    return_models: bool = False,
    seed: Optional[int] = 42,
    verbose: bool = True,
):
    """
    兩階段 LOEO 訓練：全部資料打底 → 謠言子集微調。

    Stage 1（打底）：在「全部 thread（謠言+非謠言）」上訓練，排除掉
    這次 LOEO 的 test event（兩階段都要排除，不然 test event 的資訊
    會在打底階段就滲進模型裡，等於變相破壞 LOEO 的事件隔離）。目的
    不是把這個階段的表現調到最好，是讓 GNN 的卷積層先學會「怎麼讀懂
    一棵傳播樹」這個底層能力——這個能力理論上不太受是不是謠言影響，
    用更多資料（6425 vs 2402）練這個底層能力，比在小樣本上從頭學更穩。
    patience 故意設得比微調階段小（預設 5），這階段只是暖身，不用
    練到完全收斂。

    Stage 2（微調）：把打底完的權重當起點（不是重新初始化模型），
    接著在「只有謠言」的訓練資料上，用跟 run_loeo() 一樣的方式訓練
    到 early stopping，讓輸出層真正針對預算限制排序這個任務校準。

    兩階段用不同的 scaler，各自對應自己的訓練母體（打底用全部資料的
    統計量，微調用謠言子集的統計量）——不然全部資料的統計量會混進
    微調階段，又是另一種母體不一致的問題，跟之前抓到的 scaler 問題
    是同一個道理。

    rumour_graphs / full_graphs：事件集合必須一致（同一份 PHEME，
    一份只留 is_rumour==1，一份保留全部），不然 LOEO 的 fold 會對不
    起來。

    回傳格式跟 run_loeo() 一致：return_models=False 回傳 results；
    return_models=True 回傳 (results, models)。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"使用裝置：{device}" + (f"，固定隨機種子={seed}" if seed is not None else "，隨機種子未固定"))
        print(f"打底資料：{len(full_graphs)} 張圖（全部 thread）"
              f"，微調資料：{len(rumour_graphs)} 張圖（只有謠言）")

    events = sorted(set(g.event_id for g in rumour_graphs))
    results = {}
    models = {} if return_models else None
    run_start_time = time.time()

    from sklearn.model_selection import train_test_split

    for fold_i, test_event in enumerate(events, 1):
        if verbose:
            print(f"\n{'='*55}")
            print(f"LOEO fold {fold_i}/{len(events)}：test_event = {test_event}"
                  f"（目前累計耗時 {(time.time() - run_start_time)/60:.1f} 分鐘）")

        # ---- Stage 1：打底，全部資料，排除 test_event ----
        pretrain_train = [g for g in full_graphs if g.event_id != test_event]
        pretrain_train_g, pretrain_val_g = train_test_split(
            pretrain_train, test_size=0.1, random_state=42
        )
        pretrain_scaler = StandardScaler()
        pretrain_train_g = apply_scaler_to_graphs(pretrain_train_g, pretrain_scaler, fit=True)
        pretrain_val_g   = apply_scaler_to_graphs(pretrain_val_g,   pretrain_scaler, fit=False)

        model = GraphSAGEDual(
            in_channels=IN_CHANNELS, hidden=hidden, num_layers=num_layers,
            dropout=dropout, alpha=alpha, rank_weight=rank_weight, rank_margin=rank_margin,
        ).to(device)

        model = _run_training_stage(
            model, pretrain_train_g, pretrain_val_g, alpha, lr,
            pretrain_epochs, pretrain_patience, batch_size, device,
            stage_label="Stage1-打底", verbose=verbose,
        )

        # ---- Stage 2：微調，只有謠言，排除 test_event，沿用打底
        #      完的權重當起點，不重新初始化 ----
        finetune_train = [g for g in rumour_graphs if g.event_id != test_event]
        test_graphs    = [g for g in rumour_graphs if g.event_id == test_event]
        finetune_train_g, finetune_val_g = train_test_split(
            finetune_train, test_size=0.1, random_state=42
        )
        finetune_scaler = StandardScaler()
        finetune_train_g = apply_scaler_to_graphs(finetune_train_g, finetune_scaler, fit=True)
        finetune_val_g   = apply_scaler_to_graphs(finetune_val_g,   finetune_scaler, fit=False)
        test_graphs_scaled = apply_scaler_to_graphs(test_graphs, finetune_scaler, fit=False)

        if verbose:
            print(f"  微調：train={len(finetune_train_g)}  "
                  f"val={len(finetune_val_g)}  test={len(test_graphs_scaled)}")

        model = _run_training_stage(
            model, finetune_train_g, finetune_val_g, alpha, lr,
            finetune_epochs, finetune_patience, batch_size, device,
            stage_label="Stage2-微調", verbose=verbose,
        )

        # ---- 用微調階段的 train 資料算 baseline，跟 test 集比較 ----
        finetune_train_y_mean = float(np.mean([g.y.item() for g in finetune_train_g]))
        finetune_train_node_y_non_source = np.concatenate([
            g.node_y.numpy()[~g.is_source_mask.numpy()] for g in finetune_train_g
        ])
        finetune_train_node_y_mean = (
            float(finetune_train_node_y_non_source.mean())
            if len(finetune_train_node_y_non_source) > 0 else 0.0
        )
        test_loader = DataLoader(test_graphs_scaled, batch_size=batch_size, shuffle=False)
        baseline_test_mae_a = constant_baseline_mae(test_loader, finetune_train_y_mean, "y")
        baseline_test_mae_b = constant_baseline_mae(test_loader, finetune_train_node_y_mean, "node_y")

        if return_models:
            model.fitted_scaler_ = finetune_scaler
            models[test_event] = model

        metrics = evaluate(model, test_loader, device)
        metrics["mae_a_baseline"] = baseline_test_mae_a
        metrics["mae_b_baseline"] = baseline_test_mae_b
        metrics["beats_baseline_A"] = 1.0 if metrics["mae_a"] < baseline_test_mae_a else 0.0
        metrics["beats_baseline_B"] = 1.0 if metrics["node_mae"] < baseline_test_mae_b else 0.0
        metrics["n_test"] = len(test_graphs_scaled)
        results[test_event] = metrics
        if verbose:
            print(
                f"  最終 mae_a={metrics['mae_a']:.4f}（baseline={baseline_test_mae_a:.4f}，"
                f"{'贏' if metrics['beats_baseline_A'] else '沒贏'}）"
                f"  spear_a={metrics['spearman_a']:+.3f} | "
                f"node_mae={metrics['node_mae']:.4f}（baseline={baseline_test_mae_b:.4f}，"
                f"{'贏' if metrics['beats_baseline_B'] else '沒贏'}）"
                f"  spear_b={metrics['spearman_b']:+.3f}"
            )

    # 彙總（跟 run_loeo 同樣的格式，方便直接對照）
    if verbose:
        print(f"\n{'='*55}")
        print("LOEO（打底+微調）結果彙總：")
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