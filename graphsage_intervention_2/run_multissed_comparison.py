"""
多種子對照實驗：rank_weight=0（純 MSE）vs rank_weight=0.5（加排序損失）
================================================
用途：單次訓練的結果雜訊太大（同樣的超參數，光是隨機初始化不同，
就能讓 Head A 從贏過 observed_size_rank 變成輸給它）。這支腳本對
每一組設定跑多個不同的隨機種子，比較平均值 ± 標準差，才能分清楚
「這個差異是超參數真的有效果」還是「剛好抽到比較好/比較差的初始化」。

判斷標準：如果 rank_weight=0.5 的平均值，扣掉一個標準差之後還贏過
rank_weight=0 的平均值（加一個標準差），才算是站得住腳的訊號；
兩組的範圍如果重疊，代表現在還看不出真的有差異，需要更多種子或
更大的效果才能判斷。

用法：
    python run_multiseed_comparison.py

預設跑 rank_weight ∈ {0.0, 0.5}，各 3 個種子（42, 123, 2024），
共 6 輪完整的 LOEO 訓練 + 模擬。每輪耗時視你的機器而定，參考先前
單輪大約 5-10 分鐘，6 輪總共可能要 30-60 分鐘，建議放著跑，不用
盯著看。
"""

import os
import time
import numpy as np
import pandas as pd
import torch

from pheme_intervention_sim import run_intervention_simulation

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------

RANK_WEIGHTS = [0.0, 0.5]
SEEDS = [42, 123, 2024]
MIN_THREADS_WITH_FUTURE_FOR_SUMMARY = 30  # 跟 pheme_intervention_sim.py 預設一致

# 這幾個策略是判讀時真正要看的，其他 Oracle 策略只是輔助診斷，
# 這裡先不重複列出，跑完個別結果還是會印出完整內容
KEY_STRATEGIES = {
    "crr_head_a_rank__head_b_nodes": "A+B(現況)",
    "crr_observed_size_rank__head_b_nodes": "size+B",
    "crr_oracle_growth_rank__head_b_nodes": "OracleG+B",
}


def compute_excluded_mean(event_summary, strategy_key, min_threads):
    """比照 pheme_intervention_sim.py 的排除小事件邏輯，重算一次平均值。"""
    vals = [
        s.get(strategy_key, 0.0)
        for s in event_summary.values()
        if s.get("n_threads_with_future", 0) >= min_threads
    ]
    if not vals:
        return None
    return float(np.mean(vals))


def main():
    GRAPHS_30MIN = os.getenv("GRAPHS_30MIN", "pheme_graphs_roberta_30min_v2.pt")
    REPLY_CSV = os.getenv("REPLY_CSV", "pheme_reply_level_v3.csv")

    print("載入圖資料集...")
    graphs_30min = torch.load(GRAPHS_30MIN, weights_only=False)
    print(f"30min：{len(graphs_30min)} 張圖")

    print("載入 reply-level 表...")
    reply_df = pd.read_csv(
        REPLY_CSV,
        dtype={"thread_id": str, "tweet_id": str, "parent_id": str},
        low_memory=False,
    )
    print(f"reply_df：{len(reply_df)} 列\n")

    # results[rank_weight][strategy_key] = [各種子跑出來的平均值]
    results = {rw: {k: [] for k in KEY_STRATEGIES} for rw in RANK_WEIGHTS}

    run_start = time.time()
    total_runs = len(RANK_WEIGHTS) * len(SEEDS)
    run_i = 0

    for rw in RANK_WEIGHTS:
        for seed in SEEDS:
            run_i += 1
            print(f"\n{'#'*70}")
            print(f"# 第 {run_i}/{total_runs} 輪：rank_weight={rw}, seed={seed}"
                  f"（累計耗時 {(time.time()-run_start)/60:.1f} 分鐘）")
            print(f"{'#'*70}")

            sim_result = run_intervention_simulation(
                graphs_30min=graphs_30min,
                reply_df=reply_df,
                budget_threads=10,
                k_nodes=5,
                min_threads_with_future_for_summary=MIN_THREADS_WITH_FUTURE_FOR_SUMMARY,
                loeo_kwargs=dict(
                    hidden=256, num_layers=3, dropout=0.3, alpha=0.7,
                    rank_weight=rw, rank_margin=0.1,
                    lr=1e-3, epochs=50, batch_size=32,
                    min_test_graphs_for_summary=50,
                    seed=seed,
                ),
                device_str="auto",
                verbose=False,  # 這裡關掉逐 epoch 輸出，避免 6 輪洗版；
                                 # 想看細節可以先跑單輪 verbose=True 確認正常
            )

            event_summary = sim_result["event_summary"]
            for strategy_key in KEY_STRATEGIES:
                mean_val = compute_excluded_mean(
                    event_summary, strategy_key, MIN_THREADS_WITH_FUTURE_FOR_SUMMARY
                )
                results[rw][strategy_key].append(mean_val)
                print(f"  {KEY_STRATEGIES[strategy_key]:<12} "
                      f"（排除小事件後平均）= {mean_val:.4f}")

    # ------------------------------------------------------------------
    # 彙總：每組 rank_weight 的 mean ± std（跨種子）
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("多種子對照實驗彙總")
    print(f"{'='*70}")
    print(f"{'rank_weight':<15}" + "".join(f"{name:>18}" for name in KEY_STRATEGIES.values()))

    summary_stats = {}
    for rw in RANK_WEIGHTS:
        row = f"{rw:<15}"
        summary_stats[rw] = {}
        for strategy_key in KEY_STRATEGIES:
            vals = [v for v in results[rw][strategy_key] if v is not None]
            mean = np.mean(vals)
            std = np.std(vals)
            summary_stats[rw][strategy_key] = (mean, std)
            row += f"{mean:>10.4f}±{std:.4f}"
        print(row)

    print(f"\n各種子的原始數值（看看有沒有單一種子特別極端）：")
    for rw in RANK_WEIGHTS:
        for strategy_key in KEY_STRATEGIES:
            print(f"  rank_weight={rw}, {KEY_STRATEGIES[strategy_key]}: "
                  f"{[round(v,4) if v is not None else None for v in results[rw][strategy_key]]}")

    # 判讀：rank_weight=0.5 的 A+B 是否穩定贏過 rank_weight=0 的 A+B
    key = "crr_head_a_rank__head_b_nodes"
    mean0, std0 = summary_stats[0.0][key]
    mean5, std5 = summary_stats[0.5][key]
    print(f"\n{'='*70}")
    print("判讀：rank_weight=0.5 是否真的比 rank_weight=0 好（不是雜訊）？")
    print(f"{'='*70}")
    print(f"  rank_weight=0.0 的 A+B：{mean0:.4f} ± {std0:.4f}"
          f"（範圍約 [{mean0-std0:.4f}, {mean0+std0:.4f}]）")
    print(f"  rank_weight=0.5 的 A+B：{mean5:.4f} ± {std5:.4f}"
          f"（範圍約 [{mean5-std5:.4f}, {mean5+std5:.4f}]）")
    if mean5 - std5 > mean0 + std0:
        print("  >>> 兩組範圍不重疊，rank_weight=0.5 站得住腳地贏過 rank_weight=0")
    elif mean0 - std0 > mean5 + std5:
        print("  >>> 兩組範圍不重疊，rank_weight=0.5 站得住腳地輸給 rank_weight=0")
    else:
        print("  >>> 兩組範圍有重疊，現在的樣本（3 個種子）還不足以判斷有沒有"
              "真實差異，只能說『看不出穩定差異』，不是『證明沒有差異』——"
              "如果這個問題重要到值得深究，需要更多種子或更明顯的效果量")


if __name__ == "__main__":
    main()