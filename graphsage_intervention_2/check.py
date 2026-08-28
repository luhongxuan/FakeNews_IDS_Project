"""
檢查：observed_size（30分鐘內節點數）跟 future_growth_30min 的相關性
================================================
用途：驗證「observed_size_rank 這個土法煉鋼 baseline 為什麼這麼強」
的假設——如果 observed_size 本身就跟真正的未來成長量高度相關，
代表這是一個非常強的先驗訊號，Head A 的 global_mean_pool 架構可能
沒有把這個訊號有效保留下來（平均掉了），值得直接把它加成一個
額外特徵接在 Head A 的輸出前面。

用法：
    python check_observed_size_correlation.py --reply_level_csv pheme_reply_level_v3.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reply_level_csv", required=True)
    args = parser.parse_args()

    print(f"讀取 {args.reply_level_csv} ...")
    df = pd.read_csv(
        args.reply_level_csv,
        dtype={"thread_id": str, "tweet_id": str, "event_id": str},
        low_memory=False,
    )
    print(f"  {len(df)} 列")

    # 每個節點是否落在 30 分鐘窗口內（NaN offset 保守當作窗口內，
    # 跟 thread_to_data() 截斷邏輯一致）
    df["in_window"] = df["offset_sec"].isna() | (df["offset_sec"] <= 1800)

    # 每個 thread 的 observed_size（窗口內節點數）
    observed_size = df[df["in_window"]].groupby("thread_id").size()
    observed_size.name = "observed_size"

    # 每個 thread 的 future_growth_30min（同一個 thread 每列都一樣，取第一筆即可）
    thread_level = df.drop_duplicates(subset=["thread_id"])[
        ["thread_id", "event_id", "future_growth_30min"]
    ].set_index("thread_id")

    merged = thread_level.join(observed_size, how="left")
    merged["observed_size"] = merged["observed_size"].fillna(0).astype(int)

    print(f"\n共 {len(merged)} 個 thread")
    print(f"observed_size 統計: mean={merged['observed_size'].mean():.2f}, "
          f"median={merged['observed_size'].median():.0f}, "
          f"max={merged['observed_size'].max()}")
    print(f"future_growth_30min 統計: mean={merged['future_growth_30min'].mean():.2f}, "
          f"median={merged['future_growth_30min'].median():.0f}, "
          f"max={merged['future_growth_30min'].max()}")

    # 整體 Spearman 相關係數
    rho, p = spearmanr(merged["observed_size"], merged["future_growth_30min"])
    print(f"\n=== 整體 Spearman(observed_size, future_growth_30min) = {rho:.4f} "
          f"(p={p:.2e}) ===")

    if rho > 0.5:
        print(">>> 相關性很強：observed_size 本身就是一個很強的先驗訊號，")
        print("    這解釋了為什麼 observed_size_rank 這個土法煉鋼 baseline 這麼難打贏。")
    elif rho > 0.2:
        print(">>> 中等相關：observed_size 有一定預測力，但不是壓倒性的。")
    else:
        print(">>> 相關性弱：observed_size 本身預測力不強，")
        print("    observed_size_rank 表現好可能有其他原因，不是這個假設。")

    # 分事件看，確認不是被少數大事件拉高/拉低整體數字
    print("\n各事件分開看：")
    for event_id, grp in merged.groupby("event_id"):
        if len(grp) < 10:
            continue
        rho_ev, _ = spearmanr(grp["observed_size"], grp["future_growth_30min"])
        print(f"  {event_id:<22} n={len(grp):5d}  spearman={rho_ev:+.4f}")


if __name__ == "__main__":
    main()