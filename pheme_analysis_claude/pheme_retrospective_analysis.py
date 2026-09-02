"""
PHEME Retrospective Feature Analysis Pipeline
===============================================
研究問題：在完整 thread 結束後，rumour 與 non-rumour 是否在結構、
時間動態、情感特徵上存在可觀測差異？

本腳本執行：
1. 描述統計 (median, IQR, mean, std) — 依 rumour label 分組
2. Mann-Whitney U test (兩組獨立樣本，無母數)
3. Benjamini-Hochberg FDR 校正 (控制多重比較的偽發現率)
4. Cliff's delta 效果量 (非母數，對離群值穩健)
5. Boxplot / Violin plot / ECDF plot (長尾特徵先 log1p 只用於視覺化)
6. 綜合排序：哪些特徵最能區分 rumour / non-rumour

重要原則：
- 統計檢定 (Mann-Whitney) 使用原始尺度，因為它基於排序，對單調轉換具不變性。
- log1p 僅用於畫圖，不用於檢定，避免混淆「視覺呈現」與「統計推論」。
- 判斷特徵是否重要，以效果量 (Cliff's delta) 為主，p-value 只用來說明
  「差異是否穩定到不像雜訊」，不能單獨用來說明「差異有多大」。
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------------
# 0. 設定
# ------------------------------------------------------------------

STRUCTURE_FEATURES = [
    "max_depth", "total_nodes", "max_width", "leaf_ratio",
    "avg_branching", "linearity", "width_depth_ratio",
]
TEMPORAL_FEATURES = [
    "first_reply_sec", "early_30min_count", "early_60min_count",
    "total_duration_hr", "burst_ratio",
]
SENTIMENT_FEATURES = [
    "sent_mean", "sent_std", "sent_min",
    "neg_ratio", "avg_negative", "avg_positive",
]
ALL_FEATURES = STRUCTURE_FEATURES + TEMPORAL_FEATURES + SENTIMENT_FEATURES

# 长尾特徵：畫圖時建議先 log1p 再呈現 (只影響視覺化，不影響統計檢定)
LONG_TAIL_FEATURES = [
    "total_nodes", "max_width", "total_duration_hr",
    "first_reply_sec", "early_30min_count", "early_60min_count",
    "burst_ratio",
]

LABEL_COL = "is_rumour"  # 1 = rumour, 0 = non-rumour，請依你的資料欄位名稱調整

OUTPUT_DIR = Path("pheme_analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# 1. Cliff's Delta 實作
# ------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    計算 Cliff's delta 效果量。
    delta = P(X > Y) - P(X < Y)，範圍 [-1, 1]。
    正值代表 x 組傾向大於 y 組，負值相反，0 代表無差異。

    使用 O(n log n) 的排序法計算，而非暴力 O(n*m)，
    避免大樣本 (PHEME thread 數量多) 時運算過慢。
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan

    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)
    rank_x = ranks[:nx]

    # 利用 rank-biserial 與 Mann-Whitney U 的關係反推 Cliff's delta
    # delta = 2*U/(nx*ny) - 1，其中 U 為 x 相對於 y 的 Mann-Whitney U 統計量
    sum_rank_x = np.sum(rank_x)
    u_x = sum_rank_x - nx * (nx + 1) / 2
    delta = 2 * u_x / (nx * ny) - 1
    return delta


def interpret_cliffs_delta(delta: float) -> str:
    """Romano et al. (2006) 的慣例判讀門檻。"""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


# ------------------------------------------------------------------
# 2. 描述統計
# ------------------------------------------------------------------

def compute_descriptive_stats(df: pd.DataFrame, features: list, label_col: str) -> pd.DataFrame:
    """對每個特徵，依 rumour/non-rumour 分組計算 median, IQR, mean, std。"""
    rows = []
    for feat in features:
        for label, name in [(1, "rumour"), (0, "non_rumour")]:
            vals = df.loc[df[label_col] == label, feat].dropna()
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            rows.append({
                "feature": feat,
                "group": name,
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(),
                "median": median,
                "IQR": q3 - q1,
                "q1": q1,
                "q3": q3,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 3. 假設檢定 + 效果量 + FDR 校正
# ------------------------------------------------------------------

def run_hypothesis_tests(df: pd.DataFrame, features: list, label_col: str) -> pd.DataFrame:
    """
    對每個特徵執行 Mann-Whitney U test，並計算 Cliff's delta。
    最後統一做 Benjamini-Hochberg FDR 校正。
    """
    results = []
    for feat in features:
        rumour_vals = df.loc[df[label_col] == 1, feat].dropna().values
        nonrumour_vals = df.loc[df[label_col] == 0, feat].dropna().values

        # Mann-Whitney U test（雙尾，無母數）
        u_stat, p_val = stats.mannwhitneyu(
            rumour_vals, nonrumour_vals, alternative="two-sided"
        )

        delta = cliffs_delta(rumour_vals, nonrumour_vals)

        results.append({
            "feature": feat,
            "n_rumour": len(rumour_vals),
            "n_nonrumour": len(nonrumour_vals),
            "U_statistic": u_stat,
            "p_value_raw": p_val,
            "cliffs_delta": delta,
            "effect_size_interpretation": interpret_cliffs_delta(delta),
        })

    result_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR 校正（一次對所有特徵的 p-value 做校正）
    reject, p_adj, _, _ = multipletests(
        result_df["p_value_raw"], alpha=0.05, method="fdr_bh"
    )
    result_df["p_value_fdr"] = p_adj
    result_df["significant_after_fdr"] = reject

    # 綜合判斷：統計顯著 且 效果量至少 small 才算「有實質意義的差異」
    result_df["practically_meaningful"] = (
        result_df["significant_after_fdr"]
        & (result_df["cliffs_delta"].abs() >= 0.147)
    )

    return result_df.sort_values("cliffs_delta", key=lambda s: s.abs(), ascending=False)


# ------------------------------------------------------------------
# 4. 視覺化
# ------------------------------------------------------------------

def plot_feature_comparison(df: pd.DataFrame, feature: str, label_col: str,
                             use_log: bool, output_dir: Path):
    """
    對單一特徵畫出 boxplot、violin plot、ECDF 三合一圖。
    use_log=True 時，只在畫圖時做 log1p 轉換，圖標題會註明。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    plot_df = df[[feature, label_col]].dropna().copy()
    plot_df["group"] = plot_df[label_col].map({1: "rumour", 0: "non_rumour"})

    if use_log:
        plot_df["plot_value"] = np.log1p(plot_df[feature])
        value_label = f"log1p({feature})"
    else:
        plot_df["plot_value"] = plot_df[feature]
        value_label = feature

    # Boxplot
    sns.boxplot(data=plot_df, x="group", y="plot_value", ax=axes[0],
                hue="group", palette="Set2", legend=False)
    axes[0].set_title(f"Boxplot: {value_label}")
    axes[0].set_ylabel(value_label)

    # Violin plot
    sns.violinplot(data=plot_df, x="group", y="plot_value", ax=axes[1],
                    hue="group", palette="Set2", legend=False, inner="quartile")
    axes[1].set_title(f"Violin: {value_label}")
    axes[1].set_ylabel(value_label)

    # ECDF
    for group, color in [("rumour", "#e74c3c"), ("non_rumour", "#3498db")]:
        vals = np.sort(plot_df.loc[plot_df["group"] == group, "plot_value"])
        y = np.arange(1, len(vals) + 1) / len(vals)
        axes[2].plot(vals, y, label=group, color=color)
    axes[2].set_title(f"ECDF: {value_label}")
    axes[2].set_xlabel(value_label)
    axes[2].set_ylabel("cumulative probability")
    axes[2].legend()

    plt.suptitle(f"Feature: {feature}" + (" (log1p scale for visualization only)" if use_log else ""))
    plt.tight_layout()
    fig.savefig(output_dir / f"{feature}_comparison.png", dpi=150)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, features: list, label_col: str, output_dir: Path):
    for feat in features:
        use_log = feat in LONG_TAIL_FEATURES
        plot_feature_comparison(df, feat, label_col, use_log, output_dir)
    print(f"圖表已輸出至：{output_dir.resolve()}")


def plot_effect_size_summary(result_df: pd.DataFrame, output_dir: Path):
    """畫出所有特徵的 Cliff's delta 排序條狀圖，一眼看出哪些特徵區辨力較強。"""
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(result_df))))
    plot_data = result_df.sort_values("cliffs_delta")
    colors = ["#2ecc71" if m else "#95a5a6" for m in plot_data["practically_meaningful"]]
    ax.barh(plot_data["feature"], plot_data["cliffs_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.147, color="gray", linestyle="--", linewidth=0.8, label="small effect threshold")
    ax.axvline(-0.147, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Cliff's delta (rumour vs non-rumour)")
    ax.set_title("Effect size summary\n(green = significant after FDR AND effect size ≥ small)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "effect_size_summary.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------------

def run_full_analysis(df: pd.DataFrame, label_col: str = LABEL_COL):
    """
    輸入：df 需包含 ALL_FEATURES 欄位 + label_col (1=rumour, 0=non-rumour)
    輸出：描述統計表、檢定結果表、圖表，並印出最終排序摘要
    """
    print("=" * 60)
    print("Step 1: 描述統計")
    print("=" * 60)
    desc_stats = compute_descriptive_stats(df, ALL_FEATURES, label_col)
    desc_stats.to_csv(OUTPUT_DIR / "descriptive_stats.csv", index=False)
    print(desc_stats.round(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("Step 2-4: Mann-Whitney U + Cliff's delta + BH-FDR 校正")
    print("=" * 60)
    test_results = run_hypothesis_tests(df, ALL_FEATURES, label_col)
    test_results.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)
    print(test_results.round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("Step 6: 視覺化")
    print("=" * 60)
    generate_all_plots(df, ALL_FEATURES, label_col, OUTPUT_DIR)
    plot_effect_size_summary(test_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Step 8: 綜合排序摘要（依效果量絕對值排序）")
    print("=" * 60)
    meaningful = test_results[test_results["practically_meaningful"]]
    if len(meaningful) == 0:
        print("⚠️ 沒有特徵同時滿足「FDR校正後顯著」且「效果量 ≥ small」。")
        print("   這代表 rumour / non-rumour 在這組特徵上即使有統計顯著差異，")
        print("   實質區辨力也非常有限，不建議直接宣稱可用於分類。")
    else:
        print(meaningful[["feature", "cliffs_delta", "effect_size_interpretation",
                           "p_value_fdr"]].round(4).to_string(index=False))

    return desc_stats, test_results


# ------------------------------------------------------------------
# 使用範例（請將 df 換成你實際讀取 PHEME 特徵後的 DataFrame）
# ------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(".csv")
    desc_stats, test_results = run_full_analysis(df, label_col="is_rumour")
    print("請將此腳本 import 到你的 notebook，並傳入包含以下欄位的 DataFrame：")
    print(ALL_FEATURES + [LABEL_COL])