# 有效模型、方法與正式結果

本文件只整合目前仍由系統、正式 benchmark 或可辯護結果需要的模型。探索性與失敗模型的
原因請見 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。不同 protocol 的分數分表呈現，不建立
混合排行榜。

## 1. 評估定義

### 1.1 固定 30 分鐘

對 event `e` 與 budget `K`：

```text
ImpactCapture_e@K =
  sum(PI_i(30), i in model Top-K) / sum(PI_j(30), j in event candidates)
```

主要結果是七個 eligible events 的等權 macro average。這衡量固定 30 分鐘 ranking 在同一
時間點可涵蓋多少 corrected preventable impact。

### 1.2 Bridge A

Static 在 30 分鐘一次介入；dynamic 在 10–60 分鐘分批介入。兩者每 event 總 Budget 50，
共同分母為 10 分鐘時仍存在的 future opportunity：

```text
UnifiedCapture_e@50 =
  sum(PI_i(T_i), i in selected threads) / sum(PI_j(10), j in candidates)
```

再對七個 events 等權 macro。這是完整政策效益比較，不能取代固定 30 分鐘模型本身的
ImpactCapture@K。

## 2. 統一固定 30 分鐘 benchmark

共同 protocol：

- 2,373 個 Graph7 與 corrected-v5 都具合法資料的 rumour threads；
- 九個 events 參與 training universe，七個 candidate 數至少 100 的 events 作主要 macro；
- cutoff 30 分鐘；target `log1p(corrected_preventable_impact)`；
- nested leave-one-event-out；所有 feature／model selection 僅使用 outer-training events；
- budgets `K={1,3,5,10,20,50,100}`；thread ID 作 deterministic tie-break。

| 模型 | @1 | @3 | @5 | @10 | @20 | @50 | @100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Expected | 0.38% | 1.13% | 1.89% | 3.78% | 7.56% | 18.90% | 37.80% |
| Current Early Size | 1.58% | 3.39% | 4.58% | 10.59% | 17.54% | 33.69% | 53.78% |
| Graph7 Author-aware RF | **1.79%** | 4.80% | 8.49% | 12.58% | 22.48% | 40.15% | 59.76% |
| Graph7 Nested Discourse RF | 1.37% | 4.65% | 7.69% | 12.62% | **23.51%** | 40.15% | 60.22% |
| v5 Nested Text RF | 1.09% | 6.12% | 8.86% | **14.48%** | 21.83% | 41.73% | 61.69% |
| Active／Quiet Two-expert RF | 1.08% | **6.38%** | **8.90%** | 12.56% | 22.98% | **42.17%** | **62.63%** |

沒有單一模型在所有 budget 全勝。K=10 適合 v5 Text；K=20 適合 Discourse；較大的 K=50／100
則 Active／Quiet 最佳。

正式來源：

- [unified_model_comparison.csv](benchmarks/pheme_fixed30_unified/reference_result/20260906_163045_109094_unified_fixed30_nested_full/unified_model_comparison.csv)
- [run_record.json](benchmarks/pheme_fixed30_unified/reference_result/20260906_163045_109094_unified_fixed30_nested_full/run_record.json)

## 3. 固定 30 分鐘有效模型

| 模型 | Cutoff-safe 特徵 | 訓練／選擇方法 | 統一 benchmark 定位 |
|---|---|---|---|
| Graph7 Author-aware RF | 早期活動量、時間、樹結構、作者過往行為與 profile summary，共 15 個 schema-approved features | RF 300 trees、depth 8、min leaf 3；event LOEO | K=1 最佳；@10 12.58% |
| Graph7 Nested Discourse RF | role base、strict-past event context、早期 discourse／reply-role bundles | Nested LOEO；inner validation 在三個 feature variants 間選擇 | K=20 最佳；@10 12.62% |
| v5 Nested Text RF | temporal／structure／author baseline，加 frozen source/reply RoBERTa PCA64 | PCA train-fold-only；inner validation 決定是否加入文本；RF 300 trees、depth 8、min leaf 3 | K=10 最佳；@10 14.48% |
| Active／Quiet Two-expert RF | v5 safe features、source/reply PCA、immutable account age、recency gate | outer-train 60th-percentile gate；兩個 RF experts；inner OOF positive-slope Ridge calibration | K=3/5/50/100 最佳；@50 42.17% |

### 歷史數字的角色

Graph7 `0.1192/0.1385`、v5 `0.1438/0.1467` 與 Active／Quiet `0.1461` 使用較早 protocol，
保留用來敘述研究演進，但不能與上表統一結果直接相減。

## 4. 動態有效模型

### 4.1 Balanced Multi-checkpoint Cumulative RF

- Checkpoints：10、20、30、40、50、60 分鐘。
- 特徵：57 個累積 activity、time、tree、latency 與簡單 source-text statistics。
- Target：各 checkpoint 的 `log1p(corrected dynamic preventable impact)`。
- 訓練：pooled elapsed-time-aware RF；outer event LOEO；所有 held-out event checkpoints 一起
  排除。
- 政策：總 Budget 50，固定 quotas `[9,9,8,8,8,8]`，已介入 thread 不重複選。
- 定位：正式穩健 dynamic baseline。

### 4.2 Hawkes-inspired Cumulative RF

- 保留上述 57 features，新增 5／10／20 分鐘固定 decay intensity、short/long ratio 與
  recent share，共 62 features。
- RF 與 split 和 baseline 相同；五個預先指定 seeds 做 paired stability evaluation。
- 五 seed 舊 2,402-thread protocol：baseline 平均 `0.3252`，Hawkes 平均 `0.3381`，每個
  seed delta 皆為正。
- 限制：event bootstrap CI 跨 0，部分 events 退步；只能稱 Hawkes-inspired candidate。

## 5. Bridge A：Static 與 Dynamic 的共同 Budget-50 結果

共同條件為 2,373 threads、corrected target、七事件 macro、總 Budget 50；dynamic models
已在共同 identities 上重新 fit。

| 模型 | 決策時間 | Mean interventions | Macro Unified Capture@50 | Pooled Capture@50 | Mean action minute |
|---|---|---:|---:|---:|---:|
| Random Expected | 固定 30 分鐘 | 50.0 | 12.97% | 9.14% | 30.0 |
| Current Early Size | 固定 30 分鐘 | 50.0 | 23.11% | 14.15% | 30.0 |
| Graph7 Author-aware RF | 固定 30 分鐘 | 50.0 | 27.12% | 21.63% | 30.0 |
| Graph7 Nested Discourse RF | 固定 30 分鐘 | 50.0 | 27.17% | 20.61% | 30.0 |
| v5 Nested Text RF | 固定 30 分鐘 | 50.0 | 28.24% | 21.27% | 30.0 |
| Active／Quiet Two-expert RF | 固定 30 分鐘 | 50.0 | 28.62% | 21.37% | 30.0 |
| Balanced Cumulative RF | 動態 10–60 分鐘 | 50.0 | 32.72% | 24.93% | 34.2 |
| Hawkes-inspired Cumulative RF | 動態 10–60 分鐘 | 50.0 | **33.73%** | **25.57%** | 34.2 |

正式來源：

- [unified_bridge_a_model_comparison.csv](benchmarks/pheme_static_dynamic_bridge/reference_result/20260907_154306_938376_bridge_a_full/unified_bridge_a_model_comparison.csv)
- [run_record.json](benchmarks/pheme_static_dynamic_bridge/reference_result/20260907_154306_938376_bridge_a_full/run_record.json)

解讀限制：dynamic policy 可在 10 分鐘開始介入，卻因固定 quota schedule 的後段選擇使平均
行動時間為 34.2 分鐘。Bridge A 的差距包含模型、資訊更新頻率與政策時間三者，不應只歸因
於 RF architecture。

## 6. Twitter15／16 有效輔助模型

這些模型使用固定 source-level train／validation／test split，不是 PHEME event LOEO，因此
獨立報告。

| 模型 | 特徵／架構 | 訓練方法 | Frozen test reduction@10 |
|---|---|---|---:|
| Twitter Graph-only RF | 15 個 30 分鐘 graph/activity/time features；排除文本、label 與 future fields | Validation 選 RF depth；train+validation refit；test 一次 | Twitter15 12.88%；Twitter16 25.98% |
| Twitter Weak-order Scalar MLP | 31 inputs → Linear(32) → ReLU → Dropout(0.05) → scalar | tail-weighted Smooth-L1 + weak within-train Oracle-head order loss；validation 選 order weight 0.05 | Twitter15 19.86%；Twitter16 32.92% |

Twitter result 只能證明輔助資料上的排序可行性；Twitter15/16 缺乏可靠 event identifier，不能
宣稱 event-separated generalization，也不能取代 PHEME benchmark。

## 7. Verification Agent 與政策層

Verification Agent 不是 future-impact ranking model，因此不放進前述排行榜。目前有效的工程
能力包括：

- cutoff-safe claim／reply payload；
- 搜尋與結構化 evidence state；
- cluster-level dedup、cache、TTL、缺漏重試與 `agent_failure`；
- confirmed-true exclusion；
- RF propagation risk × evidence risk 的 deterministic ordering；
- none／soft／medium／hard／monitor 建議與 DB audit trail。

真實 smoke 已證明 supported claim 能被安全排除，但尚未有 time-aligned、跨事件的真實 Agent
效果 leaderboard。Oracle veracity replay 與 current-web demo 都不能冒充正式早期查核結果。

## 8. 有效元件與非模型目錄

| 目錄 | 類型 | 作用 |
|---|---|---|
| `data_pipeline/pheme_corrected_target` | Target／data lineage | 建立 corrected all-roots preventable-impact mapping，不是模型 |
| `benchmarks/pheme_fixed30_unified` | Benchmark | 統一 2,373-thread 固定 30 分鐘比較 |
| `benchmarks/pheme_static_dynamic_bridge` | Benchmark | 統一 static–dynamic Budget-50 政策比較 |

## 9. 結果使用原則

- 論文固定模型主表使用第 2 節。
- 固定與動態共同比較使用第 5 節。
- 歷史模型演進可以引用舊數字，但必須標註 protocol 不同。
- `analysis/` 中的模型只能放在消融、診斷或負面結果，不得列為有效模型。
- 所有 capture／blocked nodes 都是離線反事實評估，不是平台實際造成的因果減量。
