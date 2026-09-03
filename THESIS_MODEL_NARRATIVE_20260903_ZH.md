# 專題論文模型演進建議稿

## 1. 建議的論文核心主張

本研究的核心不是「嘗試很多模型後挑最高分」，而是逐步回答以下問題：

1. 早期可觀察資訊是否比隨機或只看早期規模更能找出值得介入的 thread？
2. 文本與回覆脈絡能否補足純結構特徵？
3. 活躍與安靜 thread 是否需要不同專家？
4. 安靜 thread 的後期爆發，能否由 30 分鐘內資訊穩定辨識？
5. 若單一 30 分鐘決策有資訊上限，改成多時間點介入是否更有效？
6. 模型能否在維持效果時減少介入數量，或判斷等待是否值得？

最後可以形成一個清楚結論：早期結構、時間與文本訊號確實能支援跨事件介入；active/quiet 分流能改善中等預算，但 quiet burst 在固定 30 分鐘仍高度事件相依。相比繼續增加靜態特徵，使用多 checkpoint 累積觀察與重複決策，帶來更穩定的政策增益。

## 2. 主文建議保留的模型演進

### Step 0：Random 與 Early-size baseline

最先建立兩個容易理解的參考：隨機挑選，以及依 30 分鐘內已觀察到的 thread size 排序。它們回答的是「複雜模型是否真的學到早期傳播風險，而不只是挑目前最大的 thread」。

在 Graph7 author-aware 實驗中：

| 方法 | CRR@10 |
|---|---:|
| Random | 0.0390 |
| Early size | 0.1059 |
| Author-aware RF | **0.1192** |

這是論文第一個正面結果：合法早期特徵比單看規模更有用。

### Step 1：Graph7 Author-aware RF

使用 15 個預先指定的 30 分鐘特徵，包括活動量、使用者集中度、三個時間區間、recency、早期樹形、可觀察使用者歷史與 follower 摘要。

這個模型適合作為第一個正式模型，因為結構簡單、容易解釋，而且直接勝過 early-size baseline。它的功能不是成為最高分模型，而是證明 early intervention prediction 本身可行。

### Step 2：Graph7 Nested Discourse/Context RF

下一步加入 role、context、discourse 等特徵 bundle，並以 nested LOEO 在 inner events 選擇 representation；outer held-out event 不參與選模。

| 方法 | CRR@10 |
|---|---:|
| Early size | 0.1033 |
| Nested discourse/context RF | **0.1385** |

這一步建立兩件事：較完整的早期脈絡比單純規模更有效，而且改善能在 schema-locked、event-separated protocol 下保留。它是論文中最適合作為「方法正確性確認」的模型。

### Step 3：v5 Text RF

在安全 temporal/tree baseline 上加入 train-fold PCA64 文本表示，並由 inner LOEO 決定每個 outer fold 使用 baseline 或 text-augmented bundle。

主要結果為 legacy artifact lineage 下的 `reduction@10 = 0.1467`。達到 20% reduction 平均需要 22.57 次介入，而 Oracle 只需要 6.71 次，顯示模型具有可操作訊號，但與理想政策仍有明顯差距。

這可作為最高的 legacy static 30-minute reference；但不可和 Graph7 或 corrected-target 數字混成單一 leaderboard。

### Step 4：Calibrated Active/Quiet Two-expert RF

觀察到活躍與安靜 thread 的資料分布不同後，以 outer-train event 的 recency 60th percentile 建立不使用 outcome 的 gate，分別訓練 active RF 與 quiet RF，再使用 inner OOF ridge calibration 對齊兩個 expert 的分數尺度。

在 legacy protected-v5 protocol 下：

| 模型 | reduction@10 |
|---|---:|
| Paired global RF | 0.1319 |
| Calibrated two experts | **0.1461** |
| Legacy v5 reference | 0.1467 |

它沒有超過 v5，但改善 K=5--50 的中等預算表現。這是合理的研究推進：全域模型可處理多數情況，但傳播狀態異質性值得用雙專家建模。

### Step 5：Corrected-target Two-expert 與 Quiet-tier Hybrid

後續 audit 發現原 target 對所有可介入根節點的計算需要修正，因此保留 candidate、split、cutoff 與模型流程，只版本化修正 outcome。這一步應寫成研究有效性修正，而不是宣稱模型進步。

在相同 corrected target 下：

| 模型 | reduction@10 | reduction@50 | Oracle efficiency@50 |
|---|---:|---:|---:|
| Corrected v5 | 0.1368 | — | — |
| Corrected active/quiet | **0.1412** | 0.4299 | 0.6123 |
| Corrected quiet-tier hybrid | **0.1412** | **0.4334** | **0.6164** |

Quiet-tier hybrid 在 Top-50 只比 corrected active/quiet 增加 0.0035 reduction，兩者都命中 154/350 個 Oracle Top-50；hybrid 的 quiet 命中反而由 15 降為 13。這表示 coarse tier information 只產生很小的 utility 變化，沒有解決 quiet recall。

### Step 6：Multi-checkpoint Cumulative RF

固定 30 分鐘的 quiet 訊號遇到上限後，研究問題從「再增加哪些靜態特徵」轉為「能否隨時間更新決策」。模型在 10、20、30、40、50、60 分鐘取得累積可觀察資訊，並使用 event-held-out OOF RF 預測當下介入 utility。

在共同 minute-10 denominator、總 Budget 50 下：

| 政策 | Mean horizon reduction | Blocked future nodes | Mean action minute |
|---|---:|---:|---:|
| Single-30 corrected v5 | 0.2748 | 3,790 | 30.0 |
| Single-30 quiet-tier hybrid | 0.2966 | 3,995 | 30.0 |
| Window/Delta RF | 0.3129 | 4,570 | 34.2 |
| **Balanced cumulative RF** | **0.3261** | **4,527** | 34.2 |
| Dynamic Oracle | 0.4928 | 7,356 | 34.2 |

Balanced cumulative 相對 single-30 hybrid 增加 0.0295 absolute mean horizon reduction，阻擋多 532 個 future nodes；七個 events 中六個改善、一個持平。這是目前最適合成為論文最終主要模型的結果。

### Step 7：Adaptive Cumulative Policy

固定 Budget 50 雖然有效，但真實系統不一定每次都需要用滿名額。因此使用 inner-event validation 選擇 utility threshold 或 action cost，在多 checkpoint cumulative score 上決定是否介入。

| Policy | Mean interventions | Mean horizon reduction | Blocked nodes |
|---|---:|---:|---:|
| Target 20 | 34.29 | 0.2225 | 3,123 |
| Target 25 | 38.57 | 0.2441 | 3,558 |
| Target 30 | 41.43 | 0.2558 | 3,738 |
| Cost 0.2 | 48.57 | 0.3333 | 4,025 |
| Fixed cumulative 50 | 50.00 | 0.3261 | 4,527 |

這一節不應宣稱 adaptive 全面超越 fixed policy，而應呈現「介入數量與阻擋效果的 Pareto trade-off」。若專題重視最少介入，Target 25/30 是具體可展示的操作點；若重視阻擋總節點，fixed cumulative 50 仍較穩定。

## 3. 適合放在消融或負面結果章節的模型

### 3.1 Oracle-order Scalar MLP

Huber utility loss 加 pairwise Oracle-order loss，在 30 分鐘得到 0.1245，低於 v5 RF 0.1467。它證明換成 MLP 並加入排序 loss，不會自然解決 intervention ranking。

### 3.2 Quiet Semantic、Pairwise 與 Exhaustive Feature Groups

可合併成一張消融表，不需要每個模型各寫一節：

- Direct semantic Ridge 沒有勝過 ordinary quiet RF。
- Semantic residual 只在 K=10 局部改善到 0.1483，其他 budgets 下降。
- Quiet pairwise ranker 改善 K=1，卻破壞 K=3--50。
- 255 種 feature-group combinations 改善 quiet K=1/3，但 K=5--100 與 Spearman 下降。

共同結論是：quiet thread 並非完全沒有早期訊號，但訊號弱、事件相依，而且局部 head improvement 無法形成穩定完整排序。

### 3.3 Window/Delta RF

Window/Delta 適合作為 cumulative RF 的對照。它證明短期變化有訊號，但只看當前十分鐘及 delta 會遺失累積結構；結果 0.3129 低於 cumulative 0.3261。

### 3.4 Wait-loss Hazard RF

Hazard RF 對「再等十分鐘會損失多少」具有可學習性，例如 positive ROC-AUC 約 0.746--0.754；但這個預測訊號沒有轉化成更好的 controller。它應被描述為「可預測性不等於決策價值」的負面結果。

### 3.5 Quiet Acceleration RF

此模型直接預測 quiet thread 下一個十分鐘的 acceleration。整體 ROC-AUC 0.555、Spearman 0.066；20 分鐘 checkpoint 較好（ROC-AUC 0.613），但之後接近隨機。它支持「爆發起點難以由既有特徵跨事件穩定辨識」。

### 3.6 Oracle-membership Qualification 與 Quiet Rescue Expert

Case-control 診斷顯示 relational/interactions 能比隨機多辨識部分漏掉的 quiet Oracle members，但 nested intervention evaluation 證明 membership 不等於 high impact。

| 額外 quiet 名額 | Rescue impact | 原 quiet score impact | 差距 |
|---:|---:|---:|---:|
| 5 | 83 | 165 | -82 |
| 10 | 227 | 602 | -375 |
| 15 | 332 | 720 | -388 |
| 20 | 414 | 794 | -380 |

這是很有價值的負面結果：即使能辨識一些 Oracle Top-50 membership，也可能只找到邊緣 members，無法達成「阻擋最多 future nodes」的真正目標。

## 4. 論文主表應如何分開

不可把所有數字放在同一個 leaderboard。建議至少分成三張表：

1. **Legacy static 30-minute**：early-size、Graph7、v5、legacy active/quiet。
2. **Corrected-target static 30-minute**：corrected v5、corrected active/quiet、quiet-tier hybrid。
3. **Corrected dynamic horizon**：single-30 replay、Window/Delta、cumulative、adaptive、Hazard。

不同 artifact lineage 或 denominator 的數字只能描述研究演進，不可直接宣稱前者或後者「模型架構更好」。

## 5. 建議的主文篇幅配置

主文詳細介紹五個核心階段即可：

1. Early-size/Random baseline。
2. Graph7 nested RF。
3. v5 text RF。
4. Active/quiet two-expert RF。
5. Multi-checkpoint cumulative RF，加上 adaptive policy trade-off。

Quiet tier、semantic pairwise、Hazard、acceleration、rescue expert 統一放在「針對 quiet burst 的後續研究與負面結果」。這樣既能呈現研究深度，也不會讓讀者誤以為每個模型都是最終候選。

## 6. 可直接使用的研究故事摘要

本研究首先以隨機與早期規模排序建立基準，證明早期時間、結構與使用者特徵可以提升介入效益。接著透過 event-separated nested RF 加入 discourse/context 與文本表示，使固定 30 分鐘模型取得更高的 future-impact reduction。分析錯誤案例後發現，活躍與安靜 thread 的傳播狀態不同，因此建立經 train-only calibration 的 active/quiet 雙專家模型，改善中等介入預算下的表現。然而，多種 semantic、tier、pairwise、acceleration 與 rescue 方法均顯示，quiet burst 的早期訊號弱且高度事件相依，局部分類能力也不等於能阻擋較多 future nodes。研究因此由單點排序轉向多 checkpoint 決策；累積 RF 在固定總預算下穩定改善六個事件，成為目前最具實用性的主要政策。最後，adaptive 與 Hazard 實驗進一步揭示介入數量、介入時間與阻擋效益之間的取捨。

## 7. 主要結果來源

- `PROJECT_RESULTS_20260901.md`
- `effective_models/pheme_v5_text_rf/reference_result/20260831_223827_909201_v5_best_text_oof_min_interventions/`
- `effective_models/pheme_graph7_discourse_rf/reference_result/20260830_212959_019175_nested_discourse_context_selection/`
- `effective_models/pheme_active_quiet_calibrated_rf/reference_result/20260901_210513_260899_paired_oof_pheme_quiet_expert_calibration/`
- `effective_models/pheme_v5_corrected_target/experiments/20260903_143643_168807_corrected_v5_nested_full/`
- `effective_models/pheme_active_quiet_corrected_target/experiments/20260903_214222_297421_corrected_quiet_tier_hybrid_full/`
- `effective_models/pheme_multicheckpoint_rf/reference_result/20260903_160025_balanced_cumulative_sequential_policy/`
- `effective_models/pheme_multicheckpoint_rf/experiments/20260903_162347_617042_adaptive_cumulative_policy_full/`
- `effective_models/pheme_multicheckpoint_rf/experiments/20260903_172123_236311_wait_loss_hazard_policy_full/`
- `effective_models/pheme_multicheckpoint_rf/experiments/20260903_194947_531733_quiet_acceleration_rf_full/`
- `effective_models/pheme_active_quiet_corrected_target/experiments/20260903_221403_155217_quiet_rescue_expert_full/`
