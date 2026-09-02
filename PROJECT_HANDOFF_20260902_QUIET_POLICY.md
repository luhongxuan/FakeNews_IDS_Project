# Quiet-thread Oracle-policy handoff — 2026-09-02

## 給下一個聊天的最短摘要

目前研究的是：在 PHEME rumour-only thread 中，只看 source 後 30 分鐘的
合法資訊，預測每個 thread 在此刻完整介入可阻擋的未來節點數，並依預測
分數決定介入優先順序。完整系統目前最強參考仍是 v5 text RF
`reduction@10 = 0.1467`；train-only calibrated active/quiet two-expert RF 是
`0.1461`，在中等 Budget 5--50 較有優勢，但 Top-1/3 較差。

最新完成的是 **quiet pool 專用的 exhaustive nested feature-combination
RF**。它搜尋八個合法特徵群組的全部 255 種非空組合。結果只在 quiet
Budget 1、3 改善；從 Budget 5 到 100 均不如既有 quiet RF，完整排序的
Spearman 也下降。因此它不是新 baseline，也尚未整合回完整 active/quiet
policy。它較合理的用途可能是提供「前端候選訊號」，而不是直接取代
quiet RF。

## 1. 任務與資料

- Dataset：protected PHEME v5 rumour-only snapshot artifact。
- Threads：2,402，9 個 events；正式 macro 報告使用 7 個 thread 數至少
  100 的 eligible events。
- Observation cutoff：strict 30 minutes（1,800 秒）。
- Candidate：只包含 rumour threads，不包含 non-rumour。
- Intervention：選中完整 thread，效益為 cutoff 後可阻擋的未來節點。
- Training target：`preventable_y = log1p(preventable_impact)`。
- `preventable_impact` 是訓練 outcome／事後 Oracle utility，不是模型輸入。
- Split：outer LOEO；每次完整保留一個 event 作 outer test。

保護後的必要輸入已移到：

`data/protected_research_assets/pheme_v5_strict30/`

舊 run record 中的 `graphsage_intervention_5/...` 是歷史執行路徑，不是目前
應使用的路徑。

## 2. 目前完整 policy 的來源

### v5 text RF

- 單一 global RF 預測每個 rumour thread 的 `log1p(preventable_impact)`。
- safe temporal/depth features，加上 train-fold source/reply RoBERTa PCA64。
- Nested LOEO 選 feature variant。
- Eligible-event macro `reduction@10 = 0.1467`。
- 保留程式：`effective_models/pheme_v5_text_rf/`。

### calibrated active/quiet two-expert RF

- Gate feature：`log1p_seconds_since_last_activity`。
- 每個 train fold 只用該 fold 計算第 60 percentile；recency 較高者為 quiet。
- Active 與 quiet 各自訓練一個 RF utility regressor。
- 兩個 expert 的分數以 inner event-OOF prediction 學 positive-slope Ridge
  calibration，再合併排序。
- Full-policy `reduction@10 = 0.1461`；global paired comparator 是 0.1319。
- 中等 Budget 改善，但仍未穩定超過原始 v5 0.1467。
- 保留程式：`effective_models/pheme_active_quiet_calibrated_rf/`。

## 3. 最新 exhaustive quiet feature-combination RF

### 它實際回答的問題

它只取每個 outer held-out event 中被 train-fold 60% recency gate 判定為
quiet 的 threads，在 quiet pool 內比較：

1. `current_quiet_rf` 的排序；
2. inner LOEO 從 255 組特徵組合中選出的 RF 排序。

所以這份數字是 **quiet-only Oracle efficiency**，不是 full-policy
reduction，不能直接和 0.1467／0.1461 混比。

### 八個特徵群組

1. `activity_level`：observed size、10/5-minute bins、early/middle/late
   fractions、active/peak minute counts。
2. `temporal_dynamics`：first/last reply、activity entropy/Gini/trend/
   acceleration、offset 與 interarrival quantiles。
3. `topology`：root children、leaf/internal fractions、degree/depth/
   outdegree distributions、entropy/Gini/HHI、width-to-depth。
4. `text_surface`：source/reply/snapshot length、vocabulary、entropy、URL、
   hashtag、mention、punctuation、digit/uppercase 等表面統計。
5. `sentiment`：observable text 的 VADER compound/positive/negative/neutral
   summaries。
6. `account_age`：source/reply/snapshot 的 immutable account-age summaries。
7. `semantic_summary`：embedding norm、source-reply cosine、centroid distance
   與 compact semantic components。
8. `source_reply_embedding`：source 加 observed replies embedding；scaler 與
   PCA32 僅 fit 當前 train fold。

沒有使用 retweet/favorite counts、followers/friends/statuses、verified 或
其他可變 profile 欄位。

### 模型與選擇流程

- Model：`RandomForestRegressor`。
- `n_estimators=300`
- `max_depth=8`
- `min_samples_leaf=3`
- `max_features=0.8`
- `random_state=42`
- 255 組組合全部在執行前固定。
- 每個 outer fold 內，再用 inner event LOEO 評估每個組合。
- Inner criterion：quiet Oracle efficiency 在 K={1,3,5,10,20,50} 的平均。
- tie-break：較少 group 優先，再按組合名稱穩定排序。
- 選定組合後，用全部 outer-train quiet rows 重訓，outer held-out event
  只評估一次。

## 4. 最新結果

七個 eligible events 的 mean quiet Oracle efficiency：

| Quiet Budget | Current quiet RF | Selected feature combination | Difference |
|---:|---:|---:|---:|
| 1 | 0.0871 | **0.2582** | +0.1711 |
| 3 | 0.1919 | **0.2731** | +0.0812 |
| 5 | **0.2853** | 0.2807 | -0.0045 |
| 10 | **0.3556** | 0.3236 | -0.0320 |
| 20 | **0.5090** | 0.4164 | -0.0926 |
| 50 | **0.6353** | 0.6131 | -0.0222 |
| 100 | **0.8512** | 0.8337 | -0.0174 |

- Inner search budgets 平均：0.3440 -> 0.3608，但改善幾乎來自 K=1/3。
- Macro Spearman：0.3171 -> 0.2995，完整排序反而退步。
- Outer mean efficiency：4/7 events 改善、3/7 下降。
- Top-1 的大幅提升高度受 `putinmissing` 影響：新組合剛好選中 Oracle
  第一名，efficiency 由 0.0476 變成 1.0。
- `sydneysiege`、`ferguson`、`germanwings-crash` 有明顯退步。

Group 被 outer nested selection 選中的次數：

| Group | Folds |
|---|---:|
| topology | 5/7 |
| semantic_summary | 5/7 |
| activity_level | 4/7 |
| text_surface | 4/7 |
| temporal_dynamics | 2/7 |
| source_reply_embedding | 2/7 |
| sentiment | 1/7 |
| account_age | 0/7 |

每個 fold 選到的組合都不同，沒有穩定的跨事件 feature subset。

## 5. 現在真正遇到的問題

1. **Quiet high-impact thread 的 30 分鐘訊號弱且事件相依。** 同樣安靜的
   threads 在現有合法特徵上很接近，但 future impact 可差很多。
2. **資料只有少量事件。** Inner LOEO 可選出在其他事件看似好的組合，卻
   不一定泛化到新的 outer event；winner-takes-all feature selection 方差大。
3. **頭部命中與完整排序衝突。** 某些表示可偶爾命中 Oracle Top-1/3，卻
   破壞 K>=5 的排序。
4. **目前模型仍學 utility，不是直接複製 Oracle rank。** RF 回歸
   `log1p(impact)` 後才排序；先前 order、pairwise、soft-Top-K、semantic
   residual 實驗也都只得到局部改善，沒有穩定贏 baseline。
5. **沒有可再任意加入的明顯安全特徵。** Account age、語意、sentiment、
   broad topology/time 組合都已測過；更多特徵很容易只增加事件特定雜訊。
6. **同一批 PHEME events 已反覆用於研究假設。** Nested LOEO 避免直接
   outer leakage，但最終仍需外部 untouched event/dataset 才能確認新方法。

## 6. 最值得研究的下一步假設

不要直接以 exhaustive winner 取代 quiet RF。最新結果顯示它可能包含一個
互補的 extreme-head signal。較合理的下一步是：

- 保留 current quiet RF 作主要排序；
- 以 inner event-OOF predictions 學一個有界、可收縮到 baseline 的 blend；
- exhaustive/head model 只提供第二個 score，不直接決定整條 ranking；
- blend weight、是否啟用及最大修正幅度完全由 inner LOEO 選；
- selection objective 同時約束 K=1/3 的提升與 K=5--50 不得明顯退步；
- outer held-out event 只做一次評估。

這可以檢驗「組合模型是否提供互補頭部訊號」，而不是再次進行更多特徵
搜尋。若 nested shrinkage/blend 仍不穩定，就應接受現有 30-minute quiet
signal 的可辨識上限，而不是用 test event 反覆調權重。

## 7. 重要檔案位置

- 最新完整結果：
  `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_233016_536789_nested_pheme_quiet_feature_combinations/`
- 最新歷史程式：
  `research_scratch/legacy_full/graphsage_intervention_14/run_nested_pheme_quiet_feature_combinations.py`
- 組合 helper：
  `research_scratch/legacy_full/graphsage_intervention_14/pheme_quiet_feature_combination_common.py`
- 有效 calibrated two-expert：
  `effective_models/pheme_active_quiet_calibrated_rf/`
- 有效 v5 baseline：
  `effective_models/pheme_v5_text_rf/`
- protected v5 inputs：
  `data/protected_research_assets/pheme_v5_strict30/`
- 全專案結果摘要：`PROJECT_RESULTS_20260901.md`
- 重整與封存紀錄：`REORGANIZATION_MANIFEST_20260902.md`

## 8. 接手時的執行注意事項

- 不要直接執行 archived Graph14 腳本：它保留舊的
  `graphsage_intervention_5` historical imports/path，而根目錄舊版本已封存。
- 若要做新實驗，先把必要的 latest helper 複製到一個新的 retained candidate
  目錄，改用 `data/protected_research_assets/pheme_v5_strict30/`，再做
  `py_compile -> import/path check -> leakage validation -> smoke`。
- 完整長時間 nested run 仍由使用者在前台手動執行，必須有 fold/combination
  進度和 run record。
- 不可用 outer/test 結果調 blend weight、特徵或 gate。
- 不可更改 rumour-only candidate pool、split、label、30-minute cutoff 或 metric。
