# Hawkes-inspired Multi-checkpoint Cumulative RF 結果

## 1. 研究問題

原本的 cumulative RF 已使用回覆數、reply rate、recency 與 interarrival gap，
但不同時間發生的回覆在多數統計量中仍可能具有相近權重。本實驗檢驗：將越接近
checkpoint 的回覆賦予越高權重，能否補充短期傳播動能並提升早期介入排序。

## 2. 模型與特徵

基礎模型為 57-feature Multi-checkpoint Cumulative Random Forest，涵蓋：

- 累積活動量、reply rate、first/last reply、recency 與 active span；
- 六個十分鐘活動 bins 與 reply-gap 統計；
- root children、leaf/internal fraction、reply-to-reply、parent concentration；
- depth 與 reply-latency 統計；
- source 與 observed replies 的字數、token、URL、mention、hashtag、問號、驚嘆號及 uppercase ratio。

新增五個 cutoff-safe 特徵：

1. `hawkes_exp_decay_300s`
2. `hawkes_exp_decay_600s`
3. `hawkes_exp_decay_1200s`
4. `hawkes_short_to_long_ratio`
5. `hawkes_recent_share`

固定衰減強度定義為：

```text
E_tau(T) = sum_i exp(-(T - t_i) / tau), 其中 t_i <= T
```

因此每個 checkpoint 只使用當下或更早的 reply timestamps。新增特徵後總數為 62。

## 3. 訓練與評估

- Dataset：PHEME corrected-v5 rumour-thread allowlist，2,402 threads、9 events。
- 正式 eligible events：candidate 數至少 100 的 7 events。
- Checkpoints：10、20、30、40、50、60 分鐘。
- Target：`log1p(corrected dynamic preventable impact)`。
- Model：RandomForestRegressor。
- 參數：300 trees、depth 8、minimum leaf 3、`max_features=0.8`。
- Split：outer leave-one-event-out；held-out event 的所有 checkpoints 均不參與 fitting。
- Policy：總 Budget 50，固定 quotas `[9, 9, 8, 8, 8, 8]`。
- 已介入 thread 從後續 candidate pool 移除。
- Denominator：各 event 第 10 分鐘的 future horizon。

Baseline 與 Hawkes challenger 在每個 outer event、每個 seed 使用完全相同的資料、
target、split、policy 與 RF 參數；唯一差異是五個 fixed-decay features。

## 4. Seed 42 完整比較

| 模型 | Mean horizon reduction | Blocked future nodes | Mean action minute |
|---|---:|---:|---:|
| Cumulative baseline | 0.3261 | 4,527 | 34.2 |
| Hawkes-inspired cumulative RF | **0.3406** | **4,763** | 34.2 |

絕對提升為 `+0.0145`，相對 baseline 約增加 4.5%，並多阻擋 236 個 future nodes。

## 5. Event stability audit

- 七個 events 中四個改善、三個下降。
- 平均 reduction delta：`+0.01453`。
- 中位數 delta：`+0.01528`。
- 移除任一 event 後，平均 delta 仍介於 `+0.00880` 至 `+0.01822`。
- Event bootstrap 95% CI：`[-0.00047, 0.03081]`。
- Bootstrap mean delta 為正的比例：96.2%。
- Exact paired two-sided sign-flip test：`p = 0.15625`。
- 最終 50 個介入對象的平均 Jaccard：0.678。
- 各 checkpoint Top-50 平均 Jaccard：0.729。

結果顯示正向趨勢不由單一事件完全造成，但 eligible events 只有七個，信賴區間略跨 0，
因此不可宣稱已達統計顯著。

## 6. 五組 random-seed stability

預先指定 seeds 為 7、21、42、84、168。

| Seed | Baseline reduction | Hawkes reduction | Delta | Blocked-node delta |
|---:|---:|---:|---:|---:|
| 7 | 0.3234 | 0.3380 | +0.0146 | +278 |
| 21 | 0.3265 | 0.3384 | +0.0119 | +158 |
| 42 | 0.3261 | 0.3406 | +0.0145 | +236 |
| 84 | 0.3235 | 0.3358 | +0.0123 | +245 |
| 168 | 0.3266 | 0.3375 | +0.0109 | +175 |

五組 seeds 全部為正：

- Baseline 五組平均：0.3252。
- Hawkes 五組平均：0.3381。
- 平均絕對提升：`+0.01286`。
- 相對平均提升：約 3.95%。
- Seed delta 標準差：0.00164。
- 平均 blocked-node delta：`+218.4`。

這證明改善不是 seed 42 的偶然波動，但 seeds 共用相同 events，不能把五個 seeds
視為五份獨立資料做顯著性檢定。

## 7. 跨事件限制

跨五組 seeds，Ferguson、Germanwings、OttawaShooting 與 PutinMissing 均為 5/5
改善；CharlieHebdo、PrinceToronto 與 SydneySiege 的平均變化略為負值。固定 5、10、
20 分鐘衰減尺度並非適合所有傳播節奏，因此模型通過的是 random-seed stability，
而不是「所有事件均改善」。

## 8. 論文與部署定位

可使用的結論：

> 固定指數衰減特徵能補充近期傳播動能；在五組預先指定 RF seeds 下，Hawkes-inspired
> cumulative RF 均優於相同 seed 的 cumulative baseline，平均 horizon-reduction 增益
> 為 0.0129。然而，改善仍具有事件異質性，且七事件的統計檢定未達顯著。

目前模型應稱為「通過 random-seed validation 的 candidate improvement」，不自動覆寫
Balanced Cumulative RF baseline。固定 Budget-50 只適合受控比較；小事件或真實系統需
另外驗證 capacity-aware adaptive intervention policy。

## 9. Research safety

- 原始資料與 protected derived dataset 未修改。
- Candidate identities、labels、event split 與 target 未修改。
- 每個 feature 僅使用 `timestamp <= checkpoint` 的回覆。
- Future growth 與 preventable impact 只用於 target/evaluation。
- Outer held-out events 未用於 fitting 或 feature selection。
- 所有實驗均寫入新的 timestamped output directories，既有 baseline 未覆寫。
