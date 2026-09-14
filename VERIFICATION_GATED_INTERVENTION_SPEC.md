# Verification-gated Early Intervention 正式規格

本文件是目前 Verification Agent 與分級介入政策的 canonical specification。
它承接早期 V1／V2 設計，但不以日期命名；後續規格變更直接在版本控制中追蹤。

## 1. 核心研究架構

本規格不重新訓練 propagation model，而是將系統正式拆成三層：

```text
Propagation Risk Layer
Hawkes-inspired Multi-checkpoint Cumulative RF
預測現在介入可能避免的 future impact
                    ↓
Evidence Layer
Verification Agent 只產生結構化 evidence report
                    ↓
Policy Layer
確定性規則分配 hard / medium / soft / none
```

RF 回答「哪一些 threads 值得優先查證」；Verification Agent 回答「目前有什麼證據」；
Policy 才回答「證據允許採取多強的行動」。LLM 不應直接自由決定最終處置。

## 2. 舊 replay 的處理

歷史 `agent_vs_rf_replay_full.py` 執行結果：

- 保留 terminal output 與既有輸出作診斷紀錄。
- 不納入正式模型比較。
- 不覆寫、不重新命名為正式結果。

原因是舊流程具有 veracity mapping、`cluster_thread_id`／`thread_id` lookup、初始狀態與
tool recheck 不一致，以及 action accounting 等問題。後續修正已落在目前的查核 Agent、
介入 policy 與對應 smoke／DB tests；舊 replay 不再是 canonical 執行入口。

## 3. 必須先修正的基礎錯誤

### 3.1 PHEME mapping

| misinformation | true | 類別 |
|---:|---:|---|
| 0 | 1 | true |
| 1 | 0 | false |
| 0 | 0 | unverified |
| 1 | 不存在 | false |
| 0 | 不存在 | unverified |

### 3.2 Identity

- `thread_id` 是 veracity/evidence lookup key。
- `event_id` 只表示事件，不可代替 thread identity。
- `cluster_thread_id` 若由 live system 使用，必須有明確且經測試的 thread mapping。
- Initial verification status 與 Agent tool recheck 必須讀取同一份 identity 與 evidence cache。

### 3.3 Failure semantics

- Timeout、network error、tool error、parse failure 均映射為 `search_failed`。
- `search_failed` 不可轉換成 false、refuted 或 insufficient。
- 每個 event 與 checkpoint 必須增量保存結果、tool calls 與 failure details。

Mapping、identity lookup 與 timeout semantics 均需單元測試。

## 4. Verification Agent schema

Agent 僅輸出以下結構化 evidence report：

```json
{
  "evidence_status": "supported | refuted | conflicting | insufficient | search_failed",
  "confidence": 0.0,
  "supporting_sources": [],
  "refuting_sources": [],
  "independent_source_count": 0,
  "claim_match": true,
  "evidence_published_at": []
}
```

正式 schema validation 需確認：

- `confidence` 介於 0 和 1。
- Sources 保存 URL、publisher、title、retrieved_at、published_at、支持或反駁摘要。
- `claim_match=false` 的來源不得形成 supported/refuted 結論。
- 多個網站重複引用同一原始消息時，只能計為一個獨立來源。
- 歷史嚴格評估只接受 `published_at <= checkpoint_time` 的證據。
- 缺少發布時間的來源不得用於 time-aligned hard action。

## 5. Credible-support Gate

```text
先查 cutoff-aware evidence cache
        ↓
已有可信且時間對齊的 supporting evidence → 不抑制、保留監控
        ↓
其餘 candidates 由 RF 排序
        ↓
分批送 Verification Agent
        ↓
Policy 根據 evidence report 分級行動
        ↓
名額不足時取下一批 RF candidates 補位
```

這是「當下已有可信支持證據就不抑制」，不是使用 PHEME eventual-true label 作 gate。
Evidence cache 必須以 claim identity、source identity 與 publication time 版本化，避免未來
證據洩漏到較早 checkpoint。

## 6. 四級介入政策

| Evidence status 與條件 | 行動 | 初始 action strength |
|---|---|---:|
| supported | None；持續監控 | 0.00 |
| insufficient＋低 propagation risk | None／提示 | 0.00 |
| insufficient＋高 RF risk | Soft downrank | 0.10–0.25 |
| conflicting | Medium downrank＋人工查核 | 0.25–0.50 |
| refuted，且證據充分、claim matched、time aligned | Hard hide／限制轉發 | 0.75–1.00 |
| search_failed | Monitor only | 0.00 |

Soft 與 medium intervention 必須可逆、設定到期時間、每 10–20 分鐘重新查證，並在找到
可信支持證據後立即解除。實驗中的 strength ranges 必須預先固定或由 inner events 選擇，
不可用 outer outcomes 調整。

## 7. 兩種預算與補位

- `intervention_budget`：實際可採取非零 action 的 threads 上限。
- `verification_budget`：Agent calls 上限。
- `shortlist_batch_size`：每輪交給 Agent 的 candidates 數量。
- `replacement_rounds`：因 supported、search_failed 或不符合介入條件而進行的補位輪數。

Supported、search_failed 與 none 不消耗 intervention slot，但會消耗 verification budget。
任一 budget 用完即停止；不允許假設 Agent 能無限查證直到填滿 50 個介入名額。必須保存
unused intervention slots。

固定每 event 50 仍是 controlled experimental capacity，不是所有事件規模的部署保證。

## 8. 效益與傷害計分

探索性加權代理值定義為：

```text
weighted_effect_proxy = future_nodes × action_strength
```

這個數值不可命名為「實際阻擋節點」，因為 action strength 不是從平台因果實驗估計出的
真實傳播抑制率。正式文件應稱為：

- `weighted_effect_proxy`；或
- `strength-adjusted future-impact utility`。

必須對 soft／medium／hard strengths 做預先指定的 sensitivity analysis。除非未來取得
真實平台實驗或可信 causal estimate，否則 `action_strength=0.5` 只代表政策權重，不代表
真實阻擋 50% 傳播。

正式報告至少包含：

- Refuted hard-action future impact。
- Medium／soft strength-adjusted utility。
- Supported content intervention count。
- Eventual-true hard／medium／soft harm，僅作事後 outcome audit。
- Effective horizon proxy 與 net utility。
- Agent calls、latency、補位深度與 verification cost。
- Unused intervention slots。
- Timeout、search failure 與 no-evidence counts。

原始 propagation-only `0.3261` 與 Hawkes `0.3406` 保留，不被加權代理值覆寫。

## 9. Smoke 順序

### 9.1 Mocked deterministic smoke

先用固定 evidence reports 驗證：

- Mapping 與 identity lookup 正確。
- Supported → none。
- Refuted → hard。
- Conflicting → medium。
- Insufficient 只能依 propagation-risk rule 進 soft 或 none。
- Search failure → monitor。
- 補位、兩種 budget、expiry 與 recheck 正常。
- Action strength 與 weighted proxy 計算正確。
- 每個 decision、evidence、tool call、checkpoint 與 failure 均保存。

### 9.2 單一 event 真實 Agent smoke

Mocked smoke 成功後才接真實 Agent。此 smoke 主要驗證工具與資料流；若使用 current web，
不可當成歷史 PHEME early-verification 正式成績。

## 10. 正式實驗順序

1. Propagation-only RF baseline。
2. Veracity-stratified OOF audit。
3. Eventual-label Oracle two-tier upper bound。
4. Corrected mapping、ID、schema、failure semantics 與 policy unit tests。
5. Mocked Verification Gate smoke。
6. 單一 event 真實 Agent workflow smoke。
7. Time-aligned evidence availability audit。
8. 七事件 full Agent replay。
9. 真實 Agent 與 Oracle upper bound gap analysis。
10. 將通過 provenance 與 time-alignment 檢查的結果寫入正式論文。

## 11. 不變的研究底線

- 不重訓或刪除 RF 訓練樣本。
- 不改 future-node target。
- Eventual veracity 不進入 early RF features。
- 不以 test-event veracity 過濾 RF candidate pool。
- 不把 unverified、insufficient 或 search_failed 視為 false。
- 不把 current-web evidence 冒充歷史 checkpoint evidence。
- 不覆寫 0.3261 baseline、Hawkes results 或已保存的舊 Agent replay。
