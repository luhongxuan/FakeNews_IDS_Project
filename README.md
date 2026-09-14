# 假訊息早期傳播預測與分級介入系統

本專案研究如何只使用事件發生後早期可觀察的資訊，預測哪些傳播串在未來可能造成較大
傳播影響，並結合證據查核與分級政策，產生可稽核的介入建議。

研究正確性優先於分數。所有正式比較必須遵守相同 candidate population、觀測時間、
target、split 與 metric；不同 protocol 的歷史數字不得直接相減或放在同一排行榜。

## 主要文件

- [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)：完整研究歷程。說明每次模型轉向的原因、
  成功與失敗結果、已發現的資料／洩漏問題，以及目前仍未解決的限制。
- [PROJECT_RESULTS.md](PROJECT_RESULTS.md)：目前有效模型與正式結果。集中記錄特徵、
  訓練方法、評估 protocol、主要分數與 artifact 位置。
- [VERIFICATION_GATED_INTERVENTION_SPEC.md](VERIFICATION_GATED_INTERVENTION_SPEC.md)：
  Verification Agent、證據狀態、快取與分級介入的正式技術規格。
- [AGENTS.md](AGENTS.md)：研究安全、實驗重現、資料保護與執行規範。
- [effective_models/README.md](effective_models/README.md)：仍由系統或正式 benchmark
  使用的模型目錄。
- [effective_models/CLEANUP_STATUS.md](effective_models/CLEANUP_STATUS.md)：模型資料夾
  標準化與 Git 清理進度。
- [benchmarks/pheme_fixed30_unified/README.md](benchmarks/pheme_fixed30_unified/README.md)：
  固定 30 分鐘共同 identity／metric 評估套件。
- [benchmarks/pheme_static_dynamic_bridge/README.md](benchmarks/pheme_static_dynamic_bridge/README.md)：
  固定與動態 Budget-50 的 Bridge A replay。

## 系統流程

```text
早期事件與傳播資料
        ↓
固定 30 分鐘或 10–60 分鐘動態 snapshot
        ↓
RF 傳播效益模型：預測「現在介入可能避免多少後續傳播」
        ↓
Verification Agent：搜尋並整理支持、反駁、衝突或不足的證據
        ↓
Deterministic policy：產生 none / soft / medium / hard / monitor 建議
        ↓
前端指揮台：顯示傳播圖、查核軌跡、決策理由與歷史紀錄
```

RF 與 Agent 的責任不同：RF 不判斷真假；Agent 不直接決定最終處置。最終動作由固定政策
同時考慮傳播風險、證據狀態、查核成本與介入容量。

## 目前正式比較

1. **固定 30 分鐘統一 benchmark**：2,373 個共同 PHEME rumour threads、corrected
   target、nested LOEO，報告 Corrected Preventable Impact Capture@K。
2. **Bridge A static–dynamic comparison**：同一 2,373-thread population、每事件總 Budget
   50，以 10 分鐘時仍存在的 future opportunity 作共同分母，比較固定 30 分鐘與
   10–60 分鐘動態決策。
3. **Twitter15/16 auxiliary experiments**：使用固定 source-level split，不能宣稱為
   PHEME event-separated 結果，也不能與 PHEME 分數直接比較。

完整數字與比較邊界請見 [PROJECT_RESULTS.md](PROJECT_RESULTS.md)。

## 目錄

```text
backend/             API、查核 Agent、介入政策與 Radar 服務
frontend/            情報追蹤、PHEME replay 與介入指揮台
social-frontend/     社群情境展示介面
effective_models/    系統／正式 benchmark 仍需要的模型
benchmarks/          跨模型統一評估、政策 replay 與論文報表工具
data_pipeline/       原始資料處理、corrected target 與資料 lineage
analysis/            探索、診斷與負面結果；本機保留、Git 忽略
data/                受保護與衍生資料；不得任意修改
outputs/             論文圖表與整理輸出
```

## 儲存與 Git 原則

- `artifacts/`、`reference_result/`、`experiments/` 與 `analysis/` 均為本機資料，不提交。
- CSV、pickle/joblib、NumPy arrays、Parquet 與模型權重由使用者自行保存或搬移。
- 每個有效模型應是獨立單元，具有自己的 loader、特徵、訓練流程、smoke、artifact 清單
  與刪除紀錄，不從另一個模型資料夾 import 原始碼。
- 舊實驗不覆寫；失敗結果保留在研究歷程中，避免重複走已證實無效的方向。

## 重要限制

- `preventable impact` 是資料集中的反事實後續節點數，不是已在真實平台觀察到的因果減量。
- 目前 PHEME 評估把不同 thread 視為獨立傳播樹，未建模介入一個 thread 對其他 thread
  的跨串影響。
- Verification Agent 的當代網頁搜尋只能展示查核流程；若要宣稱嚴格早期查核，證據發布
  時間必須早於當時 checkpoint。
- 本系統尚未在真實平台執行下架或降低曝光；前端顯示的是政策建議與離線 replay。
