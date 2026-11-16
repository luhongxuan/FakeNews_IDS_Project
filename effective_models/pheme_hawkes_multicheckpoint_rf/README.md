# PHEME Hawkes-inspired Multi-checkpoint Cumulative RF

這個資料夾保存目前通過 random-seed stability validation 的 Hawkes-inspired
動態介入候選模型。它以既有 Multi-checkpoint Cumulative Random Forest 為基礎，
加入五個只由當下以前回覆時間計算的固定指數衰減特徵。

## 模型定位

- 任務：在 10、20、30、40、50、60 分鐘預測「現在介入可阻擋多少未來節點」。
- 輸出：每個 thread 在各 checkpoint 的 scalar intervention-utility score。
- 正式比較政策：總 Budget 50，checkpoint quotas 為 `[9, 9, 8, 8, 8, 8]`。
- 狀態：通過五組 RF seeds 的正向穩定性判準，仍保留為 candidate improvement。
- 正式 baseline：`effective_models/pheme_multicheckpoint_rf` 中的 Balanced Cumulative RF。

固定 Budget-50 是控制實驗政策，不是所有事件規模都適用的部署保證。實際部署應另外
驗證 capacity cap、train-only utility threshold 與 unused-budget carryover。

## 文件

- `RESULTS_20260904_ZH.md`：方法、特徵、訓練、結果與限制。
- `MODEL_MANIFEST.json`：機器可讀的 protocol、來源與 artifact hashes。
- `ARTIFACTS.md`：本機需要保留但不提交 Git 的三個大型輸入與 hash。
- `REMOVED_FILES.md`：舊跨模型流程的移除與替代紀錄。
- `reference_result/ARTIFACT_INDEX.md`：不可覆寫的完整實驗來源索引。

## Canonical commands

Smoke 與 full 共用 `training/common.py`，只有樹數與 fold 數不同。先執行短 smoke；
完整七事件 paired LOEO 屬長時間工作，由使用者在前景手動執行：

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_hawkes_multicheckpoint_rf\training\run_smoke.py
.\venv\Scripts\python.exe .\effective_models\pheme_hawkes_multicheckpoint_rf\training\run_full.py
```

## 命名邊界

本方法是 **Hawkes-inspired fixed-decay feature augmentation**。它沒有估計完整的
multivariate Hawkes process，也沒有建立使用者層級 excitation matrix，因此論文中
不可簡稱為「Hawkes model」。

## 下游介入決策

這個模型只排序 propagation utility，不判斷內容真假。下一階段由 Verification Agent
查證 RF shortlist，依證據決定 hard、medium、soft、none 或 monitor。Eventual veracity
不會加入本模型特徵；完整研究邊界見 repository root 的
`VERIFICATION_GATED_INTERVENTION_SPEC.md`。
