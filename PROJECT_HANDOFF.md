# 研究歷程與模型演進 Handoff

本文件是目前研究歷程的主要入口，整合舊版 `PROJECT_HANDOFF_*`、
`THESIS_MODEL_NARRATIVE_*` 與後續實驗。舊日期版文件保留作為不可覆寫的歷史紀錄；若內容
與本文件衝突，以可追溯的最新正式 artifact 與本文件標示的 protocol 為準。

## 1. 核心研究問題

研究目標不是一般真假分類，而是：在觀測時間截止時，只使用當時已存在的資訊，找出
「若現在介入，可能避免最多後續傳播」的 threads。後續又加入第二個問題：在證據尚未
完全確定時，應採取多強、是否可逆的介入。

```text
傳播模型：哪個 thread 的未來影響可能最大？
查核 Agent：目前有什麼證據支持或反駁核心主張？
政策層：在這些證據與風險下，允許採取什麼動作？
```

固定 30 分鐘模型的 outcome 為 `log1p(preventable_impact at 30 minutes)`；動態模型在
10、20、30、40、50、60 分鐘分別預測當下仍可避免的 future impact。任何 checkpoint
以後的資料只能作 label 或事後評估，不能進入特徵、正規化、候選建立或模型選擇。

## 2. 第一階段：GraphSAGE 與節點級介入

最初採用雙頭 GraphSAGE：Head A 預測 thread 未來影響，Head B 在已選 thread 中挑選
介入節點。這條路線沒有形成有效主模型：早期 Head A 的跨事件 CRR 偏低，加入 weighted
loss、pairwise ranking、top-k checkpoint 與 two-stage 訓練仍無法穩定改善；Head B 又讓
「選 thread」與「選 thread 內節點」兩種政策混在同一評估中，並拖累 Head A。

**為何轉向：** 圖神經網路成本高、資料量有限，且早期圖本身很稀疏；複雜模型沒有比
可解釋的 tabular 方法更穩定。研究因此先固定 whole-thread intervention，改用 RF 建立
可靠 baseline。Head B 並非永遠無效，而是需要獨立的節點級 intervention protocol 才能
重新研究。

## 3. 第二階段：固定 30 分鐘 RF baseline 與文本模型

### 3.1 Temporal／structural RF

模型使用前 30 分鐘的節點量、時間區段增量、最近活動、深度、葉節點比例、children 統計
與作者摘要，預測 `log1p(preventable_impact)`。RF 在小資料、非線性與混合尺度特徵上比
GraphSAGE 更穩定，也成為後續所有比較的骨架。

### 3.2 v5 Nested Text RF

在 tabular baseline 上加入 frozen RoBERTa 的 source／reply embedding，PCA 只能在 outer
training events 內 fit，是否採用文本由 inner validation 決定。歷史 strict-30 protocol
得到約 `0.1438–0.1467` 的 CRR@10／reduction@10，是當時最好的固定 30 分鐘模型。

**成功原因：** 文本能補充單純傳播量無法表示的主張與回覆語意，而且 nested selection
避免所有 event 都被強迫使用文本。

**限制：** 文本不是每個 event 都有幫助；歷史結果使用的 candidate pool 與後來 corrected
benchmark 不完全相同，所以不能直接拿舊數字與新表相減。

## 4. 第三階段：Graph7 作者、角色、事件脈絡與語篇

Graph7 建立更嚴格的 source-anchored snapshot，只接受可驗證的單一 root 傳播樹，加入作者
歷史／profile summary、回覆角色、strict-past event context 與 discourse features。

- Author-aware fixed RF 的歷史結果約 `0.1192`。
- Nested Discourse/Context RF 的歷史結果約 `0.1385`。

**研究價值：** 證明角色、事件脈絡與語篇在部分事件有用，也建立更清楚的 schema lock。

**為何沒有取代 v5：** 不同特徵 family 的效果高度依賴事件；account profile、user history、
context 與 discourse 都沒有形成所有 held-out events 一致的提升。Graph7 少 29 個無法建立
嚴格 snapshot 的 corrected threads；audit 顯示重疊的 2,373 threads target 完全一致，差異
來自 candidate coverage，不是 target 公式。

## 5. 第四階段：Active／Quiet 雙專家

單一 RF 容易由早期已活躍 threads 主導，因此用 outer-training events 的 recency percentile
分成 active 與 quiet，分別訓練 utility RF，再以 inner OOF prediction 學習正斜率 Ridge
校正，使兩個 expert 的分數能共同排序。

歷史同 protocol 結果由 global RF `0.1319`、uncalibrated two-expert `0.1421`，提升到
calibrated two-expert `0.1461`。

**成功原因：** 分流減少 active 與 quiet 不同分布互相干擾，且校正改善跨 expert 分數尺度。

**仍未解決：** quiet expert 進入最終名單時 precision 可以不差，但 quiet Oracle threads 的
recall 很低；早期安靜、後期爆量的 threads 缺乏穩定可轉移的排序訊號。

## 6. Quiet thread 的大量探索與負面結果

為解決 quiet recall，依序測試過 tier 分層、255 組特徵群組、semantic interaction、source
claim style、reply stance／rebuttal、跨 thread user overlap、change point、one-slot defer、
quiet-head blend、Oracle membership qualification、quiet rescue、wait-loss hazard、quiet
acceleration 與 dual-threshold 等方向。

共同發現如下：

- 粗略 broad high-impact enrichment 可以觀察到，但 exact Top-10／Top-50 尾端排序不穩定。
- Top-10 相對一般 threads 的單特徵 AUC 尚可，但相對 ranks 11–40 的近鄰候選接近隨機。
- 分層本身無法讓 quiet Oracle recall 穩定提高；更複雜的 feature combination 容易在 inner
  events 選到不同組合，跨事件轉移時退化。
- Hard-negative RF 與 three-band classification 明顯變差，因離散標籤丟失 continuous impact
  magnitude，且每 event 的正例極少。
- Wait-loss／acceleration 可以描述「何時可能變快」，但沒有穩定取代 cumulative impact
  ranking；adaptive／reserve policy 多半是成本與等待權衡，而不是偵測能力提升。
- Quiet pool 再排序曾以 tier probability 建候選池，再以 quiet utility RF 重排。完整七事件
  nested LOEO 最後多數 fold 仍選擇保留全部 quiet pool；相較 quiet utility RF，K=5、10、20、
  50、100 的平均 quiet efficiency 分別下降約 0.011、0.012、0.032、0.003、0.015，只有 K=1
  出現不穩定的正增量。因此未保留為正式模型。
- v5 Text RF 與 Active／Quiet RF 的 stacking rank fusion 亦未形成穩定優勢；K=10 相對最佳
  單模型低約 0.0005，K=20、50 分別低約 0.0175、0.0282。這表示簡單線性順位融合無法穩定
  利用兩模型的互補性，故未納入有效模型。
- 早期另有一支 rumour／non-rumour retrospective 統計草稿，但沒有完整輸入 artifact contract，
  入口程式亦未建立 DataFrame，不能重現正式結果；相關概念後來由 Chapter 3 behavior
  analysis 的正式診斷流程取代。

**為何轉向：** 問題不只是模型不夠複雜，而是固定 30 分鐘的 observable features 對 near-miss
threads 缺乏足夠資訊。因此研究改成多 checkpoint 更新，而不是持續堆疊 quiet 特徵。

## 7. Corrected target 與統一固定 30 分鐘 benchmark

Legacy traversal 在 multi-root cases 的 target 定義需要修正，因此建立 corrected all-roots
mapping。Graph7 的 2,373 個 rumour rows 與 corrected mapping 重疊部分 impact 完全一致；
額外差異是 corrected v5 還有 29 個 Graph7 無法合法建特徵的 threads。

為了公平比較，建立新的 2,373-thread intersection benchmark：rumour-only、30 分鐘 cutoff、
corrected target、nested LOEO、相同 K 與 denominator。這個 benchmark 才能直接比較
Graph7 Author-aware、Graph7 Discourse、v5 Text 與 Active／Quiet；舊結果保留但不覆寫。

結論不是某一模型全勝，而是 budget-dependent：K=1 作者模型較好，K=10 v5 Text 較好，
K=3/5/50/100 Active／Quiet 較好，K=20 Discourse 較好。

## 8. 第五階段：Multi-checkpoint Cumulative RF

固定 30 分鐘只能做一次決策，無法反映 thread 隨時間改變。動態版本每 10 分鐘更新：

- `cumulative`：從 source 到 checkpoint 的所有已觀察資料；
- `window_delta`：最近 10 分鐘、前一個 10 分鐘及兩者差值。

兩者皆用 event-separated RF 預測當下 `log1p(dynamic preventable impact)`。Sequential policy
用 `[9,9,8,8,8,8]` 在 10–60 分鐘累計選 50 個且不重複介入。Balanced Cumulative RF 的
歷史 2,402-thread 結果為 `0.3261` mean horizon reduction。

**為何保留 cumulative：** window/delta 沒有形成穩定互補優勢；累積觀察較能抵抗單一短
window 的波動。固定 quota 是公平研究設定，不是部署時每輪必須硬選的規則。

## 9. 第六階段：Hawkes-inspired fixed-decay RF

在 cumulative RF 的 57 個 cutoff-safe features 上，加入 5、10、20 分鐘固定指數衰減強度、
short/long ratio 與 recent share，共 62 features。它只是 Hawkes-inspired temporal-decay
augmentation，不是完整 Hawkes process。

歷史 2,402-thread、seed 42 結果由 `0.3261` 提升至 `0.3406`；五個 seeds 均為正向，平均
由 `0.3252` 到 `0.3381`。但 event bootstrap CI 跨 0，且部分 events 退步，所以定位為
通過 random-seed stability 的 candidate improvement，而非所有事件普遍最佳。

在後來 2,373-thread Bridge A 相同條件下，Hawkes 仍高於 Balanced Cumulative RF，但差距
應以 Bridge A 的 `0.3373 vs 0.3272` 報告，不與舊 2,402-thread 數字直接相減。

## 10. Verification-gated intervention

傳播量高不代表內容為假。早期 pure RF policy 會把名額用在後來被證實為真的高傳播內容，
因此加入 Verification Agent，但將責任拆開：

1. RF 產生 propagation-risk shortlist。
2. Agent 搜尋並回報 supported、refuted、conflicting、insufficient 或 search_failed。
3. Deterministic policy 將證據與 RF risk 映射到 none／soft／medium／hard／monitor。

工程上已驗證 batch、cache、cluster dedup、獨立 DB session、缺漏重試、agent_failure 與
confirmed-true exclusion；真實網路 smoke 能把有可靠支持證據的主張排除在限制性介入外。

**重要邊界：** eventual veracity Oracle replay 只是理想上限；今天搜尋到的網頁不能證明
十年前事件發生後 10–60 分鐘就有同樣證據。Agent timeout／搜尋失敗也不能當作 false。

## 11. Agent 直接修正排序的探索

### 11.1 Semantic residual／meta-model

Agent 曾接收 RF score、cutoff-safe source/replies 與結構摘要，輸出 residual direction、growth
tier 與八個 semantic signals。小型 pilot 顯示 Agent-only 能多抓一些 quiet false threads，
但沒有增加 false future nodes；nested feature selection 後語意增量接近零且跨 fold 選出的
signals 不一致。主要問題是語意訊號高度相關、樣本少，且 Agent 幾乎總判 RF underestimate。

**結論：** 不將此 residual model 提升為有效模型；保留作負面結果。

### 11.2 Historical analogy pairwise tournament

為讓 Agent 能大幅偏離 RF，建立 active RF 主池＋quiet rescue 池，Agent 成對比較 candidates，
並可使用 training-only historical anchors。工程上完成 frozen pair graph、A/B 對稱檢索、
Bradley–Terry／Borda、cache、retry、nested leakage 修正與進度恢復。

單一 CharlieHebdo leakage-safe fold 的結果：global RF false-node recall `28.75%`；unanchored
rescue `34.17%` 且 quiet false-thread recall由 `6.25%` 到 `25%`；anchored rescue反而降到
`27.36%`，quiet recall為 `0%`。這只是一個 outer fold，不能宣稱跨事件有效；historical
anchors 在該 fold 沒有幫助，因此整個分支留在 `analysis/`，不列有效模型。

## 12. Bridge A：固定與動態模型的共同政策比較

Bridge A 重新在 2,373-thread population 訓練動態模型，固定每 event Budget 50。Static 在
30 分鐘一次選 50；dynamic 在 10–60 分鐘依 `[9,9,8,8,8,8]` 選擇。共同分母為 event 在
10 分鐘時仍存在的 future opportunity，所以是 end-to-end policy comparison，不是單純比較
30 分鐘 ranking accuracy。

結果依序為：最佳固定模型 Active／Quiet `0.2862`、Balanced Cumulative `0.3272`、
Hawkes-inspired `0.3373`。動態方法較早取得資訊但平均動作時間為 34.2 分鐘；比較時必須同時
報告行動時間，不能把分數差全部歸因於模型架構。

## 13. 已發現並修正的研究問題

- 舊 reply CSV 曾因 Excel numeric conversion 破壞大型 Twitter IDs，後續改用校正資料。
- 舊 graph construction 曾在 LOEO 前全域 fit normalization，形成 leakage risk；正式流程改成
  fold-local fit／held-out transform-only。
- Legacy multi-root target traversal 已修正，舊與 corrected 結果分開保存。
- 多次 policy prototype 曾誤用 future `dynamic_preventable_impact` 作當下 threshold，漂亮結果
  因洩漏作廢；合法版本沒有同樣優勢。
- Agent pilot 曾把「未送 Agent」誤標成 `agent_failure`，也曾讓 inner／outer candidate
  population 不一致；相關結果作廢後重新設計。
- Historical tournament 曾有 nested anchor leakage、Oracle population 定義與 cache accounting
  問題；修正後只完成單 fold，未執行全六事件昂貴實驗。
- Clustering 僅靠 embedding 會把不同地點的制式新聞合併；加入 entity corroboration 後雖減少
  false merge，singleton rate 卻上升，表示搜尋與事件蒐集仍是部署資料瓶頸。

## 14. 目前結論與接手原則

1. 固定 30 分鐘的 early activity、結構、文本、作者與語篇提供互補的 broad triage signal，
   但 near-miss tail ordering 仍困難。
2. 動態 cumulative observation 是目前最可靠的主要改進；Hawkes-style decay 提供小幅、跨 seed
   穩定但 event-dependent 的增量。
3. Agent 最有明確價值的角色是證據查核與安全閘門；直接預測 future impact 尚未證明穩定。
4. 所有舊 metric、candidate pool、target 不同的結果必須分表；正式有效數字以
   [PROJECT_RESULTS.md](PROJECT_RESULTS.md) 為準。
5. 不再為提升分數任意加入新的 quiet feature bundle。新方向必須提出新增可觀察資訊的假設，
   並先做 leakage audit 與短 smoke。
