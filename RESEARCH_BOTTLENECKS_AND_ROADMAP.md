# 現有瓶頸分析與後續路線圖

整理自 2026-09 這一輪對現有模型、資料集、即時雷達管線的實測分析與工程修復，並經過
一輪唯讀審查修正。所有數字都標明來源檔案，主張的強度已對齊實際證據等級——能證明的
才寫成結論，只是觀察到的就標成觀察，不再混用。

> 2026-11-15 更新：第 10 節新增 Bluesky 真實資料的預先登記實測（起飛預測成立、起飛後排大小無法確定、
> 假訊息排序仍無證據）。與第 3.4–3.6、7 節衝突處，以第 10 節為準。

## 0. 已確認正確的主軸（審查後維持不變）

三層分工本身是對的，也比繼續堆 RF 特徵更合理：

- **RF**：低成本估計「這則貼文可能還會擴散多少」（純結構信號，不碰內容真假）
- **Agent**：證據查核（是否為真、依據是什麼）
- **Policy**：是否介入、介入到什麼強度（結合前兩者 + 預算限制）

以下瓶頸分析與路線圖，都是在不動搖這個主軸的前提下做修正。

## 1. 瓶頸總覽

### 1.1 早期偵測 RF：四個訓練模型在共同資料與觀測條件下呈現有限增量

正確的統一結果表（[PROJECT_RESULTS.md](PROJECT_RESULTS.md:44)，7 個候選數 ≥100 的大事件
macro，protocol：2,373 threads、30 分鐘 cutoff、nested LOEO）：

| 模型 | 類型 | @10 |
|---|---|---:|
| Oracle（事後診斷值；正式 provenance artifact 待補） | 診斷上限 | 0.305 |
| v5 Nested Text RF | 訓練模型 | 0.145 |
| Graph7 Nested Discourse RF | 訓練模型 | 0.126 |
| Active／Quiet Two-expert RF | 訓練模型 | 0.126 |
| Graph7 Author-aware RF | 訓練模型 | 0.126 |
| Current Early Size | baseline（純結構捷思法，非訓練模型） | 0.106 |
| Random Expected | baseline | 0.038 |

Oracle `0.305` 是另外以臨時計算取得的事後診斷值，目前沒有與其他模型結果相同規格的
run record、計算腳本輸出與正式 artifact，因此只用來描述可能的排序改善空間，不與已有
正式 benchmark provenance 的模型分數視為同等證據。補齊候選母體、corrected target、
K=10 排序、per-event denominator、七事件 macro 與 tie-break 的可重現紀錄後，才可將它
升格為正式 benchmark row。

**修正後的敘述**：這是**四個真的訓練過、特徵工程方法不同的 RF 架構**（文字 PCA、話語
結構、雙專家校準、作者行為），在同一份資料、同一個 30 分鐘 cutoff、同一個 nested LOEO
protocol、同一個 future-impact target 下，Capture@10 集中在 0.126–0.145；純結構捷思法
baseline（Current Early Size）則是 0.106。不是「六種架構收斂」——baseline 跟 Oracle
不是獨立的「模型」，混著算會誇大收斂的證據力。

**天花板敘述的強度修正**：這四個模型仍共享相同的 PHEME 資料、相同的 30 分鐘資訊窗、
大致相近的 RF 歸納偏好、相同的 target 定義。四者收斂只能支持「**已接近目前資料、可觀察
特徵與 RF 類模型的共同天花板**」，不能推論到「已接近這個預測任務本身的理論極限」——
換一種資料、換一種模型家族（非 RF）、換一種 target 定義，都還沒被排除。

**40% 這個數字的算法要寫清楚**：
- 直接算 `0.145 / 0.305 ≈ 47.5%` —— 這是「模型吃到 oracle 絕對值的比例」
- 扣除 random baseline 後算 `(0.145-0.038) / (0.305-0.038) ≈ 40.1%` —— 這是「模型
  取得了 random-to-oracle 可改善空間的 40%」

兩個是不同的量，之前文件把後者寫成「吃到 oracle 的 40%」是錯的，混用了兩種算法。

### 1.2 目前 live radar 實際部署的模型寫錯了

- **`v5 Nested Text RF`**：只用於 PHEME 固定 30 分鐘研究比較、歷史事件 replay 排序
  （[model_scores.py](backend/app/services/model_scores.py)，指揮台 / 情境模擬用的
  就是這個）、demo 資料選取。
- **`Balanced Cumulative RF`**（[intervention_model.py:1](backend/app/services/intervention_model.py:1)，
  57 維即時累積特徵，10/20/30/40/50/60 分鐘 checkpoint）：**這才是即時雷達
  （`intervention_agent.score_thread_at_checkpoint` → `radar.py` 的
  `manual_policy_decision`）真正在用的模型**。

兩個模型分工不同，不能把「固定 30 分鐘 benchmark 表現最好的模型」直接說成「目前部署
的模型」。

### 1.3 訓練資料集偏離現實——但「PHEME 無腐蝕」這句要降級

| 面向 | PHEME | Twitter15/16 |
|---|---|---|
| Thread 規模（中位數） | 12 | 260~271 |
| 最大 cascade | 346 | 2,990~3,089 |
| 年代 | 2014-2015 | 2011-2016（更舊） |
| 事件/樣本數 | 9 個事件 | 1,490 / 818 個 thread |
| 節點不可用率（同口徑量測） | **未量測** | 43.8%~45.4%（重新 rehydrate 後量測） |
| 類別分佈 | 真實比例 | 人工刻意平衡 25/25/25/25，不真實 |

**修正**：PHEME 的原始 JSON 是收集當下就完整存檔，不需要事後用 API 重新 hydrate，這是
它跟 Twitter15/16（推文 ID 收集於 2011-2016，內容要靠 2026 年重新呼叫 API 才能拿到）
在收集方法上的根本差異。但本專案**沒有對 PHEME 做過與 Twitter15/16 相同口徑的「節點
不可用率」量測**（例如帳號後續是否已停用、內容是否曾被平台端撤下但當時的存檔快照仍
保留），所以不能直接宣稱 PHEME 的腐蝕率是零，只能說「用同一套方法量測，PHEME 目前
沒有這個問題被觀測到，因為它的收集方式先天不需要事後 rehydrate」。

還未確認的口徑問題（列出來，不假裝已核實）：
- PHEME 的中位數 12、Twitter15/16 的中位數 260+，是否用同一種定義在算（是否含 source
  node、是否只算可取得的 reply、缺失節點是否仍計入分母）——目前兩邊的統計是分開跑的
  腳本算出來的，尚未確認過完全同一口徑。

### 1.4 現代 Twitter/X 資料取得成本——固定金額要拿掉

X 官方 2023 年關閉免費學術 API，這個事實成立。但「企業級 4.2 萬美元/月」是網路上一則
說法，不是 X 官方目前的定價敘述；[X API 官方定價頁](https://docs.x.com/x-api/getting-started/pricing)
說明目前是按使用量、按 endpoint 計價，實際金額要在 Developer Console 查看，高流量
才會用到 Enterprise 方案。可以保留的結論是「大規模即時收集現代 Twitter 資料仍可能
成本不低、且不再有免費學術管道」，但不應該再引用那個固定月費數字。

### 1.5「沒有更好的替代資料集」——這句話要收斂成可查證的版本

不應寫成「沒有更好的資料集」這種絕對敘述，改成：**尚未確認存在能同時滿足「現代平台
＋完整 cascade 結構＋精確時間戳＋真假標籤＋早期介入用的 target」這五個條件的公開
資料集**。上一輪查到的 MuMiN（2,100萬推文、19.8萬 thread、連結 1.29萬筆真實 ClaimReview
查核結果）規模夠大、標籤來源也夠權威，但**還沒確認它是否真的具備完整 reply-tree
結構、以及是否有現成的 intervention/impact target 可用**——這是評估 MuMiN 時第一件
要查的事，不能只看「新」跟「大」就當作解方。

**第四輪修正**：上面五個條件是「misinformation 分類任務」的完整需求清單；3.5／3.6
已把排序模型改成純傳播圖結構、不需要真假標籤，該模型實際只需要滿足「現代平台＋
完整 cascade 結構＋精確時間戳」三個條件，拿掉真假標籤與早期介入 target 之後，
可用的公開資料集範圍明顯變大——具體選項見 3.6。

### 1.6 查證 Agent：外部查核來源其實已經在用，缺口是「證據來源等級沒有結構化記錄」

**修正**：系統已經透過 `search_fact_checks`（[search_tools.py:47](backend/app/services/search_tools.py:47) →
[fact_check.py:9](backend/app/services/fact_check.py:9)，實際呼叫
`https://factchecktools.googleapis.com/v1alpha1/claims:search`）取得 Google Fact
Check Tools API 回傳的真實 ClaimReview（含 publisher、review_url、review_date、
textual rating）——**不是「完全沒有外部查核來源」**，這條路徑本來就存在。

真正的缺口是：agent 最終吐出的 `credibility: likely_false` 這個欄位，**沒有記錄
這個判斷是基於哪一種等級的證據**——是直接命中一則專業 ClaimReview？是多篇一般新聞
報導交叉印證？還是像 Trump/Kennedy 那個案例一樣，只是「詳細報導卻隻字未提」的
沉默推論（規則 7）？這三種證據強度完全不同，但現在存進資料庫的欄位一視同仁，都只是
一個 `credibility` 字串。

**建議新增的欄位**（寫入 `VerificationReport.report_jsonb`）：

```
verdict_basis:
  - professional_fact_check       # 命中 Google Fact Check API 的 ClaimReview
  - official_primary_source       # 命中官方/權威機構原始資料
  - multi_source_news_inference   # 多篇一般新聞交叉印證
  - agent_absence_inference       # 規則 7 這類「沉默即反駁」推論
  - insufficient_evidence

authority_confirmed: true | false   # 是否有 professional_fact_check 或 official_primary_source
human_reviewed: true | false        # 是否曾被人工複核過
```

之前這份文件說「缺乏可稽核的 verdict provenance」，應該取代原本 1.5 節「沒有一條路徑
引用外部真人查核機構」這個不準確的敘述。

## 2. 方向調整（維持原三個並行方向，但排序與內容修正）

三個方向本身成立，但彼此的**前置依賴關係**需要修正：**方向 C（查證證據等級結構化）
必須先於方向 A（RAG 自建資料）**——原因見下一節，RAG 若在證據等級沒有結構化之前就
把所有 `likely_false` 一視同仁地餵回系統，會有自我強化錯誤的風險。

### 方向 A：降低對離線基準資料集精準度的依賴，用即時雷達自建當代資料（有前提條件，見 3.4）
### 方向 B：把「已接近目前資料/特徵天花板」寫成有憑有據的研究限制章節（強度已在 1.1 修正）
### 方向 C：先把查證證據等級結構化、建立回歸測試基礎，再讓下游（RAG、policy）依賴查證結果

## 3. 詳細工作包（供第 7 節執行順序引用）

本節保留各項工作的具體內容、風險與驗證要求，但**不再另外定義一套 P0～P3 優先順序**。
正式執行階段、前置依賴與可平行項目，統一以第 7 節「建議的收斂後執行順序」為準。
這樣既保留第一輪審查累積的實作細節，也避免本節與第 7 節出現兩套互相競爭的編號。

### 3.1 可信的查證評估基礎

1. 修正本文件與相關文件裡的模型數量、部署模型、X API 敘述、資料集腐蝕率敘述（本次
   已在第 1 節完成）。
2. 在 `VerificationReport.report_jsonb` 新增 `agent_reported_verdict_basis`、
   `system_verified_verdict_basis`、`authority_confirmed`、`human_reviewed`、
   `claim_match_status` 與 `evidence_source_ids`。Agent 可以在最終 JSON 裡說明自己使用的
   推理依據，但後端必須根據既有 `tool_calls`、`evidence[].source_tier`、
   `evidence[].stance` 與 claim/event matching 結果重新計算系統認可的證據等級。
   `human_reviewed` 只能由人工操作流程更新，不能由 Agent 自行宣告。Policy 與 RAG
   寫入只使用後端驗證過的欄位。
3. 建立**固定 evidence regression suite**，但四個案例不能一視同仁地當「已知正確答案」
   使用——這四個案例的證據等級本身並不相同：

   | 案例 | expected_verdict 依據 | 可否當「已知正確答案」的 fixture |
   |---|---|---|
   | Isabela 地震 | 直接查 USGS 官方地震資料庫 | 可以——獨立權威來源 |
   | Banksy 假圖 | Snopes/CNN/Fast Company 等多來源交叉印證＋當事人否認 | 可以——多來源交叉印證 |
   | 3 gunmen | PHEME 人工標註，但本專案自己已指出這個 claim 的範圍本身有歧義（辦公室攻擊 vs 整波巴黎恐攻） | **不行**——正確答案本身有爭議，不能斷言單一 expected_verdict |
   | JFK/Trump | 沒有任何外部機構查證過，`likely_false` 是 agent 自己的沉默推論（規則7） | **不行**——用 agent 自己的推論結果驗證 agent，是循環論證 |

   因此拆成兩種用途，不再統一叫「四案例」：
   - **Verdict-correctness 回歸測試**（只用 Isabela、Banksy 這類有獨立權威來源的案例）：
     固定 evidence snapshot、模型、prompt、解碼設定，斷言 `expected_verdict`／
     `acceptable_verdict_range`，檢查是否出現 forbidden reasoning。這是唯一可以合理
     斷言「答案應該是什麼」的一層，且**只有兩個案例不夠**，需要刻意再找更多有獨立
     權威來源可查證的真實案例，不能只靠聊天過程中碰巧發現的素材。
   - **行為/誠實度回歸測試**（3 gunmen、JFK/Trump 這類正確答案本身不確定的案例）：
     不斷言 `expected_verdict`，改成斷言**行為正確性**——3 gunmen 驗證 agent 有沒有
     正確標注 `claim_match_status: partial/mismatched`（而不是不管範圍歧義直接下
     肯定結論）；JFK/Trump 驗證證據不足時 agent 有沒有誠實回報
     `verdict_basis: agent_absence_inference` 且 `authority_confirmed: false`（而不是
     驗證它有沒有「判對」，因為根本不知道什麼是對的）。
   - **純程式單元測試**：固定工具輸出，deterministic 地驗證來源分級、ClaimReview
     rating normalization、claim matching、provenance 聚合與 policy gate，跟案例的
     真實答案無關，這一層不受上述證據等級問題影響。
   - **可選的 live-search evaluation**：另外保留一組會真的上網查證的測試，用來
     衡量「現實搜尋結果隨時間漂移」對結論的影響，但這組結果只能拿來觀察，不能當
     CI 裡的 pass/fail 判準。

   每個 verdict-correctness 案例保存的欄位：`claim`、`observation_time`、`expected_verdict`、
   `acceptable_verdict_range`、`evidence_snapshot`、`forbidden_reasoning`、
   `claim/event matching 條件`、`authority_level`。
4. 修正「每個 agent 改動都有正式進步數字」這句話的強度——目前這幾輪的驗證（溫度、
   搜尋輪數、規則 6/7）都是**個案級的修正 + 有做過的工程測試**，還不是能估計泛化
   accuracy 的正式 benchmark。第 3 點的 regression suite 建起來之後，才有資格說
   「有正式的 before/after 比較」。

### 3.2 安全地接入 RAG

此工作包必須等 3.1 的 evidence provenance 與 regression guard 完成後才可接入 production；
實際執行階段對應第 7 節的 P1。

原本規劃的 `known_false_claims.py` 目前只是 schema + `record()`/`find_similar()`，
**還沒有任何 production 呼叫點**，所以嚴格說 Phase 1 第 1 項尚未完成，這點修正屬實。
但更重要的問題是設計本身：現在的 `record()` 只要收到 `credibility == "likely_false"`
就寫入，不分這個判斷是命中專業 ClaimReview 還是 agent 自己推論出來的。這會形成一個
自我強化迴圈：

```
Agent 誤判 false（例如又一次規則7式的沉默推論，但其實推論錯了）
   ↓
寫入 known_false_claims
   ↓
之後語意相近的新貼文命中 RAG 檢索
   ↓
Agent 看到「過去有相似案例被判 false」，更容易也判 false
   ↓
同一個誤判透過 RAG 被放大、擴散到更多不相關案例
```

**修正後的設計**：

1. 資料庫拆成兩個強制隔離的語意層（可使用兩張表，或使用一張具 DB constraint 的表
   加上只能走不同 repository API 的 `memory_tier`）：
   - `verified_false_claims`：只保存後端確認為 `likely_false`、
     `claim_match_status == matched`、至少一筆合格 evidence 明確 `refutes`，且
     `system_verified_verdict_basis` 通過高證據等級規則，或經人工複核確認的結果。
   - `agent_inferred_claims`：保存 agent 自己推論出來的 `likely_false`（例如
     `agent_absence_inference`），**只能當搜尋線索用，不得直接提高任何候選貼文的
     可信度或介入強度**。
2. `record()` 依後端驗證過的 evidence provenance、claim matching、stance 與人工複核
   狀態決定寫進哪一層，而不是現在這樣看到 `likely_false` 就無差別寫入同一張表。
3. **RAG 比對命中之後的正確用法**：
   - 提高「查證優先度」（原本的用途，保留）
   - 提供「可能相關的舊證據」給人看
   - 可用來擴展後續搜尋 query，但第一版**不把 `agent_inferred_claims` 當證據直接放進
     Agent prompt**；若未來加入，必須標為 untrusted lead，並要求重新核對日期、地點、
     人物與核心主張是否真的是同一事件
   - **不應該**直接拿舊案例的 verdict 去提高新貼文的「介入強度」——這是本節最重要的
     一條紅線。
4. 加入「不同日期／地點的 near-match」負例測試——確保相似度比對不會把同一句話套用
   在不同事件上（例如「3 gunmen」這個詞組本身可能同時出現在完全不相關的多起真實
   槍擊案報導裡）。
5. 避免同一個 claim 因為多個轉發版本被重複寫入好幾筆幾乎一樣的向量。

執行步驟（沿用原本鋪好的底層，但拆表）：
1. 啟動 Docker，重建 `db` 容器（已改成 `pgvector/pgvector:pg15`），跑
   `CREATE EXTENSION vector;`
2. 把 `known_false_claims` 拆成 `verified_false_claims` / `agent_inferred_claims`
   兩張表
3. `record()` 依後端驗證過的 provenance、claim matching、evidence stance、confidence
   與 human-review 條件分流；權威來源本身不等於「已確認為假」
4. `find_similar()` 的第一版呼叫端只用來重新排隊、擴展搜尋 query 與提示人工複核，
   不把 inferred memory 當成 Agent 證據，也不直接改變 policy 層的介入強度計算
5. 用「3 gunmen」謠言的多個轉發版本 + 至少一組「關鍵字相似但其實是不同事件」的
   負例，一起驗證比對邏輯

### 3.3 研究成果整理

1. 畫出四個訓練模型＋兩個 baseline＋Oracle 的多 budget（1/3/5/10/20/50/100）曲線圖，
   而不是只挑 @10 一個點
2. 額外顯示 event-level 的變異（附上各事件個別的 Capture@10，不是只給 macro 平均），
   因為前一輪已經發現小樣本事件（gurlitt、ebola-essien）跟大樣本事件的差異很大，
   只給 macro mean 會掩蓋這個訊號
3. 結果只描述為「目前資料與可觀察特徵下 RF 類模型的共同天花板」，不宣稱已逼近任務
   本身的理論極限
4. 整理 PHEME/Twitter15-16 的同口徑資料品質比較表——先解決 1.3 節提到的「中位數是否
   同口徑」問題，再放進正式比較

### 3.4 現代資料飛輪（不是「只需要時間」，需要先定義資料 contract）

即時雷達現在能累積貼文和查證結果，但如果目標是未來真的拿來訓練/校準模型，光是
「讓系統繼續跑」並不夠，還需要先定義：

- 10/20/30/40/50/60 分鐘的 snapshot（不是只存最終結果）
- 固定 horizon 的最終 cascade outcome（多久算「結束」要有明確定義）
- query 與 sampling provenance（這則貼文是被哪個關鍵字搜尋抓進來的，避免抽樣偏差
  被忽略）
- cluster identity 與去重紀錄
- agent evidence provenance（呼應 3.1 的 system-verified evidence provenance）
- 人工 adjudication（有沒有被人複核過、複核結果是什麼）
- `action_time`（介入實際發生時間）
- `action_strength`（降低曝光、限制轉發或其他模擬／實際行動的強度）
- `policy_version`（產生該行動的政策版本與設定）
- `selection_probability / propensity`（若可取得，用來描述候選被選中介入的機率）
- `verification_status_at_action`（採取行動當下可見的查核狀態，而非事後最終真相）
- `control_or_no_action_status`（未介入、保留對照或因預算未選中的狀態）
- `observation_completeness`（該時段的資料收集是否完整、是否可能漏抓）
- censored／deleted posts 標記（貼文後來被刪除/下架，本身也是一個信號，不能讓它
  悄悄從資料裡消失）
- **未被選入查證候選的貼文**（如果只存「系統挑中且查過的案例」，資料本身就已經有
  selection bias，之後沒辦法拿來訓練一個「決定該不該查證」的模型，因為訓練資料裡
  完全沒有「沒被選中」的負例）

沒有先把這份 contract 定義清楚，累積再久的資料也沒辦法直接拿來重新訓練 RF。
此外，雷達能觀察到的回覆、轉發與其他 cascade nodes 不等於平台實際曝光 impressions；
兩者必須在 schema、模型 target 與論文用語中保持區分。若介入後 cascade 變小，也不能
在沒有上述 action/control 欄位時直接解讀為介入造成的因果效果。

### 3.5「現代資料飛輪」必須拆成兩條速度不同的線，且排序模型不需要真假標籤

（使用者提出，第四輪修正）自建現代資料飛輪容易被誤會成「系統一直跑，兩種資料
就會一起自動變多」，實際上要拆成兩條完全不同、累積速度也完全不同的線：

- **擴散量資料（訓練排序/初篩模型用）**：target 是「這個 checkpoint 之後實際還
  多了幾個 cascade node」，這是客觀可觀察的事實，**不需要知道內容真假**，系統
  持續觀察就能自動、快速累積，不受下一點的限制。
- **真假標籤資料（RAG 的 `verified_false_claims` 層、回歸測試 fixture、未來若要
  訓練真假分類模型）**：只能靠命中權威來源（Google ClaimReview 等）或事後花時間
  做像本文件示範過的那種retrospective 查核（Isabela 查 USGS、Banksy 查多來源新聞、
  Ferguson「3 gunmen」查證範圍歧義）才算數，agent 自己即時判斷的 `likely_false`
  不能拿來當標籤——這條線會累積得很慢，量可能長期都不大，PHEME 自己當初的標籤也是
  研究者花時間事後查證才標上去的，沒有捷徑，這是任務本身的成本，不是本專案設計
  的缺陷。

**因此，排序/初篩模型的訓練資料，不需要等真假標籤那條慢速線，可以只依賴擴散量
這條快速線** —— 這也是下一點「純傳播圖結構排序」之所以可行的前提。

### 3.6 排序依據改用「純傳播圖結構」（不含文本），跨資料集訓練

（使用者提出，第四輪修正）與其在 PHEME／Twitter15-16 各自的文字＋結構混合特徵上
繼續調整，改用**純傳播圖結構特徵**（不使用文本內容）當排序依據，理由有三個，
其中第二個是本輪核對資料後才發現的：

1. **不需要真假標籤**：擴散量 target 不需要知道內容真假，見 3.5，讓训练资料
   不再受真假標籤這條慢速線卡住。
2. **意外繞開 Twitter15/16 的資料腐蝕問題**：本輪實際抽樣驗證——在
   `twitter15/hydrated_tweets_15.jsonl` 裡被標記 `deleted_or_unavailable` 的
   8 個 tweet_id，**全部 8 個仍完整出現在 `tree/` 目錄的傳播結構檔案裡**（使用者
   id、tweet id、相對時間戳）。原因是 43.8%～45.4% 的節點不可用率，量測的是
   **2026 年重新呼叫 API 抓取文字內容**時的失敗率；傳播結構（誰在何時回覆／轉推了
   誰）是 Ma et al. 在 2017 年原始收集時就已經用純 ID／時間戳記錄下來的，不需要
   事後補值，因此不受這個腐蝕問題影響。**1.3 節列出的 Twitter15/16 腐蝕率問題，
   只傷到文字內容，不傷到傳播結構**——這代表原本被視為「有嚴重資料品質問題」的
   Twitter15/16，如果只取用它的結構資料，這個問題就不成立。
3. **文字特徵在這個任務上本來貢獻就有限**：[PROJECT_RESULTS.md](PROJECT_RESULTS.md:44)
   的統一結果表裡，加了文字特徵的 `v5 Nested Text RF`（0.145）跟幾乎純結構的
   `Graph7`／`Current Early Size`（0.106～0.126）差距不大，`inner_validation_mean_crr`
   甚至常常是 baseline（不含文字）打平或贏過 `text_augmented`。拿掉文字，不是
   犧牲已驗證有效的訊號，是拿掉一個貢獻有限、卻帶來真假標籤依賴與跨資料集比較
   困難的部分。

**仍未解決、需要誠實列出的限制**：
- 不解決 PHEME 規模太小的問題——純結構可以讓 PHEME／Twitter15-16／未來自建的
  即時雷達結構資料合併訓練，但仍需要 §8.2 已經要求的跨資料集 normalization 與
  domain-shift evaluation，不能直接合併當同一份資料。
- 不解決「十幾年前的平台生態」這個時代問題——結構形狀是否真的比絕對規模數字更
  不受年代影響，目前只是假設，必須實際驗證（同樣是 §8.2 已規劃的 O1/O2 audit 與
  materialization 範圍），不能先當結論使用。

### 3.6.1 拿掉真假標籤需求後，可用的公開資料集候選（第四輪修正新增）

排序模型只需要「現代平台＋完整 cascade 結構＋精確時間戳」，不再需要真假標籤與
intervention target（見 1.5 的對應修正），可用資料集範圍因此變大，查到兩個具體
候選：

- **Weibo（DeepHawkes）**：[CaoQi92/DeepHawkes](https://github.com/CaoQi92/DeepHawkes) 公開的
  新浪微博轉發級聯資料，119,313 則訊息，2016-06-01 收集，任務設計本來就是
  「從早期觀察預測級聯最終規模」——跟本專案排序模型要的 target 完全一致，不是
  像 PHEME/Twitter15-16 那樣從misinformation 分類任務借用結構特徵。規模比 PHEME
  大、比 Twitter15/16 略新，純結構取用不受語言影響。**未解決**：一樣是約十年前
  的資料，現代演算法推播/機器人網路生態仍未涵蓋，需要跟其他來源一起做
  domain-shift 驗證，不能單獨當作現代基準。
- **Reddit（經 Arctic Shift）**：Pushshift 已於 2023 年被 Reddit 收回公開存取權，
  但 [Arctic Shift](https://www.xpoz.ai/blog/tutorials/how-to-get-historical-reddit-data-after-pushshift/)
  是免費、免授權、涵蓋 2005 年至今、速率限制每小時 12 萬次請求的繼承服務。Reddit
  留言串是樹狀結構，跟本專案既有的 cascade graph 概念相容，且**是持續更新到現在的
  真實資料**，不像上述資料集全部停在某個歷史時間點。**未解決**：Reddit 的討論
  動態（樓中樓留言、投票排序）跟 Twitter/Bluesky 的轉推機制不完全相同，遷移到
  Bluesky 上的可轉移性一樣需要 domain-shift 驗證，不能假設直接適用。

查過但不建議的選項：cascade-prediction 文獻常見的「Twitter hashtag cascade
（2012, 1,200萬推文）」比 Twitter15/16 更舊，沒有優勢；「APS 論文引用網路」是
學術引用的擴散（以年為單位），跟社群媒體的擴散機制（以分鐘/小時為單位）屬於
不同物理過程，不適合類比使用。

## 4. 可立即推進的兩項近期工作

本節只指出目前最適合先投入的兩個具體工作，不另行建立完整階段順序；整體先後關係仍以
第 7 節為準。

如果只能先做兩件事：

1. **`verdict_basis` 結構化 + 分層固定 evidence 回歸測試**（3.1 第 2、3 項；只有
   Isabela、Banksy 這類有獨立權威來源的案例可以斷言 expected_verdict，其餘案例改為
   行為/誠實度測試，見 3.1 第 3 項的修正）——這是後面 RAG、policy 依賴查證結果做任何
   事之前的地基
2. **四模型＋兩baseline＋Oracle 的多 budget 曲線圖**（3.3 第 1、2 項）——資料已齊全，
   對論文最直接有幫助

**RAG 排在這兩項之後**，不是因為它不值得做，是因為現在的 `record()` 設計如果不先拆
`verified_false_claims` / `agent_inferred_claims` 兩層，一旦接上生產流程，agent 的
一次誤判有機會透過 RAG 檢索被永久放大、擴散到更多不相關的候選貼文——這個風險必須在
接上任何 production caller 之前解決，不是接上之後再觀察。

## 5. 本次審查後的結論

原始 roadmap 的策略方向大致正確，維持不變的部分：不再無限制堆疊 RF 特徵、把 RF
定位成傳播風險初篩、研究重點轉向查核與決策層、把負面模型結果轉化為研究限制而非
失敗。

修正的部分：模型數量與部署模型的敘述錯誤、天花板結論的強度、40% 的算法歧義、
資料集腐蝕率的因果推論、X API 定價的過時數字、「沒有更好資料集」的絕對化敘述、
查證缺口的定義（從「沒有外部來源」改成「來源等級沒有結構化」）、以及最重要的——
RAG 設計本身若不拆分證據等級，會產生自我強化的錯誤迴圈，必須在接上生產流程之前
先解決。

## 6. 第二輪審查補充：正式實作前仍需收斂的設計

前述方向與優先順序整體成立，但若要將這份 roadmap 直接轉成 production 實作規格，仍有
幾個容易讓系統「看似具有證據治理，實際上仍由 Agent 自我認證」的缺口。以下不是要推翻
既有三層架構，而是把 RF、Agent、Policy 之間的責任邊界再收緊。

### 6.1 證據等級不能只由 Agent 自己宣告

P0 目前規劃要求 Agent 在最終 JSON 中輸出 `verdict_basis`、`authority_confirmed` 與
`human_reviewed`。其中 `verdict_basis` 可以讓 Agent 提出建議，但後兩者不應直接相信
Agent 的自我回報。

原因如下：

- Agent 可能把一般新聞誤認為官方第一手來源；
- 搜尋命中 ClaimReview，不代表 ClaimReview 查核的就是目前這個核心主張；
- Agent 可能因日期、地點或人物相似而錯配不同事件；
- `human_reviewed` 是系統工作流程中的客觀狀態，不是語言模型能自行決定的欄位。

建議將欄位拆成：

```text
agent_reported_verdict_basis     # Agent 對自己推理依據的說明
system_verified_verdict_basis    # 後端根據工具紀錄重新判定的依據
authority_confirmed              # 後端根據來源、claim matching 與 stance 計算
human_reviewed                   # 只能由人工操作流程更新
claim_match_status               # matched / partial / mismatched / unknown
evidence_source_ids              # 可回溯到實際工具結果的識別資訊
```

Policy 與 RAG 寫入只能使用 `system_verified_verdict_basis`、`authority_confirmed`、
`human_reviewed` 與 `claim_match_status`，不能直接依賴 Agent 自己宣告的證據等級。這樣才能
讓每次介入決策從資料庫回推到「哪個工具、哪個來源、哪一筆證據、如何匹配核心主張」。

### 6.2 命中權威來源不等於已確認為假

`official_primary_source` 或 `professional_fact_check` 只能說明來源類型，不能單獨決定
貼文真假。官方來源可能支持主張，也可能反駁主張；ClaimReview 也可能查核的是文字相似、
但日期或事件不同的另一個 claim。

因此，寫入 `verified_false_claims` 至少應同時符合：

```text
final_verdict == likely_false
AND claim_match_status == matched
AND 至少一筆合格 evidence 的 stance == refutes
AND system_verified_verdict_basis 屬於允許的高證據等級
AND 通過預先定義的 confidence 或 human-review 條件
```

Google ClaimReview 還需要做 rating normalization，並驗證被查核的 claim、日期、地點、人物
和目前候選相符。原本「verified_false_claims 幾乎不需要額外 confidence gate」的規劃太
寬鬆；來源權威性、主張是否匹配、證據方向與最終 verdict 必須分開驗證。

### 6.3 `agent_inferred_claims` 第一版不應直接成為 Agent 的證據

兩層記憶的方向是正確的，但最安全的第一版應將 `agent_inferred_claims` 限制於：

- 提高重新查證的排隊優先度；
- 在人工複核介面顯示可能相關的歷史案例；
- 作為搜尋 query 擴展的線索。

初期不建議將其直接放回 Agent prompt 當作判斷證據。即使 prompt 標示它是弱證據，模型
仍可能產生錨定效應，使過去一次錯誤推論影響新的 verdict。若後續確實需要提供給 Agent，
payload 必須明確標記：

> 此內容只是未經權威確認的歷史推論線索，不構成支持或反駁證據；必須重新核對事件、日期、
> 地點、人物與核心主張。

至於資料庫要使用兩張實體表，或是一張表加上受 DB constraint 保護的 `memory_tier`，可依
工程維護成本決定。研究安全真正要求的是：兩層資料在寫入條件、查詢 API 與下游用途上必須
強制分離，不能只靠呼叫端記得加 filter。

### 6.4 固定證據測試應拆成三層

P0 的固定 evidence regression suite 是正確方向，但「固定 evidence 就等於 deterministic
Agent test」仍需收斂。即使 evidence 不變，LLM 輸出仍可能因模型版本或推論非決定性而波動。

建議分成三層：

1. **純程式單元測試**：測試工具輸出解析、來源分級、ClaimReview rating normalization、
   claim/event matching、provenance 聚合，以及 policy gate；這一層必須 deterministic。
2. **Frozen-evidence Agent regression**：固定 claim、evidence snapshot、模型、prompt、解碼
   設定，驗證 verdict 落在允許範圍、理由沒有觸犯 `forbidden_reasoning`。這是回歸防線，
   不是正式 accuracy benchmark。
3. **Live-search evaluation**：真的重新搜尋，用來觀察搜尋索引、新聞內容與時間漂移；結果
   不應作為 CI 的固定 pass/fail 判準。

**第三輪修正（使用者提出）**：Isabela 地震、3 gunmen、JFK/Trump、Banksy 這四個案例的
`expected_verdict` 證據等級並不相同，不應一律當「已知正確答案」使用——Isabela（USGS
官方資料）、Banksy（多來源新聞交叉印證＋當事人否認）有獨立權威來源，可以斷言
expected_verdict；3 gunmen 的正確答案本身就有範圍歧義（本專案自己拿它示範這個問題），
JFK/Trump 則完全沒有外部機構查證過，`likely_false` 只是 agent 自己的沉默推論——用這
兩案例斷言 expected_verdict，等於拿 agent 自己的推論結果去驗證 agent 判得對不對，是
循環論證。3.1 第 3 項已改為：只有 Isabela、Banksy 進 verdict-correctness 回歸測試，
3 gunmen／JFK-Trump 改成驗證行為誠實度（`claim_match_status`、`verdict_basis`、
`authority_confirmed` 有沒有正確標注），不驗證判斷本身對不對。兩個 verdict-correctness
案例仍然太少，不足以宣稱 Agent 已具備可泛化的查核準確率，且應先經人工確認、允許合理
abstention（例如 `unverified`），避免為了通過測試迫使模型在證據不足時下結論——需要
再刻意尋找更多有獨立權威來源可查證的真實案例。

### 6.5 Oracle 0.305 與資料集描述性數字仍需補齊 provenance

`PROJECT_RESULTS.md` 的統一 benchmark 正式表沒有 Oracle row，但本文件使用了 Oracle
`0.305`。若要保留這個數字，必須另外保存並連結：

- 計算腳本與 run record；
- 使用的 2,373-thread candidate population；
- corrected target 與 K=10 排序方式；
- per-event denominator 與七事件 macro 聚合方式；
- tie-break 規則與輸出表。

在 provenance 補齊前，這個數字應標為「事後診斷值」，不能與已有正式 artifact 的模型結果
視為同等證據等級。

此外，1.1 的標題「在同一資料／特徵天花板附近收斂」仍容易讓人誤以為四個模型使用相同
特徵。較準確的標題是：

> 四個訓練模型在共同資料與觀測條件下呈現有限增量

PHEME、Twitter15/16 與 MuMiN 的規模、腐蝕率和資料結構數字也應附上各自的產生腳本或
外部來源。只要計算管線尚未統一，表格就應標註為「由不同資料管線產生的描述性初步統計，
尚不可直接比較」。MuMiN 是否真的保有可重建的完整 reply tree、時間戳與 intervention
target，必須先做 license、accessibility 與 schema audit，不能只依規模判斷是否適用。

### 6.6 RAG 記憶需要比目前 schema 更完整的可稽核資訊

目前 `KnownFalseClaim` 主要保存 claim text、embedding、credibility、confidence、summary 與
時間。若要成為可用於查核的證據記憶，還需要至少保存：

```text
canonical_claim
event_time / observation_time
evidence_source_urls 或 evidence record IDs
evidence_snapshot_hash
system_verified_verdict_basis
claim_match_status
model_name / prompt_hash / schema_version
review_status / reviewer / reviewed_at
valid_until / revoked_at
deduplication_key
```

原因是網頁可能更新或消失、查核結果可能被修正、相同謠言也可能在不同日期與地點重新出現。
只保存自由文字 summary 與向量，無法證明當時依據了什麼，也無法安全地撤銷一筆後來發現有
問題的記憶。

另外，現有 `record()` 捕捉所有例外後只 rollback、沒有對外留下可觀察的錯誤。查證主流程
可以在 RAG 寫入失敗時繼續，但必須留下結構化 log、health metric 或 audit status；否則系統
可能長期沒有累積任何記憶，介面與研究紀錄卻看不出來。

### 6.7 現代資料飛輪還需記錄介入造成的 outcome 混雜

如果未來系統真的執行降低曝光或限制轉發，觀察到的 cascade 變小可能有兩個原因：

- 該 thread 原本就不會繼續擴散；
- 系統介入成功，改變了後續傳播。

若不記錄處置條件，後續模型會把「介入後被壓低的 outcome」當成自然結果，形成 treatment
confounding。3.4 的資料 contract 因此還應加入：

```text
action_time
action_strength
policy_version
selection_probability / propensity（若可取得）
verification_status_at_action
control_or_no_action_status
observation_completeness
```

也必須區分「可觀察的回覆／轉發節點」與「平台實際曝光 impressions」。目前 Bluesky 雷達
能觀察 cascade，不代表能知道每篇貼文實際被多少人看到；論文與模型 target 都不應把兩者
直接視為相同。

## 7. 建議的收斂後執行順序

綜合前述風險，建議將下一階段收斂為：

```text
P0-A  後端依工具紀錄計算 system-verified evidence provenance
  ↓
P0-B  新增向後相容欄位；既有報告預設為 unknown，不偽造回填
  ↓
P0-C  建立來源分級、claim matching、rating normalization 與 policy gate 單元測試
  ↓
P0-D  建立 frozen-evidence Agent regression suite
  ↓
P1    建立分級 claim memory；第一版只服務查證排序與人工介面
  ↓
P2    完成多 budget、event-level 變異與資料品質圖表
  ↓
P3    建立包含處置行動、觀測完整度與未選中候選的現代資料 contract
```

P2 的圖表可以和 P0 平行進行，因為統一 benchmark 的既有資料已足夠，不必等待 RAG。

這次第二輪審查的核心判斷是：roadmap 的研究方向已經正確，剩下的關鍵不是再新增一個模型，
而是避免把「Agent 說自己找到權威證據」直接等同於「系統已確認證據權威且與核心主張相符」。
只要證據 provenance 改為由後端從工具紀錄重建、RAG 寫入條件收緊，這份 roadmap 就可以
作為下一階段正式實作與論文章節安排的依據。

## 8. 第三輪補充：從風險排序走向預算限制下的邊際介入

這一節整理「權威資料初始化、Agent 即時累積、圖結構訊號、剩餘可預防影響與邊際介入
價值」的新構想。它是建立在第 7 節 P0／P1 完成後的**研究支線**，不是取代第 7 節的第二套
P0～P3，也不應在 evidence provenance 尚未完成前直接接入 production policy。

### 8.1 可保留的方向：兩層記憶、共同檢索、不同信任用途

系統需要同時處理兩種時間尺度：權威查核平台適合提供歷史上已被正式查核的 claim，但通常
無法立即涵蓋剛發生的新事件；Verification Agent 能處理新事件，但其結果不具有與專業查核
相同的證據等級。因此可以：

1. 先從 Google Fact Check API 的 ClaimReview 建立初始權威記憶；匯入時保存 publisher、
   review URL、review date、原始 textual rating、正規化 verdict、被查核 claim 與
   claim-matching provenance，不能只保存文字與 embedding。
2. 將 live Agent 新產生的結果持續寫入同一套 claim-memory 查詢服務，但在儲存層強制分成
   `verified` 與 `agent_inferred` 兩個 trust tier。
3. 查詢時可以同時檢索兩層，但用途與權重不同：
   - verified 命中可提高查證優先度，也可在通過嚴格 exact-claim reuse gate 後重用既有證據；
   - inferred 命中只作為查證排隊、query expansion 與人工複核線索，不直接進 Agent prompt，
     也不提高介入強度。

這裡不應直接使用「`predicted_impact × 語意先驗 × 語言紅旗 × 帳號分數 × 圖分數`」這種
未校準的連乘公式。各分量的尺度、缺失值與誤差不同，任一接近零的項目都可能不合理地把
其他訊號全部消掉。第一版應先把這些量保存成可稽核的獨立 features，再透過預先註冊的
離線比較決定採用簡單規則、正規化加權或可訓練模型，不能先指定一組看似合理的乘法係數。

### 8.2 圖結構特徵值得加入，但不能把結構形狀直接等同於協同行為

**與 3.6 的關係**：3.6（第四輪修正）進一步把方向收斂成「排序模型改以純傳播圖結構
取代文字＋結構混合特徵」，理由是不需要真假標籤、且意外繞開 Twitter15/16 的資料
腐蝕問題（該問題只傷文字，不傷結構，已用真實抽樣驗證）。本節列出的圖特徵清單，
就是 3.6 那個純結構模型實際會用到的特徵候選；本節原先「圖特徵疊加在文字模型上」
的定位已被 3.6 取代，改為「圖特徵本身就是排序依據」。

可在每個 checkpoint 以 cutoff-safe snapshot 計算：

- observed fan-out、最大 fan-out 與分支分布；
- observed depth、breadth、depth/breadth ratio；
- 節點抵達時間集中度與 burstiness；
- observed structural virality（平均節點對距離）；
- 重複使用者、跨 thread 使用者重疊與短時間同步行為；
- 在**目前已觀察圖**上的 community／bridge／betweenness 描述。

[Goel et al. 的 structural virality](https://5harad.com/papers/twiral.pdf)用來區分大型 broadcast
與多代 person-to-person diffusion 的形狀；它本身沒有證明星狀就是 bot／協同操作，鏈狀就
是真實有機討論。因此圖形狀只能當「傳播機制特徵」，協同行為還需要時間同步、使用者重疊、
帳號特性或其他獨立訊號支持。

同樣地，observed betweenness 只能描述目前圖上的橋接位置，不能直接宣稱某帳號「即將」
把內容帶進新社群。要預測未來 bridge event，必須另建 cutoff-safe 預測 target 與模型；
否則只能寫成待驗證假設。

PHEME、Twitter15/16 與 MuMiN 也不能直接合併後學一個共同絕對門檻。三者的平台、取樣、
thread 定義、缺失節點與規模差異會改變圖統計。若要跨資料集校準，至少需要：

- 先統一 node／edge／root／reply 的定義；
- 將特徵轉成 within-event 或 within-dataset percentile／normalized measure；
- 分別報告各資料集結果；
- 使用 leave-one-dataset-out 或 domain-shift evaluation 檢查可轉移性。

因此「圖形狀相對不受年代與規模影響」目前只能列為實驗假設，不能先寫成已成立結論。

### 8.3 極高相似度不能單獨跳過查證

可以跳過的是「本輪重新呼叫 LLM／網路搜尋」，不是跳過 claim matching 或永久停止查證。
只有 verified tier 的舊 claim 同時符合下列條件，才能暫時重用既有 verdict：

```text
canonical-claim similarity 通過嚴格門檻
AND 日期／地點／人物／數值／事件類型沒有 hard conflict
AND claim_match_status == matched
AND 舊證據仍在有效期限內、未被撤銷
AND 舊 verdict 與 refuting evidence provenance 可完整回溯
```

這個狀態應命名為 `verified_match_reuse`，以區分「重新完成一次查證」。只要是 partial match、
缺少重要實體、跨日期事件、證據過期或來源失效，就必須進入正常 Verification Agent 流程。
`agent_inferred` tier 無論相似度多高，都不能使用這條捷徑。

### 8.4 正確的最佳化目標：最大化預估邊際可預防影響

全自動介入的目標可以明確寫成：在介入 budget `K` 下選擇集合 `S`，使介入後仍發生的
預估有害傳播最小，等價於最大化被預防的有害傳播：

\[
S^* = \arg\max_{|S|\le K}\widehat{F}(S,T)
\]

其中 `F(S,T)` 應衡量在時間 `T` 介入集合 `S` 的**集合效益**，而不是把每個 thread 的
個別 RF 分數直接相加。實務上的第一版可定義為研究用代理函數：

\[
\widehat{\Delta}(i\mid S,T)
= \widehat{PI}_{remaining,i}(T)
\times Risk_i(T)
\times NovelReach_i(S,T)
\times \widehat{ActionEffect}_i(T)
\]

- `PI_remaining_hat`：在目前 checkpoint 之後仍可能發生的 impact；現有 cumulative RF 的
  即時 future-impact prediction 可作第一版估計。
- `Risk`：由後端驗證後的證據狀態與介入倫理規則決定，不由 Agent 自行宣告。
- `NovelReach`：候選相對於已選集合仍可能新增觸及的代理值。
- `ActionEffect_hat`：特定 action strength 對傳播的預估效果；目前沒有真實因果資料時只能
  明確設為 proxy，不能宣稱為已知介入效果。

`pheme_replay.realized_blocked_nodes` 使用的是事後完整 cascade，只能用來離線評分或展示
「如果當時介入，歷史樹上哪些未來節點會被切斷」，不能作為 live selection 的輸入，也不能
被描述成已經存在的即時剩餘影響估計器。

**第三輪修正（使用者提出，比原稿更根本）**：`NovelReach` 不能把語意相似度當成受眾重疊
的代理指標，**連弱備援訊號都不行**——這不只是不夠精確，而是在系統最需要正確判斷的
情境下，方向可能整個相反。協同散布假訊息的理性策略，正是用幾乎相同的文字同時貼到互不
相干的社群以擴大觸及——這種情況語意相似度會判成「高度重複、該打折扣」，但真實受眾
重疊度是刻意設計成低的；反過來，用詞完全不同但被同一批高影響力帳號轉發的貼文，可能
觸及幾乎同一群人，語意相似度卻會判成「不像、不該打折」。語意相似度量的是「內容是否
同一件事」，`NovelReach` 需要的是「觸及的人是否同一群人」，兩者是不同維度，用前者
代替後者，會讓系統在最該優先介入的協同操作場景裡系統性失靈。

`NovelReach` 只能用**直接觀察到的**重疊訊號：

- 早期跨-thread 使用者重疊（兩則候選貼文的回覆／轉發者名單是否有重疊帳號）；
- observed graph community 重疊；
- 作者或轉發者集合重疊。

語意相似度只保留在階段 1 的 claim-identity 去重（判斷是否為同一個 claim、不需要重複
真的查證），**不進入 `NovelReach` 的計算**。若上述直接觀察訊號目前系統中都還不存在
（需由 O1 確認），誠實的預設做法是**不打折扣**（視為零已知重疊、候選價值不打折），
不能為了有個數字可用就套用語意相似度這個方向可能錯誤的代理指標。

### 8.5 目前不能宣稱具有 `(1-1/e)` 理論保證

[Kempe、Kleinberg 與 Tardos](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf)的約
63% greedy approximation guarantee，建立在明確的 Independent Cascade／Linear Threshold
等擴散模型，以及目標函數可證明為 monotone submodular 的條件上。
[Budak、Agrawal 與 El Abbadi](https://www.cs.ucsb.edu/sites/default/files/documents/2010-02.pdf)
研究的則是 competing campaigns：選擇使用者散播好的 campaign，以降低採用 bad campaign
的人數。它不是直接選擇若干獨立 thread 做曝光限制的同一個問題。

本專案目前的 `RF predicted impact × risk × overlap proxy` 沒有已知擴散機率模型，也尚未
證明 monotonicity 或 submodularity；語意折扣更不是受眾重疊的數學證明。因此可以將 greedy
marginal selection 描述成「受 influence-blocking 啟發的預算式邊際選擇 heuristic」，但
不能引用 `(1-1/e)` 作為本系統的保證。

若未來要主張理論保證，必須先明確定義：

- social graph 與可介入的 node／thread 單位；
- diffusion model 與 edge activation probabilities；
- action 如何移除、延遲或降低 activation；
- 集合效益 `F(S,T)`；
- 並實際證明該函數在這些設定下 monotone、submodular。

Online 情境也不能只把離線 greedy 改成「超過門檻就選」便宣稱仍有相同保證。Online
submodular 方法需要額外的抵達順序、隨機性、可交換或 secretary／streaming 等假設，必須
另外定義與驗證。

### 8.6 建議的實驗順序

這條研究支線應依下列順序進行，不直接修改正式 policy：

```text
O1  Read-only audit：盤點各 checkpoint 可合法取得的圖、使用者與 cluster features
 ↓
O2  建立 cutoff-safe feature materialization，驗證相同輸入可決定性重建
 ↓
O3  建立 offline greedy marginal heuristic（只用 cutoff 前資訊做選擇）
 ↓
O4  用 PHEME 完整未來樹只做事後評分，比較 RF Top-K 與 marginal heuristic
 ↓
O5  做 ablation：語意 redundancy／使用者重疊／community overlap 各自的增量
 ↓
O6  通過 event-separated validation 後，才在 live radar 以 shadow mode 輸出建議
 ↓
O7  累積 action/control 資料後，才估計 ActionEffect 並研究因果介入政策
```

主要評估不應只看 Top-1。至少同時報告：固定 budget 下的 false preventable-impact capture、
unique claim coverage、重複介入率、true／unverified 誤介入、每次介入的邊際收益，以及不同
event 的 macro 與 per-event 結果。只有 empirical comparison 明確改善，才能將此支線整合
進 `intervention_policy.select_for_checkpoint`；在此之前保留現有 policy 作為 baseline。

## 9. 第四輪補充：管線分工與多 agent 自動化設計

使用者提出的整體方向：加強現有 pipeline、評估多 agent 分工，目標以全自動為主。
本節的核心原則沿用第 6.1 節已確立的教訓——**不是加更多 agent 就是進步，只在有
證據支持的地方自動化，任務拆到對的粒度、用對的工具，不確定時要讓系統誠實承認
不確定，不能為了「看起來更自動」硬湊一個結論**。

### 9.1 現況：單一 agent 同時做搜尋、判斷證據夠不夠、下結論三件事

現在的查證 agent 一次做完「搜尋 → 判斷證據是否足夠 → 下結論」，這正是 §6.1
指出「agent 自我認證」問題的根源之一——agent 自己講的推理依據不可靠，因為
搜尋、判斷、下結論全部糾纏在同一次推理裡，無法個別稽核，也無法乾淨地凍結
證據做回歸測試。

### 9.2 建議的管線分工（依任務粒度分階段，不是依「多加 agent」分階段）

```text
貼文進來
  ↓
① 結構化初篩（純規則／純傳播圖結構特徵，不用 LLM）—— 3.6 的純結構排序模型
  ↓
② RAG 查詢（純向量比對，不用 LLM）—— 比對 verified_false_claims／
   agent_inferred_claims 兩層知識庫（3.2）
  ↓ 命中 verified 層且通過 claim matching → 直接沿用既有結論（verified_match_reuse，
    見 8.3），跳過 ③④⑤
  ↓ 否則往下走
③ Claim 拆解（可用小型／快速模型或規則，不佔用大模型算力）—— 把貼文拆成
   「誰、做了什麼、何時何地、具體數字」等原子事實，屬機械性任務
  ↓
④ 證據蒐集（沿用既有 search_web／search_fact_checks／fetch_article）—— 只找
   證據，不下結論
  ↓
⑤ 判斷 agent（現有的大模型，僅此一步使用）—— 只根據④已蒐集到的證據推理
   verdict，不再自行搜尋
  ↓
⑥ Claim matching 檢查（規則為主，重用既有 event_clustering_v2／
   extract_event_signature）—— 獨立驗證⑤引用的證據是否真的對應同一事件，
   不能只靠語意相似（呼應 6.2、Ferguson「3 gunmen」案例的教訓）
  ↓
⑦ 決策政策（維持現有純規則引擎 intervention_policy，不 LLM 化）
```

### 9.3 拆分的理由，不是為了看起來更先進

1. **讓已規劃的 evidence provenance／回歸測試方法論真正對得上架構**：現在
   「邊搜邊想」的單一 agent 很難真正凍結證據做 frozen-evidence regression（3.1）；
   拆開後，⑤的輸入是④已固定的證據集合，才能乾淨地做「同一組證據、換 prompt
   版本，verdict 會不會跑掉」的回歸測試，而不是勉強湊出一個測試方法論。
2. **把便宜的任務交給便宜的資源**：③、⑥屬結構化／比對工作，不需要動用大模型的
   深度推理能力，可用小模型或純規則完成，把大模型算力保留給真正需要推理的⑤。

### 9.4 多 agent 平行化的實際限制：不能忽視共用 GPU 這個基礎設施瓶頸

「多 agent」容易被誤解成「可以平行跑更快」，但目前只有一個共用的 Ollama 服務
（RunPod 上單一 32B 模型），本文件前面才剛把並發數從 2 降到 1，因為同時多個
請求會互搶 GPU、容易逾時（見本專案這學期已修復的併發查證卡頓問題）。**如果只是
把同一個大模型複製成多個「agent」平行跑，會直接重現已修好的問題，不是進步。**

真正能從多 agent 拿到自動化效益的方式是**異質分工**，不是**同質平行**：③、⑥
用小模型或純規則、不佔用大模型資源，只有⑤排隊使用大模型——「多 agent」帶來的
效益是吞吐量提升（稀缺算力只花在真正需要推理的步驟），不是多個大模型互搶同一份
資源。

### 9.5 明確保持「不自動化」的部分

- **⑦決策政策**：維持純規則引擎，不换成 LLM 判斷——現有 hard/soft/none 邏輯
  確定性、可稽核，沒有證據顯示 LLM 判斷會更好，換成 LLM 只會讓決策更難解釋、
  更難寫進論文的可重現章節。
- **②③④⑥任一步驟顯示不確定性偏高時**：不強迫系統生出確定答案——沿用本文件
  已建立的原則（8.4 的 `NovelReach` 沒把握就不打折扣、8.5 承認模型分歧不代表
  可信度）。全自動系統的目標不是「永遠給出答案」，是「知道自己什麼時候不該
  給出確定答案」，這比任何一種硬湊出的自動判斷更符合這份文件的整體立場。

## 10. 第五輪補充：Bluesky 真實資料實測（2026-11-10 至 11-14）

本節是本文件第一次出現**在非活動篩選資料上的預先登記實測**。所有數字都來自
`analysis/` 下帶 `run_record.json` 的執行目錄；結論強度已對齊證據等級。第 3.4–3.6、7 節
談到的「資料來源」與「執行順序」，若與本節衝突，以本節為準。

### 10.1 為什麼要做：PHEME 與 Twitter15/16 都是依活動量挑出來的
- PHEME 的串是依轉發門檻挑選（Zubiaga 2016），所以它是真實貼文分布的**上尾**；Bluesky 關鍵字取樣的典型池內串，30 分鐘時只有約 1 則他人回覆。
- 因此「在 PHEME 上訓練得到的傳播量排序」不能直接外推到真實分布；比較 PHEME 與真實資料時，平台、年代、取樣方式、計數定義都混在一起。
- 結論：要回答「早期傳播預測在真實分布上有沒有用」，必須用不是依活動量挑選的資料。

### 10.2 蒐集的資料（皆為公開 API，無金鑰；不含貼文文字與帳號名稱）
| 批次 | 貼文日期 (UTC) | 原文數 | 有回覆的串 | 用途 |
|---|---|---:|---:|---|
| `runs/20261109_163551_full` | 10-18 至 11-01 | 40,179 | 7,450 | 開發期（10-18..10-28）與緩衝 |
| `runs/20261109_093759_full` | 11-01 至 11-06 | 16,720 | 3,122 | 探索性測試（10-31..11-06；已被檢視過，非乾淨留出集） |
| `runs/20261110_095950_full` | 10-04 至 10-15 | 32,406 | 6,182 | 倒著確認集（未看過） |
| `runs/20261114_165934_full` | 11-07 至 11-09 | 7,002 | 1,377 | 往前確認集（模型凍結後才收集） |

- 收集器：`analysis/bluesky_cascade_collection/collect.py`；作者以加鹽雜湊儲存；run 內含公開 AT-URI，**不可分享**。
- Bluesky 的回覆有 29.4% 是原文作者自己回覆（PHEME 為 3.0%），所以傳播量一律改定義為「他人（非原文作者）回覆」。
- 轉發／按讚沒有時間戳，只能當事後標籤，不能當早期特徵。
- 限制：單一平台、關鍵字取樣（12 個關鍵字）、**沒有真假標籤**。

### 10.3 免訓練基準（尺）
池 = 截止點 T 時已有 ≥1 則他人回覆的原文。以 `unique_repliers`（不重複他人回覆者數）排序的 skill 在未見日期約 0.72–0.77（T=30）；`first_reply_speed` 與深度幾乎沒有資訊。紀錄：`20261110_005435_bluesky_trainingfree_baseline_eval_v2_combined`。

### 10.4 第一階段（誰會起飛）：模型勝過規則，兩個方向的確認都成立
任務：只用發文後 T 分鐘內的回覆結構與時間（8 個特徵：他人回覆數、自我回覆數、不重複他人回覆者、最大／平均深度、第一則他人回覆偏移、近 5 分鐘他人回覆數、原文字數），預測最終他人回覆 ≥10（Y10，盛行率約 1%）。隨機森林分類器，只用開發期訓練，超參數與最好規則只在開發期選。協定：`TRAINING_PROTOCOL.md` 附錄 A；凍結：`analysis/20261110_confirmatory_h2_frozen/FREEZE.json`（含檔案雜湊）。

| 評估集 | 正例 | T=10 模型 | T=10 規則 | PR-AUC 差異（95% 區間） | 判定 |
|---|---:|---:|---:|---:|---|
| 探索性測試 10-31..11-06 | 213 | 0.549 | 0.505 | +0.044（+0.025, +0.066） | 候選改進（探索性） |
| **倒著確認 10-04..10-15** | 323 | 0.470 | 0.398 | **+0.072（+0.055, +0.087）** | 依凍結規則成立 |
| **往前確認 11-07..11-09** | 69 | 0.483 | 0.400 | **+0.083（+0.038, +0.128）** | 依凍結規則成立 |

- T=30、60 方向相同（倒著 +0.051／+0.042；往前 +0.068／+0.063），只報告、不作結論。
- 優勢來自回覆結構與時間：拿掉 `text_chars` 後增益幾乎不變（Y10 的差異約 0），只用 `text_chars` 的 PR-AUC 約等於盛行率（`20261110_134208_bluesky_takeoff_h2_textchars_ablation`）。事後檢查（排除 T 時已有 ≥10 則者；改用「T 之後新增 ≥5」的純未來標籤）結論方向不變。
- 只用在**前 20 名**的指標上差距很小（倒著 T=10：模型 0.411 對規則 0.397）；優勢主要是整體排序品質，不是頂端抓得更準。
- 判定規則（凍結）：T=10、Y10，模型減最好規則的 PR-AUC，作者 cluster 配對 bootstrap 95% 下界 >0 且正例 ≥50。

### 10.5 第二階段（起飛後誰最大）：無法確定
池內（T=10 時已有他人回覆）排序「T 之後還會新增多少他人回覆」。協定：`TRAINING_PROTOCOL_APPENDIX_B_H3.md`（在看結果前凍結）；紀錄：`analysis/20261110_bluesky_h3_pool_size_ranking/full_20261110_110111`。
- 倒著集 T=10，K10pct skill：模型 0.613、`unique_repliers` 0.561，差異 +0.030（−0.058, +0.109），包含 0 → **無改善／無法確定**；探索性測試期方向相反（−0.014，區間含 0）；T=30 差異約 0。
- 理智檢查（單特徵 `early_size` 的 RF = 規則 `early_size`）通過。
- 結論：沒有證據需要複雜模型來排大小；不重複回覆者數的簡單規則已接近此資料上能做到的水準。池小（T=10 約每日 100 串）且目標重尾，區間寬是預期中的限制。

### 10.6 對先前說法的修正
- 先前「篩選（誰會擴散）已經做完，用簡單規則就好」**不準確**：起飛預測上，訓練模型在兩個未看過的資料集上都穩定勝過規則（中等幅度）；只有「起飛後排大小」這段規則已足夠。
- 「把 PHEME 當作正例（起飛類）來訓練」不建議：PHEME 與 Bluesky 的差異包含平台、年代與選樣，模型會學到資料集身分而非起飛。若要用，只能作**純評估**（凍結 PHEME 模型後在 Bluesky 上評估，且需特徵對應），這項尚未做。

### 10.7 目前證據等級與尚未證明的事
| 主張 | 證據等級 |
|---|---|
| 早期回覆結構可預測起飛，且優於 `unique_repliers` | 兩個未見資料集皆成立（中等幅度、僅 Bluesky、僅傳播量） |
| 起飛後可用複雜模型排大小 | **未證實**（無法確定） |
| 能依「假訊息程度」排序 | **沒有證據**：結構特徵與真假的關聯很弱（先前診斷最大 |Cliff's δ| 約 0.167）；Bluesky 沒有真假標籤；查證 P(假) 的校準從未量測 |
| 合併 P(起飛)×大小×P(假) 的完整排序 | **未做**；P(假) 校準是最大的未驗證缺口 |

### 10.8 後續（依價值排序）
1. **量測 P(假) 的校準**：在有標註的主張集上測查證輸出的機率是否可信；在此之前，不應宣稱完整排序有效。
2. 第三階段合併實驗：P(起飛) × 預期大小 ×（校準後）P(假)，先在 PHEME 的理想 oracle 條件上估上限，再看 Bluesky。
3. 往前確認集目前只有 3 天、69 個正例；可再收集更多日期，用同一凍結模型重評（須新增增補）。
4. （選項）凍結 PHEME 模型僅評估於 Bluesky，量測跨資料集遷移。
5. 工程：`backend/` 的 pgvector／RAG 草稿（`schema.py`、`requirements.txt`、`docker-compose.yml`、`known_false_claims.py`）仍未提交，需決定去留。

### 10.9 重現與稽核索引
- 收集器與協定：`analysis/bluesky_cascade_collection/`（`collect.py`、`TRAINING_PROTOCOL.md`、`TRAINING_PROTOCOL_APPENDIX_B_H3.md`、`runs/`）。
- 第一階段：`analysis/20261110_123946_bluesky_takeoff_h2/`、`..._124333_..._posthoc_v2/`、`..._134208_..._textchars_ablation/`；確認性：`analysis/20261110_confirmatory_h2_frozen/`（`confirm_backward_20261110_101430`、`confirm_forward_20261114_170254`；`smoke_*` 為管線檢查，**不是證據**）。
- 診斷與稽核：`analysis/20261109_*`、`analysis/20261110_0*`（PHEME 與 Bluesky 形狀比較、蒐集品質稽核、非自我回覆重新定義等）；失敗的執行目錄皆保留。
- 所有執行都有新輸出目錄與 `run_record.json`；本節沒有任何結果來自已覆寫的目錄。
