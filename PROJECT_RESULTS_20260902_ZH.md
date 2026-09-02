# 假訊息早期干預研究成果整理

更新日期：2026-09-02

## 一、這個研究在做什麼

本研究的目標不是只判斷一則訊息是不是謠言，而是回答更接近實際干預的
問題：

> 當我們只能看到 thread 發布後前 30 分鐘的資訊時，應該優先介入哪些
> thread，才能阻擋最多後續傳播節點？

每個模型會為候選 thread 產生一個介入優先分數，再依分數由高到低排序。
使用者可以自由決定 Budget，例如只介入 1、5、10、20 或 50 個 thread，
不需要為每個 Budget 重新訓練模型。

Oracle 代表事後已經知道完整未來的理想排序：未來可阻擋節點越多的 thread
排得越前面。模型不可能在推論時看到這些未來資訊；Oracle 只作為訓練
outcome 與評估上限。

## 二、主要評估方式

- `Reduction@K`：模型選擇 K 個 thread 後，實際阻擋的節點占該事件全部
  future impact 的比例。
- `Oracle efficiency@K`：模型 Top-K 阻擋量除以 Oracle Top-K 阻擋量。
  1.0 代表與 Oracle 相同。
- `MinThreads@reduction`：若要阻擋 10%、20%、40% 或 60% 的未來影響，
  模型需要介入多少 thread。
- Spearman correlation：模型完整排序與真實 future impact 排序的一致程度。

所有正式 PHEME 比較都維持 rumour-only candidate pool、strict 30-minute
observation、event-separated LOEO，以及相同 label 和介入效益定義。

## 三、目前最重要的 PHEME 結果

### 1. v5 text RF：目前最強的完整 policy 參考

模型使用前 30 分鐘可見的 temporal／structure features，加上 source 與
已觀測 replies 的 RoBERTa embedding。文字 PCA、scaler 與 RF 都只在當前
train fold 擬合。

- Model：RandomForestRegressor
- Target：`log1p(preventable_impact)`
- Protocol：nested event-separated LOEO
- Eligible-event macro `Reduction@10 = 0.1467`

換句話說，每個事件介入模型排序最前面的 10 個 thread，平均可阻擋約
14.67% 的總 future impact。這仍是目前完整 PHEME policy 最強且最穩定的
參考結果。

### 2. Active／quiet 雙專家 RF

研究中發現，單一 global model 容易偏好前期已經活躍的 thread，可能漏掉
前期安靜、之後才突然擴散的 thread。因此將候選分成：

- active：最近仍有活動；
- quiet：距離最後活動時間較長。

Gate 使用 outer-train fold 的 recency 第 60 percentile，不讀 held-out event
outcome。兩群各自訓練 RF，再以 inner event-OOF prediction 校準兩位專家的
分數尺度。

| Budget | Global RF efficiency |     未校準雙專家 |       校準雙專家 |
| -----: | -------------------: | ---------------: | ---------------: |
|      1 |     **0.3435** |           0.2465 |           0.2465 |
|      3 |     **0.4214** |           0.3920 |           0.3920 |
|      5 |               0.4063 | **0.4948** | **0.4948** |
|     10 |               0.4428 |           0.4685 | **0.4800** |
|     20 |               0.4814 |           0.5006 | **0.5063** |
|     50 |               0.5994 | **0.6143** |           0.6135 |
|    100 |     **0.7209** |           0.7162 |           0.7089 |

在 Budget 10，校準雙專家的 mean reduction 是 `0.1461`，高於同一 paired
protocol 的 global RF `0.1319`，但仍略低於原始 v5 的 `0.1467`。

這個模型的重要價值是：它確實改善了 K=5--50 的中等資源區間，但付出的
代價是 K=1、3 的 extreme-head 排序變差。因此它是有價值的 candidate，
尚不能直接取代 v5 baseline。

### 3. Graph7 discourse/context RF

加入早期 discourse response 與 event-context 特徵後，在正式 nested LOEO
得到 `CRR@10 = 0.1385`。它證明早期互動語境有訊號，但仍沒有超過 v5
text RF。

## 四、最新版：quiet thread 的 255 組特徵組合搜尋

為了改善 quiet thread，最新版把所有合法 30 分鐘特徵分成八群：

1. activity level
2. temporal dynamics
3. topology
4. text surface
5. sentiment
6. immutable account age
7. semantic summary
8. source/reply embedding

接著列舉全部 255 種非空組合。每個 outer LOEO fold 都只使用 inner events
選擇組合，再對 outer held-out event 評估一次。RF 設定為 300 trees、depth
8、minimum leaf 3、`max_features=0.8`、seed 42。

這裡的結果只針對 quiet candidate pool，不是完整 policy reduction：

| Quiet Budget |      原 quiet RF |      特徵組合 RF |    差異 |
| -----------: | ---------------: | ---------------: | ------: |
|            1 |           0.0871 | **0.2582** | +0.1711 |
|            3 |           0.1919 | **0.2731** | +0.0812 |
|            5 | **0.2853** |           0.2807 | -0.0045 |
|           10 | **0.3556** |           0.3236 | -0.0320 |
|           20 | **0.5090** |           0.4164 | -0.0926 |
|           50 | **0.6353** |           0.6131 | -0.0222 |
|          100 | **0.8512** |           0.8337 | -0.0174 |

完整排序 Spearman 也從 `0.3171` 降到 `0.2995`。

因此它找到了一些可能有助於 Top-1／Top-3 的訊號，但沒有形成更好的完整
排序。Top-1 的大幅提升又高度受到 `putinmissing` 事件剛好命中 Oracle
第一名影響，不能解讀為穩定突破。

各事件選到的特徵組合不同。選取較頻繁的是 topology 與 semantic summary
（各 5/7 folds），account age 則是 0/7。這表示早期結構與語意可能含有
弱訊號，但沒有一組可以穩定跨事件泛化。

## 五、Twitter15／Twitter16 的獨立結果

Twitter15/16 使用重新建立的 strict early graph features 與固定
train/validation/test protocol，不包含 non-rumour candidates。最乾淨的
scalar MLP 使用 31 個 activity、fine temporal 與 topology features，並在
validation 選到很弱的 Oracle-order loss（lambda 0.05）。設定凍結後，224
個 test threads 只評估一次：

| Corpus    | Top-10 blocked / Oracle | Reduction@10 | Oracle efficiency@10 |
| --------- | ----------------------: | -----------: | -------------------: |
| Twitter15 |             664 / 1,209 |       0.1986 |               0.5492 |
| Twitter16 |               532 / 797 |       0.3292 |               0.6675 |

Twitter16 在 Budget 40 可達 Oracle efficiency `0.9149`，但 Twitter15 明顯
較難。這說明相同早期特徵與模型在不同 corpus 的效果差異很大，不能因為
某一個資料集表現好就宣稱方法普遍有效。

## 六、已經測過但沒有穩定成功的方向

- utility regression 加強 Oracle order loss；
- pairwise ranking；
- soft-Top-K loss；
- source/reply semantic model；
- semantic residual correction；
- account-age ablation；
- sparse hand-selected quiet features；
- graph/time 雙分支模型；
- bounded structural residual re-ranking；
- Oracle Top-50 set classification；
- 八群特徵的 255 種 exhaustive combinations。

其中一些方法會改善特定事件、特定 Budget 或 Top-1，但目前沒有任何一種
能在完整曲線上穩定勝過 baseline。這些不是沒有價值的失敗：它們排除了
很多直覺上合理、但實際只會過度擬合事件或破壞其他 Budget 的做法。

## 七、目前真正遇到的困難

### 1. Quiet thread 的可辨識資訊不足

在前 30 分鐘同樣安靜的 threads，現有 activity、time、topology、text、
semantics 和 account-age features 經常很接近，但未來擴散量可以差很多。
模型因而只能學到 quiet 群體的平均風險，難以預先知道哪一個會突然爆發。

### 2. Event shift 很強

某組特徵在一個事件有效，到另一個事件可能反向。只有少數 events 時，
inner selection 很容易選到不會泛化到新 outer event 的組合。

### 3. Top-1 與完整排序不是同一件事

提高某個極端高 impact thread 的分數，可能改善 Top-1，卻同時讓大量普通
threads 的相對順序變差，導致 K=5--50 或完整 Spearman 下降。

### 4. Oracle 擁有模型看不到的未來

Oracle 直接按真實 future impact 排序；模型只看 30 分鐘。要求模型
「學 Oracle」不等於模型能取得 Oracle 的資訊。改 loss 可以讓模型更重視
排序錯誤，但無法補回 observation window 中根本不存在的訊號。

## 八、這個研究目前做到的實際成果

- 建立了嚴格 rumour-only、30-minute early-intervention protocol。
- 釐清了 Oracle、model 與 intervention accounting 使用相同效益定義。
- 實作任意 Budget 的完整 priority ranking 與 intervention curves。
- 建立 nested event-separated selection，避免用 outer test 選模型。
- 驗證 temporal snapshot、node、edge 與 train-only scaler/PCA 邊界。
- 系統性比較 global RF、雙專家、scalar MLP、pairwise、semantic、topology、
  account-age 與大量特徵組合。
- 保存正面與負面結果，沒有因結果不好而更改 label、split、candidate pool、
  cutoff 或 metric。
- 將有效模型、受保護輸入、實驗結果與舊版程式分開整理，保留可重現性。

所以目前的成果不是「已經接近 Oracle」，而是更誠實且可辯護地找出了：

> 早期活躍 threads 有一定程度可排序；quiet threads 的極端 future impact
> 在目前 30 分鐘訊號下高度不穩定，而且單純增加特徵或 ranking loss 並不
> 能穩定解決。

## 九、下一步最合理的方向

最新版 feature-combination model 不應直接取代 quiet RF，但它可能提供
互補的 extreme-head signal。下一步建議：

1. 保留 current quiet RF 作主要排序。
2. 把 feature-combination／head model 當成第二個 score。
3. 只用 inner event-OOF predictions 學習有界的 blend 或 residual correction。
4. 讓修正可以在證據不足時收縮回 baseline。
5. Selection objective 同時要求 K=1/3 改善，且限制 K=5--50 不得明顯退步。
6. Outer held-out event 仍只評估一次。
7. 最後需要新的 untouched event 或外部資料集確認，而不是持續利用同一批
   PHEME events 調整方法。

## 十、重要檔案

- 技術交接：`PROJECT_HANDOFF_20260902_QUIET_POLICY.md`
- 完整結果紀錄：`PROJECT_RESULTS_20260901.md`
- 有效模型：`effective_models/`
- 最新 exhaustive 結果：
  `research_scratch/legacy_full/graphsage_intervention_14/experiments/20260901_233016_536789_nested_pheme_quiet_feature_combinations/`
- Protected v5 inputs：`data/protected_research_assets/pheme_v5_strict30/`
- 專案整理紀錄：`REORGANIZATION_MANIFEST_20260902.md`

## 一句話結論

本研究已建立一套研究安全、可重現的早期干預評估流程，並證明模型可以
學到部分介入優先順序；但前 30 分鐘安靜、後續突然爆發的 threads 仍是
主要瓶頸，目前各種新增特徵與 Oracle-ranking loss 只能局部改善，尚未
形成穩定超越 v5 baseline 的完整方法。
