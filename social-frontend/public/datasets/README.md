# 資料集擺放規格

這個資料夾存放前端可直接讀取的 PHEME 示範資料。現有事件資料由
`social-frontend/scripts/build_pheme_dataset.py` 從受保護的 PHEME 原始資料與
v5 Text RF 歷史 OOF 排名產生；瀏覽器執行時只讀取這些 JSON，不會執行 builder。

## 目錄結構

```
public/datasets/
├── manifest.json              ← 資料集清單，下拉選單讀這份
├── sydneysiege_holdout/
│   └── thread.json            ← 這個資料集實際的貼文/回覆內容
├── charliehebdo_holdout/
│   └── thread.json
└── ...（最多預期 1～3 組）
```

## 1. `manifest.json`

下拉選單顯示的清單，陣列裡每一項對應一個資料夾：

```json
[
  { "id": "sydneysiege_holdout", "label": "雪梨人質事件（保留集）" },
  { "id": "charliehebdo_holdout", "label": "查理週刊事件（保留集）" }
]
```

- `id` 必須跟資料夾名稱完全一致，前端會用它去抓 `public/datasets/<id>/thread.json`
- `label` 是下拉選單顯示給使用者看的名稱，可以隨意命名
- 若陣列為空 `[]`，下拉選單會顯示「尚無可用資料集」；目前 repository 已包含九個 PHEME 事件的示範資料。

## 2. `<id>/thread.json`

單一資料集的實際內容，欄位刻意對齊 `graph_analysis/builder.py` 既有的建圖邏輯（`source_user` / `reply_user` / `thread_id` / `is_rumour`），這樣載入前端的同時，也能直接把同樣的邊送給後端重新算一次 centrality，不用另外轉換格式：

```json
{
  "id": "sydneysiege_holdout",
  "label": "雪梨人質事件（保留集）",
  "threads": [
    {
      "thread_id": "552783238415265792",
      "is_rumour": true,
      "source": {
        "user": "alice_92",
        "content": "內部消息，馬丁廣場那邊已經封鎖了，聽說是人質事件",
        "ts": "2014-12-15T09:12:00Z"
      },
      "reactions": [
        {
          "user": "bob_news",
          "content": "轉發：馬丁廣場人質事件",
          "reply_to": "alice_92",
          "ts": "2014-12-15T09:13:10Z"
        },
        {
          "user": "carol_sydney",
          "content": "真的假的，我朋友剛好在附近",
          "reply_to": "bob_news",
          "ts": "2014-12-15T09:14:40Z"
        }
      ]
    }
  ]
}
```

**欄位說明**

| 欄位 | 說明 |
|---|---|
| `threads[].thread_id` | 這一串討論的唯一 id，可以直接沿用 PHEME 原始的 thread id |
| `threads[].is_rumour` | `true` / `false`，整串討論共用同一個標籤（沿用 PHEME 的標記方式） |
| `threads[].source` | 這一串討論最原始的那則貼文 |
| `threads[].reactions[]` | 依時間序排列的所有回覆／轉發 |
| `reactions[].reply_to` | 這則回覆是回應給誰（帳號 handle）。省略的話預設視為回覆給 `source.user` |

一個 `thread.json` 裡可以放多個 `threads`（例如同一個事件底下有好幾串不同的討論），前端會全部載入、依時間排序顯示。

## 3. 帳號顯示

前端目前**不需要**額外的使用者資料表（大頭貼、認證標記等），會直接用 `user` 這個 handle 字串當顯示名稱，並依 handle 自動配一個固定顏色的預設頭像。如果之後想讓帳號有更完整的資訊（例如標記官方帳號），可以之後再擴充 `thread.json` 加一個 `users` 欄位，前端這邊到時候再對接。
