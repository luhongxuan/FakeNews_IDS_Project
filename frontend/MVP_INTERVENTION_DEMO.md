# 自動干預建議 MVP 展示

此 MVP 直接整合在原有 `frontend/` 的「自動干預建議」頁面，不是另一套前端。

## 啟動

```powershell
cd C:\FakeNews_IDS_Project\frontend
npm run dev
```

進入首頁後點選「自動干預建議」。頁面提供三種模式：

- `真實決策`：後端 Agent 已經完成並存檔的決策。
- `真實事件預覽`：使用雷達收集到的真實 Bluesky 事件內容與互動量，用透明的前端規則預覽 Soft / None；點選可進入真實傳播圖、RF 與查證頁面。
- `合成流程示範`：不需後端的 Hard / Medium / Soft / None 完整流程示範。

在「真實事件預覽」開啟任一雷達事件後，右側另有「真實政策決策 MVP」。按下「產生真實介入建議」會執行：

```text
當下可觀察的 Bluesky 回覆樹
        ↓
凍結的 Hawkes-inspired RF（預估後續 reply-cascade impact）
        ↓
Verification Agent（查找支持／反駁核心主張的證據）
        ↓
確定性政策（證據狀態 × RF 風險 × 當下已釋出配額）
        ↓
保存並顯示 Hard／Soft／None／Deferred 建議
```

此按鈕需要後端、資料庫與本機 Ollama/查證工具可用。它是單一貼文的手動觸發 MVP；目前尚未取代既有背景排程器，也不是同一事件多候選的完整批次排名。

## 展示內容

- RF 預估後續擴散量
- Verification Agent 的證據狀態
- Hard / Medium / Soft / None 四級政策建議
- 模擬介入強度與決策理由

真實事件預覽卡片的介入層級仍是前端預覽規則；只有事件詳細頁中按下「產生真實介入建議」後顯示的結果，才來自後端 RF、查證 Agent 與確定性政策的完整 MVP 路徑。合成流程資料會明確標示為模擬案例。所有模式都不會實際刪文、隱藏內容或調整平台曝光。
