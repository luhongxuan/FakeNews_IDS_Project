# 使用者端早期介入助理 MVP

```powershell
cd C:\FakeNews_IDS_Project\social-frontend
npm run dev
```

開啟 `http://localhost:5174`，選擇事件資料集，點右上角 AI 圖示，再以「框選貼文」選取內容。

系統依序顯示固定 30 分鐘傳播風險、已保存的查證狀態與 `hard / medium / soft / none` 政策建議。後端不可用時會自動切換成「離線展示規則」，只使用畫面上可觀測的文字與互動數，確保展示不中斷；這個分數不是研究模型結果，也不得放入模型成效表。

本 MVP 不會對真實平台執行隱藏、降觸及或限制轉發。
