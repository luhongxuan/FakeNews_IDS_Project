# 任務指令：為 docx_pipeline 新增 `figure_image` 區塊（插入圖片）

交給 Codex。本任務只改 `docx_pipeline/tools/` 的程式與文件；**不訓練、不連網、不裝套件、不碰正式 docx、不改 `analysis/` 既有結果**。與 `AGENTS.md` / `CODEX.md` 衝突時以它們為準。

## 1. 目標
讓 `spec.json` 能用一個新區塊型別 `figure_image` 插入一張 PNG（不含圖說；圖說仍用既有 `caption_figure`，緊接在圖片段落之後）。編號與圖目錄由既有圖說欄位機制處理，不在本任務範圍。

spec 寫法（`insert_after` 的 `blocks` 內）：
```json
{"type": "figure_image", "path": "C:/FakeNews_IDS_Project/analysis/20261122_report_figures/fig_rules_vs_models.png",
 "width_cm": 15.0, "alt": "規則與模型的政策層涵蓋率長條圖"}
```
- `path`：本機 PNG 絕對路徑，必須存在且為 PNG（檢查檔頭 `\x89PNG`）。
- `width_cm`：選填，預設 15.0；上限為文件可用版寬（讀 `sectPr` 的頁寬減左右邊界；超過就報錯，不要自動縮放）。
- 高度由 PNG 像素的長寬比換算，**不得變形**。像素尺寸從 PNG 的 IHDR 讀，不得引入新套件（用 `struct` 即可）。
- `alt`：必填，寫入 `wp:docPr/@descr` 與 `pic:cNvPr/@descr`。

## 2. 實作方式（複製既有結構，不自創）
1. **模板**：從文件中找「圖說段落（style af4、含 `SEQ 圖`）的前一段」且該段含 `w:drawing/wp:inline` 的段落，當作正文圖片的版型模板。若有多個，取第一個並在 `docx_cli index` 或報告中寫出它的位置；找不到就報錯，不要退而使用封面校徽那一段。
2. 深拷貝模板段落，只替換：
   - `a:blip/@r:embed` → 新增的 relationship id；
   - `wp:extent`、`pic:spPr/a:xfrm/a:ext` 的 cx/cy（EMU，1 cm = 360000）；
   - `wp:docPr/@id`（必須是全文件唯一的新值：取現有最大 id + 1）、`@name`（如「圖片 N」）、`@descr`；
   - 清掉 `wp14:anchorId`、`wp14:editId`、`w14:paraId`、`w14:textId`（或換成未使用的唯一值；paraId 規則同圖說：有效、未使用、唯一；textId 用 `77777777`）。
   - 其餘（段落樣式、置中、`jc`、`noProof`、`effectExtent` 等）一律保留；`effectExtent` 全設 0。
3. **package 部件**（這是本任務的核心，稽核必須對應放寬，且只放寬這幾項）：
   - 新增 `word/media/imageN.png`，N＝現有最大編號 + 1，內容為指定 PNG 的原始位元組。
   - `word/_rels/document.xml.rels` 新增一條 image relationship（Id 為未使用的 `rIdN`，Target `media/imageN.png`）。
   - `[Content_Types].xml` 目前已有 `Default Extension="png"`，**不應被修改**；若缺少才新增，並在稽核中標示。
   - 其他所有 zip 部件仍須逐位元組相同。
4. **追蹤修訂**：新圖片段落比照既有 `new_paragraph` 的做法，用 `w:ins` 包住 run，並在段落標記上加 `w:rPr/w:ins`，使 Reject-All 還原為原文件（包含：被拒絕後新段落消失；孤兒的 media/rels 由稽核另行說明，見下）。
5. 同一個 spec 內可含多個 `figure_image`；每個都要拿到不同的 media 檔名、rId 與 docPr id。

## 3. 稽核（`audit.py`）
- 現有「zip 部件除 document.xml 外逐位元組相同」改為：**除了預期新增的 media 檔與 rels 條目外相同**。預期集合由 spec 推得（每個 `figure_image` 一個 media、一條 rel），多出或少掉任何東西都 FAIL。
- 新增檢查：
  - `<drawing>` 數量剛好 +（spec 內 `figure_image` 數）；`wp:inline` 同樣 +N；`wp:anchor` 不變。
  - 新 media 的 sha256 等於來源 PNG 的 sha256。
  - 新 relationship 的 Type 為 image、Target 存在於 zip、Id 唯一且被新段落引用。
  - 新 `docPr/@id` 在全文件唯一；新段落 `paraId` 有效、未使用、唯一。
  - 新圖尺寸：`wp:extent` 與 `a:ext` 一致，長寬比與 PNG 像素比誤差 < 0.5%，寬度 ≤ 可用版寬。
  - 新圖片段落結構與模板一致（去除可變屬性後）。
  - `pic:blipFill` 的 embed 指向新 rel，而不是既有圖片。
- 既有檢查（Reject-All == 原文件、Accept-All == 預期結果、其他計數不變等）保持。Reject-All 的比較對象是 document.xml 的文字／結構；**孤兒 media 與 rel**（拒絕修訂後無人引用）在稽核報告寫明「預期、無害」，不要算 FAIL。
- 沒有 `figure_image` 的 spec，行為與稽核結果必須與現在**完全一致**（回歸）。

## 4. 測試
- 在 `test_caption.py` 旁新增 `test_figure_image.py`，只用**驗證副本**（`CAPTION_TEST_DOCX` 同款環境變數，如 `FIGURE_TEST_DOCX=<副本路徑>`），不碰正式檔。
- 至少涵蓋：單張圖 + 圖說（插入後 media/rels/drawing 計數、docPr id 唯一、長寬比）、同一 spec 兩張圖、PNG 不存在／非 PNG／寬度超版寬／缺 alt 的報錯、找不到模板的報錯、無 `figure_image` 時回歸（與修改前稽核輸出一致）、Reject-All 與 Accept-All 檢查。
- 驗證副本請放新目錄 `docx_pipeline/figure_validation_20261122/`（由正式檔複製；不得覆寫既有 `caption_validation_*`）。
- 用這兩個檔當實測輸入（已存在）：
  - `analysis/20261122_report_figures/fig_rules_vs_models.png`
  - `analysis/20261122_report_figures/fig_two_stage_assumed_accuracy.png`

## 5. 文件
- 更新 `docx_pipeline/README.md`：新增 `figure_image` 的語法、限制（只支援 PNG、inline、置中、不支援文字環繞、不支援表格內圖片）、稽核放寬了什麼。
- 更新 `style_map.json` 只在必要時。

## 6. 禁止事項
- 不修改正式檔 `第08組_…docx`、`versions/`、`reports/`；不執行 `apply` 於正式檔（只能在驗證副本上）。
- 不改既有區塊型別的行為；不放寬與圖片無關的稽核。
- 不連網、不裝套件（`lxml` 與標準函式庫已足夠）、不用 `soffice` 煙霧測試（本機會連 Docker 失敗，沿用 `--no-smoke`）。
- 不要自行把圖片放進 `analysis/` 以外的地方或複製成新圖檔。

## 7. 回報格式
1. 執行了哪些指令與測試，輸出摘要（含 `SUCCESS` 行與驗證副本絕對路徑）。
2. 稽核通過／失敗清單，新增與修改的稽核項目逐條列出。
3. 模板段落是哪一個（位置、圖說文字），為何選它。
4. 已知限制與尚未解決的問題。
5. **到此停止，交給使用者**：使用者會在驗證副本中用 Word 開啟確認圖片顯示正常、編號與圖目錄更新（Ctrl+A、F9，先關閉追蹤修訂）後，才由 Claude 對正式檔套用。
