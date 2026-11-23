# docx_pipeline — 主報告 Word 就地修改管線

Word 檔是事實來源（你會手動在 Word 修改）；管線只做「章節級、追蹤修訂、可回溯」的修改。

## 流程
1. `python -I tools/docx_cli.py status`            看工作檔與版本；你在 Word 改過的檔案會在下次操作前自動存成 `manual-edit` 版
2. `python -I tools/docx_cli.py index`             章節樹（章節編號 = Word 自動編號）
3. `python -I tools/docx_cli.py show 3.2`          列出該節每個區塊的序號、樣式、可否編輯（plain / img / field / math / tbl）
4. 寫 spec.json → `apply spec.json --dry-run`      先跑稽核不寫檔 → 通過後去掉 `--dry-run`
5. 在 Word 開啟檢視追蹤修訂；要更新目錄/圖表索引：Ctrl+A 後 F9
6. `restore vNNN`                                  任何版本都能還原（還原本身也是新版本）

## spec.json
```json
{"section":"3.2","title":"傳播風險與動態介入實驗結果","ops":[
 {"op":"replace_text","para":4,"expect":"段落開頭至少4字","find":"舊片語","replace":"新片語"},
 {"op":"replace_paragraph","para":5,"expect":"…","new_text":"整段新文字"},
 {"op":"delete_paragraph","para":6,"expect":"…"},
 {"op":"insert_after","para":7,"expect":"…","blocks":[{"type":"paragraph","text":"…"}]}
]}
```
- `para` 為 `show` 顯示的序號；`-1` = 章節標題本身（只能在其後插入）
- `expect` 必填：目標段落目前文字的開頭，對不上就中止（避免你手動改過後規格過期）
- block type：`paragraph`、`h1`~`h4`（不要自己寫編號）、`list_number`、`list_para`、`caption_figure`、`caption_table`、`figure_image`
- 圖片與圖說可依序放入同一個 `blocks`：
  ```json
  [{"type":"figure_image","path":"C:/absolute/path/figure.png","width_cm":15.0,"alt":"圖像內容的替代文字"},
   {"type":"caption_figure","text":"圖說說明文字"}]
  ```
- `figure_image.path` 須為現有 PNG 的絕對路徑；`alt` 必填。`width_cm` 預設 15.0，不能超過所在節的可用版寬；高度依 PNG IHDR 像素比例計算
- 圖說／表說的 `text` 只填說明文字，不填「圖 3-x」或「表 3-x」；必須開啟追蹤修訂，整段會標成插入
- 一般新增內容只用 `style_map.json` 的樣式；圖說／表說深複製既有同類段落及其直接格式。圖說使用 `af4`，表說沿用現有表說的預設樣式
- 欄位顯示數字只是估計值；在 Word 接受修訂後按 Ctrl+A、F9，更新編號、圖目錄與表目錄

## 稽核（任一項失敗就不寫檔）
除下述圖片插入例外，其他 zip 部件（內嵌字型、既有圖片、樣式、編號、頁首頁尾）位元組完全相同；目標章節以外的每個區塊完全相同；
表格/公式/section 數量不變；只有 `figure_image` 可使 drawing 與 `wp:inline` 各精確增加一個，`wp:anchor` 不變。只有預期新增的 PNG media 與 document.xml.rels 條目可改動；其他 zip 部件位元組不變。新增圖片的來源雜湊、relationship、docPr id、替代文字、尺寸比例、版寬及模板結構均須通過檢查；
欄位與書籤的新增量須與 spec 中圖說／表說的範本數量完全一致，且新段落結構須與同類範本一致；追蹤修訂 id、書籤 id／名稱唯一；
Reject-All == 原文、Accept-All == 預期結果；python-docx 與 LibreOffice 能開啟。

## 版本
`versions/vNNN_時間_標籤.docx` + `manifest.jsonl`，不進 git（資料夾內自帶 `.gitignore`）。

## 限制（刻意）
含表格、圖片、欄位、公式、tab、超連結的既有段落只能保留不能改寫。新增圖片只支援 PNG、置中的 inline 圖片，不支援文字環繞或表格內圖片；必須有可用的正文圖片模板。拒絕追蹤修訂後，新圖片段落會消失，但 package 內的 media/relationship 會成為預期且無害的孤兒項。頁碼/分頁以 Word 為準（LibreOffice 缺新細明體，頁數不同）。
