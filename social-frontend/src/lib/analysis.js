/* ==========================================================================
   analysis.js —— 分析功能的唯一進出口
   --------------------------------------------------------------------------
   目前預留兩條分析路線，兩條都還沒接上後端：

     MODE.MODEL   模型評估 —— 傳播結構已成形時，交給我們自己的模型
     MODE.SEARCH  網路查證 —— 還沒有結構可用時，改上網找佐證

   另外預留了「真假判斷」的欄位（verdict），這塊還沒開始做，
   但兩條路線的回傳結構都已經留好位置，畫面也會在有資料時自動顯示。
   ========================================================================== */

export const MODE = {
  MODEL: 'model',
  SEARCH: 'search',
};

/* --------------------------------------------------------------------------
   路線判定 —— 這裡的規則目前只是暫定的假設，還沒定案
   ----------------------------------------------------------------------------
   先以「貼文發布是否滿 30 分鐘」當作暫時的分流依據，方便畫面跑起來。
   實際要用什麼條件分流（時間門檻、回覆數、是否已有傳播結構、
   或乾脆讓使用者手動選）都還沒決定，之後只要改這一個函式即可，
   其他程式都不用動。
   -------------------------------------------------------------------------- */

/** 暫定門檻：貼文發布後 30 分鐘（單位：毫秒） */
export const OBSERVATION_CUTOFF_MS = 30 * 60 * 1000;

export function decideMode(threadPosts) {
  const earliest = threadPosts.reduce((min, p) => {
    const t = new Date(p.ts).getTime();
    return Number.isFinite(t) && t < min ? t : min;
  }, Infinity);

  if (!Number.isFinite(earliest)) return MODE.SEARCH;
  return Date.now() - earliest >= OBSERVATION_CUTOFF_MS ? MODE.MODEL : MODE.SEARCH;
}

/* --------------------------------------------------------------------------
   路線一：模型評估
   ----------------------------------------------------------------------------
   TODO(後端就緒後替換)：
     const res = await fetch('/api/score-thread', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ threadId, source, replies }),
     });
     const data = await res.json();

   預期回傳：
     score   0~100 的介入優先度
     rank    在全部候選裡排第幾（可省略）
     total   候選總數（可省略）
     basis   判斷依據，[{ label, detail }]
     verdict 真假判斷，尚未實作，格式見下方 makeVerdict()
   -------------------------------------------------------------------------- */

export async function analyzeByModel(threadPosts) {
  await new Promise((r) => setTimeout(r, 600 + Math.random() * 400));

  return {
    mode: MODE.MODEL,
    ready: false,
    note: '模型推論 API 尚未接上',
    score: null,
    rank: null,
    total: null,
    basis: [],
    verdict: null, // 預留：真假判斷
  };
}

/* --------------------------------------------------------------------------
   路線二：網路查證
   ----------------------------------------------------------------------------
   TODO(後端就緒後替換)：
     const res = await fetch('/api/verify', { ... });
     const data = await res.json();

   預期回傳：
     reasoning 判斷理由（一段文字）
     sources   證據出處，[{ title, url, stance }]
               stance: 'support' 佐證 | 'refute' 反駁 | 'context' 背景
     verdict   真假判斷，尚未實作
   -------------------------------------------------------------------------- */

export async function analyzeBySearch(threadPosts) {
  await new Promise((r) => setTimeout(r, 600 + Math.random() * 400));

  return {
    mode: MODE.SEARCH,
    ready: false,
    note: '查證 agent 尚未接上',
    reasoning: null,
    sources: [],
    verdict: null, // 預留：真假判斷
  };
}

/* --------------------------------------------------------------------------
   真假判斷 —— 預留，尚未實作
   ----------------------------------------------------------------------------
   之後不論由模型、查證 agent 或另外的分類器產生，都用這個格式回傳，
   放進上面兩條路線回傳物件的 `verdict` 欄位，畫面就會自動顯示出來。

   label      顯示文字，例如「可能為不實訊息」
   stance     'false' 不實 | 'true' 屬實 | 'unclear' 無法判定
   confidence 0~1，沒有信心值時給 null
   -------------------------------------------------------------------------- */

export function makeVerdict({ label, stance = 'unclear', confidence = null }) {
  return { label, stance, confidence };
}

/** 分析一整串討論，依目前的暫定規則決定走哪條路線。 */
export async function analyzeThread(threadPosts) {
  const mode = decideMode(threadPosts);
  return mode === MODE.MODEL ? analyzeByModel(threadPosts) : analyzeBySearch(threadPosts);
}

/* --------------------------------------------------------------------------
   問答 —— 預留，尚未實作
   ----------------------------------------------------------------------------
   TODO：
     const res = await fetch('/api/ask', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ threadId, question, context: threadPosts }),
     });
     const data = await res.json(); // 預期 { answer, sources }
   -------------------------------------------------------------------------- */

export async function askAboutThread(threadPosts, question) {
  await new Promise((r) => setTimeout(r, 400));
  return { ready: false, note: '問答功能尚未開放', answer: null, sources: [] };
}
