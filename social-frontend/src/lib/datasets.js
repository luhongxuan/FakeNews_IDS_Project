/* ==========================================================================
   datasets.js —— 資料集載入器
   --------------------------------------------------------------------------
   讀取 public/datasets/ 底下的資料集（格式規格見 public/datasets/README.md）。

   Vite 會把 public/ 底下的檔案原封不動放到網站根目錄，所以這裡直接用
   fetch('/datasets/...') 就抓得到，不需要 import、不需要打包，
   資料集到位後直接覆蓋檔案即可，前端完全不用重新編譯。
   ========================================================================== */

const BASE = '/datasets';

/** 讀取資料集清單（下拉選單用）。manifest 還沒有內容時回傳空陣列，不丟錯。 */
export async function listDatasets() {
  const res = await fetch(`${BASE}/manifest.json`, { cache: 'no-store' });
  if (!res.ok) return [];
  const list = await res.json();
  return Array.isArray(list) ? list : [];
}

/**
 * 載入單一資料集，轉換成畫面需要的兩份資料：
 *   posts —— 依時間排序，餵給 feed 直接渲染
 *   edges —— from → to 的邊列表，餵給後端重新計算 centrality
 */
export async function loadDataset(id) {
  const res = await fetch(`${BASE}/${id}/thread.json`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`找不到資料集「${id}」，確認 public/datasets/${id}/thread.json 是否存在`);
  }
  const raw = await res.json();
  return normalize(raw);
}

function normalize(raw) {
  const posts = [];
  const edges = [];

  for (const thread of raw.threads ?? []) {
    const { thread_id: threadId, is_rumour: isRumour, source, reactions = [] } = thread;

    posts.push({
      id: `${threadId}_source`,
      threadId,
      user: source.user,
      content: source.content,
      ts: source.ts,
      isRumour,
      likes: 0,
      shares: 0,
      comments: 0,
    });

    // 用來讓每則回覆找到「自己回覆的對象」實際說了什麼，畫成引用區塊
    const contentByUser = { [source.user]: source.content };

    reactions.forEach((r, i) => {
      const replyTo = r.reply_to ?? source.user;
      posts.push({
        id: `${threadId}_r${i}`,
        threadId,
        user: r.user,
        content: r.content,
        sharedFrom: replyTo,
        sourceContent: contentByUser[replyTo] ?? source.content,
        ts: r.ts,
        isRumour,
        likes: 0,
        shares: 0,
        comments: 0,
      });

      contentByUser[r.user] = r.content;
      edges.push({ from: replyTo, to: r.user, threadId, isRumour });
    });
  }

  // 依時間排序，較早的貼文在前
  posts.sort((a, b) => new Date(a.ts) - new Date(b.ts));

  return { posts, edges };
}
