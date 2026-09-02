/* ==========================================================================
   analysis.js —— AI 模型分析的唯一進出口
   --------------------------------------------------------------------------
   目前模型還沒訓練完成，這裡先回傳一個明確標示「待訓練」的假結果，
   讓前端的框選 → 送分析 → 顯示結果整條流程可以先跑起來。

   之後模型準備好，只需要把 analyzePost() 內部換成真正打後端 API 的
   fetch 呼叫即可，不需要更動任何呼叫這個函式的元件。
   ========================================================================== */

/**
 * 分析單一則貼文。
 *
 * TODO(模型就緒後替換): 換成呼叫後端推論 API，例如：
 *
 *   const res = await fetch('/api/analyze', {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body: JSON.stringify({ postId: post.id, content: post.content, user: post.user }),
 *   });
 *   const data = await res.json(); // 預期 { verdict: 'rumour' | 'real', confidence: 0~1 }
 *   return { postId: post.id, verdict: data.verdict, confidence: data.confidence, label: null };
 *
 * @param {{ id: string, content: string, user: string }} post
 * @returns {Promise<{ postId: string, verdict: 'rumour'|'real'|'pending', confidence: number|null, label: string|null }>}
 */
export async function analyzePost(post) {
  // 模擬推論耗時，讓「分析中…」狀態在畫面上看得到
  await new Promise((resolve) => setTimeout(resolve, 500 + Math.random() * 500));

  return {
    postId: post.id,
    verdict: 'pending',
    confidence: null,
    label: '模型尚未訓練完成',
  };
}

/** 批次分析多則貼文（框選一次通常會擷取到好幾則） */
export async function analyzePosts(posts) {
  return Promise.all(posts.map(analyzePost));
}
