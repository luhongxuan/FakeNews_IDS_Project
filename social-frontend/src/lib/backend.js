/* ==========================================================================
   backend.js —— 與後端溝通的一切都在這裡
   本檔案合併了兩個職責：
     1. 事件格式定義（EVENT_TYPE、makePostEvent 等）
     2. WebSocket 連線與離線佇列（createEventBridge）
   ========================================================================== */

/* --------------------------------------------------------------------------
   一、事件格式定義 —— 前後端的共同約定
   ----------------------------------------------------------------------------
   欄位刻意對齊 graph_analysis/builder.py 既有的建圖邏輯：

       G.add_edge(source_user, reply_user, thread_id=..., is_rumour=...)

   所以每一筆 share / comment 事件都直接對應圖上的一條有向邊：
       from  →  source_user（訊息從誰流出）
       to    →  reply_user （訊息流向誰）

   這樣後端收到事件後不需要再做欄位轉換，可以直接餵進建圖流程。

   is_rumour 沿用 PHEME 的語意：標記在「整串討論（thread）」上，
   而不是單則貼文，因此 share / comment 一律繼承源頭貼文的值。
   -------------------------------------------------------------------------- */

export const EVENT_TYPE = {
  POST: 'post', // 原發文，開啟一串新的 thread
  SHARE: 'share', // 轉發，產生一條 from → to 的邊
  COMMENT: 'comment', // 留言，同樣產生一條 from → to 的邊
};

let seq = 0;
const nextId = (prefix) => `${prefix}_${Date.now().toString(36)}${(seq++).toString(36)}`;

/** 原發文：開啟一串新的討論 */
export function makePostEvent({ user, content, isRumour = false, clusterId = null }) {
  const id = nextId('p');
  return {
    type: EVENT_TYPE.POST,
    id,
    thread_id: id, // 原發文的 id 即為整串討論的 thread_id
    user,
    content,
    is_rumour: isRumour,
    event_cluster_id: clusterId, // 語意分群由後端負責填入，前端只負責顯示
    ts: new Date().toISOString(),
  };
}

/** 轉發：訊息從 from 流向 to，對應圖上一條邊 */
export function makeShareEvent({ from, to, sourcePostId, threadId, isRumour, clusterId, content = '' }) {
  return {
    type: EVENT_TYPE.SHARE,
    id: nextId('s'),
    thread_id: threadId,
    source_post_id: sourcePostId,
    from,
    to,
    content, // 轉發時附加的文字，供後端做語意相似度比對
    is_rumour: isRumour,
    event_cluster_id: clusterId,
    ts: new Date().toISOString(),
  };
}

/** 留言：同樣構成一條 from → to 的邊 */
export function makeCommentEvent({ from, to, sourcePostId, threadId, isRumour, clusterId, content }) {
  return {
    type: EVENT_TYPE.COMMENT,
    id: nextId('c'),
    thread_id: threadId,
    source_post_id: sourcePostId,
    from,
    to,
    content, // 留言內容，供後端做語意相似度比對
    is_rumour: isRumour,
    event_cluster_id: clusterId,
    ts: new Date().toISOString(),
  };
}

/* --------------------------------------------------------------------------
   二、事件橋接層 —— 社群平台前端與後端之間唯一的溝通介面
   ----------------------------------------------------------------------------
   規劃書對應：
   - 社群平台端每發生一次互動，就把事件推給後端
   - 後端負責儲存（不管關係圖前端有沒有開著），並在有連線時即時推播出去
   - 因此本層只負責「送出」，不負責畫圖，也不負責計算風險分數

   後端還沒好也不會壞掉：連不上時自動退回離線模式，事件先排進佇列，
   等後端上線再一次補送，畫面操作完全不受影響。
   -------------------------------------------------------------------------- */

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000/ws/events';

/** 連線狀態：offline 尚未連上 / connecting 連線中 / live 已連線 */
export const STATUS = {
  OFFLINE: 'offline',
  CONNECTING: 'connecting',
  LIVE: 'live',
};

export function createEventBridge({ url = WS_URL, onStatus, onRemoteEvent } = {}) {
  let socket = null;
  let retryTimer = null;
  let retryCount = 0;
  let disposed = false;
  const pending = [];

  const emitStatus = (status) => {
    if (!disposed) onStatus?.(status, { queued: pending.length, url });
  };

  function flushQueue() {
    while (pending.length && socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(pending.shift()));
    }
  }

  function scheduleRetry() {
    if (disposed) return;
    // 重試間隔逐次拉長，最多 15 秒，避免後端沒開時瘋狂重連
    const delay = Math.min(1000 * 2 ** retryCount, 15000);
    retryCount += 1;
    clearTimeout(retryTimer);
    retryTimer = setTimeout(connect, delay);
  }

  function connect() {
    if (disposed || socket?.readyState === WebSocket.OPEN) return;
    emitStatus(STATUS.CONNECTING);

    try {
      socket = new WebSocket(url);
    } catch {
      emitStatus(STATUS.OFFLINE);
      scheduleRetry();
      return;
    }

    socket.onopen = () => {
      retryCount = 0;
      emitStatus(STATUS.LIVE);
      flushQueue();
    };

    socket.onmessage = (raw) => {
      try {
        onRemoteEvent?.(JSON.parse(raw.data));
      } catch {
        /* 後端傳來非 JSON 內容時忽略，不影響畫面 */
      }
    };

    socket.onclose = () => {
      emitStatus(STATUS.OFFLINE);
      scheduleRetry();
    };

    // onerror 後瀏覽器必定接著觸發 onclose，重連交給 onclose 處理即可
    socket.onerror = () => socket?.close();
  }

  function send(event) {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(event));
    } else {
      pending.push(event);
      emitStatus(socket ? STATUS.CONNECTING : STATUS.OFFLINE);
    }
  }

  function dispose() {
    disposed = true;
    clearTimeout(retryTimer);
    socket?.close();
    socket = null;
  }

  return { connect, send, dispose };
}
