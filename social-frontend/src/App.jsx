import React, { useCallback, useEffect, useRef, useState } from 'react';
import PostCard from './components/PostCard.jsx';
import ControlRail from './components/ControlRail.jsx';
import AIAgentPanel from './components/AIAgentPanel.jsx';
import SelectionOverlay from './components/SelectionOverlay.jsx';
import { createEventBridge, makeShareEvent, STATUS } from './lib/backend.js';
import { listDatasets, loadDataset } from './lib/datasets.js';
import { analyzeThread } from './lib/analysis.js';

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetId] = useState('');
  const [datasetState, setDatasetState] = useState('idle'); // idle / loading / error
  const [datasetError, setDatasetError] = useState(null);

  const [posts, setPosts] = useState([]);
  const [log, setLog] = useState([]);
  const [status, setStatus] = useState(STATUS.CONNECTING);
  const [queued, setQueued] = useState(0);
  const [railOpen, setRailOpen] = useState(true);

  const [aiOpen, setAiOpen] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [captured, setCaptured] = useState([]); // [{ threadId, posts, result }]

  const bridgeRef = useRef(null);

  /* --- 建立與後端的連線。後端沒開也不影響操作 --- */
  useEffect(() => {
    const bridge = createEventBridge({
      onStatus: (next, meta) => {
        setStatus(next);
        setQueued(meta.queued);
      },
    });
    bridgeRef.current = bridge;
    bridge.connect();
    return () => bridge.dispose();
  }, []);

  /* --- 開頁時讀取資料集清單（manifest 還是空的也不會報錯） --- */
  useEffect(() => {
    listDatasets().then(setDatasets);
  }, []);

  /** 送出事件並記入日誌 */
  const emit = useCallback((event) => {
    bridgeRef.current?.send(event);
    setLog((prev) => [...prev.slice(-199), event]);
  }, []);

  /* --- 切換資料集：清空目前畫面、載入新的一批貼文，並把邊整批送給後端 --- */
  const handleSelectDataset = useCallback(
    async (id) => {
      setDatasetId(id);
      setPosts([]);
      setLog([]);

      if (!id) {
        setDatasetState('idle');
        setDatasetError(null);
        return;
      }

      setDatasetState('loading');
      setDatasetError(null);

      try {
        const { posts: loadedPosts, edges } = await loadDataset(id);
        setPosts(loadedPosts);
        setDatasetState('idle');

        // 把這個資料集的傳播關係整批送給後端，讓關係圖前端能重新算一次 centrality
        edges.forEach((edge) => {
          emit(
            makeShareEvent({
              from: edge.from,
              to: edge.to,
              sourcePostId: null,
              threadId: edge.threadId,
              isRumour: edge.isRumour,
              clusterId: null,
              content: '',
            })
          );
        });
      } catch (err) {
        setDatasetState('error');
        setDatasetError(err.message);
      }
    },
    [emit]
  );

  /* --- 選取整串討論 → 依貼文年齡自動分派到模型或查證 --- */
  const handleCaptureThreads = useCallback(
    (threadIds) => {
      setSelecting(false);
      if (!threadIds.length) return;

      setCaptured((prev) => {
        const already = new Set(prev.map((c) => c.threadId));
        const newThreads = threadIds
          .filter((tid) => !already.has(tid))
          .map((tid) => ({
            threadId: tid,
            // 整串取出：原貼文加上所有回覆，不受畫面當下顯示範圍影響
            posts: posts.filter((p) => p.threadId === tid),
          }))
          .filter((t) => t.posts.length > 0);

        if (newThreads.length === 0) return prev;

        newThreads.forEach(({ threadId, posts: threadPosts }) => {
          analyzeThread(threadPosts).then((result) => {
            setCaptured((cur) =>
              cur.map((item) => (item.threadId === threadId ? { ...item, result } : item))
            );
          });
        });

        return [...prev, ...newThreads.map((t) => ({ ...t, result: null }))];
      });
    },
    [posts]
  );

  const handleClearCaptured = () => setCaptured([]);

  // 已選取的討論串，用來在動態牆上把對應貼文標示出來
  const capturedThreadIds = new Set(captured.map((c) => c.threadId));


  return (
    <div className="shell">
      <div className="platform">
        {/* 平台頂欄 */}
        <header className="topbar">
          <div className="topbarInner">
            <div className="brand">
              <span className="mark" aria-hidden="true" />
              <span className="brandName">IDS</span>
            </div>

            <label className="datasetPicker">
              <select
                className="datasetSelect"
                value={datasetId}
                onChange={(e) => handleSelectDataset(e.target.value)}
              >
                <option value="">— 未選擇 —</option>
                {datasets.length === 0 && <option disabled>尚無可用資料集</option>}
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </header>

        <main className="feed">
          {datasetState === 'loading' && (
            <div className="stateNotice">
              <p className="stateTitle">載入中</p>
              <p className="stateBody">正在讀取資料集內容⋯⋯</p>
            </div>
          )}

          {datasetState === 'error' && (
            <div className="stateNotice stateError">
              <p className="stateTitle">載入失敗</p>
              <p className="stateBody">{datasetError}</p>
            </div>
          )}

          {datasetState === 'idle' && posts.length === 0 && (
            <div className="empty">
              <p className="emptyTitle">尚未載入任何資料集</p>
              <p className="emptyBody">
                從右上角的下拉選單選擇一個資料集，這裡會顯示該事件的貼文與回覆內容。
              </p>
            </div>
          )}

          {posts.map((post) => (
            <PostCard
              key={post.id}
              post={post}
              users={{}}
              picked={capturedThreadIds.has(post.threadId)}
              onShare={() => {}}
              onComment={() => {}}
            />
          ))}
        </main>
      </div>

      {railOpen ? (
        <ControlRail status={status} queued={queued} log={log} onCollapse={() => setRailOpen(false)} />
      ) : (
        <button type="button" className="railHandle" onClick={() => setRailOpen(true)}>
          導播台
        </button>
      )}

      {/* AI 助手浮動開關：模擬瀏覽器側邊欄 AI agent 的入口 */}
      <button
        type="button"
        className={`aiToggle ${aiOpen ? 'aiToggleActive' : ''}`}
        onClick={() => setAiOpen((v) => !v)}
        aria-label={aiOpen ? '關閉 AI 助手' : '開啟 AI 助手'}
      >
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <path
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M4 10.5c0-3.6 2.7-6.5 6-6.5s6 2.9 6 6.5-2.7 6.5-6 6.5c-.9 0-1.8-.2-2.5-.6L4 17.5l1.1-3.1A6.4 6.4 0 014 10.5z"
          />
        </svg>
      </button>

      <AIAgentPanel
        open={aiOpen}
        onClose={() => setAiOpen(false)}
        captured={captured}
        selecting={selecting}
        onStartSelect={() => setSelecting(true)}
        onCancelSelect={() => setSelecting(false)}
        onClear={handleClearCaptured}
      />

      <SelectionOverlay
        active={selecting}
        onCaptureThreads={handleCaptureThreads}
        onCancel={() => setSelecting(false)}
      />
    </div>
  );
}
