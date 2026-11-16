import React, { useCallback, useEffect, useRef, useState } from 'react';
import PostCard from './components/PostCard.jsx';
import ControlRail from './components/ControlRail.jsx';
import AIAgentPanel from './components/AIAgentPanel.jsx';
import SelectionOverlay from './components/SelectionOverlay.jsx';
import { createEventBridge, makeShareEvent, STATUS } from './lib/backend.js';
import { listDatasets, loadDataset } from './lib/datasets.js';
import { analyzePosts } from './lib/analysis.js';

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
  const [captured, setCaptured] = useState([]); // [{ id, post, result }]

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

  /* --- AI 助手：框選擷取到的貼文 id 清單 → 找出對應貼文 → 送去分析 --- */
  const handleCapture = useCallback(
    (ids) => {
      setSelecting(false);
      if (!ids.length) return;

      setCaptured((prev) => {
        const already = new Set(prev.map((c) => c.id));
        const newlyHit = ids
          .filter((id) => !already.has(id))
          .map((id) => posts.find((p) => p.id === id))
          .filter(Boolean);

        if (newlyHit.length === 0) return prev;

        // 非同步送分析，先讓這些項目以「分析中」狀態（result: null）出現在清單裡
        analyzePosts(newlyHit, { eventId: datasetId, allPosts: posts }).then((results) => {
          setCaptured((cur) =>
            cur.map((item) => {
              const found = results.find((r) => r.postId === item.id);
              return found ? { ...item, result: found } : item;
            })
          );
        });

        return [...prev, ...newlyHit.map((post) => ({ id: post.id, post, result: null }))];
      });
    },
    [datasetId, posts]
  );

  const handleClearCaptured = () => setCaptured([]);

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
            <PostCard key={post.id} post={post} users={{}} onShare={() => {}} onComment={() => {}} />
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
            fill="currentColor"
            d="M10 2l1.4 3.9L15.3 7l-3.9 1.4L10 12.3 8.6 8.4 4.7 7l3.9-1.1L10 2zM4 13.5l.8 2.1 2.1.8-2.1.8-.8 2.1-.8-2.1-2.1-.8 2.1-.8.8-2.1zM16 12.5l.9 2.4 2.4.9-2.4.9-.9 2.4-.9-2.4-2.4-.9 2.4-.9.9-2.4z"
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
        onCapture={handleCapture}
        onCancel={() => setSelecting(false)}
      />
    </div>
  );
}
