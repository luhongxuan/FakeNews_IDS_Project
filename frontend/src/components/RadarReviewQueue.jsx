// 檔案位置：frontend/src/components/RadarReviewQueue.jsx
// 研究/維運視角的「即時雷達待審核」佇列：跟 TrendingRadar 共用同一套
// 後端分群資料（/api/radar/threads）。純瀏覽用的清單，點一列會開到
// RadarEventDetail 看傳播圖、查證、介入預估——跟主頁「待分析事件」卡片
// 點進去是同一個目的地，避免同一份資料在兩個地方各自長出不同的詳情 UI。
import { useState, useEffect, useCallback } from 'react';
import styles from './RadarReviewQueue.module.css';

const POLL_INTERVAL_MS = 45000;

function toUtcMillis(iso) {
  // Backend serializes naive UTC datetimes (no offset suffix) via
  // .isoformat(). Without a trailing Z or +HH:MM, the Date constructor
  // treats the string as local time, silently shifting every timestamp by
  // the browser's UTC offset (e.g. something that just happened shows as
  // "8 hours ago" in UTC+8).
  return new Date(/Z$|[+-]\d{2}:\d{2}$/.test(iso) ? iso : `${iso}Z`).getTime();
}

function formatRelativeTime(iso) {
  if (!iso) return '';
  const diffMs = Date.now() - toUtcMillis(iso);
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return '剛剛';
  if (mins < 60) return `${mins} 分鐘前`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} 小時前`;
  return `${Math.floor(hours / 24)} 天前`;
}

const RadarReviewQueue = ({ onBack, onLogoClick, onOpenThread, onInterventionReviewNav }) => {
  const [threads, setThreads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Research view defaults to showing everything (including brand-new,
  // zero-reply posts) -- unlike TrendingRadar's general-user default, an
  // operator watching this queue wants full visibility, not a curated feed.
  const [minEngagement, setMinEngagement] = useState(0);

  const fetchThreads = useCallback(() => {
    const controller = new AbortController();
    fetch(`/api/radar/threads?limit=30&min_engagement=${minEngagement}`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => { setThreads(data.threads || []); setError(null); })
      .catch((err) => { if (err.name !== 'AbortError') setError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return controller;
  }, [minEngagement]);

  useEffect(() => {
    setLoading(true);
    const controller = fetchThreads();
    const interval = setInterval(fetchThreads, POLL_INTERVAL_MS);
    return () => { controller.abort(); clearInterval(interval); };
  }, [fetchThreads]);

  const toggleReviewed = (event, id, current) => {
    event.stopPropagation();
    fetch(`/api/radar/threads/${id}/mark_reviewed?reviewed=${!current}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
      .then((updated) => setThreads((prev) => prev.map((t) => (t.id === id ? updated : t))))
      .catch(() => {});
  };

  const pendingCount = threads.filter((t) => !t.reviewed).length;

  return (
    <div className={styles.stage}>
      <header className={styles.topNav}>
        <button type="button" className={styles.brand} onClick={onLogoClick}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
          即時雷達・待審核
        </button>
        <div className={styles.liveStatus}><span className={styles.liveDot} />{pendingCount} 則待審核</div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className={styles.linkBtn} onClick={onInterventionReviewNav}>自動干預建議</button>
          <button className={styles.linkBtn} onClick={onBack}>返回</button>
        </div>
      </header>

      <div className={styles.explainer}>
        真實 Bluesky 公開貼文，依語意相似度分群成「同一事件」。點一列進去可以看傳播圖、查證、跑介入預估。
      </div>

      <div className={styles.filterBar}>
        <label className={styles.filterLabel} htmlFor="radar-review-engagement-filter">顯示範圍</label>
        <select
          id="radar-review-engagement-filter"
          className={styles.filterSelect}
          value={minEngagement}
          onChange={(e) => setMinEngagement(Number(e.target.value))}
        >
          <option value={0}>全部（含剛發生、還沒人回應的）</option>
          <option value={3}>已經有人在討論</option>
          <option value={20}>討論熱烈</option>
          <option value={100}>正在瘋傳</option>
        </select>
      </div>

      <main className={styles.stageBody}>
        {loading && <div className={styles.centerHint}><span className={styles.spinner} />載入中...</div>}
        {error && <div className={styles.centerHint}>載入失敗：{error}</div>}

        <div className={styles.queueList}>
          {threads.map((thread) => (
            <div
              key={thread.id}
              className={`${styles.row} ${thread.reviewed ? styles.rowReviewed : ''}`}
              onClick={() => onOpenThread(thread.id)}
            >
              <div className={styles.rowMain}>
                <div className={styles.rowBadges}>
                  {thread.member_count > 1 && <span className={styles.memberTag}>{thread.member_count} 則同事件</span>}
                  {thread.reviewed && <span className={styles.reviewedTag}>已審核</span>}
                </div>
                <div className={styles.rowText}>{thread.representative_text}</div>
                <div className={styles.rowMeta}>
                  <span>{thread.matched_keywords.join('、')}</span>
                  <span>{formatRelativeTime(thread.last_seen_at)}</span>
                  <button className={styles.reviewToggleBtn} onClick={(e) => toggleReviewed(e, thread.id, thread.reviewed)}>
                    {thread.reviewed ? '取消已審核' : '標記已審核'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
};

export default RadarReviewQueue;
