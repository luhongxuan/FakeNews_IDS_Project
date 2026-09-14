// 檔案位置：frontend/src/components/TrendingRadar.jsx
// 給一般使用者：「現在很紅」——真的即時資料（Bluesky 公開貼文＋語意分群），
// 不是模擬。跟 FactCheckTool 同一種亮色、日常的視覺風格，刻意不提「介入」
// 「cutoff」「節點」這些操作者才需要的詞。只回答「現在大家在傳什麼、
// 查起來可不可信」。
import { useState, useEffect, useCallback } from 'react';
import styles from './TrendingRadar.module.css';

const POLL_INTERVAL_MS = 45000;

const CREDIBILITY_META = {
  likely_true: { label: '看起來是真的', className: 'true' },
  disputed: { label: '證據互相矛盾', className: 'disputed' },
  likely_false: { label: '看起來是假的', className: 'false' },
  unverified: { label: '查無足夠證據', className: 'unverified' },
};

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

const TrendingRadar = ({ onBack }) => {
  const [threads, setThreads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [verifications, setVerifications] = useState({}); // id -> { loading, data, error }
  const [minEngagement, setMinEngagement] = useState(3); // hide brand-new posts with no replies yet, by default

  const fetchThreads = useCallback(() => {
    const controller = new AbortController();
    fetch(`/api/radar/threads?limit=20&min_engagement=${minEngagement}`, { signal: controller.signal })
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

  const runVerification = (id) => {
    setVerifications((prev) => ({ ...prev, [id]: { loading: true, data: null, error: null } }));
    fetch(`/api/radar/threads/${id}/verify`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setVerifications((prev) => ({ ...prev, [id]: { loading: false, data, error: null } })))
      .catch((err) => setVerifications((prev) => ({ ...prev, [id]: { loading: false, data: null, error: err.message } })));
  };

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.brand}>🔥 現在很紅</div>
        <button type="button" className={styles.backLink} onClick={onBack}>返回</button>
      </header>

      <main className={styles.main}>
        <h1 className={styles.title}>現在大家在傳什麼？</h1>
        <p className={styles.subtitle}>即時監控網路上正在流傳的內容，點「查一下」幫你找找看有沒有獨立來源可以佐證。</p>

        <div className={styles.filterRow}>
          <label className={styles.filterLabel} htmlFor="radar-engagement-filter">顯示範圍</label>
          <select
            id="radar-engagement-filter"
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

        {loading && (
          <div className={styles.loadingCard}>
            <span className={styles.spinner} />
            正在監控中...
          </div>
        )}
        {error && <div className={styles.errorCard}>載入失敗：{error}</div>}
        {!loading && !error && threads.length === 0 && (
          <div className={styles.emptyCard}>目前沒有監控到明顯的熱門話題。</div>
        )}

        <div className={styles.list}>
          {threads.map((thread) => {
            const v = verifications[thread.id];
            const meta = v?.data ? (CREDIBILITY_META[v.data.credibility] || CREDIBILITY_META.unverified) : null;
            return (
              <div key={thread.id} className={styles.card}>
                <div className={styles.cardTop}>
                  {thread.member_count > 1 && (
                    <span className={styles.memberTag}>{thread.member_count} 則相關貼文都在討論這件事</span>
                  )}
                  <span className={styles.timeText}>{formatRelativeTime(thread.last_seen_at)}</span>
                </div>
                <p className={styles.cardText}>{thread.representative_text}</p>

                {!v && (
                  <button type="button" className={styles.checkBtn} onClick={() => runVerification(thread.id)}>
                    查一下
                  </button>
                )}
                {v?.loading && (
                  <div className={styles.verifyingHint}>
                    <span className={styles.spinnerSmall} />
                    正在查證，本機推理可能需要數十秒...
                  </div>
                )}
                {v?.error && <div className={styles.errorText}>查證失敗：{v.error}</div>}
                {meta && (
                  <div className={styles.resultBlock}>
                    <div className={styles.resultTop}>
                      <span className={`${styles.verdictBadge} ${styles[meta.className]}`}>{meta.label}</span>
                      <span className={styles.confidence}>信心程度 {Math.round((v.data.confidence || 0) * 100)}%</span>
                    </div>
                    <p className={styles.summary}>{v.data.summary}</p>
                    {v.data.evidence?.length > 0 && (
                      <ul className={styles.evidenceList}>
                        {v.data.evidence.map((item, index) => (
                          <li key={index} className={styles.evidenceItem}>
                            <a href={item.url} target="_blank" rel="noreferrer" className={styles.evidenceLink}>{item.title || item.url}</a>
                            <span className={styles.evidenceSource}>{item.source}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </main>
    </div>
  );
};

export default TrendingRadar;
