// 檔案位置：frontend/src/components/LiveScenarioSimulator.jsx
// 「即時監控儀表板」：給社群平台維運/信任與安全團隊用的操作介面。
// 不是遊戲——同時呈現多則正在監控的貼文、量化的擴散預估，維運人員可以
// 依風險排序自行挑選處理順序，而不是被迫一則一則被動反應。節奏模擬
// （貼文陸續「抵達」）是為了呈現即時監控的感覺，底層資料仍是同一套
// 已驗證的 v5 模型排序結果（不需要 social-frontend 真的送 WebSocket 事件）。
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import styles from './LiveScenarioSimulator.module.css';

const QUEUE_SIZE = 12;
const FIRST_ARRIVAL_DELAY_MS = 300;
const ARRIVAL_DELAY_MS = 2200;
const NEW_BADGE_MS = 5000;

function riskLabel(score) {
  const value = Number(score) || 0;
  if (value >= 80) return { text: '高風險', className: 'riskHigh' };
  if (value >= 50) return { text: '中風險', className: 'riskMedium' };
  return { text: '低風險', className: 'riskLow' };
}

const CREDIBILITY_LABEL = {
  likely_true: '證據傾向支持', disputed: '證據互相矛盾',
  likely_false: '證據傾向反駁', unverified: '尚無足夠證據',
};

const LiveScenarioSimulator = ({ eventId, onBack, onLogoClick }) => {
  const [pool, setPool] = useState([]);
  const [loadingPool, setLoadingPool] = useState(true);
  const [poolError, setPoolError] = useState(null);

  const [running, setRunning] = useState(false);
  const [queueIndex, setQueueIndex] = useState(-1);
  const timerRef = useRef(null);

  // items keyed by thread_id, insertion order preserved for stable rendering.
  const [items, setItems] = useState({});
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    setPool([]); setPoolError(null); setQueueIndex(-1); setRunning(false); setItems({}); setExpandedId(null);
    clearTimeout(timerRef.current);
    if (!eventId) { setLoadingPool(false); return; }
    setLoadingPool(true);
    const controller = new AbortController();
    fetch(`/api/events/${eventId}/threads?limit=${QUEUE_SIZE}`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setPool(data.threads || []))
      .catch((error) => { if (error.name !== 'AbortError') setPoolError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingPool(false); });
    return () => controller.abort();
  }, [eventId]);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  const scheduleNextArrival = useCallback((fromIndex) => {
    const delay = fromIndex === -1 ? FIRST_ARRIVAL_DELAY_MS : ARRIVAL_DELAY_MS;
    timerRef.current = setTimeout(() => {
      setQueueIndex((i) => i + 1);
    }, delay);
  }, []);

  const startMonitoring = () => {
    setRunning(true);
    scheduleNextArrival(queueIndex);
  };

  // Whenever queueIndex advances, materialize that thread into `items` and
  // fetch its cascade (for the predicted-spread stat) + any cached
  // verification, then schedule the next arrival if there's more in the pool.
  useEffect(() => {
    if (queueIndex < 0 || queueIndex >= pool.length) return;
    const thread = pool[queueIndex];
    const arrivedAt = Date.now();
    setItems((prev) => ({
      ...prev,
      [thread.thread_id]: {
        thread, arrivedAt,
        cascade: null, cascadeLoading: true,
        verification: null, verifyLoading: true,
        decision: null, deciding: false, decisionError: null,
      },
    }));

    const cascadeController = new AbortController();
    fetch(`/api/threads/${thread.thread_id}/cascade?event_id=${eventId}`, { signal: cascadeController.signal })
      .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
      .then((data) => setItems((prev) => (prev[thread.thread_id]
        ? { ...prev, [thread.thread_id]: { ...prev[thread.thread_id], cascade: data, cascadeLoading: false } }
        : prev)))
      .catch(() => setItems((prev) => (prev[thread.thread_id]
        ? { ...prev, [thread.thread_id]: { ...prev[thread.thread_id], cascadeLoading: false } }
        : prev)));

    const verifyController = new AbortController();
    fetch(`/api/threads/${thread.thread_id}/verify`, { signal: verifyController.signal })
      .then((res) => { if (res.status === 404) return null; if (!res.ok) throw new Error(); return res.json(); })
      .then((data) => setItems((prev) => (prev[thread.thread_id]
        ? { ...prev, [thread.thread_id]: { ...prev[thread.thread_id], verification: data, verifyLoading: false } }
        : prev)))
      .catch(() => setItems((prev) => (prev[thread.thread_id]
        ? { ...prev, [thread.thread_id]: { ...prev[thread.thread_id], verifyLoading: false } }
        : prev)));

    if (running && queueIndex + 1 < pool.length) scheduleNextArrival(queueIndex);

    return () => { cascadeController.abort(); verifyController.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queueIndex]);

  const decide = (threadId, choice) => {
    const item = items[threadId];
    if (!item || item.deciding || item.decision) return;
    setItems((prev) => ({ ...prev, [threadId]: { ...prev[threadId], deciding: true, decisionError: null } }));
    fetch(`/api/threads/${threadId}/intervene?event_id=${eventId}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((result) => setItems((prev) => ({
        ...prev, [threadId]: { ...prev[threadId], deciding: false, decision: choice, interveneResult: result },
      })))
      .catch((error) => setItems((prev) => ({
        ...prev, [threadId]: { ...prev[threadId], deciding: false, decisionError: error.message },
      })));
  };

  const orderedItems = useMemo(() => {
    return Object.values(items).sort((a, b) => {
      const scoreDiff = (Number(b.thread.percentile_score) || 0) - (Number(a.thread.percentile_score) || 0);
      if (a.decision && !b.decision) return 1;
      if (!a.decision && b.decision) return -1;
      return scoreDiff;
    });
  }, [items]);

  const pendingHighRisk = orderedItems.filter((it) => !it.decision && riskLabel(it.thread.percentile_score).className === 'riskHigh').length;
  const blockedCount = orderedItems.filter((it) => it.decision === 'blocked').length;
  const predictedAdditionalSpread = orderedItems.reduce((sum, it) => {
    if (it.decision || !it.cascade) return sum;
    const total = it.cascade.nodes.length;
    const observed = it.cascade.nodes.filter((n) => n.observed_by_cutoff).length;
    return sum + Math.max(0, total - observed);
  }, 0);
  const monitoredCount = orderedItems.length;

  const finished = queueIndex >= 0 && queueIndex >= pool.length - 1 && pool.length > 0
    && Object.values(items).every((it) => !it.cascadeLoading);

  return (
    <div className={styles.stage}>
      <header className={styles.topNav}>
        <button type="button" className={styles.brand} onClick={onLogoClick}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
          即時監控儀表板
        </button>
        <div className={styles.liveStatus}>
          {running && <><span className={styles.liveDot} />監控中</>}
          {!running && queueIndex === -1 && '尚未開始'}
          {!running && queueIndex >= 0 && '已暫停'}
        </div>
        <button className={styles.linkBtn} onClick={onBack}>返回研究視角</button>
      </header>

      <div className={styles.explainer}>
        排序分數與擴散預估皆來自同一套已驗證的 v5 模型結果；此畫面以節奏模擬即時抵達，
        呈現維運團隊「同時監控多則貼文、依風險自行排序處理」的操作情境。
      </div>

      {!loadingPool && !poolError && (
        <div className={styles.kpiBar}>
          <div className={styles.kpiCard}>
            <div className={styles.kpiValue}>{monitoredCount}</div>
            <div className={styles.kpiLabel}>監控中貼文</div>
          </div>
          <div className={`${styles.kpiCard} ${pendingHighRisk > 0 ? styles.kpiWarn : ''}`}>
            <div className={styles.kpiValue}>{pendingHighRisk}</div>
            <div className={styles.kpiLabel}>待處理・高風險</div>
          </div>
          <div className={styles.kpiCard}>
            <div className={styles.kpiValue}>{predictedAdditionalSpread}</div>
            <div className={styles.kpiLabel}>預估後續擴散量（未處理項目合計）</div>
          </div>
          <div className={styles.kpiCard}>
            <div className={styles.kpiValue}>{blockedCount}</div>
            <div className={styles.kpiLabel}>已攔截</div>
          </div>
        </div>
      )}

      <main className={styles.stageBody}>
        {loadingPool && <div className={styles.centerHint}><span className={styles.spinner} />載入中...</div>}
        {poolError && <div className={styles.centerHint}>載入失敗：{poolError}</div>}

        {!loadingPool && !poolError && queueIndex === -1 && (
          <div className={styles.startCard}>
            <div className={styles.startTitle}>準備開始監控</div>
            <div className={styles.startBody}>啟動後，系統會依模型判斷的優先度陸續回報新監控到的貼文，可同時處理多則。</div>
            <button className={styles.primaryBtn} onClick={startMonitoring} disabled={pool.length === 0}>
              ▶ 開始即時監控
            </button>
          </div>
        )}

        {queueIndex >= 0 && (
          <div className={styles.queueList}>
            {orderedItems.map((item) => {
              const risk = riskLabel(item.thread.percentile_score);
              const isNew = Date.now() - item.arrivedAt < NEW_BADGE_MS;
              const total = item.cascade ? item.cascade.nodes.length : null;
              const observed = item.cascade ? item.cascade.nodes.filter((n) => n.observed_by_cutoff).length : null;
              const expanded = expandedId === item.thread.thread_id;
              return (
                <div key={item.thread.thread_id} className={`${styles.row} ${item.decision ? styles.rowDecided : ''}`}>
                  <div className={styles.rowMain} onClick={() => setExpandedId(expanded ? null : item.thread.thread_id)}>
                    <div className={styles.rowBadges}>
                      {isNew && !item.decision && <span className={styles.newTag}>NEW</span>}
                      <span className={`${styles.riskTag} ${styles[risk.className]}`}>{risk.text}</span>
                    </div>
                    <div className={styles.rowText}>{item.thread.preview_text || '(無法讀取原文)'}</div>
                    <div className={styles.rowStat}>
                      {item.cascadeLoading && <span className={styles.spinnerSmall} />}
                      {!item.cascadeLoading && total !== null && (
                        <span>已觀察 <b>{observed}</b> → 預估 <b>{total}</b></span>
                      )}
                    </div>
                    <div className={styles.rowVerify}>
                      {item.verifyLoading && <span className={styles.spinnerSmall} />}
                      {!item.verifyLoading && item.verification && (
                        <span className={`${styles.credBadge} ${styles[`cred_${item.verification.credibility}`] || ''}`}>
                          {CREDIBILITY_LABEL[item.verification.credibility] || item.verification.credibility}
                        </span>
                      )}
                      {!item.verifyLoading && !item.verification && <span className={styles.credBadgeMuted}>尚無查證</span>}
                    </div>
                    <div className={styles.rowActions} onClick={(e) => e.stopPropagation()}>
                      {!item.decision && (
                        <>
                          <button className={styles.blockBtnSmall} disabled={item.deciding}
                            onClick={() => decide(item.thread.thread_id, 'blocked')}>
                            {item.deciding ? '處理中' : '介入'}
                          </button>
                          <button className={styles.ignoreBtnSmall} disabled={item.deciding}
                            onClick={() => decide(item.thread.thread_id, 'observed')}>
                            忽略
                          </button>
                        </>
                      )}
                      {item.decision === 'blocked' && (
                        <span className={styles.decidedTagBlocked}>
                          已攔截・減少 {item.interveneResult?.blocked_node_count ?? '—'} 則
                        </span>
                      )}
                      {item.decision === 'observed' && (
                        <span className={styles.decidedTagObserved}>
                          已標記觀察・後續 {item.interveneResult?.blocked_node_count ?? '—'} 則未攔截
                        </span>
                      )}
                    </div>
                  </div>

                  {expanded && (
                    <div className={styles.rowDetail}>
                      {item.decisionError && <div className={styles.errorText}>操作失敗：{item.decisionError}</div>}
                      {item.verifyLoading && <div className={styles.rowDetailHint}>查證載入中...</div>}
                      {!item.verifyLoading && !item.verification && (
                        <div className={styles.rowDetailHint}>此貼文尚無獨立查證紀錄。</div>
                      )}
                      {item.verification && (
                        <>
                          <div className={styles.verifySummary}>{item.verification.summary}</div>
                          {item.verification.queries_used?.length > 0 && (
                            <div className={styles.verifyQueries}>搜尋關鍵字：{item.verification.queries_used.join('、')}</div>
                          )}
                          {item.verification.evidence?.length > 0 && (
                            <ul className={styles.evidenceList}>
                              {item.verification.evidence.map((ev, index) => (
                                <li key={index} className={styles.evidenceItem}>
                                  <span className={styles.evidenceStance}>{ev.stance || '-'}</span>
                                  <a href={ev.url} target="_blank" rel="noreferrer" className={styles.evidenceLink}>{ev.title || ev.url}</a>
                                  <span className={styles.evidenceSource}>{ev.source}{ev.date ? ` · ${ev.date}` : ''}</span>
                                </li>
                              ))}
                            </ul>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {finished && (
          <div className={styles.finishedBanner}>
            本輪監控結束，共 {pool.length} 則貼文、攔截 {blockedCount} 則。
          </div>
        )}
      </main>
    </div>
  );
};

export default LiveScenarioSimulator;
