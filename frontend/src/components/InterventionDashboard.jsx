// 檔案位置：frontend/src/components/InterventionDashboard.jsx
// 早期介入建議儀表板：讀取真實模型 (v5 text RF, 30 分鐘截止點) 的排序分數，
// 並在真實 PHEME 傳播鏈上模擬「封鎖來源貼文」的介入效果。
import { useState, useEffect, useMemo, useRef } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import styles from './InterventionDashboard.module.css';
import { cascadeStylesheet, cascadeLayout } from './graphConfig';

const InterventionDashboard = ({ eventId, onBack, onLogoClick, onProfileNav, onGraphNav, onLiveScenarioNav, onRadarReviewNav }) => {
  const [threads, setThreads] = useState([]);
  const [loadingThreads, setLoadingThreads] = useState(true);
  const [threadsError, setThreadsError] = useState(null);

  const [selectedThreadId, setSelectedThreadId] = useState(null);
  const [cascade, setCascade] = useState(null);
  const [loadingCascade, setLoadingCascade] = useState(false);
  const [cascadeError, setCascadeError] = useState(null);
  const [viewMode, setViewMode] = useState('decision');

  const [interveneResult, setInterveneResult] = useState(null);
  const [intervening, setIntervening] = useState(false);
  const [interventionError, setInterventionError] = useState(null);

  const [verification, setVerification] = useState(null);
  const [verifying, setVerifying] = useState(false);
  const [checkingVerification, setCheckingVerification] = useState(false);
  const [verificationError, setVerificationError] = useState(null);
  // agent 查證進行中，即時同步顯示它目前做到哪一步了（見 runVerification）；
  // 跟 verification（最終報告）分開放，跑完後這個就沒用了，避免兩份資料互相覆蓋。
  const [liveToolCalls, setLiveToolCalls] = useState([]);
  const cyRef = useRef(null);
  const cascadeRequestRef = useRef(null);
  const progressPollRef = useRef(null);

  const stopProgressPolling = () => {
    if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }
  };

  useEffect(() => {
    setThreads([]); setThreadsError(null); setSelectedThreadId(null);
    setCascade(null); setCascadeError(null); setInterveneResult(null); setInterventionError(null); setViewMode('decision');
    setVerification(null); setVerificationError(null);
    if (!eventId) { setLoadingThreads(false); return; }
    setLoadingThreads(true);
    const controller = new AbortController();
    fetch(`/api/events/${eventId}/threads?limit=1000`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setThreads(data.threads || []))
      .catch((error) => { if (error.name !== 'AbortError') setThreadsError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingThreads(false); });
    return () => controller.abort();
  }, [eventId]);

  useEffect(() => () => stopProgressPolling(), []);

  const selectThread = (threadId) => {
    cascadeRequestRef.current?.abort();
    const controller = new AbortController();
    cascadeRequestRef.current = controller;
    stopProgressPolling();
    setSelectedThreadId(threadId);
    setCascade(null);
    setCascadeError(null);
    setInterveneResult(null);
    setInterventionError(null);
    setViewMode('decision');
    setVerification(null);
    setVerificationError(null);
    setLiveToolCalls([]);
    setLoadingCascade(true);
    fetch(`/api/threads/${threadId}/cascade?event_id=${eventId}`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setCascade(data))
      .catch((error) => { if (error.name !== 'AbortError') setCascadeError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingCascade(false); });

    // 換 thread 時先問一次有沒有已經查證過的結果，有就直接顯示，
    // 不用每次都手動按「產生查證報告」——只有從沒查過的 thread 才需要手動觸發。
    setCheckingVerification(true);
    fetch(`/api/threads/${threadId}/verify`, { signal: controller.signal })
      .then((res) => { if (res.status === 404) return null; if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setVerification(data))
      .catch((error) => { if (error.name !== 'AbortError') setVerificationError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setCheckingVerification(false); });
  };

  const runIntervention = () => {
    if (!selectedThreadId) return;
    setIntervening(true);
    setInterventionError(null);
    fetch(`/api/threads/${selectedThreadId}/intervene?event_id=${eventId}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => { setInterveneResult(data); setViewMode('replay'); })
      .catch((error) => { setInterveneResult(null); setInterventionError(error.message); })
      .finally(() => setIntervening(false));
  };

  const runVerification = (force = false) => {
    if (!selectedThreadId) return;
    setVerifying(true);
    setVerificationError(null);
    setLiveToolCalls([]);

    // 這次呼叫自己取一個 id，讓後端的 verification_progress 用同一把 key
    // 邊做邊記錄；下面邊跑邊用同一個 id 去輪詢，畫面就能跟著 agent 目前做
    // 到哪一步即時更新，而不是整個查證跑完（可能數十秒）才一次全部顯示。
    const requestId = (crypto.randomUUID ? crypto.randomUUID() : `rv-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    stopProgressPolling();
    progressPollRef.current = setInterval(() => {
      fetch(`/api/verification_progress/${requestId}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => { if (data) setLiveToolCalls(data.tool_calls || []); })
        .catch(() => {}); // 輪詢失敗不影響主要查證流程，安靜略過就好
    }, 700);

    const url = `/api/threads/${selectedThreadId}/verify?event_id=${eventId}&request_id=${requestId}${force ? '&force=true' : ''}`;
    fetch(url, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setVerification(data))
      .catch((error) => { setVerification(null); setVerificationError(error.message); })
      .finally(() => { stopProgressPolling(); setLiveToolCalls([]); setVerifying(false); });
  };

  const CREDIBILITY_LABEL = {
    likely_true: '證據傾向支持', disputed: '證據互相矛盾',
    likely_false: '證據傾向反駁', unverified: '尚無足夠證據',
  };

  const TOOL_NAME_LABEL = {
    search_web: '🌐 一般網頁搜尋', search_news: '📰 新聞搜尋', search_fact_checks: '✅ 查核資料庫搜尋',
    fetch_article: '📄 讀取全文',
  };

  const blockedIds = useMemo(
    () => new Set(interveneResult?.blocked_node_ids || []),
    [interveneResult],
  );

  const isReplayMode = viewMode === 'replay';
  const elements = useMemo(() => {
    if (!cascade) return [];
    const visibleNodes = cascade.nodes.filter((node) => isReplayMode || node.observed_by_cutoff);
    const visibleIds = new Set(visibleNodes.map((node) => node.id));
    const nodeElements = visibleNodes.map((node) => ({
      data: {
        id: node.id,
        label: '@' + node.screen_name,
        text: node.text,
        offset_sec: node.offset_sec,
        isSource: node.is_source,
        observed: node.observed_by_cutoff,
        blocked: blockedIds.has(node.id),
      },
    }));
    const edgeElements = cascade.edges
      .filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target))
      .map((edge) => ({
        data: { source: edge.source, target: edge.target, blockedEdge: blockedIds.has(edge.target) },
      }));
    return [...nodeElements, ...edgeElements];
  }, [cascade, isReplayMode, blockedIds]);

  // react-cytoscapejs only runs the `layout` prop once, on mount. When
  // `elements` changes afterwards (toggling full-truth, or coloring blocked
  // nodes after an intervention), newly added/changed elements are patched
  // into the graph model but keep stacking at their default position unless
  // the layout is re-run explicitly.
  useEffect(() => {
    if (cyRef.current && elements.length > 0) {
      cyRef.current.layout(cascadeLayout).run();
      // A new decision snapshot should start at a readable, consistent scale.
      // Replay mode intentionally preserves this scale instead of shrinking
      // every node to fit the full historical cascade on screen.
      if (!isReplayMode) {
        cyRef.current.zoom(1);
        cyRef.current.center();
      }
    }
  }, [elements, isReplayMode]);

  return (
    <div className={styles.dashboard}>
      <header className={styles.topNav}>
        <div className={styles.navLeft}>
          <button type="button" className={styles.navTitle} onClick={onLogoClick}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            情報追蹤系統
          </button>
        </div>
        <div className={styles.navCenter}>早期介入建議 — {eventId ? eventId.toUpperCase() : '未選取事件'}</div>
        <div className={styles.navRight}>
          {onLiveScenarioNav && <button className={styles.linkBtn} onClick={onLiveScenarioNav}>▶ 即時情境模擬</button>}
          {onRadarReviewNav && <button className={styles.linkBtn} onClick={onRadarReviewNav}>📡 即時雷達待審核</button>}
          <button className={styles.linkBtn} onClick={onGraphNav}>使用者拓撲圖（原型）</button>
          <button className={styles.linkBtn} onClick={onProfileNav}>實體關聯檔案</button>
          <button className={styles.linkBtn} onClick={onBack}>返回事件總覽</button>
        </div>
      </header>

      <div className={styles.explainer}>
        <b>決策模式</b>只顯示貼文發出後 <b>30 分鐘內</b>可觀察的資訊與 v5 text RF 的優先排序。
        執行「研究回放驗證」後，才會顯示 cutoff 後的歷史節點與回溯結果；它們不是模型輸入，也不是即時預測。
      </div>

      <div className={styles.bodyLayout}>
        <aside className={styles.threadList}>
          <div className={styles.listHeader}>建議介入清單（依模型分數排序，共 {threads.length} 則）</div>
          {loadingThreads && <div className={styles.emptyHint}><span className={styles.spinner} />載入中...</div>}
          {threadsError && <div className={styles.emptyHint}>載入失敗：{threadsError}</div>}
          {!loadingThreads && !threadsError && threads.length === 0 && (
            <div className={styles.emptyHint}>此事件目前沒有可用的模型分數。</div>
          )}
          {threads.map((thread) => (
            <button
              type="button"
              key={thread.thread_id}
              className={`${styles.threadCard} ${selectedThreadId === thread.thread_id ? styles.threadCardActive : ''}`}
              onClick={() => selectThread(thread.thread_id)}
              aria-pressed={selectedThreadId === thread.thread_id}
            >
              <div className={styles.threadRankRow}>
                <span className={styles.threadRank}>#{thread.rank}</span>
                <span className={styles.threadScore}>優先度 {Number(thread.percentile_score || 0).toFixed(0)}</span>
              </div>
              <div className={styles.threadPreview}>{thread.preview_text || '(無法讀取原文)'}</div>
              <div className={styles.threadFooter}>僅依 30 分鐘 snapshot 排序</div>
            </button>
          ))}
        </aside>

        <main className={styles.cascadePane}>
          {!selectedThreadId && (
            <div className={styles.emptyHint}>請從左側清單選擇一則貼文，查看其真實傳播鏈。</div>
          )}

          {selectedThreadId && (
            <>
              <div className={styles.cascadeHeader}>
                <div className={styles.modeSummary} aria-live="polite">
                  <span className={`${styles.modeBadge} ${isReplayMode ? styles.modeBadgeReplay : ''}`}>
                    {isReplayMode ? '研究回放驗證' : '30 分鐘決策快照'}
                  </span>
                  {isReplayMode && '目前含 cutoff 後的歷史節點，僅供驗證。'}
                </div>
                <div className={styles.actionGroup}>
                  {interveneResult && <button className={styles.secondaryBtn} onClick={() => setViewMode(isReplayMode ? 'decision' : 'replay')}>
                    {isReplayMode ? '返回決策快照' : '查看回放結果'}
                  </button>}
                  <button className={styles.interveneBtn} onClick={runIntervention} disabled={intervening}>
                    {intervening ? '回放載入中...' : '執行研究回放驗證'}
                  </button>
                </div>
              </div>

              {loadingCascade && <div className={styles.emptyHint}><span className={styles.spinner} />載入傳播鏈中...</div>}
              {cascadeError && <div className={styles.emptyHint} role="alert">傳播鏈載入失敗：{cascadeError}</div>}
              {interventionError && <div className={styles.inlineError} role="alert">回放驗證失敗：{interventionError}</div>}

              {!loadingCascade && cascade && (
                <div className={styles.graphArea}>
                  <CytoscapeComponent
                    elements={elements}
                    stylesheet={cascadeStylesheet}
                    layout={cascadeLayout}
                    style={{ width: '100%', height: '100%' }}
                    cy={(cy) => {
                      cyRef.current = cy;
                      cy.removeAllListeners('tap');
                      cy.on('tap', 'node', (evt) => {
                        cy.elements().removeClass('dimmed');
                        cy.elements().not(evt.target.closedNeighborhood()).addClass('dimmed');
                      });
                      cy.on('tap', (evt) => { if (evt.target === cy) cy.elements().removeClass('dimmed'); });
                    }}
                  />
                </div>
              )}

              {interveneResult && isReplayMode && (
                <div className={styles.resultPanel}>
                  <div className={styles.resultTitle}>介入前 / 介入後對比（回溯重播）</div>
                  <div className={styles.resultGrid}>
                    <div className={styles.resultBox}>
                      <div className={styles.resultLabel}>介入前：完整最終節點數</div>
                      <div className={styles.resultValue}>{interveneResult.total_nodes_full_cascade}</div>
                    </div>
                    <div className={styles.resultBox}>
                      <div className={styles.resultLabel}>30 分鐘當下已觀察節點數</div>
                      <div className={styles.resultValue}>{interveneResult.observed_nodes_at_cutoff}</div>
                    </div>
                    <div className={styles.resultBoxHighlight}>
                      <div className={styles.resultLabel}>介入後：成功阻止節點數</div>
                      <div className={styles.resultValueHighlight}>{interveneResult.blocked_node_count}</div>
                    </div>
                  </div>
                  <div className={styles.resultNote}>{interveneResult.note}</div>
                </div>
              )}

              <div className={styles.verifyPanel}>
                <div className={styles.verifyHeader}>
                  <div className={styles.resultTitle}>查證 agent 報告（事後決策輔助，非模型輸入）</div>
                  <button className={styles.secondaryBtn} onClick={() => runVerification(!!verification)} disabled={verifying || checkingVerification}>
                    {verifying ? '查證中...(本機 LLM 推理，可能需要數十秒)' : checkingVerification ? '確認快取中...' : verification ? '重新查證' : '產生查證報告'}
                  </button>
                </div>
                {verificationError && <div className={styles.inlineError} role="alert">查證失敗：{verificationError}</div>}
                {!verification && !verifying && !checkingVerification && !verificationError && (
                  <div className={styles.verifyNoEvidence}>這則貼文還沒有查證紀錄，按上方按鈕產生。</div>
                )}
                {verifying && (
                  <div className={styles.liveProgress}>
                    <div className={styles.liveProgressHeader}>
                      <span className={styles.livePulseDot} />
                      Agent 正在查證・已呼叫 {liveToolCalls.length} 次工具
                    </div>
                    {liveToolCalls.length > 0 && (
                      <ul className={styles.evidenceList}>
                        {liveToolCalls.map((call, index) => (
                          <li key={index} className={styles.evidenceItem}>
                            <span className={styles.evidenceStance}>{TOOL_NAME_LABEL[call.name] || call.name}</span>
                            {call.arguments?.query && <span className={styles.evidenceLink}>「{call.arguments.query}」</span>}
                            {call.arguments?.url && <span className={styles.evidenceLink}>{call.arguments.url}</span>}
                            <span className={styles.evidenceSource}>
                              回傳 {call.result_count ?? 0} 筆{call.automatic && '・自動預查'}{call.repeated && '・重複呼叫'}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
                {verification && (
                  <div className={styles.verifyBody} key={selectedThreadId}>
                    <div className={styles.verifyTopRow}>
                      <span className={`${styles.credBadge} ${styles[`cred_${verification.credibility}`] || ''}`}>
                        {CREDIBILITY_LABEL[verification.credibility] || verification.credibility}
                      </span>
                      <span className={styles.confidenceText}>信心程度 {Math.round((verification.confidence || 0) * 100)}%</span>
                      {verification.cached && <span className={styles.cachedTag}>（已快取，{verification.generated_at ? new Date(verification.generated_at).toLocaleString() : ''}）</span>}
                    </div>
                    <div className={styles.verifySummary}>{verification.summary}</div>
                    {verification.queries_used?.length > 0 && (
                      <div className={styles.verifyQueries}>搜尋關鍵字：{verification.queries_used.join('、')}</div>
                    )}
                    {verification.evidence?.length > 0 ? (
                      <ul className={styles.evidenceList}>
                        {verification.evidence.map((item, index) => (
                          <li key={index} className={styles.evidenceItem}>
                            <span className={styles.evidenceStance}>{item.stance || '-'}</span>
                            <a href={item.url} target="_blank" rel="noreferrer" className={styles.evidenceLink}>{item.title || item.url}</a>
                            <span className={styles.evidenceSource}>{item.source}{item.date ? ` · ${item.date}` : ''}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className={styles.verifyNoEvidence}>沒有找到可引用的獨立證據。</div>
                    )}
                    {/* 查完不代表過程就不重要了 -- 跑的時候即時看到的那份工具呼叫
                        紀錄，查完後仍原封不動留在這裡（收合展示，不佔版面），而
                        不是像 liveProgress 那樣跑完就清掉。用同一個 report_jsonb
                        裡本來就有的 tool_calls，不需要另外呼叫。 */}
                    {verification.tool_calls?.length > 0 && (
                      <details className={styles.processTrace}>
                        <summary className={styles.processTraceSummary}>
                          🔎 查證過程（{verification.model || '本機模型'}・{verification.tool_calls.length} 次工具呼叫）
                        </summary>
                        <ul className={styles.evidenceList}>
                          {verification.tool_calls.map((call, index) => (
                            <li key={index} className={styles.evidenceItem}>
                              <span className={styles.evidenceStance}>{TOOL_NAME_LABEL[call.name] || call.name}</span>
                              {call.arguments?.query && <span className={styles.evidenceLink}>「{call.arguments.query}」</span>}
                              {call.arguments?.url && <span className={styles.evidenceLink}>{call.arguments.url}</span>}
                              <span className={styles.evidenceSource}>
                                回傳 {call.result_count ?? 0} 筆{call.automatic && '・自動預查'}{call.repeated && '・重複呼叫'}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
};

export default InterventionDashboard;
