// 檔案位置：frontend/src/components/RadarEventDetail.jsx
// 「待分析事件」點進去一則雷達分群後的詳情頁：對應 PHEME 事件點進去會
// 看到的 InterventionDashboard——事件底下每一則貼文（thread）都可以個別
// 點選，切換傳播圖／查證／介入預估的對象，跟原本「選一個 thread 換一張
// cascade 圖」的操作方式一致，不是只能看群組代表貼文那一則。
// 介入模擬刻意不做 PHEME 那種「回放真實未來」：直播中的貼文還沒有已知
// 的未來，所以改成「依目前擴散速度推算」的預估。
import { useState, useEffect, useMemo, useRef } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import styles from './RadarEventDetail.module.css';
import { cascadeStylesheet, cascadeLayout } from './graphConfig';

const CREDIBILITY_LABEL = {
  likely_true: '證據傾向支持', disputed: '證據互相矛盾',
  likely_false: '證據傾向反駁', unverified: '尚無足夠證據',
};

const STANCE_LABEL = {
  supports: '支持', refutes: '反駁', context: '背景資訊', unrelated: '不相關',
};

const TOOL_NAME_LABEL = {
  search_web: '🌐 一般網頁搜尋', search_news: '📰 新聞搜尋', search_fact_checks: '✅ 查核資料庫搜尋',
  fetch_article: '📄 讀取全文', get_rf_prediction: '📈 RF 傳播分數', get_verification_status: '🗂 查詢既有查證快取',
};

// 把 verification_agent.py／intervention_agent.py 已經算好、也已經放進 API
// 回應的 tool_calls／queries_used／evidence 實際畫出來，讓人看到 agent 不是
// 黑箱：它查了什麼關鍵字、呼叫了哪些工具、每次拿回幾筆結果、有沒有重複呼叫
// 被快取擋掉，以及最後引用的證據分別是「支持／反駁／背景／不相關」。
const AgentTrace = ({ report }) => {
  const toolCalls = report?.tool_calls || [];
  const queries = report?.queries_used || [];
  const evidence = report?.evidence || [];
  if (toolCalls.length === 0 && queries.length === 0 && evidence.length === 0) return null;
  return (
    <details className={styles.agentTrace}>
      <summary className={styles.agentTraceSummary}>
        🔎 Agent 執行紀錄（{report.model || '本機模型'}・{toolCalls.length} 次工具呼叫）
      </summary>
      {queries.length > 0 && (
        <div className={styles.traceQueries}>
          {queries.map((q, i) => <span key={i} className={styles.auditChip}>🔍 {q}</span>)}
        </div>
      )}
      {toolCalls.length > 0 && (
        <ul className={styles.traceList}>
          {toolCalls.map((call, i) => (
            <li key={i} className={styles.traceItem}>
              <span className={styles.traceToolName}>{TOOL_NAME_LABEL[call.name] || call.name}</span>
              {call.arguments?.query && <span className={styles.traceArg}>「{call.arguments.query}」</span>}
              {call.arguments?.url && <span className={styles.traceArg}>{call.arguments.url}</span>}
              <span className={styles.traceMeta}>回傳 {call.result_count ?? 0} 筆</span>
              {call.automatic && <span className={styles.traceBadgeAuto}>自動預查</span>}
              {call.repeated && <span className={styles.traceBadgeRepeat}>重複呼叫・未耗網路</span>}
            </li>
          ))}
        </ul>
      )}
      {evidence.length > 0 && (
        <ul className={styles.traceList}>
          {evidence.map((e, i) => (
            <li key={i} className={styles.traceItem}>
              <span className={styles.traceStance}>{STANCE_LABEL[e.stance] || e.stance || '未標示'}</span>
              {e.url ? (
                <a href={e.url} target="_blank" rel="noreferrer" className={styles.traceLink}>{e.title || e.url}</a>
              ) : (
                <span className={styles.traceLink}>{e.title || '（無標題）'}</span>
              )}
              {e.source_tier && e.source_tier !== 'unknown' && <span className={styles.traceMeta}>{e.source_tier}</span>}
            </li>
          ))}
        </ul>
      )}
    </details>
  );
};

// 查證／政策決策還在跑（loading=true）的時候顯示：跟 AgentTrace 用同一套
// traceList/traceItem 樣式，但永遠展開（不是收合的 <details>）並帶一個
// 脈動小圓點，讓人一眼就知道「這是正在發生的事」而不是已經做完可以收合
// 的歷史紀錄。呼叫方每 700ms 輪詢一次 /api/verification_progress/{id}
// 把 calls 灌進來，agent 每呼叫一次新工具，這裡就多一行。
const LiveAgentTrace = ({ calls }) => (
  <div className={styles.liveTrace}>
    <div className={styles.liveTraceHeader}>
      <span className={styles.livePulseDot} />
      Agent 正在查證・已呼叫 {calls.length} 次工具
    </div>
    {calls.length > 0 && (
      <ul className={styles.traceList}>
        {calls.map((call, i) => (
          <li key={i} className={styles.traceItem}>
            <span className={styles.traceToolName}>{TOOL_NAME_LABEL[call.name] || call.name}</span>
            {call.arguments?.query && <span className={styles.traceArg}>「{call.arguments.query}」</span>}
            {call.arguments?.url && <span className={styles.traceArg}>{call.arguments.url}</span>}
            <span className={styles.traceMeta}>回傳 {call.result_count ?? 0} 筆</span>
            {call.automatic && <span className={styles.traceBadgeAuto}>自動預查</span>}
            {call.repeated && <span className={styles.traceBadgeRepeat}>重複呼叫・未耗網路</span>}
          </li>
        ))}
      </ul>
    )}
  </div>
);

const INTERVENTION_ACTION_LABEL = {
  escalate_now: '建議立即升級',
  escalate_next_checkpoint: '建議下個 checkpoint 再看',
  hold: '低急迫性，持續追蹤',
  dismiss: '無需處理',
  hard: 'Hard：大幅降低曝光／限制轉發',
  soft: 'Soft：可逆的小幅降低曝光',
  none: 'None：不進行限制',
  deferred_agent_failure: '查證失敗：延後決策',
  deferred_inflight: '查證進行中：延後決策',
  deferred_budget: '配額未釋出：下輪再評估',
};

const INTERVENTION_ACTION_CLASS = {
  escalate_now: 'interventionEscalateNow',
  escalate_next_checkpoint: 'interventionEscalateNext',
  hold: 'interventionHold',
  dismiss: 'interventionDismiss',
  hard: 'interventionHard',
  soft: 'interventionSoft',
  none: 'interventionNone',
  deferred_agent_failure: 'interventionDeferred',
  deferred_inflight: 'interventionDeferred',
  deferred_budget: 'interventionDeferred',
};

const RadarEventDetail = ({ threadId, initialPostUri, onBack, onLogoClick }) => {
  const [thread, setThread] = useState(null);
  const [loadingThread, setLoadingThread] = useState(true);
  const [threadError, setThreadError] = useState(null);

  const [selectedUri, setSelectedUri] = useState(null);

  const [graph, setGraph] = useState(null);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [graphError, setGraphError] = useState(null);
  const cyRef = useRef(null);

  const [verifications, setVerifications] = useState({}); // uri -> { loading, data, error }
  const [estimates, setEstimates] = useState({}); // uri -> { loading, data, error }
  const [selectedNode, setSelectedNode] = useState(null); // clicked graph node's full data
  const [interventions, setInterventions] = useState({ loading: true, data: [], error: null });
  const [eventBudget, setEventBudget] = useState(null);
  const [policyRuns, setPolicyRuns] = useState({}); // uri -> { loading, data, error }
  // 查證／政策決策正在跑的時候，即時同步顯示 agent 目前做到哪一步
  // （見 runVerify／runPolicyDecision 對 /api/verification_progress/{id} 的輪詢）。
  // uri -> tool_calls[]；跑完就清掉，不跟最終結果混在一起。
  const [liveToolCalls, setLiveToolCalls] = useState({});
  const progressPollRef = useRef(null);
  const stopProgressPolling = () => {
    if (progressPollRef.current) { clearInterval(progressPollRef.current); progressPollRef.current = null; }
  };
  useEffect(() => () => stopProgressPolling(), []);

  useEffect(() => {
    if (!threadId) return;
    const controller = new AbortController();
    setLoadingThread(true);
    fetch(`/api/radar/threads/${threadId}`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => { setThread(data); setSelectedUri(initialPostUri || data.representative_uri); setThreadError(null); })
      .catch((err) => { if (err.name !== 'AbortError') setThreadError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingThread(false); });
    return () => controller.abort();
  }, [threadId, initialPostUri]);

  useEffect(() => {
    // Decisions are made per individual post (its own discussion thread),
    // not per whole clustered event -- filter by both the cluster and the
    // specific post currently selected, so switching which post you're
    // looking at (in the left panel) shows that post's own history.
    if (!threadId || !selectedUri) return;
    const controller = new AbortController();
    fetch(`/api/radar/interventions?thread_id=${threadId}&post_uri=${encodeURIComponent(selectedUri)}&limit=20`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setInterventions({ loading: false, data: data.decisions || [], error: null }))
      .catch((err) => { if (err.name !== 'AbortError') setInterventions({ loading: false, data: [], error: err.message }); });
    return () => controller.abort();
  }, [threadId, selectedUri]);

  useEffect(() => {
    // Budget is scoped per event (this cluster), not shared across all
    // events -- see backend/app/routers/radar.py's _remaining_budget.
    if (!threadId) return;
    const controller = new AbortController();
    fetch(`/api/radar/threads/${threadId}/budget_state`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setEventBudget(data))
      .catch(() => {});
    return () => controller.abort();
  }, [threadId, interventions]);

  useEffect(() => {
    if (!threadId || !selectedUri) return;
    const controller = new AbortController();
    setLoadingGraph(true);
    setSelectedNode(null);
    fetch(`/api/radar/threads/${threadId}/graph?post_uri=${encodeURIComponent(selectedUri)}`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => { setGraph(data); setGraphError(null); })
      .catch((err) => { if (err.name !== 'AbortError') setGraphError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingGraph(false); });
    return () => controller.abort();
  }, [threadId, selectedUri]);

  const elements = useMemo(() => {
    if (!graph) return [];
    const nodeElements = graph.nodes.map((node) => ({
      data: {
        id: node.id,
        label: '@' + node.screen_name,
        text: node.text,
        offset_sec: node.offset_sec,
        isSource: node.is_source,
        observed: node.observed_by_cutoff,
      },
    }));
    const edgeElements = graph.edges.map((edge) => ({ data: { source: edge.source, target: edge.target } }));
    return [...nodeElements, ...edgeElements];
  }, [graph]);

  useEffect(() => {
    if (cyRef.current && elements.length > 0) {
      cyRef.current.layout(cascadeLayout).run();
      cyRef.current.zoom(1);
      cyRef.current.center();
    }
  }, [elements]);

  const selectedMember = useMemo(() => {
    if (!thread) return null;
    if (selectedUri === thread.representative_uri) {
      return { uri: thread.representative_uri, text: thread.representative_text, author_handle: null };
    }
    return thread.members.find((m) => m.uri === selectedUri) || null;
  }, [thread, selectedUri]);

  // 幫查證／政策決策這兩個會跑本機 LLM、可能要數十秒的動作起一個
  // request_id，邊跑邊輪詢 /api/verification_progress/{id}，讓畫面上
  // liveToolCalls[uri] 跟著 agent 實際呼叫的工具即時同步更新。
  const startProgressPolling = (uri, requestId) => {
    stopProgressPolling();
    setLiveToolCalls((prev) => ({ ...prev, [uri]: [] }));
    progressPollRef.current = setInterval(() => {
      fetch(`/api/verification_progress/${requestId}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => { if (data) setLiveToolCalls((prev) => ({ ...prev, [uri]: data.tool_calls || [] })); })
        .catch(() => {});
    }, 700);
  };

  // 改成打 /api/radar/threads/{id}/verify（跟下面「真實政策決策」共用同一支
  // radar.py::_verification_context）而不是通用的 /api/factcheck。之前這裡固定
  // 查證整個 cluster 的 representative_text、也沒有帶 post_date，MVP 那邊查的
  // 卻是「這則貼文自己的文字」+「這則貼文真實的發文日期」——同一個 agent，兩份
  // 不同的輸入，兩個不同的快取 key，難怪常常兩邊結果對不上。現在兩邊固定吃
  // 同一個 cache key，"重新查證"／"重新產生介入建議" 之後永遠看到一致的結果。
  const runVerify = (uri, _text, force = false) => {
    setVerifications((prev) => ({ ...prev, [uri]: { loading: true, data: null, error: null } }));
    const requestId = crypto.randomUUID ? crypto.randomUUID() : `rv-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    startProgressPolling(uri, requestId);
    const params = new URLSearchParams({ post_uri: uri, request_id: requestId });
    if (force) params.set('force', 'true');
    fetch(`/api/radar/threads/${threadId}/verify?${params.toString()}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setVerifications((prev) => ({ ...prev, [uri]: { loading: false, data, error: null } })))
      .catch((err) => setVerifications((prev) => ({ ...prev, [uri]: { loading: false, data: null, error: err.message } })))
      .finally(() => { stopProgressPolling(); setLiveToolCalls((prev) => ({ ...prev, [uri]: [] })); });
  };

  const runEstimate = (uri) => {
    setEstimates((prev) => ({ ...prev, [uri]: { loading: true, data: null, error: null } }));
    fetch(`/api/radar/threads/${threadId}/estimate_intervene?post_uri=${encodeURIComponent(uri)}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setEstimates((prev) => ({ ...prev, [uri]: { loading: false, data, error: null } })))
      .catch((err) => setEstimates((prev) => ({ ...prev, [uri]: { loading: false, data: null, error: err.message } })));
  };

  const runPolicyDecision = (uri, force = false) => {
    setPolicyRuns((prev) => ({ ...prev, [uri]: { loading: true, data: null, error: null } }));
    const requestId = crypto.randomUUID ? crypto.randomUUID() : `pd-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    startProgressPolling(uri, requestId);
    // Same issue as runVerify above: without &force=true, a post that
    // already has an MVP decision at this checkpoint just gets handed back
    // that same cached row (see manual_policy_decision's own docstring) --
    // clicking the button again looked like it "did nothing new".
    const forceParam = force ? '&force=true' : '';
    fetch(`/api/radar/threads/${threadId}/manual_policy_decision?post_uri=${encodeURIComponent(uri)}&request_id=${requestId}${forceParam}`, { method: 'POST' })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
        return data;
      })
      .then((data) => {
        setPolicyRuns((prev) => ({ ...prev, [uri]: { loading: false, data, error: null } }));
        setInterventions((prev) => ({
          ...prev,
          loading: false,
          error: null,
          data: [data, ...prev.data.filter((item) => item.id !== data.id)],
        }));
        // MVP 這條路徑（/manual_policy_decision -> run_batch_verification）跟上面
        // 「查證這則貼文」按鈕（/api/factcheck）是兩個完全獨立的查證快取，各自
        // 可能在不同時間點跑出不同的真實 LLM 結果，畫面上會出現「上面跟下面
        // 查證結果不一致」的錯覺。data.verification 就是 MVP 這次實際採用、拿去
        // 做決策的證據（不管是剛查的還是命中冷卻快取），直接同步進 verifications
        // state，讓「查證這則貼文」永遠顯示跟決策依據一致的最新結果。
        if (data.verification) {
          setVerifications((prev) => ({ ...prev, [uri]: { loading: false, data: data.verification, error: null } }));
        }
      })
      .catch((err) => setPolicyRuns((prev) => ({ ...prev, [uri]: { loading: false, data: null, error: err.message } })))
      .finally(() => { stopProgressPolling(); setLiveToolCalls((prev) => ({ ...prev, [uri]: [] })); });
  };

  const selectPost = (uri) => {
    setInterventions({ loading: true, data: [], error: null });
    setSelectedUri(uri);
  };

  const markReviewed = () => {
    fetch(`/api/radar/threads/${threadId}/mark_reviewed?reviewed=${!thread.reviewed}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
      .then((data) => setThread(data))
      .catch(() => {});
  };

  const v = selectedUri ? verifications[selectedUri] : null;
  const est = selectedUri ? estimates[selectedUri] : null;
  const policyRun = selectedUri ? policyRuns[selectedUri] : null;

  return (
    <div className={styles.stage}>
      <header className={styles.topNav}>
        <button type="button" className={styles.brand} onClick={onLogoClick}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
          待分析事件・即時雷達
        </button>
        <button className={styles.linkBtn} onClick={onBack}>返回事件總覽</button>
      </header>

      {loadingThread && <div className={styles.centerHint}><span className={styles.spinner} />載入中...</div>}
      {threadError && <div className={styles.centerHint}>載入失敗：{threadError}</div>}

      {thread && (
        <>
          <div className={styles.summaryBar}>
            <p className={styles.summaryText}>{thread.representative_text}</p>
            <div className={styles.summaryMeta}>
              <span>{thread.matched_keywords.join('、')}</span>
              <span>{thread.member_count} 則相關貼文</span>
              <button className={styles.reviewBtn} onClick={markReviewed}>
                {thread.reviewed ? '取消已審核' : '標記已審核'}
              </button>
            </div>
          </div>

          <div className={styles.mainLayout}>
            <div className={styles.threadListPanel}>
              <div className={styles.panelTitle}>這個事件裡的貼文（點一則切換傳播圖）</div>
              <div className={styles.threadList}>
                <button
                  className={`${styles.threadItem} ${selectedUri === thread.representative_uri ? styles.threadItemActive : ''}`}
                  onClick={() => selectPost(thread.representative_uri)}
                >
                  <span className={styles.threadItemTag}>群組代表</span>
                  <span className={styles.threadItemText}>{thread.representative_text}</span>
                </button>
                {thread.members.filter((m) => m.uri !== thread.representative_uri).map((m) => (
                  <button
                    key={m.uri}
                    className={`${styles.threadItem} ${selectedUri === m.uri ? styles.threadItemActive : ''}`}
                    onClick={() => selectPost(m.uri)}
                  >
                    <span className={styles.threadItemHandle}>@{m.author_handle}</span>
                    <span className={styles.threadItemText}>{m.text}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.graphPanel}>
              <div className={styles.panelTitle}>傳播圖（這則貼文的真實回覆樹）</div>
              {loadingGraph && <div className={styles.centerHint}><span className={styles.spinner} />載入圖形中...</div>}
              {graphError && <div className={styles.centerHint}>圖形載入失敗：{graphError}</div>}
              {!loadingGraph && graph && (
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
                        setSelectedNode(evt.target.data());
                      });
                      cy.on('tap', (evt) => {
                        if (evt.target === cy) {
                          cy.elements().removeClass('dimmed');
                          setSelectedNode(null);
                        }
                      });
                    }}
                  />
                  {selectedNode && (
                    <div className={styles.nodeDetailCard}>
                      <div className={styles.nodeDetailTop}>
                        <span className={styles.nodeDetailHandle}>
                          {selectedNode.isSource ? '📍 來源貼文' : selectedNode.label}
                        </span>
                        <button className={styles.nodeDetailClose} onClick={() => { cyRef.current?.elements().removeClass('dimmed'); setSelectedNode(null); }}>✕</button>
                      </div>
                      <div className={styles.nodeDetailText}>{selectedNode.text || '(無法讀取內容)'}</div>
                      <div className={styles.nodeDetailMeta}>
                        {selectedNode.isSource
                          ? '這是傳播鏈的起點'
                          : selectedNode.offset_sec != null
                            ? `發文後 ${Math.round(selectedNode.offset_sec / 60)} 分鐘回覆${selectedNode.observed ? '（在 30 分鐘觀察窗內）' : '（30 分鐘之後）'}`
                            : '時間資訊不明'}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className={styles.sidePanel}>
              {selectedMember && (
                <div className={styles.selectedPreview}>
                  {selectedMember.author_handle && <div className={styles.selectedHandle}>@{selectedMember.author_handle}</div>}
                  <div className={styles.selectedText}>{selectedMember.text}</div>
                </div>
              )}

              <div className={styles.block}>
                <div className={styles.blockTitle}>查證這則貼文</div>
                {!v && (
                  <button className={styles.actionBtn} onClick={() => runVerify(selectedUri, selectedMember?.text || '')}>
                    🔍 查證
                  </button>
                )}
                {v?.loading && (
                  <>
                    <div className={styles.hintText}><span className={styles.spinnerSmall} />查證中，本機推理可能需要數十秒...</div>
                    <LiveAgentTrace calls={liveToolCalls[selectedUri] || []} />
                  </>
                )}
                {v?.error && <div className={styles.errorText}>{v.error}</div>}
                {v?.data && (
                  <div className={styles.resultBlock}>
                    <span className={`${styles.credBadge} ${styles[`cred_${v.data.credibility}`] || ''}`}>
                      {CREDIBILITY_LABEL[v.data.credibility] || v.data.credibility}
                    </span>
                    <p className={styles.verifySummary}>{v.data.summary}</p>
                    <AgentTrace report={v.data} />
                    <button className={styles.actionBtnMuted} onClick={() => runVerify(selectedUri, selectedMember?.text || '', true)}>
                      重新查證
                    </button>
                  </div>
                )}
              </div>

              <div className={styles.block}>
                <div className={styles.blockTitle}>🤖 真實政策決策</div>
                <p className={styles.hintText}>
                  按下按鈕後，這則真實貼文會使用當下回覆樹跑 RF、由 Verification Agent 取得證據，再交給 deterministic policy 產生建議。只會寫入決策紀錄，不會真的刪文或調整曝光。
                </p>
                <button
                  className={styles.actionBtn}
                  onClick={() => runPolicyDecision(selectedUri, !!policyRun?.data)}
                  disabled={policyRun?.loading || !selectedUri}
                >
                  {policyRun?.loading ? '正在執行 RF 與查證 Agent...' : policyRun?.data ? '重新產生介入建議' : '產生真實介入建議'}
                </button>
                {policyRun?.error && <div className={styles.errorText}>{policyRun.error}</div>}
                {policyRun?.loading && <LiveAgentTrace calls={liveToolCalls[selectedUri] || []} />}
                {policyRun?.data && (
                  <div className={`${styles.resultBlock} ${styles.policyAudit}`}>
                    <div className={styles.policyTopLine}>
                      <span className={`${styles.credBadge} ${styles[INTERVENTION_ACTION_CLASS[policyRun.data.action]] || ''}`}>
                        {INTERVENTION_ACTION_LABEL[policyRun.data.action] || policyRun.data.action}
                      </span>
                      <span className={styles.auditChip}>模擬強度 {policyRun.data.action_strength}</span>
                      {policyRun.data.cached_decision && <span className={styles.auditChip}>已存決策</span>}
                    </div>
                    <div className={styles.auditGrid}>
                      <span>Checkpoint <b>{policyRun.data.checkpoint_minutes} 分鐘</b></span>
                      <span>RF 預估 <b>+{Math.round(policyRun.data.predicted_reach || 0)}</b></span>
                      <span>證據 <b>{CREDIBILITY_LABEL[policyRun.data.effective_credibility] || policyRun.data.effective_credibility}</b></span>
                      <span>查證信心 <b>{Math.round((policyRun.data.verification_confidence || 0) * 100)}%</b></span>
                      <span>Risk weight <b>{policyRun.data.risk_weight ?? '排除'}</b></span>
                      <span>Priority <b>{policyRun.data.priority == null ? '無' : policyRun.data.priority.toFixed(2)}</b></span>
                      <span>已釋出名額 <b>{policyRun.data.released_cap}</b></span>
                      <span>本事件已用 <b>{policyRun.data.already_intervened_count}</b></span>
                    </div>
                    <p className={styles.verifySummary}>{policyRun.data.evidence_summary || '沒有證據摘要。'}</p>
                    <p className={styles.verifySummary}>{policyRun.data.reasoning}</p>
                    <AgentTrace report={policyRun.data.verification} />
                    {policyRun.data.rf_top_features?.length > 0 && (
                      <details className={styles.agentTrace}>
                        <summary className={styles.agentTraceSummary}>
                          📈 RF 傳播分數・前 {policyRun.data.rf_top_features.length} 個影響最大的特徵
                        </summary>
                        <ul className={styles.traceList}>
                          {policyRun.data.rf_top_features.map((f, i) => (
                            <li key={i} className={styles.traceItem}>
                              <span className={styles.traceToolName}>{f.name || f.feature}</span>
                              <span className={styles.traceMeta}>值 {typeof f.value === 'number' ? f.value.toFixed(3) : f.value}</span>
                              {f.importance != null && <span className={styles.traceMeta}>重要度 {Number(f.importance).toFixed(3)}</span>}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    <div className={styles.recommendationNotice}>這是決策支援建議，未執行任何平台介入。</div>
                  </div>
                )}
                {eventBudget && (
                  <p className={styles.hintText}>
                    這個事件（{thread.member_count} 則貼文共用）本身的配額：過去 24 小時已用 <b>{eventBudget.total_used_last_24h}</b> / <b>{eventBudget.total_budget}</b> 則。
                    每個事件各自有自己的配額，不會被其他不相關的事件搶走。
                  </p>
                )}
                {interventions.loading && <div className={styles.hintText}><span className={styles.spinnerSmall} />載入中...</div>}
                {interventions.error && <div className={styles.errorText}>{interventions.error}</div>}
                {!interventions.loading && !interventions.error && interventions.data.length === 0 && (
                  <div className={styles.hintBlock}>這則貼文還沒有 agent 決策紀錄，背景排程會在它到達下一個 checkpoint（10/20/30...分鐘）時自動處理。</div>
                )}
                {interventions.data.map((d) => (
                  <div key={d.id} className={styles.resultBlock}>
                    <span className={`${styles.credBadge} ${styles[INTERVENTION_ACTION_CLASS[d.action]] || ''}`}>
                      {INTERVENTION_ACTION_LABEL[d.action] || d.action}
                    </span>
                    <p className={styles.verifySummary}>
                      第 {d.checkpoint_minutes} 分鐘 checkpoint・信心 {Math.round(d.confidence * 100)}%
                      {d.predicted_reach != null && <> ・預估後續擴散 +{Math.round(d.predicted_reach)} 則</>}
                      {d.action_strength && <> ・模擬強度 {d.action_strength}</>}
                    </p>
                    <p className={styles.verifySummary}>{d.reasoning}</p>
                  </div>
                ))}
              </div>

              <div className={styles.block}>
                <div className={styles.blockTitle}>介入模擬（預估，非真實回放）</div>
                <p className={styles.hintText}>
                  這是直播中的內容，還沒有已知的「未來」。以下是依目前擴散速度推算的預估，不是像 PHEME
                  歷史事件那樣回放真實發生過的結果。
                </p>
                <button className={styles.actionBtn} onClick={() => runEstimate(selectedUri)} disabled={est?.loading}>
                  {est?.loading ? '估算中...' : '⛔ 估算刪除影響'}
                </button>
                {est?.error && <div className={styles.errorText}>{est.error}</div>}
                {est?.data?.estimable === false && (
                  <div className={styles.hintBlock}>目前資料不足以推算，可能是貼文剛發布還沒有任何回覆。</div>
                )}
                {est?.data?.estimable && (
                  <div className={styles.resultBlock}>
                    <p className={styles.verifySummary}>
                      這則貼文目前 <b>{est.data.age_minutes}</b> 分鐘內累積了 <b>{est.data.current_reply_count}</b> 則回覆，
                      平均每分鐘約 <b>{est.data.rate_per_minute}</b> 則。依這個速度推算，若不處理，
                      未來 {est.data.horizon_minutes} 分鐘內可能再新增約 <b>{est.data.projected_additional_replies}</b> 則回覆——
                      這是趨勢預估，不是已經發生的事實。
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default RadarEventDetail;
