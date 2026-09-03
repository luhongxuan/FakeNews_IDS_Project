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

const RadarEventDetail = ({ threadId, onBack, onLogoClick }) => {
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

  useEffect(() => {
    if (!threadId) return;
    const controller = new AbortController();
    setLoadingThread(true);
    fetch(`/api/radar/threads/${threadId}`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => { setThread(data); setSelectedUri(data.representative_uri); setThreadError(null); })
      .catch((err) => { if (err.name !== 'AbortError') setThreadError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingThread(false); });
    return () => controller.abort();
  }, [threadId]);

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

  const runVerify = (uri, text) => {
    setVerifications((prev) => ({ ...prev, [uri]: { loading: true, data: null, error: null } }));
    fetch('/api/factcheck', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setVerifications((prev) => ({ ...prev, [uri]: { loading: false, data, error: null } })))
      .catch((err) => setVerifications((prev) => ({ ...prev, [uri]: { loading: false, data: null, error: err.message } })));
  };

  const runEstimate = (uri) => {
    setEstimates((prev) => ({ ...prev, [uri]: { loading: true, data: null, error: null } }));
    fetch(`/api/radar/threads/${threadId}/estimate_intervene?post_uri=${encodeURIComponent(uri)}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setEstimates((prev) => ({ ...prev, [uri]: { loading: false, data, error: null } })))
      .catch((err) => setEstimates((prev) => ({ ...prev, [uri]: { loading: false, data: null, error: err.message } })));
  };

  const markReviewed = () => {
    fetch(`/api/radar/threads/${threadId}/mark_reviewed?reviewed=${!thread.reviewed}`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
      .then((data) => setThread(data))
      .catch(() => {});
  };

  const v = selectedUri ? verifications[selectedUri] : null;
  const est = selectedUri ? estimates[selectedUri] : null;

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
                  onClick={() => setSelectedUri(thread.representative_uri)}
                >
                  <span className={styles.threadItemTag}>群組代表</span>
                  <span className={styles.threadItemText}>{thread.representative_text}</span>
                </button>
                {thread.members.filter((m) => m.uri !== thread.representative_uri).map((m) => (
                  <button
                    key={m.uri}
                    className={`${styles.threadItem} ${selectedUri === m.uri ? styles.threadItemActive : ''}`}
                    onClick={() => setSelectedUri(m.uri)}
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
                {v?.loading && <div className={styles.hintText}><span className={styles.spinnerSmall} />查證中，本機推理可能需要數十秒...</div>}
                {v?.error && <div className={styles.errorText}>{v.error}</div>}
                {v?.data && (
                  <div className={styles.resultBlock}>
                    <span className={`${styles.credBadge} ${styles[`cred_${v.data.credibility}`] || ''}`}>
                      {CREDIBILITY_LABEL[v.data.credibility] || v.data.credibility}
                    </span>
                    <p className={styles.verifySummary}>{v.data.summary}</p>
                    <button className={styles.actionBtnMuted} onClick={() => runVerify(selectedUri, selectedMember?.text || '')}>
                      重新查證
                    </button>
                  </div>
                )}
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
