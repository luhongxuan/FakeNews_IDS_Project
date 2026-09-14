// 檔案位置：frontend/src/components/InterventionReviewPanel.jsx
// intervention_agent.py 的決策留痕：每一列是 agent 已經跑完的一次判斷
// （RF 分數 + 查證狀態 + 配額），不是它自己動手做的事 -- 這裡永遠只是
// 建議，實際要不要處理由人決定。資料來自背景排程（每 3 分鐘）或手動
// 觸發 /intervention_review，兩者共用同一張表，所以這裡兩種都看得到。
import { useState, useEffect, useCallback } from 'react';
import styles from './InterventionReviewPanel.module.css';

const POLL_INTERVAL_MS = 30000;

const ACTION_LABEL = {
  escalate_now: '建議立即升級',
  escalate_next_checkpoint: '建議下個 checkpoint 再看',
  hold: '低急迫性，持續追蹤',
  dismiss: '無需處理',
  hard: 'Hard：大幅降低曝光／限制轉發',
  medium: 'Medium：中度降低曝光／人工複核',
  soft: 'Soft：小幅降低曝光／後續重查',
  none: 'None：不限制／持續監測',
};

const ACTION_STYLE = {
  escalate_now: 'actionEscalateNow',
  escalate_next_checkpoint: 'actionEscalateNext',
  hold: 'actionHold',
  dismiss: 'actionDismiss',
  hard: 'actionHard',
  medium: 'actionMedium',
  soft: 'actionSoft',
  none: 'actionNone',
};

const DEMO_DECISIONS = [
  {
    id: 'demo-hard', action: 'hard', checkpoint_minutes: 30, confidence: 0.94,
    predicted_reach: 186, evidence_state: '已遭可靠證據反駁', risk_level: '高風險', action_strength: '90%',
    reasoning: '核心主張已被第一手來源直接反駁，且 RF 預測後續擴散量高；建議大幅降低曝光並限制轉發，保留人工最終確認。',
    decided_at: new Date(Date.now() - 3 * 60000).toISOString(), is_demo: true,
  },
  {
    id: 'demo-medium', action: 'medium', checkpoint_minutes: 30, confidence: 0.81,
    predicted_reach: 92, evidence_state: '證據相互衝突', risk_level: '高風險', action_strength: '40%',
    reasoning: '多個來源對同一主張給出不一致結論，但傳播加速跡象明顯；建議中度降低曝光、標記爭議狀態並送人工複核。',
    decided_at: new Date(Date.now() - 8 * 60000).toISOString(), is_demo: true,
  },
  {
    id: 'demo-soft', action: 'soft', checkpoint_minutes: 20, confidence: 0.68,
    predicted_reach: 47, evidence_state: '證據不足', risk_level: '中高風險', action_strength: '20%',
    reasoning: '目前無法證實或反駁，但早期結構與 RF 分數顯示可能繼續擴散；建議可逆地小幅降低曝光，10 分鐘後重新查核。',
    decided_at: new Date(Date.now() - 13 * 60000).toISOString(), is_demo: true,
  },
  {
    id: 'demo-none', action: 'none', checkpoint_minutes: 30, confidence: 0.91,
    predicted_reach: 71, evidence_state: '已獲可靠證據支持', risk_level: '高傳播／低誤訊風險', action_strength: '0%',
    reasoning: '雖然 RF 預測傳播量較高，但查證結果支持核心主張；不進行限制性處理，釋出介入名額並維持一般監測。',
    decided_at: new Date(Date.now() - 18 * 60000).toISOString(), is_demo: true,
  },
];

function radarThreadToPreview(thread) {
  const engagement = Number(thread.total_engagement || 0);
  const memberCount = Number(thread.member_count || 0);
  const elevated = engagement >= 15 || memberCount >= 3;
  const createdAt = thread.representative_created_at || thread.first_seen_at;
  const elapsedMinutes = createdAt
    ? Math.max(0, Math.floor((Date.now() - toUtcMillis(createdAt)) / 60000))
    : 0;
  const checkpoint = Math.min(60, Math.max(0, Math.floor(elapsedMinutes / 10) * 10));

  return {
    id: `radar-preview-${thread.id}`,
    thread_id: thread.id,
    post_uri: thread.representative_uri,
    action: elevated ? 'soft' : 'none',
    checkpoint_minutes: checkpoint,
    confidence: elevated ? 0.62 : 0.55,
    observed_engagement: engagement,
    member_count: memberCount,
    evidence_state: '尚未查證',
    risk_level: elevated ? '可見互動較高' : '可見互動較低',
    action_strength: elevated ? '10–20%' : '0%',
    reasoning: elevated
      ? `真實雷達事件目前包含 ${memberCount} 則相關貼文，可見互動指標為 ${engagement}。因為還沒有查證證據，預覽只建議可逆的小幅降低曝光與優先查核，不會建議隱藏。`
      : `真實雷達事件目前包含 ${memberCount} 則相關貼文，可見互動指標為 ${engagement}。目前沒有足夠的傳播風險或查證證據支持限制處理，先維持監測。`,
    decided_at: thread.last_seen_at || new Date().toISOString(),
    representative_text: thread.representative_text,
    matched_keywords: thread.matched_keywords || [],
    is_real_preview: true,
    display_status: thread.display_status,
    display_status_reasons: thread.display_status_reasons || [],
    ambiguous: Boolean(thread.ambiguous),
  };
}

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

const InterventionReviewPanel = ({ onBack, onLogoClick, onOpenThread }) => {
  const [decisions, setDecisions] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actionFilter, setActionFilter] = useState('');
  const [displayMode, setDisplayMode] = useState('decisions');
  const [radarPreviews, setRadarPreviews] = useState([]);
  const [radarLoading, setRadarLoading] = useState(false);
  const [radarError, setRadarError] = useState(null);
  // Live Radar Event Discovery V2, phase 2: default to only the server-classified
  // display_eligible threads (multiple source posts, or a real reply cascade) -- the API
  // itself still returns everything (see radar.py's _serialize), this is purely a
  // frontend default so low-signal singletons don't drown out real events by sheer count.
  const [showAllRadar, setShowAllRadar] = useState(false);

  const fetchDecisions = useCallback(() => {
    const controller = new AbortController();
    const query = actionFilter ? `&action=${actionFilter}` : '';
    fetch(`/api/radar/interventions?limit=50${query}`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => { setDecisions(data.decisions || []); setSummary(data.summary || null); setError(null); })
      .catch((err) => { if (err.name !== 'AbortError') setError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return controller;
  }, [actionFilter]);

  useEffect(() => {
    const controller = fetchDecisions();
    const interval = setInterval(fetchDecisions, POLL_INTERVAL_MS);
    return () => { controller.abort(); clearInterval(interval); };
  }, [fetchDecisions]);

  const previewSource = displayMode === 'demo'
    ? DEMO_DECISIONS
    : displayMode === 'radar'
      ? radarPreviews
      : decisions;
  // ambiguous_review threads are shown by default alongside display_eligible (never
  // hidden as if they were low-signal) -- they may be genuinely graph-rich, but their
  // own members disagree on location/incident-type, so they always render with a
  // visible "可能包含不同事件" warning (see the badge below) rather than being silently
  // treated the same as a verified-consistent event.
  const visibleDecisions = previewSource
    .filter((d) => !actionFilter || d.action === actionFilter)
    .filter((d) => displayMode !== 'radar' || showAllRadar || d.display_status === 'display_eligible' || d.display_status === 'ambiguous_review');
  const hiddenSingletonCount = displayMode === 'radar' && !showAllRadar
    ? radarPreviews.filter((d) => d.display_status !== 'display_eligible' && d.display_status !== 'ambiguous_review').length
    : 0;
  const escalateCount = visibleDecisions.filter((d) => ['escalate_now', 'hard'].includes(d.action)).length;

  const changeMode = (mode) => {
    if (mode === 'radar') {
      setRadarLoading(true);
      setRadarError(null);
    }
    setDisplayMode(mode);
    setActionFilter('');
  };

  useEffect(() => {
    if (displayMode !== 'radar') return undefined;
    const controller = new AbortController();
    fetch('/api/radar/threads?limit=20&min_engagement=0', { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => {
        setRadarPreviews((data.threads || []).map(radarThreadToPreview));
        setRadarError(null);
      })
      .catch((err) => { if (err.name !== 'AbortError') setRadarError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setRadarLoading(false); });
    return () => controller.abort();
  }, [displayMode]);

  return (
    <div className={styles.stage}>
      <header className={styles.topNav}>
        <button type="button" className={styles.brand} onClick={onLogoClick}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M13 2 3 14h7l-1 8 10-12h-7l1-8z"></path></svg>
          自動干預建議
        </button>
        <div className={styles.liveStatus}>
          <span className={displayMode === 'decisions' ? styles.liveDot : styles.demoDot} />
          {displayMode === 'decisions' && `${escalateCount} 則建議立即升級`}
          {displayMode === 'radar' && `${visibleDecisions.length} 個真實雷達事件${hiddenSingletonCount > 0 ? `（另有 ${hiddenSingletonCount} 個低訊號 singleton 已隱藏）` : ''}`}
          {displayMode === 'demo' && 'MVP 合成流程示範'}
        </div>
        <div className={styles.navActions}>
          <button className={displayMode === 'decisions' ? styles.demoBtnActive : styles.demoBtn} onClick={() => changeMode('decisions')}>真實決策</button>
          <button className={displayMode === 'radar' ? styles.demoBtnActive : styles.demoBtn} onClick={() => changeMode('radar')}>真實事件預覽</button>
          {displayMode === 'radar' && (
            <button className={styles.demoBtn} onClick={() => setShowAllRadar((v) => !v)}>
              {showAllRadar ? '只看 display_eligible' : `展開全部（含 ${hiddenSingletonCount} 個 singleton）`}
            </button>
          )}
          <button className={displayMode === 'demo' ? styles.demoBtnActive : styles.demoBtn} onClick={() => changeMode('demo')}>合成流程示範</button>
          <button className={styles.linkBtn} onClick={onBack}>返回</button>
        </div>
      </header>

      <div className={styles.explainer}>
        {displayMode === 'decisions' && 'intervention_agent 已完成的真實決策紀錄：綜合 Hawkes RF 傳播分數、查證結果與配額狀況。全部都只是建議，不會自動刪文或隱藏內容。'}
        {displayMode === 'radar' && '貼文內容、時間、事件分群與互動量來自真實 Bluesky 雷達資料；介入層級是在尚未完成 RF／Agent 時的透明前端預覽規則，只能產生 Soft 或 None，不是模型成績。預設只顯示有多個同事件貼文或有足夠回覆的事件（display_eligible，工程展示門檻，非驗證過的研究結果），以及標示為「可能包含不同事件」的 ambiguous_review 事件（會顯示紅色警示，不代表已確認同一事件）；點「展開全部」可看到被隱藏的低訊號 singleton。點選事件可進入真實傳播圖與查證頁面。'}
        {displayMode === 'demo' && '這是可獨立展示的合成範例：展現 RF 傳播風險、Verification Agent 證據狀態與分級介入建議如何組合。數字不是研究成績。'}
      </div>

      {displayMode === 'decisions' && summary && (
        <div className={styles.budgetBar}>
          <span>過去 24 小時全站共介入 <strong>{summary.escalated_last_24h}</strong> 則・累計 <strong>{summary.escalated_all_time}</strong> 則</span>
          <span className={styles.budgetDivider} />
          <span>配額是「每個事件各自」50 則，不是全站共用一個配額——點進個別事件才看得到它自己的配額用量</span>
        </div>
      )}

      <div className={styles.filterBar}>
        <label className={styles.filterLabel} htmlFor="intervention-action-filter">篩選</label>
        <select
          id="intervention-action-filter"
          className={styles.filterSelect}
          value={actionFilter}
          onChange={(e) => setActionFilter(e.target.value)}
        >
          <option value="">全部決策</option>
          {displayMode !== 'decisions' ? (
            <>
              <option value="hard">Hard</option>
              <option value="medium">Medium</option>
              <option value="soft">Soft</option>
              <option value="none">None</option>
            </>
          ) : (
            <>
              <option value="escalate_now">建議立即升級</option>
              <option value="escalate_next_checkpoint">建議下個 checkpoint 再看</option>
              <option value="hold">低急迫性</option>
              <option value="dismiss">無需處理</option>
            </>
          )}
        </select>
      </div>

      <main className={styles.stageBody}>
        {displayMode === 'decisions' && loading && <div className={styles.centerHint}><span className={styles.spinner} />載入中...</div>}
        {displayMode === 'decisions' && error && (
          <div className={styles.emptyState}>
            <strong>真實後端目前無法讀取</strong>
            <span>{error}</span>
            <button className={styles.demoBtnActive} onClick={() => changeMode('radar')}>改看真實雷達事件</button>
          </div>
        )}
        {displayMode === 'decisions' && !loading && !error && decisions.length === 0 && (
          <div className={styles.emptyState}>
            <strong>還沒有真實決策紀錄</strong>
            <span>背景排程會在 thread 到達 checkpoint 時處理；也可先開啟展示模式查看完整界面。</span>
            <button className={styles.demoBtnActive} onClick={() => changeMode('radar')}>開啟真實事件預覽</button>
          </div>
        )}
        {displayMode === 'radar' && radarLoading && <div className={styles.centerHint}><span className={styles.spinner} />載入真實雷達事件...</div>}
        {displayMode === 'radar' && radarError && <div className={styles.centerHint}>真實雷達資料載入失敗：{radarError}</div>}
        {displayMode === 'radar' && !radarLoading && !radarError && radarPreviews.length === 0 && (
          <div className={styles.centerHint}>目前雷達還沒有收集到事件。</div>
        )}

        <div className={styles.decisionList}>
          {visibleDecisions.map((decision) => (
            <div
              key={decision.id}
              className={`${styles.row} ${decision.is_demo ? styles.demoRow : ''}`}
              onClick={decision.is_demo ? undefined : () => onOpenThread(decision.thread_id, decision.post_uri)}
            >
              <div className={styles.rowMain}>
                <div className={styles.rowBadges}>
                  {decision.ambiguous && (
                    <span className={styles.actionTag} style={{ background: '#5c1a1a', color: '#ffb4b4', border: '1px solid #a33' }} title="此事件群組內的貼文彼此地點或事件類型不一致，可能誤合併了不同真實事件；查證與介入建議需個別確認，不可套用同一結論。">
                      ⚠ 可能包含不同事件（ambiguous_review）
                    </span>
                  )}
                  <span className={`${styles.actionTag} ${styles[ACTION_STYLE[decision.action]] || ''}`}>
                    {ACTION_LABEL[decision.action] || decision.action}
                  </span>
                  <span className={styles.checkpointTag}>第 {decision.checkpoint_minutes} 分鐘</span>
                  <span className={styles.confidenceTag}>信心 {Math.round(decision.confidence * 100)}%</span>
                  {decision.evidence_state && <span className={styles.evidenceTag}>{decision.evidence_state}</span>}
                  {decision.risk_level && <span className={styles.riskTag}>{decision.risk_level}</span>}
                  {decision.action_strength && <span className={styles.strengthTag}>模擬強度 {decision.action_strength}</span>}
                  {decision.observed_engagement != null && (
                    <span className={styles.reachTag}>真實可見互動 {decision.observed_engagement}</span>
                  )}
                  {decision.predicted_reach != null && (
                    <span className={styles.reachTag} title="模型預估：若不介入，這波傳播後續還會再產生的貼文/回覆數量（不是精確的『看到人數』，是傳播規模的代理指標）">
                      預估後續擴散 +{Math.round(decision.predicted_reach)} 則
                    </span>
                  )}
                </div>
                {decision.representative_text && <div className={styles.eventText}>{decision.representative_text}</div>}
                <div className={styles.rowText}>{decision.reasoning}</div>
                <div className={styles.rowMeta}>
                  <span>{formatRelativeTime(decision.decided_at)}</span>
                  {decision.is_demo && <span>模擬案例・不會實際介入</span>}
                  {decision.is_real_preview && <span>真實事件・點選查看傳播圖與查證</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
};

export default InterventionReviewPanel;
