// 檔案位置：frontend/src/components/LiveScenarioSimulator.jsx
// 「即時情境模擬」指揮台：把一個 PHEME 歷史事件重播成有模擬時鐘的時間軸，
// 而不是像舊版那樣用寫死的 setTimeout 節奏假裝即時、按下介入後把所有未來
// 節點一次全部揭露。核心規則（跟後端 pheme_replay.py 對齊）：
//   - Threads 依真實來源貼文時間陸續「抵達」，不是依 RF 分數排序假裝抵達。
//   - 任何畫面上看得到的節點，其真實事件時間都必須 <= 目前模擬時間 simTime；
//     這個過濾在後端 cascade_window/intervene_window 做，前端不能繞過去
//     直接呼叫舊的、會一次回傳完整未來資料的 /cascade、/intervene。
//   - 介入後「已阻擋節點」是隨模擬時間累積解鎖的（intervene_window 的
//     realized_blocked_count），不是按下當下就全部揭露。
//   - 完整結局（總共漏掉幾則、最終標籤）只在模擬跑完、進入 After-action
//     報告時才顯示。
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import styles from './LiveScenarioSimulator.module.css';
import { cascadeStylesheet, cascadeLayout } from './graphConfig';

const TICK_MS = 400; // 真實時間；每個 tick 依 speed 推進模擬時鐘
const CHECKPOINTS_SECONDS = [600, 1200, 1800, 2400, 3000, 3600]; // 10/20/30/40/50/60 分鐘
const POST_EVENT_HORIZON_SECONDS = 3600; // 最後一則 thread 抵達後，再多跑 1 小時才算模擬結束（沒設時數上限時的預設）
const SPEED_OPTIONS = [1, 5, 30, 60];
const DURATION_PRESET_HOURS = [1, 2, 3, 6, 12, null]; // null = 跑到真實事件自然結束

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

function formatClock(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const hh = String(Math.floor(s / 3600)).padStart(2, '0');
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}

const LiveScenarioSimulator = ({ eventId, onBack, onLogoClick }) => {
  const [manifest, setManifest] = useState(null); // {threads: [...], excluded_missing_timestamp}
  const [loadingManifest, setLoadingManifest] = useState(true);
  const [manifestError, setManifestError] = useState(null);

  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [simTime, setSimTime] = useState(0); // 模擬事件時鐘，秒，從事件最早那則 source 起算
  const tickRef = useRef(null);

  // 使用者自訂「只跑幾小時」：null 代表跑到真實事件自然結束（最後一則
  // thread 抵達後再 +1 小時）。設定了時數上限的話，時間一到就直接視為
  // finished、跳進 After-action 報告，不用等真實事件跑完（部分 PHEME
  // 事件真實跨度長達數十小時，不設上限的話展示時根本等不到結局）。
  const [durationLimitHours, setDurationLimitHours] = useState(null);

  // Agent 自動決策模式：開啟後，只要一個 thread 已經有查證結果（不管是
  // 重播的快取還是使用者手動觸發的真實查證），就交給後端 auto_decision
  // （重用 intervention_policy 既有門檻）自動下決策，不用等人工點擊。
  // 證據不夠強時 auto_decision 回 "undetermined"，仍然保留給人工判斷，
  // 不會因為開了自動模式就變成亂猜。
  const [autoAgentMode, setAutoAgentMode] = useState(false);

  // items keyed by thread_id -- 只有已經「抵達」(arrival_offset_seconds <= simTime)
  // 的 thread 才會出現在這裡，符合「依真實時間陸續進場」。
  const [items, setItems] = useState({});
  // 三欄版面：左邊清單選一則 thread，中間圖、右邊查證+建議都跟著換，
  // 不再是每列自己展開/收合（呼應 RadarEventDetail.jsx 的三欄配置）。
  const [selectedThreadId, setSelectedThreadId] = useState(null);
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const naturalHorizonSeconds = useMemo(() => {
    if (!manifest || manifest.threads.length === 0) return 0;
    const maxArrival = Math.max(...manifest.threads.map((t) => t.arrival_offset_seconds));
    return maxArrival + POST_EVENT_HORIZON_SECONDS;
  }, [manifest]);

  const horizonSeconds = durationLimitHours
    ? Math.min(naturalHorizonSeconds || durationLimitHours * 3600, durationLimitHours * 3600)
    : naturalHorizonSeconds;

  const finished = manifest && manifest.threads.length > 0 && simTime >= horizonSeconds;

  useEffect(() => {
    setManifest(null); setManifestError(null); setRunning(false); setSimTime(0); setItems({}); setSelectedThreadId(null);
    clearInterval(tickRef.current);
    if (!eventId) { setLoadingManifest(false); return; }
    setLoadingManifest(true);
    const controller = new AbortController();
    fetch(`/api/events/${eventId}/replay_manifest`, { signal: controller.signal })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setManifest(data))
      .catch((error) => { if (error.name !== 'AbortError') setManifestError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingManifest(false); });
    return () => controller.abort();
  }, [eventId]);

  useEffect(() => () => clearInterval(tickRef.current), []);

  // 模擬時鐘本身：真實時間每 TICK_MS 推進一次，前進量 = TICK_MS * speed。
  // 播放速度只影響「模擬時鐘走多快」，不影響任何一次 cascade_window/
  // intervene_window 呼叫的結果本身（那些是 simTime 的純函式）。
  useEffect(() => {
    if (!running || finished) return;
    tickRef.current = setInterval(() => {
      setSimTime((t) => Math.min(t + (TICK_MS / 1000) * speed, horizonSeconds || t + (TICK_MS / 1000) * speed));
    }, TICK_MS);
    return () => clearInterval(tickRef.current);
  }, [running, speed, finished, horizonSeconds]);

  useEffect(() => { if (finished) setRunning(false); }, [finished]);

  const restart = () => {
    clearInterval(tickRef.current);
    setRunning(false); setSimTime(0); setItems({}); setSelectedThreadId(null);
  };

  // 新 thread 抵達：simTime 跨過某個 thread 的 arrival_offset_seconds 時，
  // 把它加進 items 並立刻取一次當下可見的 cascade window + 快取查證。
  useEffect(() => {
    if (!manifest) return;
    const toArrive = manifest.threads.filter(
      (t) => t.arrival_offset_seconds <= simTime && !itemsRef.current[t.thread_id],
    );
    if (toArrive.length === 0) return;
    setItems((prev) => {
      const next = { ...prev };
      for (const thread of toArrive) {
        next[thread.thread_id] = {
          thread, arrivedAtSimTime: simTime,
          cascade: null, cascadeLoading: true,
          verification: null, verifyLoading: true,
          decision: null, deciding: false, decisionError: null,
          intervened: null, // {realized_blocked_count, pending_blocked_count}
        };
      }
      return next;
    });
    for (const thread of toArrive) {
      fetch(`/api/threads/${thread.thread_id}/verify`)
        .then((res) => { if (res.status === 404) return null; if (!res.ok) throw new Error(); return res.json(); })
        .then((data) => setItems((prev) => (prev[thread.thread_id]
          ? { ...prev, [thread.thread_id]: { ...prev[thread.thread_id], verification: data, verifyLoading: false } }
          : prev)))
        .catch(() => setItems((prev) => (prev[thread.thread_id]
          ? { ...prev, [thread.thread_id]: { ...prev[thread.thread_id], verifyLoading: false } }
          : prev)));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest, simTime]);

  // 已抵達但還沒決策的 thread：定期重拉一次「目前模擬時間看得到的 cascade
  // window」，讓「已觀察 N」隨模擬時間真的長大。已決策為 blocked 的則輪詢
  // intervene_window。
  //
  // 這個輪詢刻意跟模擬時鐘的 tick（TICK_MS=400ms，為了時鐘顯示流暢）分開，
  // 用自己較慢的真實時間間隔（POLL_MS）獨立跑，並且只挑「目前最需要更新」
  // 的一批 item，不是每個 tick 對「所有已抵達、還沒決策」的 thread 全部發
  // 一次請求——長時間模擬下已抵達的 thread 可能有幾百則，若每 400ms 對全部
  // 發請求，瀏覽器很快就會把連線數用完（實測：跑到中後段真的會噴
  // ERR_INSUFFICIENT_RESOURCES，這是修這個問題之前的真實錯誤，不是假設）。
  const simTimeRef = useRef(simTime);
  simTimeRef.current = simTime;
  useEffect(() => {
    if (!manifest) return;
    const POLL_MS = 1500;
    const MAX_PENDING_POLLED_PER_ROUND = 40; // 依風險排序只更新前 N 則，其餘留到下一輪
    const pollOnce = () => {
      const all = Object.values(itemsRef.current);
      const pending = all.filter((it) => !it.decision)
        .sort((a, b) => (Number(b.thread.percentile_score) || 0) - (Number(a.thread.percentile_score) || 0))
        .slice(0, MAX_PENDING_POLLED_PER_ROUND);
      const t = simTimeRef.current;
      for (const item of pending) {
        fetch(`/api/threads/${item.thread.thread_id}/cascade_window?event_id=${eventId}&thread_arrival_offset_seconds=${item.thread.arrival_offset_seconds}&sim_time_seconds=${t}`)
          .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
          .then((data) => setItems((prev) => (prev[item.thread.thread_id]
            ? { ...prev, [item.thread.thread_id]: { ...prev[item.thread.thread_id], cascade: data, cascadeLoading: false } }
            : prev)))
          .catch(() => {});
      }
      const blocked = all.filter((it) => it.decision === 'blocked');
      for (const item of blocked) {
        fetch(`/api/threads/${item.thread.thread_id}/intervene_window?event_id=${eventId}&thread_arrival_offset_seconds=${item.thread.arrival_offset_seconds}&sim_time_seconds=${t}`)
          .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
          .then((data) => setItems((prev) => (prev[item.thread.thread_id]
            ? { ...prev, [item.thread.thread_id]: { ...prev[item.thread.thread_id], intervened: data } }
            : prev)))
          .catch(() => {});
      }
    };
    pollOnce();
    const pollInterval = setInterval(pollOnce, POLL_MS);
    return () => clearInterval(pollInterval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest]);

  // 手動觸發「這則貼文真的問一次查證 agent」——跟重播模式下預設只讀取
  // 「已經存在的快取查證」(GET /verify，見上面新 thread 抵達那個 effect)
  // 不一樣：這是真的呼叫 verification_agent，會花費真實的本機/RunPod
  // 推理時間（數十秒），所以刻意做成「使用者主動點」而不是自動對每一則
  // 抵達的貼文都觸發，避免一次跑起來就是幾百次真實呼叫。
  const runRealVerify = (threadId) => {
    const item = itemsRef.current[threadId];
    if (!item || item.verifying) return;
    setItems((prev) => ({ ...prev, [threadId]: { ...prev[threadId], verifying: true } }));
    fetch(`/api/threads/${threadId}/verify?event_id=${eventId}&force=true`, { method: 'POST' })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setItems((prev) => (prev[threadId]
        ? { ...prev, [threadId]: { ...prev[threadId], verification: data, verifyLoading: false, verifying: false, recommendationChecked: false } }
        : prev)))
      .catch((error) => setItems((prev) => (prev[threadId]
        ? { ...prev, [threadId]: { ...prev[threadId], verifying: false, verifyError: error.message } }
        : prev)));
  };

  const decide = (threadId, choice) => {
    const item = itemsRef.current[threadId];
    if (!item || item.deciding || item.decision) return;
    setItems((prev) => ({ ...prev, [threadId]: { ...prev[threadId], deciding: true, decisionError: null } }));
    // 這一步只「記錄決策」本身是即時的（跟真實維運介面一樣，按下的當下
    // 就該有反饋）；但介入之後「阻擋了多少節點」不會在這裡一次算完，
    // 是靠上面那個 useEffect 隨 simTime 前進持續輪詢 intervene_window。
    setTimeout(() => {
      setItems((prev) => (prev[threadId]
        ? { ...prev, [threadId]: { ...prev[threadId], deciding: false, decision: choice } }
        : prev));
    }, 350); // 短暫「處理中」反饋，呼應原始規劃裡「按鈕 0.5 秒確認狀態」的精神
  };

  // Agent 全自動模式的第一步：查證本身也要自動化，不能只在「剛好已經有
  // 查證結果」時才自動決策——不然開了自動模式，大部分貼文永遠停在
  // "尚無查證"，實質上還是要人工按查證。
  //
  // 真的觸發 verification_agent 是有成本的真實呼叫（本機/RunPod 推理，
  // 數十秒），所以刻意限制「同時最多 MAX_CONCURRENT_AUTO_VERIFY 則在查證
  // 中」，並且每輪只挑目前風險最高、還沒查過、也還沒在查的候選——不是
  // 對所有已抵達貼文一次全部發真實查證請求。設為 1：兩個並發請求會讓
  // RunPod 上的單一 Ollama 實例同時處理兩組推理，兩者互相排隊延長個別
  // 回應時間，容易撞到 httpx 逾時或連線被中斷，改成完全序列化一次只跑
  // 一個，穩定性優先於併發吞吐量。
  useEffect(() => {
    if (!autoAgentMode) return;
    const MAX_CONCURRENT_AUTO_VERIFY = 1;
    const AUTO_VERIFY_POLL_MS = 3000;
    const tick = () => {
      const all = Object.values(itemsRef.current);
      const inFlight = all.filter((it) => it.verifying).length;
      const slots = MAX_CONCURRENT_AUTO_VERIFY - inFlight;
      if (slots <= 0) return;
      const candidates = all
        .filter((it) => !it.decision && !it.verification && !it.verifyLoading && !it.verifying && !it.autoVerifyRequested)
        .sort((a, b) => (Number(b.thread.percentile_score) || 0) - (Number(a.thread.percentile_score) || 0))
        .slice(0, slots);
      for (const item of candidates) {
        const threadId = item.thread.thread_id;
        setItems((prev) => (prev[threadId] ? { ...prev, [threadId]: { ...prev[threadId], autoVerifyRequested: true } } : prev));
        runRealVerify(threadId);
      }
    };
    tick();
    const interval = setInterval(tick, AUTO_VERIFY_POLL_MS);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoAgentMode]);

  // 建議（recommendation）：只要一則貼文有查證結果、還沒決策，就去問後端
  // auto_decision（重用 intervention_policy 既有門檻）「現在的證據支持怎麼
  // 做」，存成 item.recommendation 顯示在右側面板——這一步永遠會做，不管
  // 有沒有開 Agent 全自動，人工決策時也看得到「系統為什麼建議介入／不建
  // 議介入」。"undetermined" 代表證據還不夠強，畫面上就老實顯示「證據不
  // 足以自動判斷」，不會硬掰一個建議出來。
  useEffect(() => {
    const candidates = Object.values(itemsRef.current).filter(
      (it) => !it.decision && it.verification && !it.recommendationChecked,
    );
    for (const item of candidates) {
      const threadId = item.thread.thread_id;
      setItems((prev) => (prev[threadId] ? { ...prev, [threadId]: { ...prev[threadId], recommendationChecked: true } } : prev));
      const params = new URLSearchParams({
        predicted_impact: String(item.thread.prediction ?? 0),
        credibility: item.verification.credibility || '',
        confidence: String(item.verification.confidence ?? ''),
      });
      fetch(`/api/threads/${threadId}/auto_decision?${params}`)
        .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
        .then((data) => setItems((prev) => (prev[threadId] ? { ...prev, [threadId]: { ...prev[threadId], recommendation: data } } : prev)))
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items]);

  // Agent 全自動模式：有了上面那個建議、還沒決策、且開了自動模式時，才
  // 真的「代替人工按下去」——建議本身（上面那個 effect）永遠都算，自動
  // 執行才是額外的一步，這樣手動模式底下也看得到同一套建議邏輯的輸出。
  useEffect(() => {
    if (!autoAgentMode) return;
    const candidates = Object.values(itemsRef.current).filter(
      (it) => !it.decision && it.recommendation && !it.autoApplyChecked
        && (it.recommendation.action === 'blocked' || it.recommendation.action === 'observed'),
    );
    for (const item of candidates) {
      const threadId = item.thread.thread_id;
      setItems((prev) => (prev[threadId] ? { ...prev, [threadId]: { ...prev[threadId], autoApplyChecked: true, autoReasoning: item.recommendation.reasoning } } : prev));
      decide(threadId, item.recommendation.action);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoAgentMode, items]);

  const orderedItems = useMemo(() => {
    return Object.values(items).sort((a, b) => {
      if (a.decision && !b.decision) return 1;
      if (!a.decision && b.decision) return -1;
      if (!a.decision && !b.decision) {
        // 待決策清單依風險排序（percentile_score 越高越優先），不是依抵達
        // 順序——呼應「根據現在有疑慮的 thread 進行風險排序」。
        return (Number(b.thread.percentile_score) || 0) - (Number(a.thread.percentile_score) || 0);
      }
      return a.thread.arrival_offset_seconds - b.thread.arrival_offset_seconds;
    });
  }, [items]);

  const selectedItem = selectedThreadId ? items[selectedThreadId] : null;

  // 還沒選任何一則的話，預設選目前風險最高的那則，避免右邊/中間永遠空著
  // 等使用者自己點第一下。
  useEffect(() => {
    if (!selectedThreadId && orderedItems.length > 0) setSelectedThreadId(orderedItems[0].thread.thread_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orderedItems.length]);

  const pendingHighRisk = orderedItems.filter((it) => !it.decision && riskLabel(it.thread.percentile_score).className === 'riskHigh').length;
  const interventionsUsed = orderedItems.filter((it) => it.decision === 'blocked').length;
  const realizedBlockedTotal = orderedItems.reduce((sum, it) => sum + (it.intervened?.realized_blocked_count || 0), 0);
  const pendingBlockedTotal = orderedItems.reduce((sum, it) => sum + (it.intervened?.pending_blocked_count || 0), 0);
  const excludedTrueCount = orderedItems.filter((it) => it.verification?.credibility === 'likely_true').length;
  const arrivedCount = orderedItems.length;
  const totalThreads = manifest?.threads.length || 0;

  const nextCheckpoint = CHECKPOINTS_SECONDS.find((c) => c > simTime);
  const passedCheckpoints = CHECKPOINTS_SECONDS.filter((c) => c <= simTime);

  return (
    <div className={styles.stage}>
      <header className={styles.topNav}>
        <button type="button" className={styles.brand} onClick={onLogoClick}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
          即時情境模擬・指揮台
        </button>
        <div className={styles.clockBar}>
          <span className={styles.clockValue}>{formatClock(simTime)}</span>
          <button className={styles.clockBtn} onClick={() => setRunning((r) => !r)} disabled={loadingManifest || !manifest?.threads.length || finished}>
            {running ? '⏸ 暫停' : '▶ 播放'}
          </button>
          {SPEED_OPTIONS.map((s) => (
            <button key={s} className={`${styles.speedBtn} ${speed === s ? styles.speedBtnActive : ''}`} onClick={() => setSpeed(s)}>
              {s}×
            </button>
          ))}
          <button className={styles.clockBtn} onClick={restart}>重新開始</button>
        </div>
        <button className={styles.linkBtn} onClick={onBack}>返回研究視角</button>
      </header>

      <div className={styles.controlBar}>
        <span className={styles.controlLabel}>模擬時數上限：</span>
        {DURATION_PRESET_HOURS.map((h) => (
          <button
            key={h ?? 'full'}
            className={`${styles.speedBtn} ${durationLimitHours === h ? styles.speedBtnActive : ''}`}
            onClick={() => setDurationLimitHours(h)}
            disabled={simTime > 0}
            title={simTime > 0 ? '模擬已開始，重新開始後才能改時數上限' : undefined}
          >
            {h ? `${h} 小時` : '不設上限'}
          </button>
        ))}
        <span className={styles.controlDivider} />
        <label className={styles.autoModeToggle}>
          <input type="checkbox" checked={autoAgentMode} onChange={(e) => setAutoAgentMode(e.target.checked)} />
          🤖 Agent 全自動（查證＋決策皆自動，會產生真實 LLM 呼叫）
        </label>
      </div>

      <div className={styles.explainer}>
        PHEME 歷史事件反事實回放；未於真實平台執行介入。時間軸依真實貼文發布時間推進——
        任何畫面上的節點，其真實時間都不會早於目前的模擬時鐘；介入後的「已阻擋節點」也是
        隨模擬時間逐步累積解鎖，不是按下當下就一次全部揭露。播放速度只改變時鐘走多快，
        不影響任何一次判定的結果。
        {nextCheckpoint != null && !finished && (
          <> 下一個 checkpoint：{Math.round((nextCheckpoint - simTime) / 60)} 分鐘後（{nextCheckpoint / 60} 分鐘整）。</>
        )}
      </div>

      {!loadingManifest && !manifestError && manifest?.threads.length > 0 && (
        <div className={styles.kpiBar}>
          <div className={styles.kpiCard}><div className={styles.kpiValue}>{arrivedCount}/{totalThreads}</div><div className={styles.kpiLabel}>已出現 threads</div></div>
          <div className={`${styles.kpiCard} ${pendingHighRisk > 0 ? styles.kpiWarn : ''}`}><div className={styles.kpiValue}>{pendingHighRisk}</div><div className={styles.kpiLabel}>待處理・高風險</div></div>
          <div className={styles.kpiCard}><div className={styles.kpiValue}>{interventionsUsed}</div><div className={styles.kpiLabel}>已介入</div></div>
          <div className={styles.kpiCard}><div className={styles.kpiValue}>{realizedBlockedTotal}</div><div className={styles.kpiLabel}>已阻擋節點（模擬時間已解鎖）</div></div>
          <div className={styles.kpiCard}><div className={styles.kpiValue}>{pendingBlockedTotal}</div><div className={styles.kpiLabel}>尚未解鎖（介入已生效，等模擬時間走到）</div></div>
          <div className={styles.kpiCard}><div className={styles.kpiValue}>{excludedTrueCount}</div><div className={styles.kpiLabel}>已排除 true</div></div>
        </div>
      )}

      <main className={styles.stageBody}>
        {loadingManifest && <div className={styles.centerHint}><span className={styles.spinner} />載入事件時間軸中...</div>}
        {manifestError && <div className={styles.centerHint}>載入失敗：{manifestError}</div>}
        {!loadingManifest && !manifestError && manifest?.threads.length === 0 && (
          <div className={styles.centerHint}>這個事件沒有可用的真實時間戳，無法建立時間軸重播。</div>
        )}

        {!loadingManifest && !manifestError && manifest?.threads.length > 0 && simTime === 0 && !running && arrivedCount === 0 && (
          <div className={styles.startCard}>
            <div className={styles.startTitle}>準備開始模擬</div>
            <div className={styles.startBody}>
              按下播放後，{totalThreads} 則貼文會依照真實發布時間陸續進場，模擬時鐘會持續推進。
              {manifest.excluded_missing_timestamp.length > 0 && (
                <> （另有 {manifest.excluded_missing_timestamp.length} 則因缺少可解析的時間戳，無法排入時間軸。）</>
              )}
            </div>
            <button className={styles.primaryBtn} onClick={() => setRunning(true)}>▶ 開始模擬</button>
          </div>
        )}

        {!finished && arrivedCount > 0 && (
          <div className={styles.commandLayout}>
            {/* 左：已抵達的貼文清單，依風險排序，點一則切換中間/右邊 */}
            <div className={styles.threadListPanel}>
              <div className={styles.panelTitle}>已抵達貼文（依風險排序，共 {arrivedCount} 則）</div>
              <div className={styles.threadListScroll}>
                {orderedItems.map((item) => {
                  const risk = riskLabel(item.thread.percentile_score);
                  const isNew = simTime - item.arrivedAtSimTime < 30 && !item.decision;
                  const isSelected = selectedThreadId === item.thread.thread_id;
                  const isAutoDecided = !!(item.decision && item.autoReasoning);
                  return (
                    <div
                      key={item.thread.thread_id}
                      className={[
                        styles.threadListItem, styles.rowEnter,
                        isSelected ? styles.threadListItemActive : '',
                        item.decision ? styles.rowDecided : '',
                        isAutoDecided ? styles.autoInterveneFlash : '',
                      ].join(' ')}
                      onClick={() => setSelectedThreadId(item.thread.thread_id)}
                    >
                      <div className={styles.rowBadges}>
                        {isNew && <span className={styles.newTag}>NEW</span>}
                        <span className={`${styles.riskTag} ${styles[risk.className]}`}>{risk.text}</span>
                        {item.decision === 'blocked' && (
                          <span className={styles.decidedTagBlocked}>{isAutoDecided && '🤖 '}已介入</span>
                        )}
                        {item.decision === 'observed' && (
                          <span className={styles.decidedTagObserved}>{isAutoDecided && '🤖 '}已觀察</span>
                        )}
                      </div>
                      <div className={styles.threadListText}>{item.thread.preview_text || '(無法讀取原文)'}</div>
                      <div className={styles.rowStat}>
                        {item.cascadeLoading && <span className={styles.spinnerSmall} />}
                        {!item.cascadeLoading && item.cascade && <span>已觀察 <b>{item.cascade.nodes.length}</b> 則</span>}
                      </div>
                      {/* 手動介入的快速按鈕：不用先點進去選取才能按，呼應
                          「左邊的 thread 也可以顯示介入的按鈕」。 */}
                      {!item.decision && (
                        <div className={styles.threadListActions} onClick={(e) => e.stopPropagation()}>
                          <button className={styles.blockBtnSmall} disabled={item.deciding}
                            onClick={() => decide(item.thread.thread_id, 'blocked')}>
                            {item.deciding ? '處理中' : '介入'}
                          </button>
                          <button className={styles.ignoreBtnSmall} disabled={item.deciding}
                            onClick={() => decide(item.thread.thread_id, 'observed')}>
                            忽略
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 中：選取貼文的傳播圖 */}
            <div className={styles.graphPanel}>
              <div className={styles.panelTitle}>傳播圖（{selectedItem ? '這則貼文目前看得到的真實回覆樹' : '尚未選取貼文'}）</div>
              <div className={styles.graphArea}>
                {selectedItem ? <ThreadCascadeGraph item={selectedItem} /> : (
                  <div className={styles.centerHint}>請從左側清單選擇一則貼文</div>
                )}
              </div>
            </div>

            {/* 右：查證結果 + 介入建議（永遠會算，不管有沒有開自動模式）+ 人工決策 */}
            <div className={styles.sidePanel}>
              {!selectedItem && <div className={styles.centerHint}>選一則貼文查看查證結果與介入建議</div>}
              {selectedItem && (
                <>
                  <div className={styles.selectedPreview}>
                    <div className={styles.selectedText}>{selectedItem.thread.preview_text}</div>
                  </div>

                  <div className={styles.block}>
                    <div className={styles.blockTitle}>查證這則貼文</div>
                    {selectedItem.verifyLoading && <div className={styles.rowDetailHint}><span className={styles.spinnerSmall} />查證載入中...</div>}
                    {!selectedItem.verifyLoading && !selectedItem.verification && (
                      <button className={styles.actionBtn} disabled={selectedItem.verifying} onClick={() => runRealVerify(selectedItem.thread.thread_id)}>
                        {selectedItem.verifying ? '查證中，本機推理可能需要數十秒...' : '🔍 查證'}
                      </button>
                    )}
                    {selectedItem.verifyError && <div className={styles.errorText}>查證失敗：{selectedItem.verifyError}</div>}
                    {selectedItem.verification && (
                      <div className={styles.resultBlock}>
                        <span className={`${styles.credBadge} ${styles[`cred_${selectedItem.verification.credibility}`] || ''}`}>
                          {CREDIBILITY_LABEL[selectedItem.verification.credibility] || selectedItem.verification.credibility}
                        </span>
                        <p className={styles.verifySummary}>{selectedItem.verification.summary}</p>
                        {selectedItem.verification.queries_used?.length > 0 && (
                          <div className={styles.verifyQueries}>搜尋關鍵字：{selectedItem.verification.queries_used.join('、')}</div>
                        )}
                        <div className={styles.rowDetailHint}>
                          {selectedItem.verification.cached ? '重播已保存的 Agent 查核紀錄（非本次即時查詢）。' : '✅ 本次即時真實查證結果（非重播快取）。'}
                        </div>
                        {!selectedItem.decision && (
                          <button className={styles.actionBtnMuted} disabled={selectedItem.verifying} onClick={() => runRealVerify(selectedItem.thread.thread_id)}>
                            重新查證
                          </button>
                        )}
                      </div>
                    )}
                  </div>

                  <div className={styles.block}>
                    <div className={styles.blockTitle}>🤖 介入建議</div>
                    {!selectedItem.verification && <div className={styles.rowDetailHint}>還沒有查證結果，無法產生建議——先查證這則貼文。</div>}
                    {selectedItem.verification && !selectedItem.recommendation && <div className={styles.rowDetailHint}><span className={styles.spinnerSmall} />建議計算中...</div>}
                    {selectedItem.recommendation && (
                      <div className={styles.resultBlock}>
                        <span className={`${styles.recommendationTag} ${styles[`rec_${selectedItem.recommendation.action}`] || ''}`}>
                          {selectedItem.recommendation.action === 'blocked' ? '建議介入'
                            : selectedItem.recommendation.action === 'observed' ? '建議不介入（證據支持為真）'
                              : '證據不足以自動判斷'}
                        </span>
                        <p className={styles.verifySummary}>{selectedItem.recommendation.reasoning}</p>
                      </div>
                    )}
                  </div>

                  <div className={styles.block}>
                    <div className={styles.blockTitle}>人工決策</div>
                    {!selectedItem.decision ? (
                      <div className={styles.rowActions}>
                        <button className={styles.blockBtnSmall} disabled={selectedItem.deciding}
                          onClick={() => decide(selectedItem.thread.thread_id, 'blocked')}>
                          {selectedItem.deciding ? '處理中' : '介入'}
                        </button>
                        <button className={styles.ignoreBtnSmall} disabled={selectedItem.deciding}
                          onClick={() => decide(selectedItem.thread.thread_id, 'observed')}>
                          忽略
                        </button>
                      </div>
                    ) : (
                      <div className={`${styles.decisionResultBanner} ${selectedItem.autoReasoning ? styles.autoInterveneFlash : ''}`}>
                        <div className={styles.decisionResultHeader}>
                          {selectedItem.autoReasoning ? '🤖 已由 Agent 自動決策' : '🧑 已由人工決策'}
                        </div>
                        {selectedItem.decision === 'blocked' ? (
                          <span className={styles.decidedTagBlocked}>
                            介入・已解鎖 {selectedItem.intervened?.realized_blocked_count ?? 0} 則
                            {selectedItem.intervened?.pending_blocked_count > 0 && <> ・{selectedItem.intervened.pending_blocked_count} 則等模擬時間到達</>}
                          </span>
                        ) : (
                          <span className={styles.decidedTagObserved}>已標記觀察</span>
                        )}
                        {selectedItem.autoReasoning && <p className={styles.verifySummary}>{selectedItem.autoReasoning}</p>}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {finished && (
          <AfterActionReport eventId={eventId} manifest={manifest} items={items} />
        )}
      </main>
    </div>
  );
};

// 事件結束後才解鎖的完整結果報告：這裡才允許讀取每則 thread 完整（未經時間
// 過濾）的真實 /cascade，因為模擬本身已經跑完，不再是「決策當下」。
// 單一 thread 的傳播圖 -- 沿用 RadarEventDetail.jsx 已經在用的同一套
// cascadeStylesheet／cascadeLayout（graphConfig.js 本來就是為了「呈現單一
// thread 的回覆樹」設計的，包含 isSource/observed/blocked 三種節點樣式，
// 不是重新發明一套）。餵進去的 item.cascade 已經是 cascade_window 回傳、
// 依模擬時間過濾過的結果，圖本身自然也只會畫出「目前看得到」的節點—— 不
// 需要額外再做一次時間過濾，過濾邏輯只在後端做那一次,這裡只是照著畫。
// 只在展開一則貼文時才掛載，不會同時起 40 個 cytoscape 實體。
const ThreadCascadeGraph = ({ item }) => {
  const cyRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const elements = useMemo(() => {
    if (!item.cascade) return [];
    const realizedBlocked = new Set(item.intervened?.realized_blocked_node_ids || []);
    const nodeElements = item.cascade.nodes.map((node) => ({
      data: {
        id: node.id,
        label: '@' + (node.screen_name || 'unknown'),
        text: node.text,
        offsetSec: node.offset_sec,
        isSource: node.is_source,
        observed: node.observed_by_cutoff,
        blocked: realizedBlocked.has(node.id),
      },
    }));
    const edgeElements = item.cascade.edges.map((edge) => ({
      data: { source: edge.source, target: edge.target, blockedEdge: realizedBlocked.has(edge.target) },
    }));
    return [...nodeElements, ...edgeElements];
  }, [item.cascade, item.intervened]);

  // 換一則貼文時清掉上一則選取的節點，不然畫面會殘留跟現在這張圖無關的
  // 節點詳情卡。
  useEffect(() => { setSelectedNode(null); }, [item.thread.thread_id]);

  // 只在「節點數真的變多」的時候才重新跑版面配置，不是每次輪詢都重跑。
  // 之前的版本用 elements（陣列參考）當依賴，但 elements 是每次輪詢
  // cascade_window 都重新 useMemo 出來的新陣列——就算內容一模一樣，參考
  // 還是不同，導致每 1.5 秒就整個 re-layout + 重新置中一次，看起來像
  // 「圖一直在動、定位不住」。改成用「目前看得到的節點數」這個純數字當
  // 依賴，數字沒變就不會重新跑版面，使用者手動拖曳/縮放的畫面也不會被
  // 每一輪輪詢打斷。
  const nodeCount = item.cascade?.nodes.length ?? 0;
  useEffect(() => {
    if (cyRef.current && nodeCount > 0) {
      cyRef.current.layout(cascadeLayout).run();
      cyRef.current.fit(undefined, 30);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeCount, item.thread.thread_id]);

  if (!item.cascade) return null;
  if (elements.length === 0) return <div className={styles.rowDetailHint}>目前模擬時間還看不到任何節點。</div>;

  return (
    <div className={styles.cascadeGraphBox}>
      <div className={styles.cascadeGraphCanvas}>
        <CytoscapeComponent
          key={item.thread.thread_id}
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
                : selectedNode.offsetSec != null
                  ? `發文後 ${Math.round(selectedNode.offsetSec / 60)} 分鐘回覆${selectedNode.observed ? '（在 30 分鐘觀察窗內）' : '（30 分鐘之後，真實歷史資料）'}`
                  : '時間資訊不明'}
              {selectedNode.blocked && ' ・已阻擋（模擬時間已解鎖的反事實節點）'}
            </div>
          </div>
        )}
      </div>
      <div className={styles.cascadeGraphLegend}>
        <span><i className={styles.legendDotSource} />來源</span>
        <span><i className={styles.legendDotObserved} />已觀察</span>
        <span><i className={styles.legendDotUnobserved} />截止後才出現（真實歷史）</span>
        {item.decision === 'blocked' && <span><i className={styles.legendDotBlocked} />已阻擋（模擬時間已解鎖）</span>}
        <span className={styles.cascadeGraphHint}>點節點看內容</span>
      </div>
    </div>
  );
};

const AfterActionReport = ({ eventId, manifest, items }) => {
  const [fullCounts, setFullCounts] = useState({}); // thread_id -> {total, blocked}
  useEffect(() => {
    let cancelled = false;
    Promise.all(manifest.threads.map((t) =>
      fetch(`/api/threads/${t.thread_id}/cascade?event_id=${eventId}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => [t.thread_id, data]),
    )).then((pairs) => {
      if (cancelled) return;
      const map = {};
      for (const [id, data] of pairs) {
        if (!data) continue;
        const blocked = data.nodes.filter((n) => !n.is_source && !n.observed_by_cutoff).length;
        map[id] = { total: data.nodes.length, blocked };
      }
      setFullCounts(map);
    });
    return () => { cancelled = true; };
  }, [eventId, manifest]);

  const rows = manifest.threads.map((t) => ({ thread: t, item: items[t.thread_id], full: fullCounts[t.thread_id] }));
  const decidedRows = rows.filter((r) => r.item?.decision);
  const blockedRows = rows.filter((r) => r.item?.decision === 'blocked');
  const observedRows = rows.filter((r) => r.item?.decision === 'observed');
  const missedRows = rows.filter((r) => !r.item?.decision); // 事件結束前完全沒被處理到
  const totalFutureNodes = rows.reduce((sum, r) => sum + (r.full?.blocked || 0), 0);
  const blockedFutureNodes = blockedRows.reduce((sum, r) => sum + (r.full?.blocked || 0), 0);
  const missedFutureNodes = totalFutureNodes - blockedFutureNodes;
  const trueMisinterventions = blockedRows.filter((r) => r.item?.verification?.credibility === 'likely_true').length;

  return (
    <div className={styles.afterAction}>
      <div className={styles.afterActionTitle}>📋 事件結束・完整成效報告</div>
      <div className={styles.afterActionNote}>
        以下解鎖完整（未經模擬時間過濾）的真實歷史資料，僅供事後檢討，不代表任何即時預測。
      </div>
      <div className={styles.kpiBar}>
        <div className={styles.kpiCard}><div className={styles.kpiValue}>{decidedRows.length}/{manifest.threads.length}</div><div className={styles.kpiLabel}>已處理／總 threads</div></div>
        <div className={styles.kpiCard}><div className={styles.kpiValue}>{blockedRows.length}</div><div className={styles.kpiLabel}>已介入</div></div>
        <div className={styles.kpiCard}><div className={styles.kpiValue}>{observedRows.length}</div><div className={styles.kpiLabel}>標記觀察</div></div>
        <div className={`${styles.kpiCard} ${missedRows.length > 0 ? styles.kpiWarn : ''}`}><div className={styles.kpiValue}>{missedRows.length}</div><div className={styles.kpiLabel}>模擬結束前未處理</div></div>
        <div className={styles.kpiCard}><div className={styles.kpiValue}>{blockedFutureNodes}</div><div className={styles.kpiLabel}>實際攔下的未來節點</div></div>
        <div className={styles.kpiCard}><div className={styles.kpiValue}>{missedFutureNodes}</div><div className={styles.kpiLabel}>漏掉的未來節點</div></div>
        <div className={`${styles.kpiCard} ${trueMisinterventions > 0 ? styles.kpiWarn : ''}`}><div className={styles.kpiValue}>{trueMisinterventions}</div><div className={styles.kpiLabel}>誤介入（後來查證為真）</div></div>
      </div>
      <table className={styles.afterActionTable}>
        <thead>
          <tr><th>貼文</th><th>抵達時間</th><th>決策</th><th>查證</th><th>總節點</th><th>攔下/漏掉</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.thread.thread_id}>
              <td className={styles.afterActionText}>{r.thread.preview_text}</td>
              <td>{formatClock(r.thread.arrival_offset_seconds)}</td>
              <td>{r.item?.decision === 'blocked' ? '介入' : r.item?.decision === 'observed' ? '觀察' : '未處理'}</td>
              <td>{CREDIBILITY_LABEL[r.item?.verification?.credibility] || '尚無查證'}</td>
              <td>{r.full?.total ?? '—'}</td>
              <td>{r.full ? (r.item?.decision === 'blocked' ? `攔下 ${r.full.blocked}` : `漏掉 ${r.full.blocked}`) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default LiveScenarioSimulator;
