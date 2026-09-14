/* 使用者端 AI 助理的單一決策入口。
 * 優先使用後端已保存的 fixed-30 OOF RF 預測與查證快取；後端離線時，
 * 退回只使用目前畫面可觀測資訊的展示規則。展示規則不使用 isRumour、
 * preventable_impact 或任何 30 分鐘後欄位，也不得被當成研究模型成績。
 */

const API_BASE = import.meta.env?.VITE_API_BASE ?? 'http://localhost:8000';
const rankedEventCache = new Map();

export function riskLevel(percentile) {
  if (percentile >= 80) return 'high';
  if (percentile >= 50) return 'medium';
  return 'low';
}

export function normalizeEvidence(report) {
  const confidence = Number(report?.confidence ?? 0);
  const raw = report?.credibility ?? 'unverified';
  if ((raw === 'likely_true' || raw === 'likely_false') && confidence < 0.7) {
    return { status: 'unverified', confidence, summary: report?.summary ?? null };
  }
  return { status: raw, confidence, summary: report?.summary ?? null };
}

export function recommendAction({ risk, evidence }) {
  if (evidence === 'likely_true') return { tier: 'none', label: '不限制，維持一般監測', strength: '0%' };
  if (evidence === 'agent_failure') return { tier: 'none', label: '查核失敗，等待後續重查', strength: '0%' };
  if (evidence === 'likely_false' && risk === 'high') {
    return { tier: 'hard', label: '送人工複核；通過後才考慮高強度限制', strength: '75–100%' };
  }
  if ((evidence === 'likely_false' || evidence === 'disputed') && risk !== 'low') {
    return { tier: 'medium', label: '中度降低曝光並送人工複核', strength: '25–50%' };
  }
  if (risk === 'high') {
    return { tier: 'soft', label: '暫時小幅降低曝光，下一 checkpoint 重查', strength: '10–25%' };
  }
  return { tier: 'none', label: '持續監測，暫不限制', strength: '0%' };
}

function observableFallback(post, context) {
  const text = String(post.content ?? '');
  const threadPosts = context.allPosts.filter((item) => String(item.threadId) === String(post.threadId));
  const interactionCount = Math.max(0, threadPosts.length - 1);
  const punctuationSignals = (text.match(/[!?！？]/g) ?? []).length;
  const textSignal = Math.min(text.length / 280, 1);
  const interactionSignal = Math.min(interactionCount / 12, 1);
  const emphasisSignal = Math.min(punctuationSignals / 4, 1);
  const score = Math.round(100 * (0.55 * interactionSignal + 0.3 * textSignal + 0.15 * emphasisSignal));
  return {
    percentileScore: score,
    predictedImpact: null,
    rank: null,
    sourceMode: 'local_demo',
    sourceLabel: '離線展示規則',
    explanation: `只依目前畫面可見的文字長度、標點與同串早期互動數（${interactionCount}）產生展示分數。`,
  };
}

async function fetchEventRanking(eventId) {
  if (!rankedEventCache.has(eventId)) {
    rankedEventCache.set(eventId, fetch(`${API_BASE}/api/events/${encodeURIComponent(eventId)}/threads?limit=1000`)
      .then((response) => {
        if (!response.ok) throw new Error(`risk API ${response.status}`);
        return response.json();
      })
      .catch((error) => {
        rankedEventCache.delete(eventId);
        throw error;
      }));
  }
  return rankedEventCache.get(eventId);
}

async function fetchCachedEvidence(threadId) {
  try {
    const response = await fetch(`${API_BASE}/api/threads/${encodeURIComponent(threadId)}/verify`);
    if (!response.ok) return normalizeEvidence(null);
    return normalizeEvidence(await response.json());
  } catch {
    return normalizeEvidence(null);
  }
}

export async function analyzePost(post, context = {}) {
  const safeContext = { eventId: context.eventId ?? '', allPosts: context.allPosts ?? [] };
  let riskData;
  try {
    const ranking = await fetchEventRanking(safeContext.eventId);
    const row = ranking.threads.find((item) => String(item.thread_id) === String(post.threadId));
    if (!row) throw new Error('thread not present in fixed-30 OOF artifact');
    riskData = {
      percentileScore: Number(row.percentile_score), predictedImpact: Number(row.prediction), rank: Number(row.rank),
      sourceMode: 'validated_oof', sourceLabel: '真實 fixed-30 OOF RF',
      explanation: '風險來自既有 Nested-LOEO OOF 預測；不是內容真偽機率。',
    };
  } catch {
    riskData = observableFallback(post, safeContext);
  }

  const evidence = riskData.sourceMode === 'validated_oof' ? await fetchCachedEvidence(post.threadId) : normalizeEvidence(null);
  const risk = riskLevel(riskData.percentileScore);
  return {
    postId: post.id, threadId: post.threadId, checkpointMinutes: 30, risk, ...riskData, evidence,
    recommendation: recommendAction({ risk, evidence: evidence.status }),
  };
}

export async function analyzePosts(posts, context = {}) {
  return Promise.all(posts.map((post) => analyzePost(post, context)));
}
