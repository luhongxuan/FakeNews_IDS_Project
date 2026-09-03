import React, { useState } from 'react';
import { MODE } from '../lib/analysis.js';

/** 把時間差寫成「發布 12 分鐘」這種說法，用來解釋為什麼走這條分析路線 */
function ageText(posts) {
  const earliest = posts.reduce((min, p) => {
    const t = new Date(p.ts).getTime();
    return Number.isFinite(t) && t < min ? t : min;
  }, Infinity);
  if (!Number.isFinite(earliest)) return null;

  const mins = Math.floor((Date.now() - earliest) / 60000);
  if (mins < 60) return `發布 ${mins} 分鐘`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `發布 ${hours} 小時`;
  return `發布 ${Math.floor(hours / 24)} 天`;
}

const STANCE_TEXT = {
  support: '佐證',
  refute: '反駁',
  context: '背景',
};

const VERDICT_STANCE = {
  false: { text: '不實', className: 'verdictFalse' },
  true: { text: '屬實', className: 'verdictTrue' },
  unclear: { text: '無法判定', className: 'verdictUnclear' },
};

/**
 * 真假判斷 —— 預留區塊，功能尚未實作。
 * 兩條路線的回傳物件裡只要帶了 verdict 欄位，這裡就會自動顯示出來。
 */
function VerdictBlock({ verdict }) {
  if (!verdict) return null;
  const stance = VERDICT_STANCE[verdict.stance] ?? VERDICT_STANCE.unclear;

  return (
    <div className="verdictBlock">
      <div className="basisLabel">真假判斷</div>
      <div className="verdictRow">
        <span className={`verdictTag ${stance.className}`}>{stance.text}</span>
        <span className="verdictLabel">{verdict.label}</span>
        {verdict.confidence != null && (
          <span className="verdictConfidence">{Math.round(verdict.confidence * 100)}%</span>
        )}
      </div>
    </div>
  );
}

/** 模型路線：傳播結構已成形，顯示介入優先度 */
function ModelResult({ result }) {
  if (!result.ready) {
    return (
      <div className="resultPending">
        <span className="pendingDot" aria-hidden="true" />
        {result.note}
      </div>
    );
  }

  return (
    <>
      <div className="scoreRow">
        <div className="scoreValue">
          {result.score}
          <span className="scoreUnit">/100</span>
        </div>
        <div className="scoreMeta">
          <div className="scoreLabel">介入優先度</div>
          {result.rank != null && (
            <div className="scoreRank">
              全部 {result.total} 串中排第 {result.rank}
            </div>
          )}
        </div>
      </div>

      <div className="scoreBar">
        <div className="scoreBarFill" style={{ width: `${result.score}%` }} />
      </div>

      {result.basis.length > 0 && (
        <div className="basisBlock">
          <div className="basisLabel">判斷依據</div>
          {result.basis.map((b) => (
            <div key={b.label} className="basisRow">
              <span className="basisName">{b.label}</span>
              <span className="basisDetail">{b.detail}</span>
            </div>
          ))}
        </div>
      )}

      <VerdictBlock verdict={result.verdict} />
    </>
  );
}

/** 查證路線：貼文太新沒有結構，顯示推論理由與證據出處 */
function SearchResult({ result }) {
  if (!result.ready) {
    return (
      <div className="resultPending">
        <span className="pendingDot" aria-hidden="true" />
        {result.note}
      </div>
    );
  }

  return (
    <>
      {result.reasoning && (
        <div className="basisBlock">
          <div className="basisLabel">判斷理由</div>
          <p className="reasoningText">{result.reasoning}</p>
        </div>
      )}

      {result.sources.length > 0 && (
        <div className="basisBlock">
          <div className="basisLabel">證據出處</div>
          {result.sources.map((s) => (
            <a key={s.url} className="sourceRow" href={s.url} target="_blank" rel="noreferrer">
              <span className={`sourceStance stance-${s.stance}`}>
                {STANCE_TEXT[s.stance] ?? s.stance}
              </span>
              <span className="sourceTitle">{s.title}</span>
            </a>
          ))}
        </div>
      )}

      <VerdictBlock verdict={result.verdict} />
    </>
  );
}

export default function AIAgentPanel({
  open,
  onClose,
  captured,
  selecting,
  onStartSelect,
  onCancelSelect,
  onClear,
}) {
  const [question, setQuestion] = useState('');

  return (
    <div className={`aiPanel ${open ? 'aiPanelOpen' : ''}`} aria-hidden={!open}>
      <div className="aiHead">
        <div className="aiHeadTitle">
          <span className="aiMark" aria-hidden="true">
            <svg viewBox="0 0 20 20">
              <path
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 10.5c0-3.6 2.7-6.5 6-6.5s6 2.9 6 6.5-2.7 6.5-6 6.5c-.9 0-1.8-.2-2.5-.6L4 17.5l1.1-3.1A6.4 6.4 0 014 10.5z"
              />
            </svg>
          </span>
          <div>
            <div className="aiEyebrow">分析助手</div>
            <h2 className="aiTitle">貼文檢視</h2>
          </div>
        </div>
        <button type="button" className="aiClose" onClick={onClose} aria-label="關閉分析助手">
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" d="M5 5l10 10M15 5L5 15" />
          </svg>
        </button>
      </div>

      <div className="aiList">
        {captured.length === 0 ? (
          <div className="aiEmpty">
            <p className="aiEmptyTitle">點一下貼文，開始檢視</p>
            <p className="aiEmptyHint">會一併帶入整串回覆內容</p>
          </div>
        ) : (
          captured.map(({ threadId, posts, result }) => {
            const source = posts[0];
            const age = ageText(posts);

            return (
              <div key={threadId} className="threadItem">
                <div className="threadHead">
                  <span className="threadUser">{source.user}</span>
                  {result ? (
                    <span className={`routeTag ${result.mode === MODE.MODEL ? 'routeModel' : 'routeSearch'}`}>
                      {result.mode === MODE.MODEL ? '模型評估' : '網路查證'}
                    </span>
                  ) : (
                    <span className="routeTag routeLoading">分析中…</span>
                  )}
                </div>

                <div className="threadMeta">
                  {age}
                  {posts.length > 1 && ` · 含 ${posts.length - 1} 則回覆`}
                </div>

                <p className="threadContent">{source.content}</p>

                {result && (
                  <div className="threadResult">
                    {result.mode === MODE.MODEL ? (
                      <ModelResult result={result} />
                    ) : (
                      <SearchResult result={result} />
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* 問答區：功能還沒開始做，先把版面留好 */}
      <div className="askBlock">
        <div className="askRow">
          <input
            type="text"
            className="askInput"
            placeholder="針對選取的內容提問"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled
          />
          <button type="button" className="askSend" disabled aria-label="送出提問">
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 10h13m-5-5l5 5-5 5"
              />
            </svg>
          </button>
        </div>
      </div>

      <div className="aiFooter">
        {captured.length > 0 && (
          <button type="button" className="aiClearBtn" onClick={onClear}>
            清除全部
          </button>
        )}

        {selecting ? (
          <button type="button" className="aiCaptureBtn aiCaptureBtnActive" onClick={onCancelSelect}>
            <span className="aiCaptureDot" aria-hidden="true" />
            選取中，點此取消
          </button>
        ) : (
          <button type="button" className="aiCaptureBtn" onClick={onStartSelect}>
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <rect
                x="3"
                y="3"
                width="14"
                height="14"
                rx="2"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeDasharray="2.5 2.5"
              />
            </svg>
            選取貼文
          </button>
        )}
      </div>
    </div>
  );
}
