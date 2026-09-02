import React from 'react';

const VERDICT_META = {
  rumour: { label: '疑似謠言', className: 'verdictRumour' },
  real: { label: '判定為真', className: 'verdictReal' },
  pending: { label: '模型尚未訓練完成', className: 'verdictPending' },
};

export default function AIAgentPanel({
  open,
  onClose,
  captured,
  selecting,
  onStartSelect,
  onCancelSelect,
  onClear,
}) {
  return (
    <div className={`aiPanel ${open ? 'aiPanelOpen' : ''}`} aria-hidden={!open}>
      <div className="aiHead">
        <div>
          <div className="aiEyebrow">AI 分析助手</div>
          <h2 className="aiTitle">畫面內容偵測</h2>
        </div>
        <button type="button" className="aiClose" onClick={onClose} aria-label="關閉 AI 助手">
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              d="M5 5l10 10M15 5L5 15"
            />
          </svg>
        </button>
      </div>

      <p className="aiNote">
        之後這裡會接上我們自己的偵測模型，即時分析畫面上的貼文是否為謠言。模型目前還在訓練，
        這裡先示範「框選 → 送分析 → 顯示結果」的完整流程。
      </p>

      <div className="aiList">
        {captured.length === 0 ? (
          <div className="aiEmpty">
            尚未擷取任何貼文。點下方「框選貼文」，在畫面上拖曳框出想分析的內容——貼文如果很長，
            拖曳到畫面邊緣會自動往下捲動，不用中斷動作。
          </div>
        ) : (
          captured.map(({ id, post, result }) => (
            <div key={id} className="aiItem">
              <div className="aiItemHead">
                <span className="aiItemUser">{post.user}</span>
                {result ? (
                  <span className={`aiVerdict ${VERDICT_META[result.verdict].className}`}>
                    {result.label ?? VERDICT_META[result.verdict].label}
                  </span>
                ) : (
                  <span className="aiVerdict verdictLoading">分析中…</span>
                )}
              </div>
              <p className="aiItemContent">{post.content}</p>
            </div>
          ))
        )}
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
            框選中，點此取消
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
            框選貼文
          </button>
        )}
      </div>
    </div>
  );
}
