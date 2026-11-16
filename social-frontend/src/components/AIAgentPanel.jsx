import React from 'react';

const RISK_LABEL = { high: '高風險', medium: '中風險', low: '低風險' };
const EVIDENCE_LABEL = {
  likely_true: '證據支持', likely_false: '證據反駁', disputed: '證據衝突',
  unverified: '尚無可靠證據', agent_failure: '查核失敗',
};

export default function AIAgentPanel({ open, onClose, captured, selecting, onStartSelect, onCancelSelect, onClear }) {
  return (
    <div className={`aiPanel ${open ? 'aiPanelOpen' : ''}`} aria-hidden={!open}>
      <div className="aiHead">
        <div><div className="aiEyebrow">DECISION SUPPORT MVP</div><h2 className="aiTitle">早期介入助理</h2></div>
        <button type="button" className="aiClose" onClick={onClose} aria-label="關閉 AI 助手">×</button>
      </div>
      <p className="aiNote">框選貼文後，系統依序顯示 30 分鐘傳播風險、已保存的查證狀態與分級政策建議。本頁只產生建議與離線模擬，不會真的隱藏貼文或改變曝光。</p>
      <div className="aiFlow"><span>30 分鐘快照</span><b>→</b><span>RF 風險</span><b>→</b><span>證據閘門</span><b>→</b><span>政策建議</span></div>
      <div className="aiList">
        {captured.length === 0 ? <div className="aiEmpty">尚未擷取貼文。點下方「框選貼文」，拖曳框住一或多則貼文即可看到完整決策卡。</div>
          : captured.map(({ id, post, result }) => (
            <div key={id} className="aiItem">
              <div className="aiItemHead"><span className="aiItemUser">{post.user}</span>
                {result ? <span className={`aiVerdict risk-${result.risk}`}>{RISK_LABEL[result.risk]}</span> : <span className="aiVerdict verdictLoading">分析中…</span>}
              </div>
              <p className="aiItemContent">{post.content}</p>
              {result && <>
                <div className="aiModeRow"><span className={`aiMode ${result.sourceMode === 'validated_oof' ? 'aiModeLive' : 'aiModeDemo'}`}>{result.sourceLabel}</span><span>觀測點 {result.checkpointMinutes} 分鐘</span></div>
                <div className="aiMetricGrid">
                  <div><small>風險百分位</small><strong>{result.percentileScore.toFixed(0)}</strong></div>
                  <div><small>事件內排名</small><strong>{result.rank ? `#${result.rank}` : '展示'}</strong></div>
                  <div><small>證據狀態</small><strong>{EVIDENCE_LABEL[result.evidence.status] ?? result.evidence.status}</strong></div>
                  <div><small>建議強度</small><strong>{result.recommendation.strength}</strong></div>
                </div>
                <div className={`aiRecommendation tier-${result.recommendation.tier}`}><small>政策建議（不自動執行）</small><strong>{result.recommendation.label}</strong></div>
                <p className="aiExplain">{result.explanation}</p>
                {result.evidence.summary && <p className="aiEvidence">查證摘要：{result.evidence.summary}</p>}
              </>}
            </div>
          ))}
      </div>
      <div className="aiFooter">
        {captured.length > 0 && <button type="button" className="aiClearBtn" onClick={onClear}>清除全部</button>}
        {selecting ? <button type="button" className="aiCaptureBtn aiCaptureBtnActive" onClick={onCancelSelect}><span className="aiCaptureDot" />框選中，點此取消</button>
          : <button type="button" className="aiCaptureBtn" onClick={onStartSelect}>＋ 框選貼文</button>}
      </div>
    </div>
  );
}
