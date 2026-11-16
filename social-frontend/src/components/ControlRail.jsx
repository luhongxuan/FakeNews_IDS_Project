import React from 'react';
import { STATUS } from '../lib/backend.js';

const STATUS_TEXT = {
  [STATUS.LIVE]: '已連線',
  [STATUS.CONNECTING]: '連線中',
  [STATUS.OFFLINE]: '後端未連線',
};

const TYPE_LABEL = { post: '發文', share: '轉發', comment: '留言' };

export default function ControlRail({ status, queued, log, onCollapse }) {
  return (
    <aside className="rail" aria-label="導播台">
      <div className="head">
        <div>
          <div className="eyebrow">導播台</div>
          <h2 className="title">後端同步狀態</h2>
        </div>
        <button type="button" className="collapse" onClick={onCollapse} aria-label="收起導播台">
          <svg viewBox="0 0 20 20" aria-hidden="true">
            <path
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M8 5l5 5-5 5"
            />
          </svg>
        </button>
      </div>

      <p className="note">
        這一欄是操作者的控制面板，不屬於模擬的社群平台。展示使用者視角時可以收起。
      </p>

      {/* 後端連線狀態 */}
      <section className="block">
        <div className="blockLabel">後端連線</div>
        <div className="status">
          <span className={`beacon ${status === STATUS.LIVE ? 'beaconLive' : ''}`} />
          <span className="statusText">{STATUS_TEXT[status] ?? status}</span>
        </div>
        <div className="hint">
          {status === STATUS.LIVE
            ? '選擇資料集後，邊列表會即時送給後端重新計算風險分數'
            : `事件暫存於本機佇列${queued > 0 ? `（${queued} 筆待送）` : ''}，後端上線後自動補送`}
        </div>
      </section>

      {/* 事件日誌：證明載入資料集時，邊真的有送給後端 */}
      <section className="block logBlock">
        <div className="blockLabel">送出事件</div>
        <div className="log">
          {log.length === 0 ? (
            <div className="logEmpty">尚無事件。從左側選擇一個資料集載入。</div>
          ) : (
            [...log].reverse().map((entry) => (
              <div key={entry.id} className="logRow">
                <span className="logTime">
                  {new Date(entry.ts).toLocaleTimeString('zh-TW', { hour12: false })}
                </span>
                <span className={`logType type_${entry.type}`}>{TYPE_LABEL[entry.type] ?? entry.type}</span>
                <span className="logEdge">
                  {entry.type === 'post' ? entry.user : `${entry.from} → ${entry.to}`}
                </span>
              </div>
            ))
          )}
        </div>
      </section>
    </aside>
  );
}
