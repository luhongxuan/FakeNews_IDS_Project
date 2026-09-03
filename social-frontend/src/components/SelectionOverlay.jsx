import React, { useCallback, useEffect } from 'react';

/**
 * 選取層。啟用時鋪在畫面最上層，點一下貼文即可選取該貼文所屬的整串討論
 * （原貼文 + 所有回覆）。
 *
 * 這裡只用點擊位置判斷「碰到了哪一則貼文」，真正的內容是之後從資料裡
 * 整串取出，所以貼文再長、需要捲動幾頁，拿到的都是完整內容，
 * 不會被畫面當下顯示的範圍截斷。
 */
export default function SelectionOverlay({ active, onCaptureThreads, onCancel }) {
  /** 回傳指定畫面座標底下的貼文卡片元素 */
  const cardAtPoint = (clientX, clientY) => {
    let hit = null;
    document.querySelectorAll('[data-post-id]').forEach((el) => {
      const r = el.getBoundingClientRect();
      if (clientX >= r.left && clientX <= r.right && clientY >= r.top && clientY <= r.bottom) {
        hit = el;
      }
    });
    return hit;
  };

  const handleClick = useCallback(
    (e) => {
      if (e.button !== 0) return;
      const el = cardAtPoint(e.clientX, e.clientY);
      if (!el || !el.dataset.threadId) {
        onCancel?.();
        return;
      }
      onCaptureThreads?.([el.dataset.threadId]);
    },
    [onCaptureThreads, onCancel]
  );

  useEffect(() => {
    if (!active) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, onCancel]);

  if (!active) return null;

  return (
    <div className="selectOverlay" onMouseDown={handleClick}>
      <div className="selectHint">點一下貼文，選取整串討論　·　Esc 取消</div>
    </div>
  );
}
