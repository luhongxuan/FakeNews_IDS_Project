import React, { useCallback, useEffect, useRef, useState } from 'react';

const EDGE = 72; // 滑鼠離視窗上/下邊緣多近開始自動捲動（px）
const MAX_SPEED = 18; // 自動捲動最大速度（px / frame）

/**
 * 框選擷取層。
 * --------------------------------------------------------------------------
 * 啟用時鋪在整個畫面最上層，讓使用者用滑鼠拖出一個選取框。放開滑鼠後，
 * 會比對選取框跟畫面上所有帶 [data-post-id] 標記的貼文卡片是否有重疊，
 * 有重疊的就整則擷取。
 *
 * 這裡刻意不是做「螢幕截圖」，而是直接比對 DOM 位置、再從資料裡取出貼文
 * 的完整內容——所以即使某則貼文長到必須下拉捲動才能看完，擷取到的內容
 * 還是完整的一則，不會被螢幕當下看得到的範圍切掉。
 *
 * 拖曳選取框時，如果滑鼠靠近視窗上緣或下緣，畫面會自動捲動，讓使用者
 * 不用中斷拖曳、也能框選到目前不在畫面範圍內的貼文。
 */
export default function SelectionOverlay({ active, onCapture, onCancel }) {
  const [rect, setRect] = useState(null); // 目前選取框（視窗座標，隨捲動即時換算）
  const startDoc = useRef(null); // 起點（文件座標 = 視窗座標 + 捲動位移）
  const currentDoc = useRef(null);
  const dragging = useRef(false);
  const scrollSpeed = useRef(0);
  const rafId = useRef(null);

  const toDocPoint = (clientX, clientY) => ({
    x: clientX + window.scrollX,
    y: clientY + window.scrollY,
  });

  const computeScreenRect = useCallback(() => {
    if (!startDoc.current || !currentDoc.current) return null;
    const x1 = Math.min(startDoc.current.x, currentDoc.current.x) - window.scrollX;
    const x2 = Math.max(startDoc.current.x, currentDoc.current.x) - window.scrollX;
    const y1 = Math.min(startDoc.current.y, currentDoc.current.y) - window.scrollY;
    const y2 = Math.max(startDoc.current.y, currentDoc.current.y) - window.scrollY;
    return { left: x1, top: y1, width: x2 - x1, height: y2 - y1 };
  }, []);

  /** 拖曳期間持續執行：處理自動捲動、更新選取框視覺位置 */
  const tick = useCallback(() => {
    if (!dragging.current) return;
    if (scrollSpeed.current !== 0) {
      window.scrollBy(0, scrollSpeed.current);
    }
    setRect(computeScreenRect());
    rafId.current = requestAnimationFrame(tick);
  }, [computeScreenRect]);

  const handleMouseDown = useCallback(
    (e) => {
      if (!active || e.button !== 0) return;
      dragging.current = true;
      startDoc.current = toDocPoint(e.clientX, e.clientY);
      currentDoc.current = startDoc.current;
      setRect(computeScreenRect());
      rafId.current = requestAnimationFrame(tick);
    },
    [active, computeScreenRect, tick]
  );

  const handleMouseMove = useCallback((e) => {
    if (!dragging.current) return;
    currentDoc.current = toDocPoint(e.clientX, e.clientY);

    // 靠近視窗上下邊緣時，依距離算出自動捲動速度
    const vh = window.innerHeight;
    if (e.clientY < EDGE) {
      scrollSpeed.current = -Math.ceil(((EDGE - e.clientY) / EDGE) * MAX_SPEED);
    } else if (e.clientY > vh - EDGE) {
      scrollSpeed.current = Math.ceil(((e.clientY - (vh - EDGE)) / EDGE) * MAX_SPEED);
    } else {
      scrollSpeed.current = 0;
    }
  }, []);

  const finishSelection = useCallback(() => {
    dragging.current = false;
    scrollSpeed.current = 0;
    cancelAnimationFrame(rafId.current);

    if (!startDoc.current || !currentDoc.current) {
      setRect(null);
      return;
    }

    const selLeft = Math.min(startDoc.current.x, currentDoc.current.x);
    const selRight = Math.max(startDoc.current.x, currentDoc.current.x);
    const selTop = Math.min(startDoc.current.y, currentDoc.current.y);
    const selBottom = Math.max(startDoc.current.y, currentDoc.current.y);

    setRect(null);
    startDoc.current = null;
    currentDoc.current = null;

    // 幾乎沒拖出範圍，視為誤點、取消本次框選
    if (selRight - selLeft < 6 && selBottom - selTop < 6) {
      onCancel?.();
      return;
    }

    const hitIds = [];
    document.querySelectorAll('[data-post-id]').forEach((el) => {
      const r = el.getBoundingClientRect();
      const elLeft = r.left + window.scrollX;
      const elRight = r.right + window.scrollX;
      const elTop = r.top + window.scrollY;
      const elBottom = r.bottom + window.scrollY;

      const overlap = elLeft < selRight && elRight > selLeft && elTop < selBottom && elBottom > selTop;
      if (overlap) hitIds.push(el.dataset.postId);
    });

    onCapture?.(hitIds);
  }, [onCapture, onCancel]);

  useEffect(() => {
    if (!active) return undefined;
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', finishSelection);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', finishSelection);
      cancelAnimationFrame(rafId.current);
    };
  }, [active, handleMouseMove, finishSelection]);

  if (!active) return null;

  return (
    <div className="selectOverlay" onMouseDown={handleMouseDown}>
      <div className="selectHint">拖曳滑鼠框選貼文，靠近畫面上下邊緣會自動捲動</div>
      {rect && (
        <div
          className="selectBox"
          style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
        />
      )}
    </div>
  );
}
