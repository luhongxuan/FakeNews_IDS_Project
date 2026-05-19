// 檔案位置：frontend/src/components/NetworkGraph.jsx
import React, { useState, useRef, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import styles from './NetworkGraph.module.css';
import { USE_MOCK_DATA, MOCK_ELEMENTS, cyStylesheet, cyLayout } from './graphConfig';

const NetworkGraph = ({ eventId, onBack, onLogoClick, onEventChange, onUserClick, onProfileNav }) => {
  const [selectedNode, setSelectedNode] = useState(null);
  const [elements, setElements] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isLeftMenuOpen, setIsLeftMenuOpen] = useState(false);
  const cyRef = useRef(null);

  useEffect(() => {
    closePanel();
    if (USE_MOCK_DATA) {
      setLoading(true); setElements([]);
      const timer = setTimeout(() => { setElements(MOCK_ELEMENTS); setLoading(false); }, 600);
      return () => clearTimeout(timer);
    }
  }, [eventId]);

  const closePanel = () => {
    setSelectedNode(null);
    if (cyRef.current) {
      cyRef.current.elements().unselect(); 
      cyRef.current.elements().removeClass('dimmed'); 
    }
  };

  // 🌟 實體按鈕：依據畫面正中央為基準點進行縮放
  const handleZoomIn = () => { 
    if (cyRef.current) {
      const cy = cyRef.current;
      cy.zoom({
        level: cy.zoom() * 1.2,
        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
      });
    }
  };

  const handleZoomOut = () => { 
    if (cyRef.current) {
      const cy = cyRef.current;
      cy.zoom({
        level: cy.zoom() * 0.8,
        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 }
      });
    } 
  };

  const handleFit = () => { if (cyRef.current) { cyRef.current.fit(); cyRef.current.center(); } };

  const eventsArray = Array.isArray(selectedNode?.participated_events) ? selectedNode.participated_events : [];
  const sortedEvents = [...eventsArray].sort((a, b) => (b.impact_score || 0) - (a.impact_score || 0));
  const top5Events = sortedEvents.slice(0, 5);
  const hasMoreEvents = sortedEvents.length > 5;

  return (
    <div className={styles.dashboard}>
      <header className={styles.topNav}>
        <div className={styles.navLeft}>
          <button className={styles.menuBtn} onClick={() => setIsLeftMenuOpen(!isLeftMenuOpen)}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
          </button>
          <div className={styles.navTitle} onClick={onLogoClick} style={{ cursor: 'pointer' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            情報追蹤系統
          </div>
        </div>
        <div className={styles.navCenter}>
          <div className={styles.searchBar}>
            <svg className={styles.searchIcon} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            <input 
              type="text" 
              className={styles.searchInput} 
              placeholder="輸入事件 ID 搜尋 (例如: election2020)..." 
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.target.value) onEventChange(e.target.value);
              }}
            />
          </div>
        </div>
        <div className={styles.navRight}></div>
      </header>

      <div className={styles.bodyLayout}>
        <aside className={`${styles.leftSidebar} ${!isLeftMenuOpen ? styles.leftSidebarClosed : ''}`}>
          <div className={styles.menuItem} onClick={onBack}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>事件總覽看板
          </div>
          <div className={styles.menuItemActive}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>當前拓撲分析
          </div>
          <div className={styles.menuItem} onClick={onProfileNav} style={{ cursor: 'pointer' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>實體關聯檔案
          </div>
        </aside>

        <div className={styles.contentWrapper}>
          <section className={styles.kpiGrid}>
            <div className={styles.card}>
              <div className={styles.kpiLabel}>分析節點總數</div>
              <div className={styles.kpiValue}>{loading ? '-' : elements.filter(e => e.data.id && !e.data.source).length}</div>
            </div>
            <div className={styles.card}>
              <div className={styles.kpiLabel}>異常涉入帳號</div>
              <div className={styles.kpiValueDanger}>{loading ? '-' : elements.filter(e => e.data.is_rumour && !e.data.source).length}</div>
            </div>
            <div className={styles.card}>
              <div className={styles.kpiLabel}>當前分析事件集</div>
              <div className={styles.kpiValue} style={{ color: '#E3E3E3', fontSize: '18px', marginTop: '4px' }}>
                {loading ? '載入中...' : (eventId ? eventId.toUpperCase() : '未選取事件 (請搜尋)')}
              </div>
            </div>
          </section>

          <main className={styles.mainWorkspace}>
            <div className={styles.graphContainer}>
              <div className={styles.graphHeader}>
                <span>網絡傳播拓撲矩陣</span>
                <div className={styles.legendContainer}>
                  <div className={styles.legendItem}><div className={styles.dotBlue}></div> 正常節點</div>
                  <div className={styles.legendItem}><div className={styles.dotRed}></div> 異常/謠言節點</div>
                </div>
              </div>

              {/* 🌟 核心修改區：接管滑鼠滾輪，使用一致的比例縮放，並跟隨滑鼠座標 */}
              <div 
                style={{ flex: 1, position: 'relative' }}
                onWheel={(e) => {
                  if (!cyRef.current || loading || elements.length === 0) return;
                  const cy = cyRef.current;
                  const oldZoom = cy.zoom();
                  
                  // 使用 1.15 倍率進行比例縮放 (適中且平滑的速度，不會忽快忽慢)
                  const zoomFactor = 1.15;
                  let newZoom = e.deltaY < 0 ? oldZoom * zoomFactor : oldZoom / zoomFactor;
                  
                  if (newZoom < 0.3) newZoom = 0.3;
                  if (newZoom > 3) newZoom = 3;

                  // 取得滑鼠在畫布中的相對座標，讓縮放中心緊緊跟隨滑鼠
                  const rect = e.currentTarget.getBoundingClientRect();
                  cy.zoom({
                    level: newZoom,
                    renderedPosition: { 
                      x: e.clientX - rect.left, 
                      y: e.clientY - rect.top 
                    }
                  });
                }}
              >
                {!loading && elements.length > 0 && (
                  <div className={styles.zoomControls}>
                    <button className={styles.zoomBtn} onClick={handleZoomIn}><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg></button>
                    <button className={styles.zoomBtn} onClick={handleZoomOut}><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="5" y1="12" x2="19" y2="12"></line></svg></button>
                    <button className={styles.zoomBtn} onClick={handleFit}><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"></path></svg></button>
                  </div>
                )}
                
                {!loading && elements.length === 0 && (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#5F6368', fontSize: '14px' }}>
                    請由上方搜尋列輸入事件 ID (如: sydneysiege) 以初始化傳播矩陣
                  </div>
                )}

                {elements.length > 0 && (
                  <CytoscapeComponent 
                    elements={elements} 
                    stylesheet={cyStylesheet} 
                    layout={cyLayout} 
                    userZoomingEnabled={false} /* 關閉原生縮放，改用上方完美的自訂滑鼠縮放 */
                    minZoom={0.3} 
                    maxZoom={3} 
                    style={{ width: '100%', height: '100%' }}
                    cy={(cy) => {
                      cyRef.current = cy; cy.removeAllListeners('tap');
                      cy.on('tap', 'node', (event) => {
                        const node = event.target; setSelectedNode(JSON.parse(JSON.stringify(node.data())));
                        cy.elements().removeClass('dimmed'); cy.elements().not(node.closedNeighborhood()).addClass('dimmed');
                        cy.animate({ center: { eles: node }, zoom: Math.max(cy.zoom(), 1.5) }, { duration: 400, easing: 'ease-out-cubic' });
                      });
                      cy.on('tap', (event) => { if (event.target === cy) closePanel(); });
                    }}
                  />
                )}
              </div>

              <aside className={`${styles.sidePanel} ${selectedNode ? styles.sidePanelOpen : ''}`}>
                <div className={styles.panelHeader}>
                  節點屬性
                  <button className={styles.closeBtn} onClick={closePanel}><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg></button>
                </div>

                {selectedNode && (
                  <div className={styles.panelContent}>
                    
                    <div className={styles.nodeIdentityClickable} onClick={() => onUserClick(selectedNode)} title="查看完整使用者分析檔案">
                      <div className={styles.avatar}>{selectedNode.label ? selectedNode.label.substring(0, 1) : '@'}</div>
                      <div className={styles.nodeTitles}>
                        <div style={{ fontSize: '11px', fontWeight: '500', marginBottom: '6px', fontFamily: 'system-ui, sans-serif', opacity: 0.9 }}>
                          {(() => {
                            const evs = Array.isArray(selectedNode.participated_events) ? selectedNode.participated_events : [];
                            const sortedEvs = [...evs].sort((a, b) => (b.impact_score || 0) - (a.impact_score || 0));
                            const eCount = evs.length;
                            const mImpact = eCount > 0 ? (sortedEvs[0].impact_score || 0) : 0;
                            if (selectedNode.is_rumour) {
                              const score = Math.min(99, Math.round(40 + (mImpact * 0.4) + (eCount * 5)));
                              if (score >= 80) return '🔴 核心作惡節點 (APT / 網軍頭目)';
                              if (score >= 60) return '🟠 高風險異常實體 (側翼 / 機器人)';
                              return '🟡 中等風險 (受誤導之傳播者)';
                            } else {
                              return '🔵 官方源 / 認證安全節點';
                            }
                          })()}
                        </div>
                        
                        <div className={styles.nodeName} style={{ fontSize: '18px', fontWeight: '600', color: '#E3E3E3' }}>@{selectedNode.label || '未知'}</div>
                        <div style={{ fontSize: '13px', color: '#9AA0A6', fontFamily: 'monospace', marginTop: '4px' }}>ID: {selectedNode.id}</div>
                      </div>
                    </div>

                    <div className={styles.divider}></div>

                    <div className={styles.statsGrid}>
                      <div className={styles.statBox}><span className={styles.statLabel}>PageRank 權重</span><span className={styles.statValue}>{selectedNode.pagerank || '0.0'}</span></div>
                      <div className={styles.statBox}><span className={styles.statLabel}>追蹤人數</span><span className={styles.statValue}>{selectedNode.followers || '-'}</span></div>
                      <div className={styles.statBox}><span className={styles.statLabel}>帳號建立年份</span><span className={styles.statValue}>{selectedNode.account_age || '-'}</span></div>
                      <div className={styles.statBox}><span className={styles.statLabel}>發文設備</span><span className={styles.statValue}>{selectedNode.source_device || '-'}</span></div>
                    </div>

                    <div className={styles.divider}></div>

                    <div className={styles.historySection}>
                      <div className={styles.sectionTitle}>高影響力參與事件 (Top 5)</div>
                      {top5Events.length > 0 ? (
                        top5Events.map(ev => (
                          <div key={ev.id} className={styles.historyCard} onClick={() => onEventChange(ev.id)}>
                            <div className={styles.historyTitle}>
                              {ev.title}
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M9 18l6-6-6-6"/></svg>
                            </div>
                            <div className={styles.historyDate}>影響力評分：{ev.impact_score}</div>
                          </div>
                        ))
                      ) : (
                        <div className={styles.historyEmpty}>此帳號無跨事件參與紀錄</div>
                      )}

                      {hasMoreEvents && (
                        <button className={styles.viewMoreBtn} onClick={() => onUserClick(selectedNode)}>
                          查看所有事件 ({sortedEvents.length})
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </aside>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default NetworkGraph;