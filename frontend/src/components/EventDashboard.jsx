// 檔案位置：frontend/src/components/EventDashboard.jsx
import React, { useState } from 'react';
import styles from './EventDashboard.module.css';

const MOCK_EVENTS = [
  { id: 'sydneysiege', title: '雪梨人質事件', date: '2014-12-15', nodes: 1221, rumours: 522, status: 'monitoring', severity: 'high', description: '發生於澳洲雪梨的咖啡館人質挾持事件，社群媒體上出現大量關於嫌犯身份的未經證實消息。' },
  { id: 'charliehebdo', title: '查理週刊槍擊案', date: '2015-01-07', nodes: 2079, rumours: 458, status: 'monitoring', severity: 'high', description: '法國巴黎《查理週刊》總部遭恐怖攻擊，Twitter 上迅速爆發大規模聲援與陰謀論。' },
  { id: 'covid19-5g', title: '5G 傳播新冠病毒', date: '2020-04-05', nodes: 5600, rumours: 3200, status: 'monitoring', severity: 'high', description: '疫情期間，網路上謠傳 5G 基地台會傳播新冠病毒，導致多地基地台遭縱火。' },
  { id: 'election2020', title: '美國大選舞弊指控', date: '2020-11-04', nodes: 8900, rumours: 4500, status: 'monitoring', severity: 'high', description: '美國總統大選期間，社群媒體上出現大量關於計票機作弊的虛假訊息。' },
  { id: 'vaccine-microchip', title: '疫苗植入微晶片', date: '2020-12-10', nodes: 3100, rumours: 2100, status: 'monitoring', severity: 'medium', description: '陰謀論指控藉由新冠疫苗向人體植入追蹤微晶片。' },
  
  { id: 'ottawashooting', title: '渥太華槍擊案', date: '2014-10-22', nodes: 890, rumours: 310, status: 'pending', severity: 'high', description: '加拿大國會山莊發生槍擊，初期 Twitter 出現關於多名槍手的錯誤警報。' },
  { id: 'putinmissing', title: '普丁失蹤傳聞', date: '2015-03-11', nodes: 238, rumours: 115, status: 'pending', severity: 'low', description: '俄羅斯總統普丁連續數日未公開露面，社群湧現政變甚至身亡的陰謀論。' },
  { id: 'ukraine-ghost', title: '基輔之鬼', date: '2022-02-25', nodes: 1500, rumours: 800, status: 'pending', severity: 'medium', description: '俄烏戰爭初期，流傳一名烏克蘭王牌飛行員擊落多架俄軍戰機的故事。' },
  { id: 'taiwan-quake', title: '花蓮地震不實畫面', date: '2024-04-03', nodes: 950, rumours: 320, status: 'pending', severity: 'low', description: '台灣花蓮發生強震，社群平台上混雜了過去其他地震的舊畫面。' },
  { id: 'paris-olympics', title: '巴黎奧運假新聞', date: '2024-07-26', nodes: 1200, rumours: 400, status: 'pending', severity: 'medium', description: '奧運期間針對選手性別、賽事安排的各類錯假訊息傳播。' },

  { id: 'bostonbombing', title: '波士頓馬拉松爆炸案', date: '2013-04-15', nodes: 3450, rumours: 890, status: 'archived', severity: 'high', description: '終點線發生爆炸，網路上湧現大量尋人與假嫌犯指控。' },
  { id: 'ferguson', title: '佛格森騷亂', date: '2014-08-09', nodes: 1143, rumours: 284, status: 'archived', severity: 'medium', description: '美國密蘇里州非裔青年遭警察槍殺引發抗議，網路上充斥對立假訊息。' },
  { id: 'hurricanesandy', title: '颶風珊迪假照片', date: '2012-10-29', nodes: 2100, rumours: 1050, status: 'archived', severity: 'medium', description: '颶風侵襲美國東岸期間，Twitter 上瘋傳多張合成的淹水假照片。' },
  { id: 'mh370', title: '馬航 MH370 失聯', date: '2014-03-08', nodes: 4200, rumours: 1800, status: 'archived', severity: 'high', description: '航班失聯，網路上出現各種關於航班下落與劫機的陰謀論。' },
  { id: 'germanwings', title: '德國之翼空難', date: '2015-03-24', nodes: 469, rumours: 112, status: 'archived', severity: 'medium', description: '德國之翼航空墜毀，網傳副機長動機與機械故障的猜測。' },
  { id: 'ebola-essien', title: '埃辛感染伊波拉謠言', date: '2014-10-12', nodes: 14, rumours: 14, status: 'archived', severity: 'low', description: '網傳知名足球員 Michael Essien 感染伊波拉病毒，為惡作劇。' },
  { id: 'gurlitt', title: '古利特藝術品收藏案', date: '2013-11-03', nodes: 138, rumours: 45, status: 'archived', severity: 'low', description: '慕尼黑發現納粹掠奪藝術品，推特上出現去向錯誤資訊。' },
];

const EventDashboard = ({ onEventClick, onLogoClick, onGraphNav, onProfileNav }) => {
  const [isLeftMenuOpen, setIsLeftMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterSeverity, setFilterSeverity] = useState('all');

  const filteredEvents = MOCK_EVENTS.filter(event => {
    const matchSearch = event.title.includes(searchQuery) || event.description.includes(searchQuery) || event.id.includes(searchQuery);
    const matchSeverity = filterSeverity === 'all' || event.severity === filterSeverity;
    return matchSearch && matchSeverity;
  });

  const pendingEvents = filteredEvents.filter(e => e.status === 'pending');
  const monitoringEvents = filteredEvents.filter(e => e.status === 'monitoring');
  const archivedEvents = filteredEvents.filter(e => e.status === 'archived');

  const getSeverityTag = (severity) => {
    switch (severity) {
      case 'high': return <span className={`${styles.tag} ${styles.tagHigh}`}>高風險</span>;
      case 'medium': return <span className={`${styles.tag} ${styles.tagMedium}`}>中風險</span>;
      case 'low': return <span className={`${styles.tag} ${styles.tagLow}`}>低風險</span>;
      default: return null;
    }
  };

  const renderCard = (event) => (
    <div key={event.id} className={styles.eventCard} onClick={() => onEventClick(event.id)}>
      <div className={styles.cardTitle}>
        {event.title}
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5F6368" strokeWidth="2"><path d="M9 18l6-6-6-6"/></svg>
      </div>
      <div className={styles.cardDesc}>{event.description}</div>
      <div className={styles.cardTags}>
        <span className={styles.tag}>{event.id}</span>
        {getSeverityTag(event.severity)}
      </div>
      <div className={styles.cardFooter}>
        <div className={styles.stats}>
          <div className={styles.statItem}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
            {event.nodes}
          </div>
          <div className={styles.statItem} style={{ color: '#F28B82' }}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            {event.rumours}
          </div>
        </div>
        <div>{event.date}</div>
      </div>
    </div>
  );

  // 🌟 修正：首頁專屬「純靜態、無條紋跑動、無文字提示」的工業風虛線留白框
  const renderEmptyPlaceholders = () => (
    [1, 2, 3].map(id => (
      <div 
        key={id} 
        style={{
          backgroundColor: 'rgba(30, 31, 34, 0.4)',
          border: '1px dashed rgba(60, 64, 67, 0.4)',
          borderRadius: '8px',
          height: '140px',
          width: '100%',
          marginBottom: '16px',
          pointerEvents: 'none'
        }}
      ></div>
    ))
  );

  return (
    <div className={styles.dashboard}>
      <header className={styles.topNav}>
        <div className={styles.navLeft}>
          <button className={styles.menuBtn} onClick={() => setIsLeftMenuOpen(!isLeftMenuOpen)}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
          </button>
          <div className={styles.navTitle} onClick={onLogoClick} style={{ cursor: 'pointer' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            情報追蹤系統
          </div>
        </div>

        <div className={styles.navCenter}>
          <div className={styles.searchBar}>
            <svg className={styles.searchIcon} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <input 
              type="text" 
              className={styles.searchInput} 
              placeholder="搜尋事件名稱、描述或 ID..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
        </div>

        <div className={styles.navRight}>
          <div className={styles.searchFilterWrapper}>
            <select className={styles.selectInput} value={filterSeverity} onChange={(e) => setFilterSeverity(e.target.value)}>
              <option value="all">所有風險</option>
              <option value="high">高風險</option>
              <option value="medium">中風險</option>
              <option value="low">低風險</option>
            </select>
          </div>
        </div>
      </header>

      <div className={styles.bodyLayout}>
        {/* 🌟 修正：完全解鎖首頁側邊欄！移除 cursor 限制並綁定對應的隨時切換路由處理器 */}
        <aside className={`${styles.leftSidebar} ${!isLeftMenuOpen ? styles.leftSidebarClosed : ''}`}>
          <div className={styles.menuItemActive}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
            事件總覽看板
          </div>
          <div className={styles.menuItem} onClick={onGraphNav} style={{ cursor: 'pointer' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
            當前拓撲分析
          </div>
          <div className={styles.menuItem} onClick={onProfileNav} style={{ cursor: 'pointer' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            實體關聯檔案
          </div>
        </aside>

        <div className={styles.kanbanBoard}>
          <div className={styles.kanbanColumn}>
            <div className={styles.columnHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#9AA0A6' }}></div>
                待分析事件
              </div>
              <span className={styles.columnCount}>{pendingEvents.length}</span>
            </div>
            <div className={styles.columnBody}>
              {pendingEvents.length > 0 ? pendingEvents.map(renderCard) : renderEmptyPlaceholders()}
            </div>
          </div>

          <div className={styles.kanbanColumn}>
            <div className={styles.columnHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#8AB4F8' }}></div>
                分析與監控中
              </div>
              <span className={styles.columnCount}>{monitoringEvents.length}</span>
            </div>
            <div className={styles.columnBody}>
              {monitoringEvents.length > 0 ? monitoringEvents.map(renderCard) : renderEmptyPlaceholders()}
            </div>
          </div>

          <div className={styles.kanbanColumn}>
            <div className={styles.columnHeader}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#81C995' }}></div>
                已結案 / 歸檔
              </div>
              <span className={styles.columnCount}>{archivedEvents.length}</span>
            </div>
            <div className={styles.columnBody}>
              {archivedEvents.length > 0 ? archivedEvents.map(renderCard) : renderEmptyPlaceholders()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EventDashboard;