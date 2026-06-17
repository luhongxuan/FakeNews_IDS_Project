// 檔案位置：frontend/src/components/UserProfile.jsx
import React, { useState } from 'react';
import styles from './UserProfile.module.css';
import { MOCK_ELEMENTS } from './graphConfig';

const UserProfile = ({ user, onBack, onEventClick, onLogoClick, onUserClick }) => {
  const [isLeftMenuOpen, setIsLeftMenuOpen] = useState(false);
  const [sortKey, setSortKey] = useState('impact-desc');

  // 🌟 核心防呆與空白判定：若從選單點擊進入且無傳入資料，建立佔位骨架物件
  const isDataEmpty = !user;
  const userToDisplay = user || {
    id: '---',
    label: '未選取實體 (請使用上方搜尋欄)',
    followers: '-',
    source_device: '-',
    activity_level: '-',
    is_rumour: false,
    participated_events: []
  };

  const eventsArray = Array.isArray(userToDisplay.participated_events) ? userToDisplay.participated_events : [];
  
  const sortedEvents = [...eventsArray].sort((a, b) => {
    if (sortKey === 'impact-desc') return (b.impact_score || 0) - (a.impact_score || 0);
    if (sortKey === 'impact-asc') return (a.impact_score || 0) - (b.impact_score || 0);
    
    const timeA = a.date ? new Date(a.date).getTime() : 0;
    const timeB = b.date ? new Date(b.date).getTime() : 0;
    if (sortKey === 'date-desc') return timeB - timeA;
    if (sortKey === 'date-asc') return timeA - timeB;
    return 0;
  });

  const eventCount = eventsArray.length;
  const maxImpact = eventCount > 0 ? [...eventsArray].sort((a, b) => b.impact_score - a.impact_score)[0].impact_score : 0;
  
  // 專業級 OSINT 評級分析演算法
  let riskScore = 0;
  let riskLevel = 'safe';
  let riskLabel = '🔵 官方源 / 認證安全節點';
  let scoreLabel = '公眾影響力 (Influence)';

  if (userToDisplay.is_rumour) {
    scoreLabel = '綜合威脅指數 (Threat Index)';
    riskScore = Math.min(99, Math.round(40 + (maxImpact * 0.4) + (eventCount * 5)));
    if (riskScore >= 80) {
      riskLevel = 'critical';
      riskLabel = '🔴 核心作惡節點 (APT / 網軍頭目)';
    } else if (riskScore >= 60) {
      riskLevel = 'high';
      riskLabel = '🟠 高風險異常實體 (側翼 / 機器人)';
    } else {
      riskLevel = 'medium';
      riskLabel = '🟡 中等風險 (受誤導之傳播者)';
    }
  } else {
    riskScore = Math.min(99, Math.round((maxImpact * 0.5) + (eventCount * 2) + (userToDisplay.pagerank || 0)));
  }

  const displayScore = eventCount > 0 ? riskScore : (userToDisplay.pagerank || 0);

  return (
    <div className={styles.dashboard}>
      <header className={styles.topNav}>
        <div className={styles.navLeft}>
          <button className={styles.menuBtn} onClick={() => setIsLeftMenuOpen(!isLeftMenuOpen)}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
          </button>
          <div className={styles.navTitle} onClick={onLogoClick} style={{ cursor: 'pointer' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            情報追蹤系統
          </div>
        </div>

        {/* 🌟 修正：在帳戶頁面中央精確嵌入全域搜尋欄位，支援 Enter 檢索反查帳號 */}
        <div className={styles.navCenter}>
          <div className={styles.searchBar}>
            <svg className={styles.searchIcon} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            <input 
              type="text" 
              className={styles.searchInput} 
              placeholder="輸入帳號 ID 進行全局檢索 (例如: rumour_center)..." 
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.target.value && onUserClick) {
                  const targetId = e.target.value.trim();
                  const found = MOCK_ELEMENTS.find(el => el.data.id === targetId || el.data.label === targetId);
                  if (found) onUserClick(found.data);
                  else alert(`系統情報庫中查無此節點實體: ${targetId}`);
                }
              }}
            />
          </div>
        </div>

        <div className={styles.navRight}>
          <button className={styles.backBtn} onClick={onBack}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
            返回拓撲圖
          </button>
        </div>
      </header>

      <div className={styles.bodyLayout}>
        <aside className={`${styles.leftSidebar} ${!isLeftMenuOpen ? styles.leftSidebarClosed : ''}`}>
          <div className={styles.menuItem} onClick={onLogoClick}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>事件總覽看板
          </div>
          <div className={styles.menuItem} onClick={onBack}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>當前拓撲分析
          </div>
          <div className={styles.menuItemActive}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>實體關聯檔案
          </div>
        </aside>

        <main className={styles.mainWorkspace}>
          <section className={styles.profileCard}>
            <div className={styles.avatarSection}>
              <div className={styles.avatar}>{userToDisplay.label ? userToDisplay.label.substring(0, 1) : '@'}</div>
            </div>

            <div className={styles.infoSection}>
              <div className={styles.nameBlock}>
                {/* 🌟 修正 1：對齊新視覺標準 —— 名稱 ➔ ID ➔ 縮小化情報標籤 */}
                <div className={styles.userName}>@{userToDisplay.label}</div>
                <div className={styles.userId}>ID: {userToDisplay.id}</div>
                {!isDataEmpty && (
                  <div className={`${styles.riskText} ${styles[riskLevel]}`} style={{ fontSize: '11px', fontWeight: '500', marginTop: '6px', opacity: 0.95 }}>
                    {riskLabel}
                  </div>
                )}
              </div>
              <div className={styles.statsGrid}>
                <div className={styles.statBox}>
                  <span className={styles.statLabel}>{scoreLabel}</span>
                  <span className={`${styles.statValue} ${userToDisplay.is_rumour ? styles.riskCriticalText : ''}`}>{isDataEmpty ? '-' : displayScore} <span style={{fontSize: 12, color: '#5F6368'}}>{!isDataEmpty && '/ 99'}</span></span>
                </div>
                <div className={styles.statBox}><span className={styles.statLabel}>跨事件活動次數</span><span className={styles.statValue}>{isDataEmpty ? '-' : eventCount} <span style={{fontSize: 12, color: '#5F6368'}}>{!isDataEmpty && '次'}</span></span></div>
                <div className={styles.statBox}><span className={styles.statLabel}>主要發文設備</span><span className={styles.statValue}>{userToDisplay.source_device || '-'}</span></div>
                <div className={styles.statBox}><span className={styles.statLabel}>帳號活躍度</span><span className={styles.statValue}>{userToDisplay.activity_level || '-'}</span></div>
              </div>
            </div>
          </section>

          <section className={styles.eventsSection}>
            <div className={styles.sectionHeader}>
              <div className={styles.sectionTitle}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9AA0A6" strokeWidth="2"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>
                歷史活動軌跡 (Historical Trace)
              </div>
              {!isDataEmpty && (
                <select className={styles.selectInput} value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
                  <option value="impact-desc">權重指標：從高到低</option>
                  <option value="impact-asc">權重指標：從低到高</option>
                  <option value="date-desc">事件年份：從新到舊</option>
                  <option value="date-asc">事件年份：從舊到新</option>
                </select>
              )}
            </div>
            
            {/* 🌟 修正 2：當無資料導入時，卡片絕不塌陷消失，改用純靜態、內部無跑馬燈流動線條的留白虛線框，且不顯示任何文字提醒 */}
            {isDataEmpty ? (
              <div className={styles.cardList}>
                {[1, 2, 3].map((placeholderId) => (
                  <div key={placeholderId} className={styles.skeletonCardStatic}></div>
                ))}
              </div>
            ) : sortedEvents.length > 0 ? (
              <div className={styles.cardList}>
                {sortedEvents.map(ev => (
                  <div key={ev.id} className={styles.eventCard} onClick={() => onEventClick(ev.id)}>
                    <div className={styles.eventCardHeader}>
                      <div className={styles.eventCardTitle}>
                        {ev.title}
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8AB4F8" strokeWidth="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                      </div>
                      <span className={styles.eventDate}>{ev.date}</span>
                    </div>
                    <div className={styles.eventCardBody}>
                      <div className={styles.eventImpactBadge}>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                        個體涉入權重: {ev.impact_score}
                      </div>
                      <p className={styles.eventDesc}>{ev.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </section>
        </main>
      </div>
    </div>
  );
};

export default UserProfile;