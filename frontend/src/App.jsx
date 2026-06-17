// 檔案位置：frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import NetworkGraph from './components/NetworkGraph'; 
import EventDashboard from './components/EventDashboard';
import UserProfile from './components/UserProfile';

function App() {
  const [currentPage, setCurrentPage] = useState(() => localStorage.getItem('currentPage') || 'dashboard');
  const [selectedEventId, setSelectedEventId] = useState(() => localStorage.getItem('selectedEventId') || null);
  
  const [selectedUser, setSelectedUser] = useState(() => {
    try {
      const item = localStorage.getItem('selectedUser');
      if (item && item !== 'undefined' && item !== 'null' && item !== '[object Object]') {
        return JSON.parse(item);
      }
      return null;
    } catch (error) {
      return null;
    }
  });

  useEffect(() => {
    localStorage.setItem('currentPage', currentPage);
    if (selectedEventId) localStorage.setItem('selectedEventId', selectedEventId);
    else localStorage.removeItem('selectedEventId');
    if (selectedUser) localStorage.setItem('selectedUser', JSON.stringify(selectedUser));
    else localStorage.removeItem('selectedUser');
  }, [currentPage, selectedEventId, selectedUser]);

  const handleEventClick = (eventId) => {
    setSelectedEventId(eventId);
    setCurrentPage('graph');
  };

  const handleUserClick = (userData) => {
    setSelectedUser(userData);
    setCurrentPage('userProfile');
  };

  const handleBackToDashboard = () => {
    setCurrentPage('dashboard');
  };

  const handleBackToGraph = () => {
    setCurrentPage('graph');
  };

  return (
    <>
      {currentPage === 'dashboard' && (
        <EventDashboard 
          onEventClick={handleEventClick} 
          onLogoClick={handleBackToDashboard} 
          onGraphNav={() => handleEventClick(null)} /* 🌟 解鎖：允許首頁直接跳轉空圖表 */
          onProfileNav={() => handleUserClick(null)} /* 🌟 解鎖：允許首頁直接跳轉空帳戶 */
        />
      )}
      
      {currentPage === 'graph' && (
        <NetworkGraph 
          eventId={selectedEventId} 
          onBack={handleBackToDashboard} 
          onLogoClick={handleBackToDashboard}
          onEventChange={handleEventClick} 
          onUserClick={handleUserClick}
          onProfileNav={() => handleUserClick(null)} /* 🌟 解鎖：允許圖表頁隨時切換至空帳戶 */
        />
      )}

      {currentPage === 'userProfile' && (
        <UserProfile 
          user={selectedUser}
          onBack={handleBackToGraph} 
          onEventClick={handleEventClick} 
          onLogoClick={handleBackToDashboard}
          onUserClick={handleUserClick}
        />
      )}
    </>
  );
}

export default App;