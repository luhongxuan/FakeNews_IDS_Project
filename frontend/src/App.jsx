// 檔案位置：frontend/src/App.jsx
import { useCallback, useEffect, useState } from 'react';
import NetworkGraph from './components/NetworkGraph';
import EventDashboard from './components/EventDashboard';
import UserProfile from './components/UserProfile';
import InterventionDashboard from './components/InterventionDashboard';
import FactCheckTool from './components/FactCheckTool';
import LiveScenarioSimulator from './components/LiveScenarioSimulator';
import TrendingRadar from './components/TrendingRadar';
import RadarReviewQueue from './components/RadarReviewQueue';
import RadarEventDetail from './components/RadarEventDetail';

const HISTORY_KEY = 'fakenews-navigation';

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

  const navigate = useCallback((page, eventId = selectedEventId, user = selectedUser) => {
    const state = {
      [HISTORY_KEY]: true,
      page,
      selectedEventId: eventId,
      selectedUser: user,
    };
    window.history.pushState(state, '', window.location.href);
    setCurrentPage(page);
    setSelectedEventId(eventId);
    setSelectedUser(user);
  }, [selectedEventId, selectedUser]);

  useEffect(() => {
    const currentState = window.history.state;
    if (!currentState?.[HISTORY_KEY]) {
      window.history.replaceState({
        [HISTORY_KEY]: true,
        page: currentPage,
        selectedEventId,
        selectedUser,
      }, '', window.location.href);
    }

    const handlePopState = (event) => {
      const state = event.state;
      if (!state?.[HISTORY_KEY]) return;
      setCurrentPage(state.page || 'dashboard');
      setSelectedEventId(state.selectedEventId || null);
      setSelectedUser(state.selectedUser || null);
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
    // This effect initializes the current browser entry once. Navigation is
    // handled by `navigate`, while popstate restores entries created by it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleEventClick = (eventId) => {
    navigate('intervention', eventId);
  };

  const handleGraphClick = (eventId) => {
    navigate('graph', eventId);
  };

  const handleUserClick = (userData) => {
    navigate('userProfile', selectedEventId, userData);
  };

  const handleBackToDashboard = () => {
    navigate('dashboard', null, null);
  };

  const handleBackToGraph = () => {
    navigate('graph');
  };

  const handleFactCheckClick = () => {
    navigate('factcheck', null, null);
  };

  const handleLiveScenarioClick = (eventId) => {
    navigate('live-scenario', eventId);
  };

  const handleRadarClick = () => {
    navigate('radar', null, null);
  };

  const handleRadarReviewClick = () => {
    navigate('radar-review', null, null);
  };

  const handleOpenRadarThread = (threadId) => {
    navigate('radar-thread', threadId, null);
  };

  return (
    <>
      {currentPage === 'dashboard' && (
        <EventDashboard
          onEventClick={handleEventClick}
          onLogoClick={handleBackToDashboard}
          onGraphNav={() => handleGraphClick(null)} /* 🌟 解鎖：允許首頁直接跳轉空圖表 */
          onProfileNav={() => handleUserClick(null)} /* 🌟 解鎖：允許首頁直接跳轉空帳戶 */
          onFactCheckNav={handleFactCheckClick}
          onRadarNav={handleRadarClick}
          onRadarEventClick={handleOpenRadarThread}
        />
      )}

      {currentPage === 'factcheck' && (
        <FactCheckTool onBack={handleBackToDashboard} />
      )}

      {currentPage === 'radar' && (
        <TrendingRadar onBack={handleBackToDashboard} />
      )}

      {currentPage === 'radar-review' && (
        <RadarReviewQueue
          onOpenThread={handleOpenRadarThread}
          onBack={handleBackToDashboard}
          onLogoClick={handleBackToDashboard}
        />
      )}

      {currentPage === 'radar-thread' && (
        <RadarEventDetail
          threadId={selectedEventId}
          onBack={handleBackToDashboard}
          onLogoClick={handleBackToDashboard}
        />
      )}

      {currentPage === 'intervention' && (
        <InterventionDashboard
          eventId={selectedEventId}
          onBack={handleBackToDashboard}
          onLogoClick={handleBackToDashboard}
          onProfileNav={() => handleUserClick(null)}
          onGraphNav={() => handleGraphClick(selectedEventId)}
          onLiveScenarioNav={() => handleLiveScenarioClick(selectedEventId)}
          onRadarReviewNav={handleRadarReviewClick}
        />
      )}

      {currentPage === 'live-scenario' && (
        <LiveScenarioSimulator
          eventId={selectedEventId}
          onBack={() => handleEventClick(selectedEventId)}
          onLogoClick={handleBackToDashboard}
        />
      )}

      {currentPage === 'graph' && (
        <NetworkGraph
          eventId={selectedEventId}
          onBack={handleBackToDashboard}
          onLogoClick={handleBackToDashboard}
          onEventChange={handleGraphClick}
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
