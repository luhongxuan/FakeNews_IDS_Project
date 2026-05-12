import React from 'react';
// 引入我們剛剛寫的元件
import NetworkGraph from './components/NetworkGraph'; 

function App() {
  return (
    <div>
      <h1 style={{ textAlign: 'center', marginTop: '20px' }}>我的專案前端</h1>
      
      {/* 呼叫網路圖元件 */}
      <NetworkGraph />
      
    </div>
  );
}

export default App;