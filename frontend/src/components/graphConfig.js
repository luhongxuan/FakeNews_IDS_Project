// 檔案位置：frontend/src/components/graphConfig.js

export const USE_MOCK_DATA = true; 

export const MOCK_ELEMENTS = [
  // --- 🌟 節點 (Nodes) 區塊：已將所有帳號完整補齊，避免連線找不到源頭 ---
  { data: { id: 'source_media', label: '官方新聞源', pagerank: 45, is_rumour: false, followers: '1.2M', account_age: '8 年', source_device: 'TweetDeck', activity_level: '極高', participated_events: [{ id: 'bostonbombing', title: '波士頓馬拉松爆炸案', date: '2013-04-15', impact_score: 40 }] } },
  { data: { id: 'user_verified_01', label: '認證機構帳號', pagerank: 35, is_rumour: false, followers: '450K', account_age: '5 年', source_device: 'Web App', activity_level: '高', participated_events: [{ id: 'charliehebdo', title: '查理週刊槍擊案', date: '2015-01-07', impact_score: 35 }] } },
  
  // 這裡就是原本被我不小心刪掉的 user_normal_02，現在補回來了！
  { data: { id: 'user_normal_02', label: '中繼轉發節點 A', pagerank: 22, is_rumour: false, followers: '1,204', account_age: '2 年', source_device: 'iPhone', activity_level: '中', participated_events: [] } },
  { data: { id: 'user_normal_03', label: '中繼轉發節點 B', pagerank: 25, is_rumour: false, followers: '3,420', account_age: '3 年', source_device: 'Android', activity_level: '中', participated_events: [{ id: 'taiwan-quake', title: '花蓮地震不實畫面', date: '2024-04-03', impact_score: 15 }] } },
  
  // 謠言核心與網軍節點
  { data: { 
    id: 'rumour_center', label: '謠言擴散源頭', pagerank: 38, is_rumour: true, followers: '85K', account_age: '1.5 年', source_device: 'Web App', activity_level: '極高 (異常)', 
    participated_events: [
      { id: 'covid19-5g', title: '5G 傳播新冠病毒', date: '2020-04-05', impact_score: 95, description: '該節點在此事件中作為第一波假消息發射源。' }, 
      { id: 'election2020', title: '美國大選舞弊指控', date: '2020-11-04', impact_score: 88, description: '大量轉發未經驗證的計票機作弊影片。' }, 
      { id: 'ukraine-ghost', title: '基輔之鬼', date: '2022-02-25', impact_score: 72, description: '發布遊戲畫面偽裝成真實戰爭影片。' },
      { id: 'paris-olympics', title: '巴黎奧運假新聞', date: '2024-07-26', impact_score: 65, description: '散播針對特定選手性別的惡意言論。' },
      { id: 'taiwan-quake', title: '花蓮地震不實畫面', date: '2024-04-03', impact_score: 80, description: '將日本海嘯舊畫面移花接木為台灣地震。' },
      { id: 'putinmissing', title: '普丁失蹤傳聞', date: '2015-03-11', impact_score: 45, description: '參與早期的政治陰謀論擴散。' }
    ] 
  } },
  { data: { id: 'bot_01', label: '異常帳號 Alpha', pagerank: 20, is_rumour: true, followers: '12', account_age: '3 天', source_device: 'API/Bot', activity_level: '高 (自動化)', participated_events: [{ id: 'election2020', title: '美國大選舞弊指控', date: '2020-11-04', impact_score: 10 }] } },
  { data: { id: 'bot_02', label: '異常帳號 Beta', pagerank: 20, is_rumour: true, followers: '8', account_age: '2 天', source_device: 'API/Bot', activity_level: '高 (自動化)', participated_events: [{ id: 'putinmissing', title: '普丁失蹤傳聞', date: '2015-03-11', impact_score: 5 }] } },
  { data: { id: 'bot_03', label: '自動化轉發機器人', pagerank: 20, is_rumour: true, followers: '150', account_age: '1 個月', source_device: 'API/Bot', activity_level: '高 (批次處理)', participated_events: [{ id: 'hurricanesandy', title: '颶風珊迪假照片', date: '2012-10-29', impact_score: 25 }, { id: 'mh370', title: '馬航 MH370 失聯', date: '2014-03-08', impact_score: 30 }] } },
  { data: { id: 'user_misled_04', label: '受影響受眾 X', pagerank: 24, is_rumour: true, followers: '450', account_age: '4 年', source_device: 'iPhone', activity_level: '低', participated_events: [] } },
  { data: { id: 'user_misled_05', label: '受影響受眾 Y', pagerank: 22, is_rumour: true, followers: '890', account_age: '5 年', source_device: 'Android', activity_level: '低', participated_events: [] } },

  // --- 🌟 連線 (Edges) 區塊：確保上面的節點全部都在，才不會報錯 ---
  { data: { source: 'source_media', target: 'user_verified_01', is_rumour: false } },
  { data: { source: 'source_media', target: 'user_normal_02', is_rumour: false } },
  { data: { source: 'user_verified_01', target: 'user_normal_03', is_rumour: false } },
  { data: { source: 'user_normal_02', target: 'rumour_center', is_rumour: true } }, 
  { data: { source: 'rumour_center', target: 'bot_01', is_rumour: true } },
  { data: { source: 'rumour_center', target: 'bot_02', is_rumour: true } },
  { data: { source: 'rumour_center', target: 'bot_03', is_rumour: true } },
  { data: { source: 'bot_01', target: 'user_misled_04', is_rumour: true } },
  { data: { source: 'bot_02', target: 'user_misled_05', is_rumour: true } },
  { data: { source: 'user_normal_03', target: 'user_misled_04', is_rumour: true } }
];

export const cyStylesheet = [
  { selector: 'node, edge', style: { 'transition-property': 'opacity', 'transition-duration': '0.3s' } },
  { selector: 'node', style: { 'label': 'data(label)', 'text-valign': 'bottom', 'text-margin-y': 6, 'font-size': '11px', 'font-family': 'system-ui, -apple-system, sans-serif', 'width': 'data(pagerank)', 'height': 'data(pagerank)', 'color': '#E8EAED', 'text-background-opacity': 0.85, 'text-background-color': '#202124', 'text-background-padding': '4px', 'text-background-shape': 'roundrectangle', 'border-width': 1.5 } },
  { selector: 'node[!is_rumour]', style: { 'background-color': '#8AB4F8', 'border-color': '#A8C7FA' } },
  { selector: 'node[?is_rumour]', style: { 'background-color': '#F28B82', 'border-color': '#FF8A65' } },
  { selector: 'edge', style: { 'width': 1.5, 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'arrow-scale': 1.0 } },
  { selector: 'edge[!is_rumour]', style: { 'line-color': '#5F6368', 'target-arrow-color': '#5F6368' } },
  { selector: 'edge[?is_rumour]', style: { 'line-color': '#8C4A46', 'target-arrow-color': '#8C4A46' } },
  { selector: '.dimmed', style: { 'opacity': 0.15 } }
];

export const cyLayout = { name: 'concentric', padding: 60, minNodeSpacing: 65, animate: false };