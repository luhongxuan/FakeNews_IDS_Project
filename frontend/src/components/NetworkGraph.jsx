// 檔案位置：frontend/src/components/NetworkGraph.jsx
import React, { useState, useRef, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

// =======================================================
// 🌟 核心控制開關：true = 使用精心設計的模擬資料 / false = 連接真實後端 Docker
// =======================================================
const USE_MOCK_DATA = true; 

// --- 模擬資料集 (設計前端樣式專用，保證緊密連線且不卡頓) ---
const MOCK_ELEMENTS = [
  // 核心與正常節點 (藍色)
  { data: { id: 'source_media', label: '官方新聞源 (中心)', pagerank: 60, is_rumour: false } },
  { data: { id: 'user_verified_01', label: '認證時事評論員', pagerank: 45, is_rumour: false } },
  { data: { id: 'user_normal_02', label: '一般讀者_小明', pagerank: 25, is_rumour: false } },
  { data: { id: 'user_normal_03', label: '海外轉發帳號', pagerank: 30, is_rumour: false } },
  
  // 謠言與惡意擴散節點 (紅色)
  { data: { id: 'rumour_center', label: '爆料不貼圖粉專 ⚠️', pagerank: 50, is_rumour: true } },
  { data: { id: 'bot_01', label: '水軍帳號_A', pagerank: 20, is_rumour: true } },
  { data: { id: 'bot_02', label: '水軍帳號_B', pagerank: 20, is_rumour: true } },
  { data: { id: 'bot_03', label: '自動轉發機器人', pagerank: 22, is_rumour: true } },
  { data: { id: 'user_misled_04', label: '被誤導的吃瓜群眾', pagerank: 28, is_rumour: true } },
  { data: { id: 'user_misled_05', label: '社團轉發大媽', pagerank: 25, is_rumour: true } },

  // --- 訊息傳播邊線 (Edges) ---
  // 正常擴散路徑
  { data: { source: 'source_media', target: 'user_verified_01', is_rumour: false } },
  { data: { source: 'source_media', target: 'user_normal_02', is_rumour: false } },
  { data: { source: 'user_verified_01', target: 'user_normal_03', is_rumour: false } },
  
  // 謠言扭曲與爆發路徑
  { data: { source: 'user_normal_02', target: 'rumour_center', is_rumour: true } }, // 從這點開始被扭曲
  { data: { source: 'rumour_center', target: 'bot_01', is_rumour: true } },
  { data: { source: 'rumour_center', target: 'bot_02', is_rumour: true } },
  { data: { source: 'rumour_center', target: 'bot_03', is_rumour: true } },
  { data: { source: 'bot_01', target: 'user_misled_04', is_rumour: true } },
  { data: { source: 'bot_02', target: 'user_misled_05', is_rumour: true } },
  { data: { source: 'user_normal_03', target: 'user_misled_04', is_rumour: true } }
];

const NetworkGraph = () => {
  const [selectedNode, setSelectedNode] = useState(null);
  const [elements, setElements] = useState([]);
  const [loading, setLoading] = useState(true);
  const cyRef = useRef(null);

  useEffect(() => {
    if (USE_MOCK_DATA) {
      // 【模擬模式】模擬半秒鐘的網路延遲，隨後直接套用精美假資料
      const timer = setTimeout(() => {
        setElements(MOCK_ELEMENTS);
        setLoading(false);
      }, 500);
      return () => clearTimeout(timer);
    } else {
      // 【真實模式】未來到效能好的電腦，只需切換開關，就會跑這段去敲 Docker 後端
      fetch('http://localhost:8000/api/graph')
        .then((res) => res.json())
        .then((data) => {
          if (data.elements && data.elements.length > 0) {
            setElements(data.elements);
          }
          setLoading(false);
        })
        .catch((err) => {
          console.error("無法取得圖形資料:", err);
          setLoading(false);
        });
    }
  }, []);

  //  Cytoscape 樣式表 (加入更柔和現代的配色)
  const stylesheet = [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'text-valign': 'bottom',
        'text-margin-y': 8,
        'font-size': '11px',
        'font-family': 'Microsoft JhengHei, sans-serif',
        'width': 'data(pagerank)',
        'height': 'data(pagerank)',
        'color': '#2c3e50',
        'text-background-opacity': 0.7,
        'text-background-color': '#ffffff',
        'text-background-padding': '3px',
        'text-background-shape': 'roundrectangle',
        'border-width': 2,
        'border-color': '#ffffff'
      }
    },
    // 謠言帳號渲染為警戒紅
    { selector: 'node[is_rumour]', style: { 'background-color': '#e74c3c', 'box-shadow': '0px 0px 10px #e74c3c' } },
    // 正常帳號渲染為科技藍
    { selector: 'node[!is_rumour]', style: { 'background-color': '#3498db' } },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': 'data(is_rumour)',
        // 根據是不是謠言傳播線來決定線條顏色
        'line-style': 'solid',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'arrow-scale': 1.2
      }
    },
    // 動態判斷邊線顏色
    { selector: 'edge[is_rumour]', style: { 'line-color': '#fab1a0', 'target-arrow-color': '#e74c3c' } },
    { selector: 'edge[!is_rumour]', style: { 'line-color': '#dfe6e9', 'target-arrow-color': '#3498db' } }
  ];

  // 同心圓排版設定 (不傷效能，結構清晰)
  const layout = { 
    name: 'concentric', 
    padding: 40,
    minNodeSpacing: 60,
    animate: false 
  };

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: '1000px', margin: '0 auto', padding: '20px', color: '#2c3e50' }}>
      
      {/* 標題區 */}
      <div style={{ borderBottom: '2px solid #eceff1', paddingBottom: '10px', marginBottom: '20px' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
          🌐 假新聞訊息擴散路徑分析平台
          {USE_MOCK_DATA && <span style={{ fontSize: '12px', backgroundColor: '#ffeaa7', color: '#d63031', padding: '3px 8px', borderRadius: '20px', fontWeight: 'normal' }}>UI 模擬設計模式</span>}
        </h2>
        <p style={{ margin: '5px 0 0 0', color: '#7f8c8d', fontSize: '14px' }}>本圖表呈現特定社群事件中，核心帳號之間的互動轉推與假訊息滲透軌跡。</p>
      </div>

      {/* 上方快顯統計看板 (KPI Cards) */}
      <div style={{ display: 'flex', gap: '15px', marginBottom: '20px' }}>
        <div style={{ flex: 1, background: '#f8f9fa', padding: '15px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.02)', borderLeft: '5px solid #3498db' }}>
          <div style={{ fontSize: '12px', color: '#7f8c8d' }}>分析節點總數 (Users)</div>
          <div style={{ fontSize: '22px', fontWeight: 'bold', marginTop: '5px' }}>{loading ? '--' : elements.filter(e => e.data.id && !e.data.source).length} 員</div>
        </div>
        <div style={{ flex: 1, background: '#f8f9fa', padding: '15px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.02)', borderLeft: '5px solid #e74c3c' }}>
          <div style={{ fontSize: '12px', color: '#7f8c8d' }}>惡意/謠言傳播源</div>
          <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#e74c3c', marginTop: '5px' }}>{loading ? '--' : elements.filter(e => e.data.is_rumour && !e.data.source).length} 站</div>
        </div>
        <div style={{ flex: 1, background: '#f8f9fa', padding: '15px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.02)', borderLeft: '5px solid #2ecc71' }}>
          <div style={{ fontSize: '12px', color: '#7f8c8d' }}>當前監控事件範圍</div>
          <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '8px', color: '#27ae60' }}>Sydney Siege</div>
        </div>
      </div>

      {/* 主工作區：左側圖表 + 右側詳細資訊與圖例 */}
      <div style={{ display: 'flex', gap: '20px' }}>
        
        {/* 左側：Cytoscape 畫布區容器 */}
        <div style={{ position: 'relative', width: '650px', height: '480px', border: '1px solid #cfd8dc', borderRadius: '12px', backgroundColor: '#ffffff', boxShadow: '0 4px 12px rgba(0,0,0,0.05)', overflow: 'hidden' }}>
          
          {/* 圖例說明 (浮動在畫布右上角) */}
          <div style={{ position: 'absolute', top: '15px', right: '15px', zIndex: 10, background: 'rgba(255,255,255,0.9)', padding: '10px', borderRadius: '6px', border: '1px solid #e0e0e0', fontSize: '11px', display: 'flex', flexDirection: 'column', gap: '5px' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>💡 圖例說明</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#3498db', borderRadius: '50%' }}></span> 正常分享帳號</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#e74c3c', borderRadius: '50%' }}></span> 謠言相關帳號</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><span style={{ display: 'inline-block', width: '20px', height: '2px', backgroundColor: '#fab1a0' }}></span> ➔ 謠言傳播方向</div>
          </div>

          {loading && (
            <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(255,255,255,0.8)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 5 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#34495e' }}>正在渲染社交傳播網路...</div>
                <div style={{ fontSize: '12px', color: '#95a5a6', marginTop: '5px' }}>正在建構拓撲矩陣</div>
              </div>
            </div>
          )}
          
          {elements.length > 0 && (
            <CytoscapeComponent 
              elements={elements} 
              stylesheet={stylesheet}
              layout={layout}
              style={{ width: '100%', height: '100%' }}
              cy={(cy) => {
                if (!cyRef.current) {
                  cyRef.current = cy;
                  cy.on('tap', 'node', (event) => setSelectedNode(event.target.data()));
                  cy.on('tap', (event) => {
                    if (event.target === cy) setSelectedNode(null);
                  });
                }
              }}
            />
          )}
        </div>

        {/* 右側：側邊資訊欄卡片 */}
        <div style={{ width: '330px', display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <div style={{ flex: 1, backgroundColor: '#ffffff', border: '1px solid #cfd8dc', borderRadius: '12px', padding: '20px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', borderBottom: '1px solid #f1f2f6', paddingBottom: '8px', color: '#2c3e50' }}>🔍 節點探針審查</h3>
            
            {selectedNode ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
                <div style={{ backgroundColor: selectedNode.is_rumour ? '#fff5f5' : '#f0f7ff', padding: '10px', borderRadius: '6px', border: selectedNode.is_rumour ? '1px solid #fed7d7' : '1px solid #ebf8ff' }}>
                  <span style={{ fontSize: '11px', fontWeight: 'bold', color: selectedNode.is_rumour ? '#e53e3e' : '#3182ce', textTransform: 'uppercase' }}>
                    {selectedNode.is_rumour ? '⚠️ 謠言涉入帳號' : '✅ 正常情報源'}
                  </span>
                  <div style={{ fontSize: '16px', fontWeight: 'bold', marginTop: '4px', wordBreak: 'break-all' }}>@{selectedNode.label}</div>
                </div>

                <div>
                  <label style={{ fontSize: '11px', color: '#7f8c8d', display: 'block' }}>使用者系統特徵碼 (ID)</label>
                  <code style={{ background: '#f1f2f6', padding: '2px 6px', borderRadius: '4px', fontSize: '12px', display: 'inline-block', marginTop: '4px' }}>{selectedNode.id}</code>
                </div>

                <div>
                  <label style={{ fontSize: '11px', color: '#7f8c8d', display: 'block' }}>傳播影響力權重 (Node Size)</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '4px' }}>
                    <div style={{ flex: 1, backgroundColor: '#e2e8f0', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ width: `${(selectedNode.pagerank / 60) * 100}%`, backgroundColor: selectedNode.is_rumour ? '#e74c3c' : '#3498db', height: '100%' }}></div>
                    </div>
                    <span style={{ fontWeight: 'bold', fontSize: '13px' }}>{selectedNode.pagerank}</span>
                  </div>
                </div>

                <div style={{ marginTop: '10px', fontSize: '12px', color: '#95a5a6', fontStyle: 'italic', lineHeight: '1.4' }}>
                  提示：您可以點擊圖中其他圓圈節點，即時切換比對不同社交個體的影響力權重與假新聞涉入狀態。
                </div>
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#95a5a6', fontSize: '13px', textAlign: 'center', padding: '0 20px', lineHeight: '1.5' }}>
                <div>
                  <span style={{ fontSize: '24px', display: 'block', marginBottom: '10px' }}>🖱️</span>
                  請用滑鼠點擊圖中的任何一個帳號節點，即可啟動情資探針查看該使用者的詳細擴散數據。
                </div>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default NetworkGraph;