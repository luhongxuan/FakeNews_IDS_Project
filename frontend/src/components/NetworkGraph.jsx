import React, { useState, useRef, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

const NetworkGraph = () => {
  // 1. 狀態管理：用來儲存「當前被點擊的節點資料」
  const [selectedNode, setSelectedNode] = useState(null);
  
  // 為了避免重複綁定點擊事件，使用 useRef 追蹤是否已經綁定過
  const cyRef = useRef(null);

  // 2. 準備圖形資料 (Elements)
  // 這裡的資料格式是 Cytoscape 規定的：必須包含 data 屬性
  // 之後這包資料會由您的 FastAPI 後端提供
  const elements = [
    // --- 節點 (Nodes) ---
    // is_rumour: 標示是否散播假新聞 / pagerank: 影響力分數 (決定節點大小)
    { data: { id: 'user_1', label: '源頭帳號 (A)', is_rumour: true, pagerank: 50 } },
    { data: { id: 'user_2', label: '一般帳號 (B)', is_rumour: false, pagerank: 30 } },
    { data: { id: 'user_3', label: '跟風帳號 (C)', is_rumour: true, pagerank: 40 } },
    { data: { id: 'user_4', label: '闢謠帳號 (D)', is_rumour: false, pagerank: 45 } },

    // --- 邊 (Edges) ---
    // source: 起點 / target: 終點
    { data: { source: 'user_1', target: 'user_2' } },
    { data: { source: 'user_1', target: 'user_3' } },
    { data: { source: 'user_3', target: 'user_4' } },
    { data: { source: 'user_2', target: 'user_4' } }
  ];

  // 3. 視覺樣式設定 (Stylesheet)
  const stylesheet = [
    // 全域節點基本樣式
    {
      selector: 'node',
      style: {
        'label': 'data(label)', // 顯示標籤文字
        'text-valign': 'bottom',// 文字顯示在節點下方
        'text-margin-y': 5,
        'font-size': '12px',
        // 動態讀取 data 中的 pagerank 來決定大小
        'width': 'data(pagerank)',
        'height': 'data(pagerank)',
      }
    },
    // 條件樣式：如果是散播謠言的節點 (is_rumour === true)，顯示為紅色
    {
      selector: 'node[is_rumour]',
      style: {
        'background-color': '#e74c3c' // 紅色
      }
    },
    // 條件樣式：如果是一般節點 (!is_rumour 或未定義)，顯示為藍色
    {
      selector: 'node[!is_rumour]',
      style: {
        'background-color': '#3498db' // 藍色
      }
    },
    // 全域邊(連線)基本樣式
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#bdc3c7',
        'target-arrow-color': '#bdc3c7',
        'target-arrow-shape': 'triangle', // 加上箭頭代表傳播方向
        'curve-style': 'bezier'
      }
    }
  ];

  // 4. 排版方式 (Layout)
  // 'cose' 是一種內建的力導向排版（會自動把節點彈開，很適合社交網路圖）
  const layout = { name: 'cose', padding: 50 };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
      <h2>假新聞擴散路徑圖</h2>
      
      {/* 圖形顯示區域 */}
      <div style={{ width: '800px', height: '500px', border: '1px solid #ccc', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
        <CytoscapeComponent 
          elements={elements} 
          stylesheet={stylesheet}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
          
          // 取得 Cytoscape 實體 (cy)，用來綁定點擊事件
          cy={(cy) => {
            if (!cyRef.current) {
              cyRef.current = cy;
              
              // 綁定點擊節點事件 (tap)
              cy.on('tap', 'node', (event) => {
                const node = event.target;
                // 將被點擊的節點資料 (node.data()) 存入 React State
                setSelectedNode(node.data());
              });

              // 點擊空白處時，清除選擇
              cy.on('tap', (event) => {
                if (event.target === cy) {
                  setSelectedNode(null);
                }
              });
            }
          }}
        />
      </div>

      {/* 5. 點擊節點後的詳細資訊面板 */}
      <div style={{ marginTop: '20px', padding: '15px', width: '800px', backgroundColor: '#eef2f5', borderRadius: '8px', minHeight: '100px' }}>
        <h3>節點詳細資訊</h3>
        {selectedNode ? (
          <div>
            <p><strong>帳號名稱：</strong> {selectedNode.label}</p>
            <p><strong>帳號 ID：</strong> {selectedNode.id}</p>
            <p><strong>影響力分數 (PageRank)：</strong> {selectedNode.pagerank}</p>
            <p><strong>狀態：</strong> 
              <span style={{ color: selectedNode.is_rumour ? 'red' : 'blue', fontWeight: 'bold', marginLeft: '5px' }}>
                {selectedNode.is_rumour ? '謠言傳播者 🚨' : '一般使用者 ✅'}
              </span>
            </p>
          </div>
        ) : (
          <p style={{ color: '#888' }}>請點擊圖中的圓圈節點來查看詳細資訊。</p>
        )}
      </div>
    </div>
  );
};

export default NetworkGraph;