// 檔案位置：frontend/src/components/NetworkGraph.jsx
import React, { useState, useRef, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

const NetworkGraph = () => {
  const [selectedNode, setSelectedNode] = useState(null);
  // 1. 新增 elements 狀態來儲存圖形資料，初始為空陣列
  const [elements, setElements] = useState([]);
  const [loading, setLoading] = useState(true);
  const cyRef = useRef(null);

  // 2. 當元件初次載入時，向後端 API 請求真實的圖形資料
  useEffect(() => {
    fetch('http://localhost:8000/api/graph')
      .then((res) => res.json())
      .then((data) => {
        if (data.elements) {
          setElements(data.elements);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("無法取得圖形資料:", err);
        setLoading(false);
      });
  }, []);

  // 視覺樣式設定 (保留您原本的設計)
  const stylesheet = [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'text-valign': 'bottom',
        'text-margin-y': 5,
        'font-size': '12px',
        'width': 'data(pagerank)',
        'height': 'data(pagerank)',
      }
    },
    { selector: 'node[is_rumour]', style: { 'background-color': '#e74c3c' } },
    { selector: 'node[!is_rumour]', style: { 'background-color': '#3498db' } },
    {
      selector: 'edge',
      style: {
        'width': 2,
        'line-color': '#bdc3c7',
        'target-arrow-color': '#bdc3c7',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier'
      }
    }
  ];

  const layout = { name: 'cose', padding: 50 };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
      <h2>假新聞擴散路徑圖 (Charlie Hebdo)</h2>
      
      <div style={{ width: '800px', height: '500px', border: '1px solid #ccc', borderRadius: '8px', backgroundColor: '#f9f9f9', position: 'relative' }}>
        {/* 若正在請求資料，顯示載入中提示 */}
        {loading && <p style={{ position: 'absolute', top: '45%', left: '45%' }}>載入圖形中...</p>}
        
        {/* 只有當 elements 內有資料時，才渲染圖形元件 */}
        {elements.length > 0 && !loading && (
          <CytoscapeComponent 
            elements={elements} 
            stylesheet={stylesheet}
            layout={layout}
            style={{ width: '100%', height: '100%' }}
            cy={(cy) => {
              if (!cyRef.current) {
                cyRef.current = cy;
                // 點擊節點時，顯示詳細資訊
                cy.on('tap', 'node', (event) => setSelectedNode(event.target.data()));
                // 點擊空白處時，清除選擇狀態
                cy.on('tap', (event) => {
                  if (event.target === cy) setSelectedNode(null);
                });
              }
            }}
          />
        )}
      </div>

      {/* 顯示選取節點的詳細資訊區塊 */}
      <div style={{ marginTop: '20px', padding: '15px', width: '800px', backgroundColor: '#eef2f5', borderRadius: '8px', minHeight: '100px' }}>
        <h3>節點詳細資訊</h3>
        {selectedNode ? (
          <div>
            <p><strong>帳號名稱：</strong> {selectedNode.label}</p>
            <p><strong>節點 ID：</strong> {selectedNode.id}</p>
            <p><strong>連線影響力：</strong> {selectedNode.pagerank}</p>
          </div>
        ) : (
          <p style={{ color: '#888' }}>請點擊圖中的圓圈節點來查看詳細資訊。</p>
        )}
      </div>
    </div>
  );
};

export default NetworkGraph;