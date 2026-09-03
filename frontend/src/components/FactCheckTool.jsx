// 檔案位置：frontend/src/components/FactCheckTool.jsx
// 給一般使用者的獨立查證工具：完全不提「介入」「節點」「傳播圖」「優先度」。
// 回答的是使用者真正在意的問題——「我看到的這段內容是真的嗎」——不是
// 「該不該封鎖哪一則」。重用同一個查證 agent (/api/factcheck)，但介面、
// 用字、視覺風格都刻意跟操作者用的「情報追蹤系統」區隔開來。
import { useState } from 'react';
import styles from './FactCheckTool.module.css';

const EXAMPLES = [
  '5G 基地台會傳播新冠病毒',
  '喝漂白水可以預防新冠肺炎',
  '某知名球星確診伊波拉病毒',
];

const CREDIBILITY_META = {
  likely_true: { label: '看起來是真的', className: 'true' },
  disputed: { label: '證據互相矛盾', className: 'disputed' },
  likely_false: { label: '看起來是假的', className: 'false' },
  unverified: { label: '查無足夠證據', className: 'unverified' },
};

const FactCheckTool = ({ onBack }) => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const runCheck = (inputText) => {
    const value = (inputText ?? text).trim();
    if (!value || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    fetch('/api/factcheck', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: value }),
    })
      .then((res) => { if (!res.ok) throw new Error(`HTTP ${res.status}`); return res.json(); })
      .then((data) => setResult(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  const useExample = (example) => {
    setText(example);
    runCheck(example);
  };

  const meta = result ? (CREDIBILITY_META[result.credibility] || CREDIBILITY_META.unverified) : null;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.brand}>🔍 查一下</div>
        <button type="button" className={styles.backLink} onClick={onBack}>研究人員入口 →</button>
      </header>

      <main className={styles.main}>
        <h1 className={styles.title}>這是真的嗎？</h1>
        <p className={styles.subtitle}>貼上你看到的一段內容，我們幫你上網找找看有沒有獨立來源可以佐證。</p>

        <div className={styles.inputCard}>
          <textarea
            className={styles.textarea}
            placeholder="例如：某某新聞說……"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            maxLength={2000}
          />
          <div className={styles.inputFooter}>
            <span className={styles.charCount}>{text.length}/2000</span>
            <button type="button" className={styles.checkBtn} onClick={() => runCheck()} disabled={loading || !text.trim()}>
              {loading ? '查證中...' : '查一下'}
            </button>
          </div>
        </div>

        {!result && !loading && (
          <div className={styles.examples}>
            <span className={styles.examplesLabel}>或試試看：</span>
            {EXAMPLES.map((example) => (
              <button key={example} type="button" className={styles.exampleChip} onClick={() => useExample(example)}>
                {example}
              </button>
            ))}
          </div>
        )}

        {loading && (
          <div className={styles.loadingCard}>
            <span className={styles.spinner} />
            正在搜尋獨立來源、比對證據……本機推理可能需要數十秒
          </div>
        )}

        {error && <div className={styles.errorCard}>查證失敗：{error}</div>}

        {result && !loading && (
          <div className={styles.resultCard}>
            <div className={styles.resultTop}>
              <span className={`${styles.verdictBadge} ${styles[meta.className]}`}>{meta.label}</span>
              <span className={styles.confidence}>信心程度 {Math.round((result.confidence || 0) * 100)}%</span>
            </div>
            <p className={styles.summary}>{result.summary}</p>

            {result.evidence?.length > 0 ? (
              <div className={styles.evidenceSection}>
                <div className={styles.evidenceLabel}>查到的獨立來源</div>
                <ul className={styles.evidenceList}>
                  {result.evidence.map((item, index) => (
                    <li key={index} className={styles.evidenceItem}>
                      <a href={item.url} target="_blank" rel="noreferrer" className={styles.evidenceLink}>{item.title || item.url}</a>
                      <span className={styles.evidenceSource}>{item.source}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className={styles.noEvidence}>沒有找到可以引用的獨立來源，建議先不要相信或轉發。</p>
            )}

            <button type="button" className={styles.againBtn} onClick={() => { setResult(null); setText(''); }}>再查一則</button>
          </div>
        )}
      </main>
    </div>
  );
};

export default FactCheckTool;
