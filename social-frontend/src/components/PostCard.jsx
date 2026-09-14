import React from 'react';

/* ==========================================================================
   Avatar —— 沒有真實頭像圖檔，改用「名稱首字 + 依帳號決定的底色」。
   同一個帳號永遠得到同一個顏色，視覺上才像真的社群平台。
   App.jsx（發文框）與本檔案的 PostCard 都會用到，所以放在同一個檔案、
   以具名匯出的方式共用。
   ========================================================================== */

const PALETTE = ['#4C6B63', '#7A6A55', '#4F6076', '#6B5468', '#5A6B4C', '#75604A'];

function pickColor(handle) {
  let hash = 0;
  for (let i = 0; i < handle.length; i += 1) {
    hash = (hash * 31 + handle.charCodeAt(i)) >>> 0;
  }
  return PALETTE[hash % PALETTE.length];
}

export function Avatar({ handle = '', name = '', size = 48 }) {
  const initial = (name || handle).trim().charAt(0) || '?';

  return (
    <div
      aria-hidden="true"
      style={{
        width: size,
        height: size,
        flexShrink: 0,
        borderRadius: '50%',
        backgroundColor: pickColor(handle),
        color: '#fff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: size * 0.42,
        fontWeight: 500,
        lineHeight: 1,
        userSelect: 'none',
      }}
    >
      {initial}
    </div>
  );
}

/* ==========================================================================
   PostCard —— 貼文卡片
   ========================================================================== */

/** 相對時間：社群平台慣用的「幾分鐘前」寫法 */
function relativeTime(iso) {
  const diff = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return `${Math.floor(diff)} 秒前`;
  if (diff < 3600) return `${Math.floor(diff / 60)} 分鐘前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小時前`;
  return new Date(iso).toLocaleDateString('zh-TW', { month: 'numeric', day: 'numeric' });
}

const VerifiedMark = () => (
  <svg className="verified" viewBox="0 0 20 20" aria-label="已認證帳號" role="img">
    <path
      fill="currentColor"
      d="M10 1.5l2.02 1.63 2.6-.13.72 2.5 2.16 1.44-1.06 2.38 1.06 2.38-2.16 1.44-.72 2.5-2.6-.13L10 18.5l-2.02-1.63-2.6.13-.72-2.5L2.5 13.06l1.06-2.38L2.5 8.3l2.16-1.44.72-2.5 2.6.13z"
    />
    <path fill="var(--paper)" d="M8.9 12.6L6.3 10l1.06-1.06 1.54 1.53 3.74-3.73L13.7 7.8z" />
  </svg>
);

export default function PostCard({ post, users, onShare, onComment, isNew }) {
  const author = users[post.user] ?? { handle: post.user, name: post.user, verified: false };
  const sourceAuthor = post.sharedFrom ? users[post.sharedFrom] : null;

  return (
    <article className={`card ${isNew ? 'enter' : ''}`} data-post-id={post.id}>
      {/* 轉發標頭：這一行就是圖上那條 from → to 的邊 */}
      {sourceAuthor && (
        <div className="repostFlag">
          <svg viewBox="0 0 20 20" className="repostIcon" aria-hidden="true">
            <path
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M4 8V6.5A2.5 2.5 0 016.5 4H14l-2-2m6 6v1.5a2.5 2.5 0 01-2.5 2.5H6l2 2"
            />
          </svg>
          <span>
            {author.name} 轉發了 {sourceAuthor.name} 的貼文
          </span>
        </div>
      )}

      <div className="body">
        <Avatar handle={author.handle} name={author.name} />

        <div className="main">
          <header className="meta">
            <span className="name">{author.name}</span>
            {author.verified && <VerifiedMark />}
            <span className="handle">@{author.handle}</span>
            <span className="dot">·</span>
            <time className="time" dateTime={post.ts}>
              {relativeTime(post.ts)}
            </time>
          </header>

          <p className="content">{post.content}</p>

          {/* 引用區塊：轉發鏈的視覺化，左側那條細線就是傳播路徑 */}
          {sourceAuthor && post.sourceContent && (
            <blockquote className="quote">
              <div className="quoteMeta">
                <Avatar handle={sourceAuthor.handle} name={sourceAuthor.name} size={24} />
                <span className="quoteName">{sourceAuthor.name}</span>
                {sourceAuthor.verified && <VerifiedMark />}
                <span className="quoteHandle">@{sourceAuthor.handle}</span>
              </div>
              <p className="quoteText">{post.sourceContent}</p>
            </blockquote>
          )}

          <footer className="actions">
            <button type="button" className="action" onClick={() => onComment(post)} aria-label="留言">
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                  d="M17 10.5c0 3.3-3.1 6-7 6-.9 0-1.7-.1-2.5-.4L3 17.5l1.3-3.2A5.7 5.7 0 013 10.5c0-3.3 3.1-6 7-6s7 2.7 7 6z"
                />
              </svg>
              <span>{post.comments || ''}</span>
            </button>

            <button
              type="button"
              className="action shareAction"
              onClick={() => onShare(post)}
              aria-label="轉發"
            >
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4 8V6.5A2.5 2.5 0 016.5 4H14l-2-2m6 6v1.5a2.5 2.5 0 01-2.5 2.5H6l2 2"
                />
              </svg>
              <span>{post.shares || ''}</span>
            </button>

            <button type="button" className="action likeAction" aria-label="喜歡">
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinejoin="round"
                  d="M10 16.5S3.5 12.7 3.5 8.4a3.4 3.4 0 016.5-1.3 3.4 3.4 0 016.5 1.3c0 4.3-6.5 8.1-6.5 8.1z"
                />
              </svg>
              <span>{post.likes || ''}</span>
            </button>
          </footer>
        </div>
      </div>
    </article>
  );
}
