import React from 'react';

interface ZeroStateProps {
  title: string;
  message?: string;
  action?: { 
    label: string; 
    onClick: () => void 
  };
}

export function ZeroState({ title, message, action }: ZeroStateProps) {
  return (
    <div role="status" aria-live="polite" className="zero">
      <div className="icon" aria-hidden>⚠️</div>
      <div>
        <div className="title">{title}</div>
        {message && <div className="msg">{message}</div>}
      </div>
      {action && <button className="btn" onClick={action.onClick}>{action.label}</button>}
      
      <style>{`
        .zero {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px;
          border: 1px dashed rgba(255,255,255,0.15);
          border-radius: 12px;
          background: rgba(255,255,255,0.03);
        }
        .icon {
          font-size: 18px;
        }
        .title {
          font-weight: 600;
        }
        .msg {
          opacity: .8;
        }
        .btn {
          margin-left: auto;
          padding: 6px 10px;
          border-radius: 10px;
          border: 1px solid rgba(255,255,255,0.1);
          background: transparent;
          color: #e5e7eb;
        }
        .btn:hover {
          background: rgba(255,255,255,0.06);
        }
      `}</style>
    </div>
  );
}

export default ZeroState;
