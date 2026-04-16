import { useState } from 'react';
import { C } from '../constants';

function TryAgainBtn({ onClick }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        padding: '12px 36px', borderRadius: 9999, fontSize: 15, fontWeight: 600,
        cursor: 'pointer', transition: 'all 300ms ease',
        background: hov ? C.error : 'transparent',
        color: hov ? '#fff' : C.error,
        border: `1.5px solid ${C.error}`,
      }}
    >
      Try Again
    </button>
  );
}

function getSubtitle(message) {
  if (!message) return 'An unexpected error occurred.';
  const m = String(message).toLowerCase();
  if (m === 'microphone_denied' || m.includes('notallowed') || m.includes('permission'))
    return 'Allow microphone access in your browser settings';
  if (m.includes('fetch') || m.includes('unreachable') || m.includes('failed to fetch') || m.includes('networkerror'))
    return 'Make sure the backend server is running: python backend.py';
  return message;
}

function getTitle(message) {
  if (!message) return 'Something went wrong';
  const m = String(message).toLowerCase();
  if (m === 'microphone_denied' || m.includes('notallowed') || m.includes('permission'))
    return 'Microphone access denied';
  if (m.includes('fetch') || m.includes('unreachable') || m.includes('failed to fetch'))
    return 'Backend unreachable';
  return 'Something went wrong';
}

export default function ErrorScreen({ message, onReset }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 24, padding: '40px 20px',
      animation: 'fadeUp 400ms ease forwards',
    }}>
      {/* Warning triangle */}
      <svg width="72" height="72" viewBox="0 0 72 72" fill="none">
        <path d="M36 8L68 62H4L36 8Z" stroke={C.error} strokeWidth="2.5"
          strokeLinejoin="round" fill="rgba(255,61,0,0.10)" />
        <line x1="36" y1="28" x2="36" y2="44" stroke={C.error} strokeWidth="2.5" strokeLinecap="round"/>
        <circle cx="36" cy="52" r="2" fill={C.error}/>
      </svg>

      <div style={{ textAlign: 'center', maxWidth: 380 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12, color: C.error }}>
          {getTitle(message)}
        </h2>
        <p style={{ color: C.muted, fontSize: 14, lineHeight: 1.6 }}>
          {getSubtitle(message)}
        </p>
      </div>

      <TryAgainBtn onClick={onReset} />
    </div>
  );
}
