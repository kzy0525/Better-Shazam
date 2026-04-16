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
        background: hov ? C.accent : 'transparent',
        color: hov ? '#000' : C.accent,
        border: `1.5px solid ${C.accent}`,
      }}
    >
      Try Again
    </button>
  );
}

export default function NoMatchScreen({ onReset }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 24, padding: '40px 20px',
      animation: 'fadeUp 400ms ease forwards',
    }}>
      {/* Icon */}
      <svg width="72" height="72" viewBox="0 0 72 72" fill="none">
        <circle cx="36" cy="36" r="34" stroke="rgba(0,212,200,0.25)" strokeWidth="2"/>
        <path d="M22 22l28 28M50 22L22 50" stroke={C.accent} strokeWidth="3"
          strokeLinecap="round" opacity="0.6"/>
      </svg>

      <div style={{ textAlign: 'center' }}>
        <h2 style={{ fontSize: 26, fontWeight: 700, marginBottom: 10 }}>No match found</h2>
        <p style={{ color: C.muted, fontSize: 15 }}>
          This song may not be in the database
        </p>
      </div>

      <TryAgainBtn onClick={onReset} />
    </div>
  );
}
