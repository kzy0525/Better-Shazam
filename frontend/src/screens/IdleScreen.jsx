import { useState } from 'react';
import { C } from '../constants';

function MicIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
      <rect x="16" y="4" width="16" height="24" rx="8" fill="white" />
      <path d="M8 24C8 33.941 15.163 42 24 42C32.837 42 40 33.941 40 24" stroke="white" strokeWidth="3" strokeLinecap="round" fill="none" />
      <line x1="24" y1="42" x2="24" y2="48" stroke="white" strokeWidth="3" strokeLinecap="round" />
      <line x1="16" y1="48" x2="32" y2="48" stroke="white" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

const PILLS = [
  { label: 'Classical Fingerprinting' },
  { label: 'ML Embeddings' },
  { label: 'Noise Removal' },
];

export default function IdleScreen({ onRecord }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 36, padding: '40px 20px' }}>
      {/* Title */}
      <div style={{ textAlign: 'center' }}>
        <h1 style={{
          fontSize: 48, fontWeight: 800, letterSpacing: '-1px',
          textShadow: `0 0 30px rgba(255,107,43,0.5), 0 0 60px rgba(255,107,43,0.2)`,
          marginBottom: 10,
        }}>
          Better Shazam
        </h1>
        <p style={{ color: C.muted, fontSize: 16, fontWeight: 500, letterSpacing: '0.5px' }}>
          Dual-path audio fingerprinting
        </p>
      </div>

      {/* Record button */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18 }}>
        <button
          onClick={onRecord}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          style={{
            width: 180, height: 180, borderRadius: '50%', border: 'none', cursor: 'pointer',
            background: `radial-gradient(circle at 40% 35%, ${C.accent}, ${C.accentDk})`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: hovered
              ? `0 0 70px rgba(255,107,43,0.60), 0 0 140px rgba(255,107,43,0.25)`
              : `0 0 40px rgba(255,107,43,0.30), 0 0 80px rgba(255,107,43,0.10)`,
            transform: hovered ? 'scale(1.05)' : 'scale(1)',
            transition: 'all 300ms ease',
            animation: 'glowPulse 3s ease-in-out infinite',
          }}
        >
          <MicIcon />
        </button>
        <p style={{ color: C.muted, fontSize: 14 }}>Press to record 10 seconds</p>
      </div>

      {/* Feature pills */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
        {PILLS.map(({ label }) => (
          <div
            key={label}
            style={{
              padding: '8px 16px', borderRadius: 9999,
              background: C.card,
              border: `1px solid ${C.border}`,
              color: C.muted, fontSize: 13, fontWeight: 500,
            }}
          >
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
