import { C } from '../constants';

// ── Icons ──────────────────────────────────────────────────────────────────
function Icon({ type }) {
  const s = { width: 18, height: 18, flexShrink: 0 };
  const stroke = C.accent;
  const sw = 1.5;
  switch (type) {
    case 'demucs': return (
      <svg {...s} viewBox="0 0 18 18" fill="none">
        <path d="M1 9 Q4.5 3 9 9 Q13.5 15 17 9" stroke={stroke} strokeWidth={sw} strokeLinecap="round"/>
      </svg>
    );
    case 'wiener': return (
      <svg {...s} viewBox="0 0 18 18" fill="none">
        <path d="M2 4h14M4 9h10M7 14h4" stroke={stroke} strokeWidth={sw} strokeLinecap="round"/>
      </svg>
    );
    case 'bandpass': return (
      <svg {...s} viewBox="0 0 18 18" fill="none">
        <rect x="6" y="3" width="6" height="12" rx="1" stroke={stroke} strokeWidth={sw}/>
        <line x1="1" y1="9" x2="6"  y2="9" stroke={stroke} strokeWidth={sw}/>
        <line x1="12" y1="9" x2="17" y2="9" stroke={stroke} strokeWidth={sw}/>
      </svg>
    );
    case 'peaks': return (
      <svg {...s} viewBox="0 0 18 18" fill="none">
        <polyline points="1,15 5,8 9,4 13,8 17,15" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    );
    case 'classical': return (
      <svg {...s} viewBox="0 0 18 18" fill="none">
        <ellipse cx="9" cy="5"  rx="7" ry="2.5" stroke={stroke} strokeWidth={sw}/>
        <path d="M2 5v4M16 5v4" stroke={stroke} strokeWidth={sw}/>
        <ellipse cx="9" cy="9"  rx="7" ry="2.5" stroke={stroke} strokeWidth={sw}/>
        <path d="M2 9v4M16 9v4" stroke={stroke} strokeWidth={sw}/>
        <ellipse cx="9" cy="13" rx="7" ry="2.5" stroke={stroke} strokeWidth={sw}/>
      </svg>
    );
    case 'ml': return (
      <svg {...s} viewBox="0 0 18 18" fill="none">
        <circle cx="9"  cy="9"  r="2" stroke={stroke} strokeWidth={sw}/>
        <circle cx="3"  cy="4"  r="1.5" stroke={stroke} strokeWidth={sw}/>
        <circle cx="15" cy="4"  r="1.5" stroke={stroke} strokeWidth={sw}/>
        <circle cx="3"  cy="14" r="1.5" stroke={stroke} strokeWidth={sw}/>
        <circle cx="15" cy="14" r="1.5" stroke={stroke} strokeWidth={sw}/>
        <line x1="4.5"  y1="5"  x2="7.5"  y2="7.5"  stroke={stroke} strokeWidth={sw}/>
        <line x1="13.5" y1="5"  x2="10.5" y2="7.5"  stroke={stroke} strokeWidth={sw}/>
        <line x1="4.5"  y1="13" x2="7.5"  y2="10.5" stroke={stroke} strokeWidth={sw}/>
        <line x1="13.5" y1="13" x2="10.5" y2="10.5" stroke={stroke} strokeWidth={sw}/>
      </svg>
    );
    case 'combining': return (
      <svg {...s} viewBox="0 0 18 18" fill="none">
        <path d="M2 5h5l5 8h4" stroke={stroke} strokeWidth={sw} strokeLinecap="round"/>
        <path d="M2 13h5l5-8h4" stroke={stroke} strokeWidth={sw} strokeLinecap="round"/>
      </svg>
    );
    default: return null;
  }
}

// ── Status indicator ────────────────────────────────────────────────────────
function StatusIndicator({ status }) {
  if (status === 'active') return (
    <div style={{
      width: 18, height: 18, borderRadius: '50%',
      border: `2px solid rgba(255,107,43,0.25)`,
      borderTopColor: C.accent,
      animation: 'spin 0.7s linear infinite',
      flexShrink: 0,
    }} />
  );
  if (status === 'complete') return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none"
      style={{ flexShrink: 0, animation: 'fadeIn 200ms ease forwards' }}>
      <circle cx="9" cy="9" r="8" fill="rgba(0,200,83,0.15)" stroke={C.success} strokeWidth="1.5"/>
      <path d="M5.5 9l2.5 2.5L12.5 6" stroke={C.success} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
  return <div style={{ width: 18, height: 18, flexShrink: 0 }} />;
}

// ── Step row ────────────────────────────────────────────────────────────────
const STEP_META = {
  demucs:    { label: 'Demucs source separation',    icon: 'demucs' },
  wiener:    { label: 'Wiener filter',               icon: 'wiener' },
  bandpass:  { label: 'Bandpass filter (80–8000 Hz)', icon: 'bandpass' },
  peaks:     { label: 'Extracting spectrogram peaks', icon: 'peaks' },
  classical: { label: 'Classical fingerprinting',    icon: 'classical' },
  ml:        { label: 'ML embedding',                icon: 'ml' },
  combining: { label: 'Combining results',           icon: 'combining' },
};

function StepRow({ stepKey, status }) {
  const meta = STEP_META[stepKey];
  if (!meta) return null;
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 14,
      padding: '12px 16px',
      background: status === 'active'
        ? 'rgba(255,107,43,0.06)'
        : 'rgba(255,255,255,0.02)',
      borderRadius: 10,
      border: `1px solid ${status === 'active' ? 'rgba(255,107,43,0.25)' : 'rgba(255,255,255,0.06)'}`,
      animation: 'fadeIn 300ms ease forwards',
      transition: 'all 300ms ease',
    }}>
      <Icon type={meta.icon} />
      <span style={{
        flex: 1, fontSize: 14, fontWeight: 500,
        color: status === 'active' ? C.text : status === 'complete' ? C.text : C.muted,
      }}>
        {meta.label}
      </span>
      <StatusIndicator status={status} />
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────
const STEP_ORDER = ['demucs', 'wiener', 'bandpass', 'peaks', 'classical', 'ml', 'combining'];

export default function ProcessingScreen({ steps }) {
  const visibleSteps = STEP_ORDER.filter(k => steps[k]);

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 24, padding: '32px 20px', width: '100%', maxWidth: 480,
    }}>
      <h2 style={{ fontSize: 24, fontWeight: 700, color: C.text }}>Analyzing...</h2>

      {/* Pipeline steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%' }}>
        {visibleSteps.map(k => (
          <StepRow key={k} stepKey={k} status={steps[k]} />
        ))}
      </div>

    </div>
  );
}
