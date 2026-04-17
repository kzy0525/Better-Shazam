import { useState } from 'react';
import { C } from '../constants';

function confidenceStyle(label) {
  if (!label) return { bg: 'rgba(136,136,136,0.2)', color: C.muted, icon: '–' };
  const l = label.toLowerCase();
  if (l.includes('high') || l.includes('confident'))
    return { bg: 'rgba(0,200,83,0.18)', color: C.success, border: 'rgba(0,200,83,0.4)', icon: '✓' };
  if (l.includes('moderate'))
    return { bg: 'rgba(255,171,0,0.18)', color: C.warning, border: 'rgba(255,171,0,0.4)', icon: '⚠' };
  return { bg: 'rgba(255,61,0,0.18)', color: C.error, border: 'rgba(255,61,0,0.4)', icon: '✕' };
}

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

function PathCard({ title, icon, children }) {
  return (
    <div style={{
      flex: 1, minWidth: 0,
      background: C.elevated, borderRadius: 12,
      borderLeft: `3px solid ${C.accent}`,
      padding: '16px 18px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
        <span style={{
          fontSize: 10, fontWeight: 800, color: C.accent,
          background: 'rgba(255,107,43,0.12)', border: '1px solid rgba(255,107,43,0.3)',
          borderRadius: 4, padding: '2px 5px', letterSpacing: '0.5px',
        }}>
          {icon}
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color: C.muted, letterSpacing: '0.5px' }}>
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function SimilarityBar({ value }) {
  return (
    <div style={{ height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 2, overflow: 'hidden', flex: 1 }}>
      <div style={{
        height: '100%', borderRadius: 2,
        width: `${Math.round(value * 100)}%`,
        background: `linear-gradient(90deg, ${C.accentDk}, ${C.accent})`,
        transition: 'width 600ms ease',
      }} />
    </div>
  );
}

export default function ResultScreen({ result, spectrogram, onReset }) {
  const [expanded, setExpanded] = useState(false);

  if (!result) return null;
  const { confidence, classical, ml, artwork_url } = result;

  // Display rule: classical confidence >= 10 → trust classical; otherwise use ML top result
  const classicalConf = classical?.confidence ?? 0;
  const useClassical  = classicalConf >= 10;

  const displayTitle  = useClassical
    ? classical?.title
    : ml?.results?.[0]?.title;
  const displayArtist = useClassical
    ? result.artist
    : ml?.results?.[0]?.artist;

  const cs = confidenceStyle(confidence);
  const hasClassical = !!classical?.title;
  const hasMl        = ml?.results?.length > 0;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 28, padding: '40px 20px', maxWidth: 560, width: '100%',
      animation: 'fadeUp 400ms ease forwards',
    }}>
      {/* ── Primary result ── */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
        {artwork_url && (
          <img
            src={artwork_url}
            alt="Album art"
            style={{
              width: 120, height: 120, borderRadius: 14, objectFit: 'cover',
              boxShadow: `0 0 30px rgba(255,107,43,0.35)`,
            }}
          />
        )}
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: 28, fontWeight: 800, marginBottom: 6, lineHeight: 1.2 }}>
            {displayTitle}
          </h2>
          <p style={{ fontSize: 18, color: C.muted, fontWeight: 500 }}>{displayArtist}</p>
        </div>

        {/* Confidence badge */}
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 7,
          padding: '6px 18px', borderRadius: 9999,
          background: cs.bg, color: cs.color,
          border: `1px solid ${cs.border || 'transparent'}`,
          fontSize: 13, fontWeight: 700, letterSpacing: '0.3px',
          animation: 'scaleBounce 400ms ease forwards',
        }}>
          <span>{cs.icon}</span>
          <span>{confidence}</span>
        </div>
      </div>

      {/* ── Expand / collapse toggle ── */}
      <button
        onClick={() => setExpanded(v => !v)}
        style={{
          background: 'none', border: 'none', cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: 6,
          color: C.muted, fontSize: 13, fontWeight: 500,
          transition: 'color 300ms ease',
          padding: '4px 0',
        }}
        onMouseEnter={e => e.currentTarget.style.color = C.accent}
        onMouseLeave={e => e.currentTarget.style.color = C.muted}
      >
        <svg
          width="14" height="14" viewBox="0 0 14 14" fill="none"
          style={{ transition: 'transform 300ms ease', transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
        >
          <path d="M2 4.5l5 5 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        {expanded ? 'Hide details' : 'Show details'}
      </button>

      {/* ── Expandable details ── */}
      {expanded && (
        <div style={{
          width: '100%', display: 'flex', flexDirection: 'column', gap: 16,
          animation: 'fadeIn 250ms ease forwards',
        }}>
          {/* Path cards */}
          <div style={{ display: 'flex', gap: 14, width: '100%', alignItems: 'stretch' }}>
            <PathCard title="CLASSICAL PATH" icon="DB">
              {hasClassical ? (
                <>
                  <p style={{ fontSize: 13, color: C.text, marginBottom: 4, fontWeight: 500 }}>{classical.title}</p>
                  <p style={{ fontSize: 12, color: C.muted }}>
                    Confidence: <strong style={{ color: C.accent }}>{classical.confidence}</strong>
                  </p>
                </>
              ) : (
                <p style={{ fontSize: 13, color: C.muted }}>No match</p>
              )}
            </PathCard>

            <PathCard title="ML PATH" icon="ML">
              {hasMl ? (
                <>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {ml.results.map((r, i) => (
                      <div key={i}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: i === 0 ? C.text : C.muted, fontWeight: i === 0 ? 600 : 400 }}>
                            {r.title}
                          </span>
                          <span style={{ fontSize: 12, color: C.accent, fontWeight: 600 }}>
                            {r.similarity.toFixed(3)}
                          </span>
                        </div>
                        <SimilarityBar value={r.similarity} />
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <p style={{ fontSize: 13, color: C.muted }}>No match</p>
              )}
            </PathCard>
          </div>

          {/* Spectrogram */}
          {spectrogram && (
            <div>
              <p style={{ color: C.muted, fontSize: 12, fontWeight: 500, marginBottom: 8, letterSpacing: '0.5px' }}>
                RECORDED SPECTROGRAM
              </p>
              <div style={{ borderRadius: 12, overflow: 'hidden', border: `1px solid ${C.border}`, background: C.card }}>
                <img
                  src={`data:image/png;base64,${spectrogram}`}
                  alt="Spectrogram"
                  style={{ width: '100%', maxHeight: 280, objectFit: 'contain', display: 'block' }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      <TryAgainBtn onClick={onReset} />
    </div>
  );
}
