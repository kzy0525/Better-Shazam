import { useState } from 'react';
import { C } from '../constants';

function confidenceStyle(label) {
  if (!label) return { bg: 'rgba(136,136,136,0.2)', color: C.muted, icon: '–' };
  const l = label.toLowerCase();
  if (l.includes('high') || l.includes('confident'))
    return { bg: 'rgba(0,200,83,0.18)', color: C.success, border: 'rgba(0,200,83,0.4)', icon: '✓' };
  if (l.includes('moderate'))
    return { bg: 'rgba(255,171,0,0.18)', color: C.warning, border: 'rgba(255,171,0,0.4)', icon: '⚠' };
  return { bg: 'rgba(255,61,0,0.18)',  color: C.error,   border: 'rgba(255,61,0,0.4)',   icon: '✕' };
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
        <span style={{ fontSize: 16 }}>{icon}</span>
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
  if (!result) return null;
  const { title, artist, confidence, classical, ml, artwork_url } = result;
  const cs = confidenceStyle(confidence);

  const hasClassical = classical?.title;
  const hasMl        = ml?.results?.length > 0;

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      gap: 28, padding: '40px 20px', maxWidth: 560, width: '100%',
      animation: 'fadeUp 400ms ease forwards',
    }}>
      {/* ── Top: art + title ── */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
        {artwork_url && (
          <img
            src={artwork_url}
            alt="Album art"
            style={{
              width: 120, height: 120, borderRadius: 14, objectFit: 'cover',
              boxShadow: `0 0 30px rgba(0,212,200,0.35)`,
            }}
          />
        )}
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: 28, fontWeight: 800, marginBottom: 6, lineHeight: 1.2 }}>{title}</h2>
          <p style={{ fontSize: 18, color: C.muted, fontWeight: 500 }}>{artist}</p>
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

      {/* ── Path cards ── */}
      <div style={{ display: 'flex', gap: 14, width: '100%', alignItems: 'stretch' }}>
        {/* Classical */}
        <PathCard title="CLASSICAL PATH" icon="🗄️">
          {hasClassical ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                <span style={{ color: C.success, fontSize: 13, fontWeight: 600 }}>✓ Match found</span>
              </div>
              <p style={{ fontSize: 13, color: C.text, marginBottom: 4, fontWeight: 500 }}>{classical.title}</p>
              <p style={{ fontSize: 12, color: C.muted }}>Confidence score: <strong style={{ color: C.accent }}>{classical.confidence}</strong></p>
            </>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: C.error, fontSize: 13, fontWeight: 600 }}>✕ No match</span>
            </div>
          )}
        </PathCard>

        {/* ML */}
        <PathCard title="ML PATH" icon="🧠">
          {hasMl ? (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                <span style={{ color: C.success, fontSize: 13, fontWeight: 600 }}>✓ Match found</span>
              </div>
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
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ color: C.error, fontSize: 13, fontWeight: 600 }}>✕ No match</span>
            </div>
          )}
        </PathCard>
      </div>

      {/* Spectrogram (if not shown on processing screen) */}
      {spectrogram && (
        <div style={{ width: '100%' }}>
          <p style={{ color: C.muted, fontSize: 12, fontWeight: 500, marginBottom: 8, letterSpacing: '0.5px' }}>
            RECORDED SPECTROGRAM
          </p>
          <div style={{ borderRadius: 12, overflow: 'hidden', border: `1px solid ${C.border}`, background: C.card }}>
            <img
              src={`data:image/png;base64,${spectrogram}`}
              alt="Spectrogram"
              style={{ width: '100%', maxHeight: 160, objectFit: 'cover', display: 'block' }}
            />
          </div>
        </div>
      )}

      <TryAgainBtn onClick={onReset} />
    </div>
  );
}
