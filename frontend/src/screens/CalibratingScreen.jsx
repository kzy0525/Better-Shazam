import { C } from '../constants';

export default function CalibratingScreen() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 28 }}>
      {/* Pulsing ring */}
      <div style={{
        width: 180, height: 180, borderRadius: '50%',
        border: `3px solid ${C.accent}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        animation: 'ringPulse 1.1s ease-in-out infinite',
        boxShadow: `0 0 30px rgba(0,212,200,0.25)`,
      }}>
        <div style={{
          width: 140, height: 140, borderRadius: '50%',
          border: `1.5px solid rgba(0,212,200,0.2)`,
        }} />
      </div>

      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>Calibrating microphone...</p>
        <p style={{ color: C.muted, fontSize: 14 }}>Capturing ambient noise profile</p>
      </div>

      {/* Flat silent waveform */}
      <div style={{
        width: 300, height: 40,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div style={{
          width: '100%', height: 2,
          background: `linear-gradient(90deg, transparent, ${C.accent}, transparent)`,
          opacity: 0.5,
        }} />
      </div>
    </div>
  );
}
