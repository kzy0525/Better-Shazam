import { useEffect, useRef, useState } from 'react';
import { C, RECORD_DURATION } from '../constants';
import { encodeWAV } from '../audioUtils';

const RADIUS = 76;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function RecordingScreen({ stream, audioContext, onComplete }) {
  const [elapsed, setElapsed]   = useState(0);
  const waveCanvasRef           = useRef(null);
  const analyserRef             = useRef(null);
  const rafRef                  = useRef(null);
  const startRef                = useRef(null);
  const doneRef                 = useRef(false);

  // Audio setup + recording capture
  useEffect(() => {
    if (!stream || !audioContext) return;

    // Resume context if suspended (browser autoplay policy)
    if (audioContext.state === 'suspended') audioContext.resume();

    const source    = audioContext.createMediaStreamSource(stream);
    const analyser  = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.6;
    analyser.minDecibels = -85;
    analyser.maxDecibels = -10;
    analyserRef.current = analyser;

    const processor  = audioContext.createScriptProcessor(4096, 1, 1);
    const silentGain = audioContext.createGain();
    silentGain.gain.value = 0;

    // analyser must be in the connected graph or Chrome won't feed it data
    source.connect(analyser);
    analyser.connect(silentGain);
    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(audioContext.destination);

    const chunks = [];
    processor.onaudioprocess = (e) => {
      if (!doneRef.current) {
        chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      }
    };

    const timeout = setTimeout(() => {
      doneRef.current = true;
      processor.disconnect();
      source.disconnect();

      // Merge all chunks
      const total  = chunks.reduce((s, c) => s + c.length, 0);
      const merged = new Float32Array(total);
      let off = 0;
      for (const chunk of chunks) { merged.set(chunk, off); off += chunk.length; }

      onComplete(encodeWAV(merged, audioContext.sampleRate));
    }, RECORD_DURATION * 1000);

    return () => {
      clearTimeout(timeout);
      try { processor.disconnect(); source.disconnect(); } catch {}
    };
  }, []); // eslint-disable-line

  // Visual: countdown + waveform via single RAF loop
  useEffect(() => {
    startRef.current = performance.now();

    const tick = (now) => {
      const secs = (now - startRef.current) / 1000;
      setElapsed(Math.min(secs, RECORD_DURATION));

      // Draw waveform
      const canvas = waveCanvasRef.current;
      const analyser = analyserRef.current;
      if (canvas && analyser) {
        const ctx = canvas.getContext('2d');
        const { width, height } = canvas;
        ctx.clearRect(0, 0, width, height);

        // Frequency domain — far more reactive to music than time domain
        const data = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(data);

        const barCount = 60;
        const barW     = width / barCount;
        // Use the lower 70% of frequency bins — that's where musical content lives
        const binRange = Math.floor(data.length * 0.7);
        for (let i = 0; i < barCount; i++) {
          const bin    = Math.floor(i * binRange / barCount);
          const value  = data[bin] / 255; // 0..1
          const barH   = Math.max(2, value * height * 0.95);
          const x      = i * barW;
          const y      = (height - barH) / 2;

          const grad = ctx.createLinearGradient(0, y, 0, y + barH);
          grad.addColorStop(0, C.accent);
          grad.addColorStop(1, C.accentDk);
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.roundRect(x + 1, y, barW - 2, barH, 2);
          ctx.fill();
        }
      }

      if (secs < RECORD_DURATION) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []); // eslint-disable-line

  const progress    = Math.max(0, 1 - elapsed / RECORD_DURATION);
  const dashOffset  = CIRCUMFERENCE * (1 - progress);
  const countdown   = Math.max(0, Math.ceil(RECORD_DURATION - elapsed));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 32 }}>
      {/* Countdown ring */}
      <div style={{ position: 'relative', width: 180, height: 180 }}>
        <svg width="180" height="180" style={{ transform: 'rotate(-90deg)' }}>
          {/* Track */}
          <circle cx="90" cy="90" r={RADIUS} fill="none"
            stroke="rgba(255,107,43,0.12)" strokeWidth="4" />
          {/* Progress */}
          <circle cx="90" cy="90" r={RADIUS} fill="none"
            stroke={C.accent} strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={dashOffset}
            style={{
              filter: 'drop-shadow(0 0 6px rgba(255,107,43,0.7))',
              transition: 'stroke-dashoffset 0.1s linear',
            }}
          />
        </svg>
        {/* Countdown number */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{ fontSize: 52, fontWeight: 800, color: C.text, lineHeight: 1 }}>
            {countdown}
          </span>
        </div>
      </div>

      {/* Label */}
      <p style={{ fontSize: 18, fontWeight: 600, color: C.text, letterSpacing: '0.5px' }}>
        Listening...
      </p>

      {/* Live waveform canvas */}
      <canvas
        ref={waveCanvasRef}
        width={400}
        height={80}
        style={{
          borderRadius: 8,
          background: 'rgba(255,107,43,0.04)',
          border: `1px solid rgba(255,107,43,0.12)`,
        }}
      />
    </div>
  );
}
