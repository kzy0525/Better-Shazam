import { useEffect, useRef } from 'react';

const WAVES = [
  { amp: 38,  freq: 0.0055, speed: 0.28, phase: 0.0 },
  { amp: 22,  freq: 0.0090, speed: 0.45, phase: 2.1 },
  { amp: 50,  freq: 0.0038, speed: 0.18, phase: 4.3 },
  { amp: 30,  freq: 0.0130, speed: 0.38, phase: 1.4 },
];

export default function AnimatedBackground({ isRecording }) {
  const canvasRef   = useRef(null);
  const rafRef      = useRef(null);
  const timeRef     = useRef(0);
  const recordRef   = useRef(isRecording);

  // Keep a ref in sync so the animation loop sees current value without restart
  useEffect(() => { recordRef.current = isRecording; }, [isRecording]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext('2d');

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const draw = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      const rec   = recordRef.current;
      const base  = rec ? 0.216 : 0.144;
      const speed = rec ? 1.9  : 1.0;

      timeRef.current += 0.016 * speed;

      WAVES.forEach((w, i) => {
        const yBase = height * (0.22 + i * 0.19);
        ctx.beginPath();
        for (let x = 0; x <= width; x += 3) {
          const y = yBase + Math.sin(x * w.freq + timeRef.current * w.speed + w.phase) * w.amp;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `rgba(255,107,43,${base})`;
        ctx.lineWidth   = 2;
        ctx.stroke();
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', resize);
    };
  }, []); // only mounts/unmounts once

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}
    />
  );
}
