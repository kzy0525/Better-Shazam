/**
 * Encode a Float32Array of mono PCM samples into a WAV Blob.
 */
export function encodeWAV(samples, sampleRate) {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  function str(offset, s) {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  }

  str(0, 'RIFF');
  view.setUint32(4,  36 + numSamples * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16,           true); // chunk size
  view.setUint16(20,  1,           true); // PCM
  view.setUint16(22,  1,           true); // mono
  view.setUint32(24, sampleRate,   true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32,  2,           true); // block align
  view.setUint16(34, 16,           true); // bits per sample
  str(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let off = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Read an SSE stream from a fetch Response and call onEvent for each parsed event.
 */
export async function parseSSEStream(response, onEvent) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() ?? '';
    for (const line of lines) {
      const t = line.trim();
      if (t.startsWith('data: ')) {
        try { onEvent(JSON.parse(t.slice(6))); } catch { /* skip malformed */ }
      }
    }
  }
}
