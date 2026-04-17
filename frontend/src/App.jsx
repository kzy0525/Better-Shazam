import { useRef, useState } from 'react';
import AnimatedBackground from './components/AnimatedBackground';
import IdleScreen       from './screens/IdleScreen';
import RecordingScreen  from './screens/RecordingScreen';
import ProcessingScreen from './screens/ProcessingScreen';
import ResultScreen     from './screens/ResultScreen';
import NoMatchScreen    from './screens/NoMatchScreen';
import ErrorScreen      from './screens/ErrorScreen';
import { API_URL }      from './constants';
import { parseSSEStream } from './audioUtils';

export default function App() {
  const [screen,      setScreen]      = useState('idle');
  const [steps,       setSteps]       = useState({});
  const [spectrogram, setSpectrogram] = useState(null);
  const [result,      setResult]      = useState(null);
  const [errorMsg,    setErrorMsg]    = useState('');

  const streamRef  = useRef(null);
  const ctxRef     = useRef(null);

  // ── Reset everything back to idle ────────────────────────────────────────
  function reset() {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    try { ctxRef.current?.close(); } catch {}
    ctxRef.current = null;
    setSteps({});
    setSpectrogram(null);
    setResult(null);
    setErrorMsg('');
    setScreen('idle');
  }

  // ── Start: request mic + 1.5s calibration ────────────────────────────────
  async function handleRecordClick() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      streamRef.current = stream;
      ctxRef.current = new AudioContext();
      setScreen('recording');
    } catch (e) {
      const name = e?.name || '';
      setErrorMsg(
        name === 'NotAllowedError' || name === 'PermissionDeniedError'
          ? 'microphone_denied'
          : e?.message || 'Failed to access microphone'
      );
      setScreen('error');
    }
  }

  // ── Recording done: send WAV to backend via SSE ──────────────────────────
  async function handleRecordingComplete(wavBlob) {
    setScreen('processing');
    setSteps({});

    // Stop mic tracks — recording is done
    streamRef.current?.getTracks().forEach(t => t.stop());

    const updateStep = (key, status) => {
      setSteps(prev => {
        const next = { ...prev, [key]: status };
        // Auto-activate "combining" when both parallel paths finish
        if ((key === 'classical' || key === 'ml') && status === 'complete') {
          if (next.classical === 'complete' && next.ml === 'complete') {
            next.combining = 'active';
          }
        }
        return next;
      });
    };

    try {
      const body = new FormData();
      body.append('file', wavBlob, 'recording.wav');

      const response = await fetch(`${API_URL}/identify`, { method: 'POST', body });

      if (!response.ok) throw new Error(`Server error ${response.status}`);

      await parseSSEStream(response, (event) => {
        const { step, status, data } = event;

        if (step === 'error') {
          setErrorMsg(data?.message || 'Backend error');
          setScreen('error');
          return;
        }

        if (step === 'spectrogram' && status === 'complete') {
          setSpectrogram(data?.image ?? null);
          return;
        }

        if (step === 'result' && status === 'complete') {
          updateStep('combining', 'complete');
          if (data?.title) {
            setResult(data);
            setTimeout(() => setScreen('result'), 700);
          } else {
            setTimeout(() => setScreen('noMatch'), 700);
          }
          return;
        }

        updateStep(step, status === 'complete' ? 'complete' : 'active');
      });

    } catch (e) {
      const msg = e?.message || '';
      setErrorMsg(
        msg.toLowerCase().includes('fetch') || msg.includes('Failed to fetch')
          ? 'Failed to fetch'
          : msg
      );
      setScreen('error');
    }
  }

  const isRecording = screen === 'recording';

  return (
    <div style={{
      position: 'relative', minHeight: '100vh',
      background: '#0a0a0a',
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
      overflow: 'hidden',
    }}>
      <AnimatedBackground isRecording={isRecording} />

      {/* Content layer */}
      <div style={{
        position: 'relative', zIndex: 1,
        minHeight: '100vh',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
      }}>
        {screen === 'idle' && (
          <IdleScreen onRecord={handleRecordClick} />
        )}
        {screen === 'recording' && (
          <RecordingScreen
            stream={streamRef.current}
            audioContext={ctxRef.current}
            onComplete={handleRecordingComplete}
          />
        )}
        {screen === 'processing' && (
          <ProcessingScreen steps={steps} spectrogram={spectrogram} />
        )}
        {screen === 'result' && (
          <ResultScreen result={result} spectrogram={spectrogram} onReset={reset} />
        )}
        {screen === 'noMatch' && (
          <NoMatchScreen onReset={reset} />
        )}
        {screen === 'error' && (
          <ErrorScreen message={errorMsg} onReset={reset} />
        )}
      </div>
    </div>
  );
}
