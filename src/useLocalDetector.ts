import { useEffect, useRef, useState } from 'react';
import { detect, loadModel } from './dfine';
import type { LiveBox } from './types';

const DEFAULT_INTERVAL_MS = 333; // 3 fps default

function readIntervalOverride(): number {
  if (typeof window === 'undefined') return DEFAULT_INTERVAL_MS;
  const fpsParam = new URLSearchParams(window.location.search).get('fps');
  const fps = fpsParam ? Number(fpsParam) : NaN;
  if (Number.isFinite(fps) && fps > 0) return Math.round(1000 / fps);
  return DEFAULT_INTERVAL_MS;
}

export function useLocalDetector(opts: {
  videoEl: HTMLVideoElement | null;
  enabled: boolean;
}) {
  const [boxes, setBoxes] = useState<LiveBox[]>([]);
  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState<'webgpu' | 'wasm' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ inferences: 0, lastError: '' });
  const scratchRef = useRef<HTMLCanvasElement | null>(null);
  const inferenceCountRef = useRef(0);
  const lastErrorRef = useRef('');

  if (!scratchRef.current && typeof document !== 'undefined') {
    scratchRef.current = document.createElement('canvas');
  }

  // Load model lazily — only when first enabled, so users in cloud mode
  // never download the ONNX file.
  useEffect(() => {
    if (!opts.enabled || ready || error) return;
    let cancelled = false;
    loadModel()
      .then((r) => {
        if (cancelled) return;
        setBackend(r.backend);
        setReady(true);
      })
      .catch((e) => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : 'モデル読み込み失敗';
        setError(msg);
      });
    return () => {
      cancelled = true;
    };
  }, [opts.enabled, ready, error]);

  useEffect(() => {
    if (!opts.enabled || !ready || !opts.videoEl) return;
    const video = opts.videoEl;
    let stopped = false;
    let inFlight = false;
    let lastRun = 0;

    const statsTimer = setInterval(() => {
      if (stopped) return;
      setStats({
        inferences: inferenceCountRef.current,
        lastError: lastErrorRef.current,
      });
    }, 1000);

    const intervalMs = readIntervalOverride();

    const loop = async (ts: number) => {
      if (stopped) return;
      if (!inFlight && ts - lastRun >= intervalMs) {
        inFlight = true;
        lastRun = ts;
        try {
          const scratch = scratchRef.current;
          if (scratch && video.videoWidth) {
            const dets = await detect(video, scratch);
            if (!stopped) setBoxes(dets);
            inferenceCountRef.current++;
          }
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          lastErrorRef.current = msg.slice(0, 120);
          console.error('Local detect error', e);
        } finally {
          inFlight = false;
        }
      }
      if (!stopped) requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    return () => {
      stopped = true;
      clearInterval(statsTimer);
      setBoxes([]);
    };
  }, [opts.enabled, ready, opts.videoEl]);

  return { boxes, ready, backend, error, stats };
}
