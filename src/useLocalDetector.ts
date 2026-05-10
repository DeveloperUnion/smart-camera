import { useEffect, useRef, useState } from 'react';
import { detect, loadModel } from './yolo11';
import type { LiveBox } from './types';

const DEFAULT_INTERVAL_MS = 333; // 3 fps default
// iOS WebKit kills tabs that sustain ~30%+ CPU for several seconds, so back
// off the inference cadence on iPhone/iPad where WASM is the only path.
const MOBILE_WEBKIT_INTERVAL_MS = 666; // ~1.5 fps

function isIOSWebKit(): boolean {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  return (
    /iPad|iPhone|iPod/.test(ua) ||
    (ua.includes('Mac') &&
      typeof document !== 'undefined' &&
      'ontouchend' in document)
  );
}

function readIntervalOverride(): number {
  const base = isIOSWebKit() ? MOBILE_WEBKIT_INTERVAL_MS : DEFAULT_INTERVAL_MS;
  if (typeof window === 'undefined') return base;
  const fpsParam = new URLSearchParams(window.location.search).get('fps');
  const fps = fpsParam ? Number(fpsParam) : NaN;
  if (Number.isFinite(fps) && fps > 0) return Math.round(1000 / fps);
  return base;
}

export function useLocalDetector(opts: {
  videoEl: HTMLVideoElement | null;
  enabled: boolean;
}) {
  // boxes is exposed as a ref (not state) so that 3fps inference doesn't
  // re-render the entire App tree — the overlay rAF loop reads the ref directly.
  const boxesRef = useRef<LiveBox[]>([]);
  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState<'wasm' | null>(null);
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
      // Pause inference while the tab is hidden — sustained WASM/WebGPU work
      // in the background is what triggers iOS WebKit's memory-pressure tab kill.
      if (typeof document !== 'undefined' && document.hidden) {
        if (!stopped) requestAnimationFrame(loop);
        return;
      }
      if (!inFlight && ts - lastRun >= intervalMs) {
        inFlight = true;
        lastRun = ts;
        try {
          const scratch = scratchRef.current;
          if (scratch && video.videoWidth) {
            const dets = await detect(video, scratch);
            if (!stopped) boxesRef.current = dets;
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
      boxesRef.current = [];
    };
  }, [opts.enabled, ready, opts.videoEl]);

  return { boxesRef, ready, backend, error, stats };
}
