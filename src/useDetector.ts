import { useEffect, useRef, useState, type RefObject } from 'react';
import { detect, loadModel } from './yolo';
import type { Detection } from './types';

const TARGET_INTERVAL_MS = 333; // 3 fps (battery + memory)

export function useDetector(opts: {
  videoRef: RefObject<HTMLVideoElement | null>;
  enabled: boolean;
}) {
  const [detections, setDetections] = useState<Detection[]>([]);
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

  useEffect(() => {
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
  }, []);

  useEffect(() => {
    if (!opts.enabled || !ready) return;
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

    const loop = async (ts: number) => {
      if (stopped) return;
      if (!inFlight && ts - lastRun >= TARGET_INTERVAL_MS) {
        inFlight = true;
        lastRun = ts;
        try {
          const video = opts.videoRef.current;
          const scratch = scratchRef.current;
          if (video && scratch && video.videoWidth) {
            const dets = await detect(video, scratch);
            if (!stopped) setDetections(dets);
            inferenceCountRef.current++;
          }
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          lastErrorRef.current = msg.slice(0, 120);
          console.error('Detection error', e);
        } finally {
          inFlight = false;
        }
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);

    return () => {
      stopped = true;
      clearInterval(statsTimer);
      setDetections([]);
    };
  }, [opts.enabled, opts.videoRef, ready]);

  return { detections, ready, backend, error, stats };
}
