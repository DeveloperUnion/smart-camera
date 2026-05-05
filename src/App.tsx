import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { useRecorder, MAX_DURATION_MS } from './useRecorder';
import { useVideoDetector } from './useVideoDetector';
import { activeBoxesAt } from './playbackOverlay';
import type { Detection } from './types';
import './App.css';

type Phase = 'idle' | 'recording' | 'analyzing' | 'review' | 'cart';

type Flash = { bbox: [number, number, number, number]; expiry: number };

const FLASH_MS = 700;

const DEBUG =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).get('debug') === '1';

type CartEntry = { instance_id: number; label: string };

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  // Map keyed by instance_id so the same physical object can't be added twice.
  const [cart, setCart] = useState<Map<number, CartEntry>>(new Map());
  const flashesRef = useRef<Flash[]>([]);

  // Refs for the live preview (recording) and playback (review).
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const reviewVideoRef = useRef<HTMLVideoElement>(null);
  const recordedUrlRef = useRef<string | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);

  const camera = useCamera();
  const recorder = useRecorder(camera.stream);
  const detector = useVideoDetector();

  // Latest detections via ref so the playback rAF loop can read without
  // re-binding on every state update.
  const detectionsRef = useRef<Detection[]>([]);
  detectionsRef.current = detector.detections;
  const cartRef = useRef(cart);
  cartRef.current = cart;

  const handleStart = useCallback(async () => {
    setPhase('recording');
    if (!camera.active) {
      await camera.start();
    }
  }, [camera]);

  // Once camera becomes active during recording phase, kick off the recorder.
  // The recorder.start() promise resolves when recording finishes.
  useEffect(() => {
    if (phase !== 'recording' || !camera.active || recorder.recording) return;
    let cancelled = false;
    (async () => {
      const result = await recorder.start();
      if (cancelled || !result) return;
      // Camera no longer needed — free it before playback.
      camera.stop();
      const url = URL.createObjectURL(result.blob);
      if (recordedUrlRef.current) URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = url;
      setRecordedUrl(url);
      setPhase('analyzing');
      await detector.analyze(result.blob, result.mimeType);
    })();
    return () => {
      cancelled = true;
    };
    // We intentionally only react to phase + camera.active here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, camera.active]);

  // When analysis finishes, transition to review.
  useEffect(() => {
    if (phase === 'analyzing' && detector.status === 'done') {
      setPhase('review');
      // Auto-play once we land on review.
      requestAnimationFrame(() => {
        reviewVideoRef.current?.play().catch(() => {});
      });
    }
  }, [phase, detector.status]);

  // Cleanup object URL on unmount.
  useEffect(() => {
    return () => {
      if (recordedUrlRef.current) URL.revokeObjectURL(recordedUrlRef.current);
    };
  }, []);

  // Playback overlay — draws bboxes synced to currentTime.
  useEffect(() => {
    if (phase !== 'review') return;
    const canvas = overlayRef.current;
    const video = reviewVideoRef.current;
    if (!canvas || !video) return;

    let raf = 0;
    const draw = () => {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (vw && vh) {
        if (canvas.width !== vw) canvas.width = vw;
        if (canvas.height !== vh) canvas.height = vh;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        raf = requestAnimationFrame(draw);
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const t = video.currentTime;
      const boxes = activeBoxesAt(
        detectionsRef.current,
        t,
        canvas.width,
        canvas.height,
      );

      ctx.font =
        '16px -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Yu Gothic UI", sans-serif';
      for (const b of boxes) {
        const inCart = cartRef.current.has(b.instance_id);
        ctx.lineWidth = inCart ? 3 : 2;
        ctx.strokeStyle = inCart ? '#3B82F6' : '#9CA3AF';
        if (inCart) ctx.setLineDash([]);
        else ctx.setLineDash([6, 4]);
        const [x1, y1, x2, y2] = b.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const padding = 6;
        const labelHeight = 22;
        const textWidth = ctx.measureText(b.label).width + padding * 2;
        const labelY = Math.max(0, y1 - labelHeight);
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(x1, labelY, textWidth, labelHeight);
        ctx.fillStyle = '#fff';
        ctx.fillText(b.label, x1 + padding, labelY + 16);
      }

      const now = performance.now();
      flashesRef.current = flashesRef.current.filter((f) => f.expiry > now);
      ctx.strokeStyle = '#3B82F6';
      ctx.setLineDash([]);
      ctx.lineWidth = 4;
      for (const f of flashesRef.current) {
        const [x1, y1, x2, y2] = f.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  const handleTap = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = overlayRef.current;
      const video = reviewVideoRef.current;
      if (!canvas || !video) return;
      const rect = canvas.getBoundingClientRect();

      const canvasAspect = canvas.width / canvas.height;
      const rectAspect = rect.width / rect.height;
      let scale: number;
      let offsetX = 0;
      let offsetY = 0;
      if (canvasAspect > rectAspect) {
        scale = canvas.height / rect.height;
        offsetX = (canvas.width - rect.width * scale) / 2;
      } else {
        scale = canvas.width / rect.width;
        offsetY = (canvas.height - rect.height * scale) / 2;
      }
      const x = (e.clientX - rect.left) * scale + offsetX;
      const y = (e.clientY - rect.top) * scale + offsetY;

      const boxes = activeBoxesAt(
        detectionsRef.current,
        video.currentTime,
        canvas.width,
        canvas.height,
      );

      let hit: (typeof boxes)[number] | null = null;
      let smallestArea = Infinity;
      for (const b of boxes) {
        const [x1, y1, x2, y2] = b.bbox;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
          const area = (x2 - x1) * (y2 - y1);
          if (area < smallestArea) {
            smallestArea = area;
            hit = b;
          }
        }
      }
      if (!hit) {
        // Tap outside any box: toggle play/pause.
        if (video.paused) video.play().catch(() => {});
        else video.pause();
        return;
      }

      flashesRef.current.push({
        bbox: hit.bbox,
        expiry: performance.now() + FLASH_MS,
      });
      // Already in cart? Just flash, don't double-add.
      if (cartRef.current.has(hit.instance_id)) return;
      const entry: CartEntry = {
        instance_id: hit.instance_id,
        label: hit.label,
      };
      setCart((prev) => {
        const next = new Map(prev);
        next.set(entry.instance_id, entry);
        return next;
      });
    },
    [],
  );

  const handleStopRecording = useCallback(() => {
    recorder.stop();
  }, [recorder]);

  const handleFinishReview = useCallback(() => {
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
    }
    setRecordedUrl(null);
    setPhase('cart');
  }, []);

  const handleRetryAnalyze = useCallback(async () => {
    // Re-analyze the same blob? The blob is gone after URL.createObjectURL,
    // but the URL still works. Easier path: re-fetch from the URL.
    if (!recordedUrl) {
      setPhase('idle');
      return;
    }
    try {
      const res = await fetch(recordedUrl);
      const blob = await res.blob();
      setPhase('analyzing');
      await detector.analyze(blob, blob.type || 'video/mp4');
    } catch {
      setPhase('idle');
    }
  }, [recordedUrl, detector]);

  const handleReset = useCallback(() => {
    setCart(new Map());
    detector.reset();
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
    }
    setRecordedUrl(null);
    setPhase('idle');
  }, [detector]);

  // Cart aggregation: group instances by label.
  const cartGroups = (() => {
    const groups = new Map<string, number[]>();
    for (const entry of cart.values()) {
      const arr = groups.get(entry.label) ?? [];
      arr.push(entry.instance_id);
      groups.set(entry.label, arr);
    }
    return Array.from(groups.entries());
  })();
  const cartCount = cart.size;

  const recordingProgress = Math.min(1, recorder.elapsedMs / MAX_DURATION_MS);
  const remainingS = Math.max(
    0,
    Math.ceil((MAX_DURATION_MS - recorder.elapsedMs) / 1000),
  );

  return (
    <div className="app">
      {phase === 'idle' && (
        <div className="screen idle">
          <h1>SmartCamera</h1>
          <p className="lead">
            最大 10 秒の動画を撮影 → AI が物体を検出 → 動画を見ながらタップでカゴに追加。
          </p>
          <p className="lead lead-tip">
            ヒント: カメラはゆっくり動かしてください (速いとブレで検出精度が落ちます)。
          </p>
          {camera.error && <div className="status err">{camera.error}</div>}
          <button className="primary" onClick={handleStart}>
            📹 撮影開始
          </button>
        </div>
      )}

      {phase === 'recording' && (
        <div className="screen running">
          <video
            ref={camera.videoRef}
            autoPlay
            playsInline
            muted
            className="video"
          />
          <div className="rec-indicator">
            <span className="rec-dot" /> REC {remainingS}s
          </div>
          <div
            className="rec-progress"
            style={{ transform: `scaleX(${recordingProgress})` }}
          />
          <button className="stop" onClick={handleStopRecording}>
            停止
          </button>
          {camera.error && <div className="error">{camera.error}</div>}
        </div>
      )}

      {phase === 'analyzing' && detector.status !== 'error' && (
        <div className="screen running">
          {recordedUrl && (
            <video
              src={recordedUrl}
              playsInline
              muted
              className="video"
            />
          )}
          <div className="analyzing-overlay">
            <div className="spinner" />
            <div className="analyzing-text">解析中…</div>
            <div className="analyzing-sub">
              {detector.status === 'uploading'
                ? 'アップロード中'
                : 'Gemini で物体を検出中 (5〜15 秒)'}
            </div>
          </div>
        </div>
      )}

      {phase === 'review' && recordedUrl && (
        <div className="screen running">
          <video
            ref={reviewVideoRef}
            src={recordedUrl}
            playsInline
            loop
            controls={false}
            className="video"
          />
          <canvas
            ref={overlayRef}
            className="overlay"
            onPointerDown={handleTap}
          />
          <div className="badge">🛒 {cartCount}</div>
          <button className="stop" onClick={handleFinishReview}>
            終了
          </button>
          {detector.error && (
            <div className="error">
              解析エラー: {detector.error}
              <button className="retry" onClick={handleRetryAnalyze}>
                再試行
              </button>
            </div>
          )}
          {DEBUG && (
            <div className="debug">
              <div>status: {detector.status}</div>
              <div>instances: {detector.detections.length}</div>
              <div>
                appearances:{' '}
                {detector.detections.reduce(
                  (a, d) => a + d.appearances.length,
                  0,
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {phase === 'analyzing' && detector.status === 'error' && (
        <div className="error-screen">
          <p>解析に失敗しました</p>
          <p className="lead">{detector.error}</p>
          <button className="primary" onClick={handleReset}>
            最初から
          </button>
        </div>
      )}

      {phase === 'cart' && (
        <div className="screen stopped">
          <h1>カゴの中身</h1>
          {cartGroups.length === 0 ? (
            <p className="lead">何も追加されていません。</p>
          ) : (
            <ul className="cart">
              {cartGroups.map(([label, instances]) => (
                <li key={label}>
                  <span className="label">{label}</span>
                  <span className="count">× {instances.length}</span>
                  <button
                    className="remove"
                    onClick={() =>
                      setCart((prev) => {
                        const next = new Map(prev);
                        for (const id of instances) next.delete(id);
                        return next;
                      })
                    }
                    aria-label="削除"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
          <button className="primary" onClick={handleReset}>
            最初から
          </button>
        </div>
      )}
    </div>
  );
}
