import { useEffect, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { useDetector } from './useDetector';
import { labelOf } from './coco-labels';
import type { Detection } from './types';
import './App.css';

type Phase = 'idle' | 'running' | 'stopped';

type Flash = { bbox: [number, number, number, number]; expiry: number };

const FLASH_MS = 700;

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [cart, setCart] = useState<Map<number, number>>(new Map());
  const flashesRef = useRef<Flash[]>([]);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  const camera = useCamera();
  const detector = useDetector({
    videoRef: camera.videoRef,
    enabled: phase === 'running',
  });

  useEffect(() => {
    if (phase !== 'running') return;
    const canvas = overlayRef.current;
    const video = camera.videoRef.current;
    if (!canvas || !video) return;

    let raf = 0;
    const draw = () => {
      if (video.videoWidth) {
        if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
        if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        raf = requestAnimationFrame(draw);
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2;
      ctx.strokeStyle = '#9CA3AF';
      ctx.setLineDash([6, 4]);
      ctx.font = '16px system-ui, -apple-system, sans-serif';

      for (const d of detector.detections) {
        const [x1, y1, x2, y2] = d.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = labelOf(d.classId);
        const padding = 6;
        const labelHeight = 22;
        const textWidth = ctx.measureText(text).width + padding * 2;
        const labelY = Math.max(0, y1 - labelHeight);
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(x1, labelY, textWidth, labelHeight);
        ctx.fillStyle = '#fff';
        ctx.fillText(text, x1 + padding, labelY + 16);
      }

      const now = performance.now();
      flashesRef.current = flashesRef.current.filter((f) => f.expiry > now);
      ctx.strokeStyle = '#3B82F6';
      ctx.setLineDash([]);
      ctx.lineWidth = 3;
      for (const f of flashesRef.current) {
        const [x1, y1, x2, y2] = f.bbox;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [phase, detector.detections, camera.videoRef]);

  const handleTap = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    // Account for object-fit: cover scaling.
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

    let hit: Detection | null = null;
    let smallestArea = Infinity;
    for (const d of detector.detections) {
      const [x1, y1, x2, y2] = d.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        const area = (x2 - x1) * (y2 - y1);
        if (area < smallestArea) {
          smallestArea = area;
          hit = d;
        }
      }
    }

    if (hit) {
      flashesRef.current.push({
        bbox: hit.bbox,
        expiry: performance.now() + FLASH_MS,
      });
      const classId = hit.classId;
      setCart((prev) => {
        const next = new Map(prev);
        next.set(classId, (next.get(classId) ?? 0) + 1);
        return next;
      });
    }
  };

  const handleStart = () => {
    setPhase('running');
  };

  useEffect(() => {
    if (phase === 'running' && !camera.active) {
      camera.start();
    }
  }, [phase, camera.active, camera.start]);

  const handleStop = () => {
    camera.stop();
    setPhase('stopped');
  };

  const handleReset = () => {
    setCart(new Map());
    setPhase('idle');
  };

  const cartCount = Array.from(cart.values()).reduce((a, b) => a + b, 0);

  return (
    <div className="app">
      {phase === 'idle' && (
        <div className="screen idle">
          <h1>SmartCamera</h1>
          <p className="lead">
            カメラを起動して、写った物をタップでカゴに追加します。
          </p>
          <div className="status">
            {!detector.ready && !detector.error && (
              <span>モデル読み込み中…</span>
            )}
            {detector.ready && (
              <span>
                準備完了 <small>({detector.backend})</small>
              </span>
            )}
            {detector.error && (
              <span className="err">エラー: {detector.error}</span>
            )}
          </div>
          <button
            className="primary"
            onClick={handleStart}
            disabled={!detector.ready}
          >
            カメラ開始
          </button>
        </div>
      )}

      {phase === 'running' && (
        <div className="screen running">
          <video
            ref={camera.videoRef}
            autoPlay
            playsInline
            muted
            className="video"
          />
          <canvas
            ref={overlayRef}
            className="overlay"
            onPointerDown={handleTap}
          />
          <div className="badge">🛒 {cartCount}</div>
          <button className="stop" onClick={handleStop}>
            停止
          </button>
          {camera.error && <div className="error">{camera.error}</div>}
        </div>
      )}

      {phase === 'stopped' && (
        <div className="screen stopped">
          <h1>カゴの中身</h1>
          {cart.size === 0 ? (
            <p className="lead">何も追加されていません。</p>
          ) : (
            <ul className="cart">
              {Array.from(cart.entries()).map(([classId, count]) => (
                <li key={classId}>
                  <span className="label">{labelOf(classId)}</span>
                  <span className="count">× {count}</span>
                  <button
                    className="remove"
                    onClick={() =>
                      setCart((prev) => {
                        const next = new Map(prev);
                        next.delete(classId);
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
