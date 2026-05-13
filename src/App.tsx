import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { useRecorder, MAX_DURATION_MS } from './useRecorder';
import { useVideoDetector } from './useVideoDetector';
import { useLocalDetector } from './useLocalDetector';
import { activeBoxesAt } from './playbackOverlay';
import type { Detection, TrackedBox } from './types';
import './App.css';

type Mode = 'cloud' | 'local';
type Phase =
  | 'idle'
  | 'preview' // cloud: live camera awaiting shutter
  | 'recording'
  | 'analyzing'
  | 'review'
  | 'live' // local: live camera + on-device inference + tap-to-add
  | 'cart';

type Flash = { bbox: [number, number, number, number]; expiry: number };

const FLASH_MS = 700;

const DEBUG =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).get('debug') === '1';

// The video/overlay use object-fit: cover, so on phones the canvas extends past
// the visible viewport. Compute the visible canvas region (in canvas coords)
// so labels and boxes can be clamped to what the user actually sees.
function visibleCanvasRegion(canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return { x1: 0, y1: 0, x2: canvas.width, y2: canvas.height };
  }
  const canvasAspect = canvas.width / canvas.height;
  const rectAspect = rect.width / rect.height;
  if (canvasAspect > rectAspect) {
    const scale = canvas.height / rect.height;
    const cropX = (canvas.width - rect.width * scale) / 2;
    return { x1: cropX, y1: 0, x2: canvas.width - cropX, y2: canvas.height };
  }
  const scale = canvas.width / rect.width;
  const cropY = (canvas.height - rect.height * scale) / 2;
  return { x1: 0, y1: cropY, x2: canvas.width, y2: canvas.height - cropY };
}

type CartEntry = { instance_id: number; label: string };

export default function App() {
  const [mode, setMode] = useState<Mode>('cloud');
  const [phase, setPhase] = useState<Phase>('idle');
  // Cart keyed by instance_id — cloud uses Gemini's stable id, local uses the
  // sticky id assigned by LocalTracker. Either way, tapping the same physical
  // object twice is deduped, while distinct instances of the same label
  // aggregate as count.
  const [cart, setCart] = useState<Map<number, CartEntry>>(new Map());
  const flashesRef = useRef<Flash[]>([]);

  // Refs for the live preview/recording (camera) and playback (review).
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const reviewVideoRef = useRef<HTMLVideoElement>(null);
  const progressFillRef = useRef<HTMLDivElement>(null);
  const recordedUrlRef = useRef<string | null>(null);
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [isVideoPaused, setIsVideoPaused] = useState(true);

  const camera = useCamera();
  const recorder = useRecorder(camera.stream);
  const detector = useVideoDetector();
  const localDetector = useLocalDetector({
    videoEl: camera.videoEl,
    enabled: mode === 'local' && phase === 'live',
  });

  // Latest results via ref so the playback rAF loop can read without
  // re-binding on every state update.
  const detectionsRef = useRef<Detection[]>([]);
  detectionsRef.current = detector.detections;
  const localBoxesRef = localDetector.boxesRef;
  const cartRef = useRef(cart);
  cartRef.current = cart;

  const handleSetMode = useCallback((m: Mode) => {
    setMode(m);
  }, []);

  const handleStart = useCallback(async () => {
    if (mode === 'cloud') setPhase('preview');
    else setPhase('live');
    if (!camera.active) {
      await camera.start();
    }
  }, [mode, camera]);

  const handleShutter = useCallback(() => {
    setPhase('recording');
  }, []);

  const handleCancelPreview = useCallback(() => {
    camera.stop();
    setPhase('idle');
  }, [camera]);

  // Cloud: drive the MediaRecorder + analysis pipeline from 'recording'.
  useEffect(() => {
    if (phase !== 'recording' || recorder.recording) return;
    let cancelled = false;
    (async () => {
      const result = await recorder.start();
      if (cancelled || !result) return;
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  useEffect(() => {
    if (phase === 'analyzing' && detector.status === 'done') {
      setIsVideoPaused(true);
      setPhase('review');
    }
  }, [phase, detector.status]);

  useEffect(() => {
    return () => {
      if (recordedUrlRef.current) URL.revokeObjectURL(recordedUrlRef.current);
    };
  }, []);

  // Cloud: bbox overlay synced to the recorded video's currentTime.
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
      const visible = visibleCanvasRegion(canvas);
      for (const b of boxes) {
        const inCart = cartRef.current.has(b.instance_id);
        ctx.lineWidth = inCart ? 3 : 2;
        ctx.strokeStyle = inCart ? '#3B82F6' : '#9CA3AF';
        if (inCart) ctx.setLineDash([]);
        else ctx.setLineDash([6, 4]);
        const [rx1, ry1, rx2, ry2] = b.bbox;
        if (rx2 <= visible.x1 || rx1 >= visible.x2 || ry2 <= visible.y1 || ry1 >= visible.y2) {
          continue;
        }
        const x1 = Math.max(visible.x1, Math.min(visible.x2, rx1));
        const y1 = Math.max(visible.y1, Math.min(visible.y2, ry1));
        const x2 = Math.max(visible.x1, Math.min(visible.x2, rx2));
        const y2 = Math.max(visible.y1, Math.min(visible.y2, ry2));
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const padding = 6;
        const labelHeight = 22;
        const visibleW = visible.x2 - visible.x1;
        const textWidth = Math.min(
          visibleW,
          ctx.measureText(b.label).width + padding * 2,
        );
        const labelX = Math.max(visible.x1, Math.min(x1, visible.x2 - textWidth));
        const labelY =
          y1 - labelHeight >= visible.y1
            ? y1 - labelHeight
            : Math.min(y1, visible.y2 - labelHeight);
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(labelX, labelY, textWidth, labelHeight);
        ctx.fillStyle = '#fff';
        ctx.fillText(b.label, labelX + padding, labelY + 16);
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

      const fill = progressFillRef.current;
      if (fill && video.duration > 0) {
        fill.style.transform = `scaleX(${Math.min(1, t / video.duration)})`;
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  // Local: bbox overlay synced to the live camera video.
  useEffect(() => {
    if (phase !== 'live') return;
    const canvas = overlayRef.current;
    const video = camera.videoEl;
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

      ctx.font =
        '16px -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Yu Gothic UI", sans-serif';

      const visibleLive = visibleCanvasRegion(canvas);
      for (const b of localBoxesRef.current) {
        const inCart = cartRef.current.has(b.instance_id);
        ctx.lineWidth = inCart ? 3 : 2;
        ctx.strokeStyle = inCart ? '#3B82F6' : '#9CA3AF';
        if (inCart) ctx.setLineDash([]);
        else ctx.setLineDash([6, 4]);
        const [rx1, ry1, rx2, ry2] = b.bbox;
        if (
          rx2 <= visibleLive.x1 ||
          rx1 >= visibleLive.x2 ||
          ry2 <= visibleLive.y1 ||
          ry1 >= visibleLive.y2
        ) {
          continue;
        }
        const x1 = Math.max(visibleLive.x1, Math.min(visibleLive.x2, rx1));
        const y1 = Math.max(visibleLive.y1, Math.min(visibleLive.y2, ry1));
        const x2 = Math.max(visibleLive.x1, Math.min(visibleLive.x2, rx2));
        const y2 = Math.max(visibleLive.y1, Math.min(visibleLive.y2, ry2));
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const padding = 6;
        const labelHeight = 22;
        const visibleW = visibleLive.x2 - visibleLive.x1;
        const textWidth = Math.min(
          visibleW,
          ctx.measureText(b.label).width + padding * 2,
        );
        const labelX = Math.max(
          visibleLive.x1,
          Math.min(x1, visibleLive.x2 - textWidth),
        );
        const labelY =
          y1 - labelHeight >= visibleLive.y1
            ? y1 - labelHeight
            : Math.min(y1, visibleLive.y2 - labelHeight);
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(labelX, labelY, textWidth, labelHeight);
        ctx.fillStyle = '#fff';
        ctx.fillText(b.label, labelX + padding, labelY + 16);
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
  }, [phase, camera.videoEl]);

  const handleTapReview = useCallback(
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
        if (video.paused) video.play().catch(() => {});
        else video.pause();
        return;
      }

      flashesRef.current.push({
        bbox: hit.bbox,
        expiry: performance.now() + FLASH_MS,
      });
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

  const handleTapLive = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = overlayRef.current;
      if (!canvas) return;
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

      let hit: TrackedBox | null = null;
      let smallestArea = Infinity;
      for (const b of localBoxesRef.current) {
        const [x1, y1, x2, y2] = b.bbox;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
          const area = (x2 - x1) * (y2 - y1);
          if (area < smallestArea) {
            smallestArea = area;
            hit = b;
          }
        }
      }
      if (!hit) return;

      flashesRef.current.push({
        bbox: hit.bbox,
        expiry: performance.now() + FLASH_MS,
      });
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

  const handleTogglePlay = useCallback(() => {
    const video = reviewVideoRef.current;
    if (!video) return;
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  }, []);

  const handleRestart = useCallback(() => {
    const video = reviewVideoRef.current;
    if (!video) return;
    video.currentTime = 0;
    video.play().catch(() => {});
  }, []);

  const handleFinishReview = useCallback(() => {
    if (recordedUrlRef.current) {
      URL.revokeObjectURL(recordedUrlRef.current);
      recordedUrlRef.current = null;
    }
    setRecordedUrl(null);
    setPhase('cart');
  }, []);

  const handleStopLive = useCallback(() => {
    camera.stop();
    setPhase('cart');
  }, [camera]);

  const handleRetryAnalyze = useCallback(async () => {
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
          <img src="/dustalk-logo.png" alt="Dustalk" className="dustalk-logo" />
          <h1>SmartCamera</h1>
          <p className="lead">
            {mode === 'cloud'
              ? '最大 10 秒の動画を撮影 → AI が物体を検出 → 動画を見ながらタップでカゴに追加。'
              : 'カメラを起動 → 写った物体に枠が出る → タップでカゴに追加。'}
          </p>

          <div className="mode-toggle" role="tablist" aria-label="モード">
            <button
              role="tab"
              className={`mode-btn ${mode === 'cloud' ? 'on' : ''}`}
              onClick={() => handleSetMode('cloud')}
              aria-selected={mode === 'cloud'}
            >
              クラウド
            </button>
            <button
              role="tab"
              className={`mode-btn ${mode === 'local' ? 'on' : ''}`}
              onClick={() => handleSetMode('local')}
              aria-selected={mode === 'local'}
            >
              ローカル
            </button>
          </div>

          {mode === 'cloud' && (
            <p className="lead lead-tip">
              ヒント: カメラはゆっくり動かしてください (速いとブレで検出精度が落ちます)。
            </p>
          )}
          {mode === 'local' && localDetector.error && (
            <div className="status err">{localDetector.error}</div>
          )}
          {camera.error && <div className="status err">{camera.error}</div>}

          <button className="primary" onClick={handleStart}>
            {mode === 'cloud' ? '撮影開始' : 'カメラ開始'}
          </button>
        </div>
      )}

      {phase === 'preview' && (
        <div className="screen running">
          <video
            ref={camera.videoRef}
            autoPlay
            playsInline
            muted
            className="video"
          />
          <button
            className="back-btn"
            onClick={handleCancelPreview}
            aria-label="戻る"
          >
            ×
          </button>
          {!camera.active && !camera.error && (
            <div className="preview-tip">カメラ起動中…</div>
          )}
          {camera.active && (
            <div className="preview-tip">
              準備ができたら録画ボタンを押してください
              <br />
              <span className="preview-tip-sub">
                最大 10 秒。カメラはゆっくり動かしてください
              </span>
            </div>
          )}
          <button
            className="shutter"
            onClick={handleShutter}
            aria-label="録画開始"
            disabled={!camera.active}
          >
            <span className="shutter-inner" />
          </button>
          {camera.error && <div className="error">{camera.error}</div>}
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
          <button
            className="shutter recording"
            onClick={handleStopRecording}
            aria-label="録画停止"
          >
            <span className="shutter-inner" />
          </button>
          {camera.error && <div className="error">{camera.error}</div>}
        </div>
      )}

      {phase === 'analyzing' && detector.status !== 'error' && (
        <div className="screen running">
          {recordedUrl && (
            <video src={recordedUrl} playsInline muted className="video" />
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
            controls={false}
            className="video"
            onPlay={() => setIsVideoPaused(false)}
            onPause={() => setIsVideoPaused(true)}
            onEnded={() => setIsVideoPaused(true)}
          />
          <canvas
            ref={overlayRef}
            className="overlay"
            onPointerDown={handleTapReview}
          />
          {isVideoPaused && (
            <div className="play-icon" aria-hidden="true">
              ▶
            </div>
          )}
          <div className="player-controls">
            <button
              className="player-btn"
              onClick={handleTogglePlay}
              aria-label={isVideoPaused ? '再生' : '一時停止'}
            >
              {isVideoPaused ? '▶' : '❚❚'}
            </button>
            <div className="player-progress">
              <div ref={progressFillRef} className="player-progress-fill" />
            </div>
            <button
              className="player-btn"
              onClick={handleRestart}
              aria-label="最初から"
            >
              ↻
            </button>
          </div>
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
              <div>mode: cloud</div>
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

      {phase === 'live' && (
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
            onPointerDown={handleTapLive}
          />
          <div className="live-cart-panel">
            {cartGroups.length === 0 ? (
              <div className="live-cart-empty">
                枠をタップしてカゴに追加
              </div>
            ) : (
              cartGroups.map(([label, instances]) => (
                <div className="live-cart-chip" key={label}>
                  <span className="live-cart-chip-label">{label}</span>
                  <span className="live-cart-chip-count">
                    ×{instances.length}
                  </span>
                  <button
                    className="live-cart-chip-remove"
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
                </div>
              ))
            )}
          </div>
          <button className="stop" onClick={handleStopLive}>
            停止
          </button>
          {!localDetector.ready && !localDetector.error && (
            <div className="preview-tip">モデル読み込み中…</div>
          )}
          {localDetector.error && (
            <div className="error">エラー: {localDetector.error}</div>
          )}
          {camera.error && <div className="error">{camera.error}</div>}
          {DEBUG && (
            <div className="debug">
              <div>mode: local</div>
              <div>backend: {localDetector.backend ?? '—'}</div>
              <div>infs: {localDetector.stats.inferences}</div>
              {localDetector.stats.lastError && (
                <div className="debug-err">
                  err: {localDetector.stats.lastError}
                </div>
              )}
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
