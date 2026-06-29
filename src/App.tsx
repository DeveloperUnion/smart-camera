import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { useLocalDetector } from './useLocalDetector';
import { captureFrameJpeg } from './captureSnapshot';
import { Button } from './ui/Button';
import { DetectOverlay } from './live/DetectOverlay';
import { CartView } from './cart/CartView';
import type { CartEntry, RefinedItem, TrackedBox } from './types';
import './App.css';

type Phase = 'idle' | 'live' | 'refining' | 'cart';

const MAX_SELECTIONS = 30;

const DEBUG =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).get('debug') === '1';

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  // Cart keyed by instance_id from LocalTracker — so tapping the same physical
  // object twice is deduped, while distinct instances of the same label
  // aggregate as a count after refining.
  const [cart, setCart] = useState<Map<number, CartEntry>>(new Map());
  const [refineError, setRefineError] = useState<string | null>(null);
  const [tooManyVisible, setTooManyVisible] = useState(false);
  const tooManyTimerRef = useRef<number | null>(null);

  const camera = useCamera();
  const localDetector = useLocalDetector({
    videoEl: camera.videoEl,
    enabled: phase === 'live',
  });

  const {
    boxesRef: localBoxesRef,
    ready: detectorReady,
    backend: detectorBackend,
    error: detectorError,
    stats: detectorStats,
  } = localDetector;
  const cameraError = camera.error;

  const cartRef = useRef(cart);
  useEffect(() => {
    cartRef.current = cart;
  });

  // Set of selected instance_ids for the overlay's highlight, recomputed only
  // when the cart changes.
  const selectedIds = useMemo(() => new Set(cart.keys()), [cart]);
  const cartCount = cart.size;

  const handleStart = useCallback(async () => {
    setRefineError(null);
    setPhase('live');
    if (!camera.active) {
      await camera.start();
    }
  }, [camera]);

  // A tap landed inside a detection box: add it to the cart (deduped, capped),
  // capturing the current frame as the snapshot for refine-items.
  const handlePick = useCallback(
    (hit: TrackedBox) => {
      const video = camera.videoEl;
      if (!video) return;
      if (cartRef.current.has(hit.instance_id)) return;
      if (cartRef.current.size >= MAX_SELECTIONS) {
        setTooManyVisible(true);
        if (tooManyTimerRef.current !== null) {
          window.clearTimeout(tooManyTimerRef.current);
        }
        tooManyTimerRef.current = window.setTimeout(() => {
          setTooManyVisible(false);
          tooManyTimerRef.current = null;
        }, 2200);
        return;
      }

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const [bx1, by1, bx2, by2] = hit.bbox;
      const norm: [number, number, number, number] =
        vw && vh ? [bx1 / vw, by1 / vh, bx2 / vw, by2 / vh] : [0, 0, 1, 1];

      let snapshot = '';
      try {
        snapshot = captureFrameJpeg(video);
      } catch (err) {
        console.warn('snapshot failed', err);
      }

      setCart((prev) => {
        if (prev.has(hit.instance_id)) return prev;
        const next = new Map(prev);
        next.set(hit.instance_id, {
          instance_id: hit.instance_id,
          yolo_label: hit.label,
          snapshot_b64: snapshot,
          snapshot_bbox: norm,
          source: 'tap',
        });
        return next;
      });
    },
    [camera.videoEl],
  );

  const handleStopLive = useCallback(async () => {
    camera.stop();
    const entries = Array.from(cartRef.current.values()).filter(
      (e) => (e.snapshot_b64?.length ?? 0) > 0,
    );
    if (entries.length === 0) {
      setPhase('cart');
      return;
    }
    setPhase('refining');
    setRefineError(null);
    try {
      const res = await fetch('/api/refine-items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          items: entries.map((e) => ({
            id: e.instance_id,
            yolo_label: e.yolo_label,
            image_b64: e.snapshot_b64,
            image_mime: 'image/jpeg',
            bbox: e.snapshot_bbox,
          })),
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        if (res.status === 503) {
          throw new Error('Geminiが一時的に混み合っています。少し時間をおいて再試行してください。');
        }
        throw new Error(body?.error || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as {
        items?: Array<{ id: number; refined: RefinedItem }>;
      };
      setCart((prev) => {
        const next = new Map(prev);
        for (const r of data.items ?? []) {
          const cur = next.get(r.id);
          if (cur) {
            next.set(r.id, { ...cur, refined: r.refined, snapshot_b64: '' });
          }
        }
        // Also clear snapshots for entries Gemini didn't return — keeping the
        // YOLO fallback label but freeing the JPEG bytes.
        for (const [id, cur] of next) {
          if (cur.snapshot_b64) {
            next.set(id, { ...cur, snapshot_b64: '' });
          }
        }
        return next;
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setRefineError(msg);
      // Clear snapshots even on failure — they're useless from here on and 30
      // JPEGs in memory is enough to matter on iOS.
      setCart((prev) => {
        const next = new Map(prev);
        for (const [id, cur] of next) {
          if (cur.snapshot_b64) {
            next.set(id, { ...cur, snapshot_b64: '' });
          }
        }
        return next;
      });
    } finally {
      setPhase('cart');
    }
  }, [camera]);

  const handleReset = useCallback(() => {
    setCart(new Map());
    setRefineError(null);
    setPhase('idle');
  }, []);

  const handleRemove = useCallback((ids: number[]) => {
    setCart((prev) => {
      const next = new Map(prev);
      for (const id of ids) next.delete(id);
      return next;
    });
  }, []);

  return (
    <div className="app">
      {phase === 'idle' && (
        <div className="screen idle">
          <img src="/dustalk-logo.png" alt="Dustalk" className="dustalk-logo" />
          <h1>SmartCamera</h1>
          <p className="lead">
            カメラを起動 → 写った物体に枠が出る → タップでカゴに追加 →
            停止すると AI が選択した物の詳細を返します。
          </p>
          {detectorError && <div className="status err">{detectorError}</div>}
          {cameraError && <div className="status err">{cameraError}</div>}
          <Button onClick={handleStart}>カメラ開始</Button>
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
          <DetectOverlay
            video={camera.videoEl}
            boxesRef={localBoxesRef}
            selectedIds={selectedIds}
            onPick={handlePick}
          />
          <div className="badge">🛒 {cartCount}</div>
          <div className="live-cart-panel">
            {cartCount === 0 ? (
              <div className="live-cart-empty">枠をタップしてカゴに追加</div>
            ) : (
              // While live, every cart entry is labeled "物体" since DEIMv2
              // class predictions are unreliable. Show a single "選択中 × N"
              // chip with a clear-all button instead of per-label chips.
              <div className="live-cart-chip">
                <span className="live-cart-chip-label">選択中</span>
                <span className="live-cart-chip-count">×{cartCount}</span>
                <button
                  className="live-cart-chip-remove"
                  onClick={() => setCart(new Map())}
                  aria-label="すべて削除"
                >
                  ×
                </button>
              </div>
            )}
          </div>
          {tooManyVisible && (
            <div className="error">
              選択は {MAX_SELECTIONS} 個までです。停止して詳細取得に進んでください。
            </div>
          )}
          <button className="stop" onClick={handleStopLive}>
            停止
          </button>
          {!detectorReady && !detectorError && (
            <div className="preview-tip">モデル読み込み中…</div>
          )}
          {detectorError && <div className="error">エラー: {detectorError}</div>}
          {cameraError && <div className="error">{cameraError}</div>}
          {DEBUG && (
            <div className="debug">
              <div>backend: {detectorBackend ?? '—'}</div>
              <div>infs: {detectorStats.inferences}</div>
              <div>
                maxScore: {detectorStats.maxScore.toFixed(3)} raw:{' '}
                {detectorStats.rawCount} kept: {detectorStats.keptCount}
              </div>
              {detectorStats.lastError && (
                <div className="debug-err">err: {detectorStats.lastError}</div>
              )}
            </div>
          )}
        </div>
      )}

      {phase === 'refining' && (
        <div className="screen running">
          <div
            className="analyzing-overlay"
            style={{ position: 'static', height: '100%' }}
          >
            <div className="spinner" />
            <div className="analyzing-text">解析中…</div>
            <div className="analyzing-sub">
              選択した {cart.size} 個を Gemini で詳細化中
            </div>
          </div>
        </div>
      )}

      {phase === 'cart' && (
        <CartView
          cart={cart}
          refineError={refineError}
          onRemove={handleRemove}
          onReset={handleReset}
        />
      )}
    </div>
  );
}
