import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { useLocalDetector } from './useLocalDetector';
import { captureFrameJpeg } from './captureSnapshot';
import type { RefinedItem, TrackedBox } from './types';
import './App.css';

type Phase = 'idle' | 'live' | 'refining' | 'cart';

type Flash = { bbox: [number, number, number, number]; expiry: number };

const FLASH_MS = 700;
const MAX_SELECTIONS = 30;

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

type CartEntry = {
  instance_id: number;
  // YOLO's coarse label; always present so the cart can show something
  // immediately on tap, and so refine-items has a hint to send to Gemini.
  yolo_label: string;
  // Base64 JPEG (no data: prefix). Cleared after refine-items returns to
  // free memory — we don't need it again once the cart is enriched.
  snapshot_b64: string;
  // Normalized 0-1 xyxy in the captured frame's coordinate space (after
  // letterbox removal isn't needed since we captured from <video> directly).
  snapshot_bbox: [number, number, number, number];
  refined?: RefinedItem;
};

function displayLabel(e: CartEntry): string {
  return e.refined?.name ?? e.yolo_label;
}

// Returns ordered sublines for the cart row. The first non-empty subline
// becomes line 2; remaining ones can be shown as line 3 if we want, but
// for now we collapse everything past the model number into one " · "
// joined line to stay compact on phones.
function refinedSublines(r?: RefinedItem): string[] {
  if (!r) return [];
  const lines: string[] = [];
  if (r.modelNumber) lines.push(r.modelNumber);
  if (r.description) lines.push(r.description);
  const attrs: string[] = [];
  if (r.manufacturer) attrs.push(r.manufacturer);
  if (r.yearOfManufacture) attrs.push(`${r.yearOfManufacture}年`);
  if (r.capacity) attrs.push(r.capacity);
  if (attrs.length) lines.push(attrs.join(' · '));
  return lines;
}

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  // Cart keyed by instance_id from LocalTracker — so tapping the same
  // physical object twice is deduped, while distinct instances of the same
  // label aggregate as a count after refining.
  const [cart, setCart] = useState<Map<number, CartEntry>>(new Map());
  const [refineError, setRefineError] = useState<string | null>(null);
  const [tooManyVisible, setTooManyVisible] = useState(false);
  const tooManyTimerRef = useRef<number | null>(null);
  const flashesRef = useRef<Flash[]>([]);

  const overlayRef = useRef<HTMLCanvasElement>(null);

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

  const handleStart = useCallback(async () => {
    setRefineError(null);
    setPhase('live');
    if (!camera.active) {
      await camera.start();
    }
  }, [camera]);

  // Live-mode bbox overlay synced to the camera video.
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

      const visible = visibleCanvasRegion(canvas);
      for (const b of localBoxesRef.current) {
        const inCart = cartRef.current.has(b.instance_id);
        ctx.lineWidth = inCart ? 3 : 2;
        ctx.strokeStyle = inCart ? '#3B82F6' : '#9CA3AF';
        if (inCart) ctx.setLineDash([]);
        else ctx.setLineDash([6, 4]);
        const [rx1, ry1, rx2, ry2] = b.bbox;
        if (
          rx2 <= visible.x1 ||
          rx1 >= visible.x2 ||
          ry2 <= visible.y1 ||
          ry1 >= visible.y2
        ) {
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
        const labelX = Math.max(
          visible.x1,
          Math.min(x1, visible.x2 - textWidth),
        );
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

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [phase, camera.videoEl, localBoxesRef]);

  const handleTapLive = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = overlayRef.current;
      const video = camera.videoEl;
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
      const norm: [number, number, number, number] = vw && vh
        ? [bx1 / vw, by1 / vh, bx2 / vw, by2 / vh]
        : [0, 0, 1, 1];

      // Snapshot fires async but we add the cart entry synchronously with an
      // empty snapshot first so the chip shows up instantly. Once toBlob
      // completes we patch the entry. This avoids any apparent input lag.
      const instanceId = hit.instance_id;
      const yoloLabel = hit.label;
      setCart((prev) => {
        if (prev.has(instanceId)) return prev;
        const next = new Map(prev);
        next.set(instanceId, {
          instance_id: instanceId,
          yolo_label: yoloLabel,
          snapshot_b64: '',
          snapshot_bbox: norm,
        });
        return next;
      });
      captureFrameJpeg(video).then((b64) => {
        setCart((prev) => {
          const cur = prev.get(instanceId);
          if (!cur) return prev;
          const next = new Map(prev);
          next.set(instanceId, { ...cur, snapshot_b64: b64 });
          return next;
        });
      }).catch((err) => {
        console.warn('snapshot failed', err);
      });
    },
    [camera.videoEl, localBoxesRef],
  );

  const handleStopLive = useCallback(async () => {
    camera.stop();
    const entries = Array.from(cartRef.current.values()).filter(
      (e) => e.snapshot_b64.length > 0,
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
        // Also clear snapshots for entries Gemini didn't return — keeping
        // the YOLO fallback label but freeing the JPEG bytes.
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
      // Clear snapshots even on failure — they're useless from here on and
      // 30 JPEGs in memory is enough to matter on iOS.
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

  // Group by displayed label so distinct instances of the same product
  // collapse into a single row with × count. We use displayLabel so refined
  // names group together (e.g. two "コカコーラ 350ml" become × 2) and so do
  // un-refined YOLO labels (e.g. two "缶" become × 2).
  const cartGroups = (() => {
    const groups = new Map<
      string,
      { ids: number[]; entry: CartEntry }
    >();
    for (const entry of cart.values()) {
      const key = displayLabel(entry);
      const existing = groups.get(key);
      if (existing) existing.ids.push(entry.instance_id);
      else groups.set(key, { ids: [entry.instance_id], entry });
    }
    return Array.from(groups.entries());
  })();
  const cartCount = cart.size;

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
          {detectorError && (
            <div className="status err">{detectorError}</div>
          )}
          {cameraError && <div className="status err">{cameraError}</div>}
          <button className="primary" onClick={handleStart}>
            カメラ開始
          </button>
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
          <div className="badge">🛒 {cartCount}</div>
          <div className="live-cart-panel">
            {cartGroups.length === 0 ? (
              <div className="live-cart-empty">枠をタップしてカゴに追加</div>
            ) : (
              cartGroups.map(([label, { ids }]) => (
                <div className="live-cart-chip" key={label}>
                  <span className="live-cart-chip-label">{label}</span>
                  <span className="live-cart-chip-count">×{ids.length}</span>
                  <button
                    className="live-cart-chip-remove"
                    onClick={() =>
                      setCart((prev) => {
                        const next = new Map(prev);
                        for (const id of ids) next.delete(id);
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
          {detectorError && (
            <div className="error">エラー: {detectorError}</div>
          )}
          {cameraError && <div className="error">{cameraError}</div>}
          {DEBUG && (
            <div className="debug">
              <div>backend: {detectorBackend ?? '—'}</div>
              <div>infs: {detectorStats.inferences}</div>
              <div>
                maxScore: {detectorStats.maxScore.toFixed(3)} raw:{' '}
                {detectorStats.rawCount} kept:{' '}
                {detectorStats.keptCount}
              </div>
              {detectorStats.lastError && (
                <div className="debug-err">
                  err: {detectorStats.lastError}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {phase === 'refining' && (
        <div className="screen running">
          <div className="analyzing-overlay" style={{ position: 'static', height: '100%' }}>
            <div className="spinner" />
            <div className="analyzing-text">解析中…</div>
            <div className="analyzing-sub">
              選択した {cart.size} 個を Gemini で詳細化中
            </div>
          </div>
        </div>
      )}

      {phase === 'cart' && (
        <div className="screen stopped">
          <h1>カゴの中身</h1>
          {refineError && (
            <div className="status err">
              詳細取得に失敗しました ({refineError})。YOLO の暫定ラベルで表示しています。
            </div>
          )}
          {cartGroups.length === 0 ? (
            <p className="lead">何も追加されていません。</p>
          ) : (
            <ul className="cart">
              {cartGroups.map(([label, { ids, entry }]) => {
                const sublines = refinedSublines(entry.refined);
                return (
                  <li key={label}>
                    <span className="label">
                      {label}
                      {sublines.map((line, i) => (
                        <span
                          key={i}
                          style={{
                            display: 'block',
                            fontSize: 12,
                            color: '#888',
                          }}
                        >
                          {line}
                        </span>
                      ))}
                    </span>
                    <span className="count">× {ids.length}</span>
                    <button
                      className="remove"
                      onClick={() =>
                        setCart((prev) => {
                          const next = new Map(prev);
                          for (const id of ids) next.delete(id);
                          return next;
                        })
                      }
                      aria-label="削除"
                    >
                      ×
                    </button>
                  </li>
                );
              })}
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
