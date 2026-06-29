import { useCallback, useEffect, useRef, useState } from 'react';
import { useCamera } from './useCamera';
import { Button } from './ui/Button';
import { CartView } from './cart/CartView';
import { useLiveSession } from './live/useLiveSession';
import { createCartHandlers } from './live/cartTools';
import type { CartEntry, RefinedItem } from './types';
import './App.css';

type Phase = 'idle' | 'live' | 'refining' | 'cart';

const MAX_SELECTIONS = 30;

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  // Cart keyed by instance_id from LocalTracker — so tapping the same physical
  // object twice is deduped, while distinct instances of the same label
  // aggregate as a count after refining.
  const [cart, setCart] = useState<Map<number, CartEntry>>(new Map());
  const [refineError, setRefineError] = useState<string | null>(null);

  const camera = useCamera();
  const cameraError = camera.error;

  const cartRef = useRef(cart);
  useEffect(() => {
    cartRef.current = cart;
  });

  const cartCount = cart.size;

  // Voice-mode cart talk. Voice entries get unique *negative* instance_ids so
  // they never collide with the tracker's positive ids; tools read the live
  // cart via cartRef and mutate it through setCart.
  const voiceIdRef = useRef(-1);
  // Rebuilt each render (cheap); useLiveSession mirrors it into a ref, so a
  // fresh object never reconnects the session. The handlers only read cartRef/
  // voiceIdRef at call time (in tool callbacks), never during render.
  const cartHandlers = createCartHandlers({
    getCart: () => cartRef.current,
    setCart,
    nextVoiceId: () => voiceIdRef.current--,
    max: MAX_SELECTIONS,
  });
  const talk = useLiveSession({
    videoEl: camera.videoEl,
    handlers: cartHandlers,
  });
  const talkActive = talk.status === 'active' || talk.status === 'connecting';
  const toggleTalk = useCallback(() => {
    if (talkActive) talk.stop();
    else void talk.start();
  }, [talkActive, talk]);

  const handleStart = useCallback(async () => {
    setRefineError(null);
    setPhase('live');
    if (!camera.active) {
      await camera.start();
    }
  }, [camera]);

  const stopTalk = talk.stop;
  const handleStopLive = useCallback(async () => {
    stopTalk();
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
  }, [camera, stopTalk]);

  const handleReset = useCallback(() => {
    stopTalk();
    setCart(new Map());
    setRefineError(null);
    setPhase('idle');
  }, [stopTalk]);

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
            カメラを起動 → 🎙 を押して話しかけると、AI
            が捨てる物をカゴに追加します。
          </p>
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
          <div className="badge">🛒 {cartCount}</div>
          <div className="live-cart-panel">
            {cartCount === 0 ? (
              <div className="live-cart-empty">
                🎙 を押して話しかけてください
              </div>
            ) : (
              <div className="live-cart-chip">
                <span className="live-cart-chip-label">カゴ</span>
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
          {talk.status !== 'idle' && (
            <div className={`talk-status ${talk.status}`}>
              {talk.status === 'connecting' && '接続中…'}
              {talk.status === 'active' &&
                (talk.caption || '🎙 話しかけてください（例:「ペットボトル2つ追加して」）')}
              {talk.status === 'error' &&
                `音声エラー: ${talk.error ?? '不明'}`}
            </div>
          )}
          <button className="stop" onClick={handleStopLive}>
            停止
          </button>
          <button
            className={`talk-btn ${talkActive ? 'active' : ''}`}
            onClick={toggleTalk}
            aria-label={talkActive ? '音声モードを終了' : '音声モードを開始'}
          >
            {talkActive ? '🔴' : '🎙'}
          </button>
          {cameraError && <div className="error">{cameraError}</div>}
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
