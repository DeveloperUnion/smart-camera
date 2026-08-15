import { useCallback, useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { GoogleGenAI } from '@google/genai';
import type {
  FunctionResponse,
  LiveServerMessage,
  Session,
} from '@google/genai';
import { MicRecorder, AudioPlayer } from './audio';
import { dlog } from './debugLog';
import { captureFrameJpeg } from '../captureSnapshot';

export type LiveStatus = 'idle' | 'connecting' | 'active' | 'error';

// A higher-res JPEG of the same instant as the frame last SENT to the Live
// session. The model grounds its box_2d on the (downscaled) sent frame, but
// box_2d is normalized so it maps onto this sharper copy — cropping from it
// keeps model perception and crop source consistent while yielding a crisp
// thumbnail.
export type RetainedFrame = { data: string; ts: number };

// One (ephemeral token, locked model) pair from /api/live-token, tried in order.
type Attempt = { token: string; model: string };

// A tool the model can call to operate the cart by voice. The returned object
// is sent straight back as the function response so the model can confirm what
// happened verbally ("ペットボトルを1つ追加しました").
export type ToolHandler = (
  args: Record<string, unknown>,
) => Record<string, unknown> | Promise<Record<string, unknown>>;

type Options = {
  // Read each second to push a frame as visual context. May be null while the
  // camera is starting; frames are simply skipped until it's ready.
  videoEl: HTMLVideoElement | null;
  // Tool-call handlers, keyed by tool name. The tool DECLARATIONS, prompt and
  // temperature are baked into the ephemeral token server-side (see
  // api/live-token.ts) — the client only executes the calls.
  handlers: Record<string, ToolHandler>;
  // 0 disables video; otherwise the JPEG send interval in ms (1000 = 1fps).
  frameIntervalMs?: number;
  // Owned by the caller (it's read by tool handlers created before this hook
  // runs). Written here right after each frame is sent; cleared on start/stop.
  lastFrameRef?: RefObject<RetainedFrame | null>;
};

export type LiveSession = {
  status: LiveStatus;
  error: string | null;
  // Latest assistant speech transcript, for an on-screen caption.
  caption: string;
  start: () => Promise<void>;
  stop: () => void;
};

// Drives one Gemini Live (native-audio) session: fetches an ephemeral token,
// connects, streams mic audio in and the model's audio out, pushes ~1fps video
// frames, and routes tool calls to the supplied handlers. All the heavy state
// (session, mic, player, timers) lives in refs so re-renders never tear it down
// — only start()/stop() and unmount do.
export function useLiveSession({
  videoEl,
  handlers,
  frameIntervalMs = 1000,
  lastFrameRef,
}: Options): LiveSession {
  const [status, setStatus] = useState<LiveStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [caption, setCaption] = useState('');

  const sessionRef = useRef<Session | null>(null);
  const recRef = useRef<MicRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const frameTimerRef = useRef<number | null>(null);
  // True while we are the ones closing the socket, so the onclose callback can
  // tell a user-initiated stop from an unexpected server disconnect.
  const closingRef = useRef(false);
  // Accumulates the current turn's assistant transcript purely for diagnostics
  // (logged on interrupt / turnComplete to see how far a confirmation got).
  const turnTextRef = useRef('');

  // Mirror props into refs so the long-lived session callbacks always see the
  // current video element / handlers without reconnecting.
  const videoElRef = useRef(videoEl);
  const handlersRef = useRef(handlers);
  useEffect(() => {
    videoElRef.current = videoEl;
  }, [videoEl]);
  useEffect(() => {
    handlersRef.current = handlers;
  }, [handlers]);

  const teardown = useCallback(() => {
    if (frameTimerRef.current !== null) {
      window.clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
    recRef.current?.stop();
    recRef.current = null;
    playerRef.current?.close();
    playerRef.current = null;
    closingRef.current = true;
    try {
      sessionRef.current?.close();
    } catch {
      // already closed
    }
    sessionRef.current = null;
    // Don't leak the previous session's frame into the next one. Tool handlers
    // read the frame into a local at their start, so an in-flight add_to_cart
    // still completes with the frame it saw.
    if (lastFrameRef) lastFrameRef.current = null;
  }, [lastFrameRef]);

  const stop = useCallback(() => {
    dlog('stop');
    teardown();
    setCaption('');
    setStatus('idle');
  }, [teardown]);

  const handleMessage = useCallback((msg: LiveServerMessage) => {
    // Audio out: enqueue every inline audio part in order.
    const parts = msg.serverContent?.modelTurn?.parts ?? [];
    for (const p of parts) {
      const data = p.inlineData?.data;
      if (data && p.inlineData?.mimeType?.startsWith('audio/')) {
        playerRef.current?.enqueue(data);
      }
    }

    // Barge-in: the user spoke over the model — drop queued speech.
    if (msg.serverContent?.interrupted) {
      // [live] DIAGNOSTIC: a barge-in here cancels the turn; if it lands before
      // the model emits its toolCall, add_to_cart is silently dropped. The
      // partial text shows how far the spoken confirmation got before the cut.
      dlog(`interrupted (partial="${turnTextRef.current}")`);
      playerRef.current?.clear();
    }

    // Assistant transcript for the on-screen caption.
    const out = msg.serverContent?.outputTranscription?.text;
    if (out) {
      turnTextRef.current += out;
      setCaption((prev) => {
        const merged = (prev + out).slice(-140);
        return merged;
      });
    }
    if (msg.serverContent?.turnComplete) {
      dlog(`turnComplete (text="${turnTextRef.current}")`);
      turnTextRef.current = '';
      setCaption('');
    }

    // [live] DIAGNOSTIC: the server withdrew already-issued tool calls (happens
    // when a turn is interrupted after toolCall but before we respond). Seeing
    // this alongside an empty cart confirms the interrupt-drops-the-call theory.
    const cancelIds = (
      msg as { toolCallCancellation?: { ids?: string[] } }
    ).toolCallCancellation?.ids;
    if (cancelIds && cancelIds.length) {
      dlog(`toolCallCancellation ${JSON.stringify(cancelIds)}`);
    }

    // Tool calls: run each handler and return the results together.
    const calls = msg.toolCall?.functionCalls;
    if (calls && calls.length) {
      dlog(
        `toolCall ${JSON.stringify(calls.map((c) => ({ name: c.name, args: c.args })))}`,
      );
      void (async () => {
        const responses: FunctionResponse[] = [];
        for (const c of calls) {
          const name = c.name ?? '';
          const h = handlersRef.current[name];
          let response: Record<string, unknown>;
          try {
            response = h
              ? await h((c.args as Record<string, unknown>) ?? {})
              : { error: `unknown tool: ${name}` };
          } catch (err) {
            response = {
              error: err instanceof Error ? err.message : String(err),
            };
          }
          responses.push({ id: c.id, name, response });
        }
        dlog(
          `tool responses ${JSON.stringify(responses.map((r) => ({ name: r.name, response: r.response })))}`,
        );
        sessionRef.current?.sendToolResponse({ functionResponses: responses });
      })();
    }
  }, []);

  const start = useCallback(async () => {
    if (sessionRef.current) return;
    dlog('start');
    setStatus('connecting');
    setError(null);
    setCaption('');
    turnTextRef.current = '';
    closingRef.current = false;
    if (lastFrameRef) lastFrameRef.current = null;

    try {
      // Resume the playback context FIRST, while still inside the user-gesture
      // chain — iOS Safari blocks AudioContext.resume() if it's deferred past an
      // await (e.g. the token fetch below).
      const player = new AudioPlayer();
      await player.resume();
      playerRef.current = player;

      const res = await fetch('/api/live-token', { method: 'POST' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || `トークン取得失敗 (HTTP ${res.status})`);
      }
      const { primary, fallback } = (await res.json()) as {
        primary: Attempt;
        fallback: Attempt | null;
        expireTime?: string;
      };
      const attempts = [primary, fallback].filter(
        (a): a is Attempt => !!a && !!a.token && !!a.model,
      );
      if (attempts.length === 0) throw new Error('トークンが空です');

      // Connect with one (token, model) attempt. Each token is locked to its
      // model, so a fresh client is needed per attempt. Crucially, the attempt
      // is only considered successful once the FIRST server message arrives
      // (setupComplete) — a bad/unavailable model opens the socket and then
      // closes it, so resolving on open alone would mask the failure and skip
      // the fallback. We resolve on first message, reject on early close/error,
      // and time out if the server goes silent.
      const connectAttempt = (attempt: Attempt) =>
        new Promise<Session>((resolve, reject) => {
          let settled = false;
          let sess: Session | null = null;
          let sawMessage = false;
          const ok = () => {
            if (!settled && sess && sawMessage) {
              settled = true;
              window.clearTimeout(timer);
              resolve(sess);
            }
          };
          const fail = (err: Error) => {
            if (settled) return;
            settled = true;
            window.clearTimeout(timer);
            reject(err);
          };
          const timer = window.setTimeout(
            () => fail(new Error('接続タイムアウト')),
            10000,
          );

          const ai = new GoogleGenAI({
            apiKey: attempt.token,
            httpOptions: { apiVersion: 'v1alpha' },
          });
          // No `config` here on purpose: with an ephemeral token the session
          // config (modalities, prompt, tools, temperature) is whatever was
          // locked into the token at mint time; client config is ignored.
          ai.live
            .connect({
              model: attempt.model,
              callbacks: {
                onopen: () => setStatus('connecting'),
                onmessage: (msg) => {
                  sawMessage = true;
                  ok();
                  if (settled) handleMessage(msg);
                },
                onerror: (e: ErrorEvent) => {
                  dlog(`onerror ${e?.message || 'live error'}`);
                  if (!settled) return fail(new Error(e.message || 'live error'));
                  setError(e.message || 'Live セッションエラー');
                  setStatus('error');
                  teardown();
                },
                onclose: (e: CloseEvent) => {
                  dlog(`onclose code=${e?.code ?? '?'} reason=${e?.reason ?? ''}`);
                  const why = e?.reason || `コード ${e?.code ?? '?'}`;
                  if (!settled) return fail(new Error(why));
                  if (closingRef.current) {
                    setStatus((s) => (s === 'error' ? s : 'idle'));
                    return;
                  }
                  setError(`接続が切れました (${why})`);
                  setStatus('error');
                  teardown();
                },
              },
            })
            .then((s) => {
              sess = s;
              ok();
            })
            .catch((err) =>
              fail(err instanceof Error ? err : new Error(String(err))),
            );
        });

      let session: Session | null = null;
      let lastErr: Error | null = null;
      for (const attempt of attempts) {
        try {
          session = await connectAttempt(attempt);
          dlog(`connected model=${attempt.model}`);
          break;
        } catch (err) {
          lastErr = err instanceof Error ? err : new Error(String(err));
          dlog(`model "${attempt.model}" failed: ${lastErr.message}`);
        }
      }
      if (!session) throw lastErr ?? new Error('接続に失敗しました');
      sessionRef.current = session;
      setStatus('active');

      // Mic → 16 kHz PCM → realtime input. Report the actual context rate.
      const rec = new MicRecorder();
      await rec.start((b64) => {
        sessionRef.current?.sendRealtimeInput({
          audio: { data: b64, mimeType: `audio/pcm;rate=${rec.sampleRate}` },
        });
      });
      recRef.current = rec;
      // [live] DIAGNOSTIC: if in != 16000 the AudioContext ignored our request
      // (common on iOS) — the resampler now handles it, out must read 16000.
      dlog(`mic in=${rec.inputRate}Hz out=${rec.sampleRate}Hz`);

      // ~1fps full-frame video as visual context.
      if (frameIntervalMs > 0) {
        frameTimerRef.current = window.setInterval(() => {
          const v = videoElRef.current;
          const s = sessionRef.current;
          if (!v || !s) return;
          try {
            const jpeg = captureFrameJpeg(v);
            s.sendRealtimeInput({ video: { data: jpeg, mimeType: 'image/jpeg' } });
            if (lastFrameRef) {
              // Retain a higher-res copy of the SAME instant for crisp crops.
              // box_2d is normalized (0-1000), so it maps to any resolution of
              // the same framing; the model still grounds on the 720px frame
              // sent above, but add_to_cart crops from this sharper one.
              lastFrameRef.current = {
                data: captureFrameJpeg(v, 1280),
                ts: Date.now(),
              };
            }
          } catch {
            // video not ready yet — skip this tick
          }
        }, frameIntervalMs);
      }

      // onopen usually fires first, but set active defensively in case the
      // implementation resolves connect() before delivering onopen.
      setStatus((s) => (s === 'connecting' ? 'active' : s));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus('error');
      teardown();
    }
  }, [handleMessage, frameIntervalMs, teardown, lastFrameRef]);

  // Tear everything down on unmount.
  useEffect(() => teardown, [teardown]);

  return { status, error, caption, start, stop };
}
