import { useCallback, useEffect, useRef, useState } from 'react';
import { GoogleGenAI, Modality } from '@google/genai';
import type {
  FunctionDeclaration,
  FunctionResponse,
  LiveServerMessage,
  Session,
} from '@google/genai';
import { MicRecorder, AudioPlayer } from './audio';
import { captureFrameJpeg } from '../captureSnapshot';

export type LiveStatus = 'idle' | 'connecting' | 'active' | 'error';

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
  tools: FunctionDeclaration[];
  handlers: Record<string, ToolHandler>;
  systemInstruction: string;
  // 0 disables video; otherwise the JPEG send interval in ms (1000 = 1fps).
  frameIntervalMs?: number;
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
  tools,
  handlers,
  systemInstruction,
  frameIntervalMs = 1000,
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
  }, []);

  const stop = useCallback(() => {
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
    if (msg.serverContent?.interrupted) playerRef.current?.clear();

    // Assistant transcript for the on-screen caption.
    const out = msg.serverContent?.outputTranscription?.text;
    if (out) {
      setCaption((prev) => {
        const merged = (prev + out).slice(-140);
        return merged;
      });
    }
    if (msg.serverContent?.turnComplete) setCaption('');

    // Tool calls: run each handler and return the results together.
    const calls = msg.toolCall?.functionCalls;
    if (calls && calls.length) {
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
        sessionRef.current?.sendToolResponse({ functionResponses: responses });
      })();
    }
  }, []);

  const start = useCallback(async () => {
    if (sessionRef.current) return;
    setStatus('connecting');
    setError(null);
    setCaption('');
    closingRef.current = false;

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
      const { token, model, fallbackModel } = (await res.json()) as {
        token: string;
        model: string;
        fallbackModel?: string;
      };

      // Ephemeral tokens are a v1alpha feature; the token is used as the apiKey.
      const ai = new GoogleGenAI({
        apiKey: token,
        httpOptions: { apiVersion: 'v1alpha' },
      });

      const connectWith = (modelName: string) =>
        ai.live.connect({
          model: modelName,
          callbacks: {
            onopen: () => setStatus('active'),
            onmessage: handleMessage,
            onerror: (e: ErrorEvent) => {
              console.error('live onerror', e);
              setError(e.message || 'Live セッションエラー');
              setStatus('error');
              teardown();
            },
            onclose: (e: CloseEvent) => {
              console.warn('live onclose', e?.code, e?.reason);
              // A close we initiated (stop/unmount/error path) is clean.
              if (closingRef.current) {
                setStatus((s) => (s === 'error' ? s : 'idle'));
                return;
              }
              // Unexpected server disconnect — surface the reason so a bad
              // model name or rejected config is visible on-device.
              const why = e?.reason || `コード ${e?.code ?? '?'}`;
              setError(`接続が切れました (${why})`);
              setStatus('error');
              teardown();
            },
          },
          config: {
            responseModalities: [Modality.AUDIO],
            systemInstruction,
            tools: [{ functionDeclarations: tools }],
            outputAudioTranscription: {},
          },
        });

      // Try the primary model; on connect failure fall back once (the token is
      // minted with uses: 2 to allow this retry).
      let session: Session;
      try {
        session = await connectWith(model);
      } catch (primaryErr) {
        if (!fallbackModel || fallbackModel === model) throw primaryErr;
        console.warn(
          `live: primary model "${model}" failed, falling back to "${fallbackModel}"`,
          primaryErr,
        );
        session = await connectWith(fallbackModel);
      }
      sessionRef.current = session;

      // Mic → 16 kHz PCM → realtime input. Report the actual context rate.
      const rec = new MicRecorder();
      await rec.start((b64) => {
        sessionRef.current?.sendRealtimeInput({
          audio: { data: b64, mimeType: `audio/pcm;rate=${rec.sampleRate}` },
        });
      });
      recRef.current = rec;

      // ~1fps full-frame video as visual context.
      if (frameIntervalMs > 0) {
        frameTimerRef.current = window.setInterval(() => {
          const v = videoElRef.current;
          const s = sessionRef.current;
          if (!v || !s) return;
          try {
            const jpeg = captureFrameJpeg(v);
            s.sendRealtimeInput({ video: { data: jpeg, mimeType: 'image/jpeg' } });
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
  }, [handleMessage, systemInstruction, tools, frameIntervalMs, teardown]);

  // Tear everything down on unmount.
  useEffect(() => teardown, [teardown]);

  return { status, error, caption, start, stop };
}
