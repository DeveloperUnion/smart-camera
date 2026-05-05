import { useCallback, useEffect, useRef, useState } from 'react';

export const MAX_DURATION_MS = 10_000;

const PREFERRED_TYPES = [
  // iOS Safari emits MP4/H.264 directly. Prefer this when available so we
  // can hand the blob straight to Gemini without transcoding.
  'video/mp4;codecs=avc1',
  'video/mp4',
  // Android Chrome: VP9/VP8 in WebM is widely supported.
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];

function pickMimeType(): string | null {
  if (typeof MediaRecorder === 'undefined') return null;
  for (const t of PREFERRED_TYPES) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return null;
}

export type RecordingResult = { blob: Blob; mimeType: string; durationMs: number };

export function useRecorder(stream: MediaStream | null) {
  const [recording, setRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef(0);
  const stopTimerRef = useRef<number | null>(null);
  const tickTimerRef = useRef<number | null>(null);
  const resolveRef = useRef<((r: RecordingResult | null) => void) | null>(null);

  const cleanupTimers = useCallback(() => {
    if (stopTimerRef.current !== null) {
      clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    if (tickTimerRef.current !== null) {
      clearInterval(tickTimerRef.current);
      tickTimerRef.current = null;
    }
  }, []);

  const start = useCallback(async (): Promise<RecordingResult | null> => {
    setError(null);
    if (!stream) {
      setError('カメラが起動していません');
      return null;
    }
    const mimeType = pickMimeType();
    if (!mimeType) {
      setError('この端末では録画に対応していません');
      return null;
    }

    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 800_000,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : '録画の初期化に失敗');
      return null;
    }

    chunksRef.current = [];
    recorderRef.current = recorder;
    startedAtRef.current = performance.now();

    const result = new Promise<RecordingResult | null>((resolve) => {
      resolveRef.current = resolve;
    });

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      cleanupTimers();
      setRecording(false);
      const blob = new Blob(chunksRef.current, { type: mimeType });
      const durationMs = Math.round(performance.now() - startedAtRef.current);
      const r = resolveRef.current;
      resolveRef.current = null;
      if (r) r({ blob, mimeType, durationMs });
    };
    recorder.onerror = (ev) => {
      cleanupTimers();
      setRecording(false);
      const err = (ev as { error?: { message?: string } }).error;
      setError(err?.message ?? '録画エラー');
      const r = resolveRef.current;
      resolveRef.current = null;
      if (r) r(null);
    };

    recorder.start(250);
    setRecording(true);
    setElapsedMs(0);

    tickTimerRef.current = window.setInterval(() => {
      setElapsedMs(Math.round(performance.now() - startedAtRef.current));
    }, 100);

    stopTimerRef.current = window.setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, MAX_DURATION_MS);

    return result;
  }, [stream, cleanupTimers]);

  const stop = useCallback(() => {
    const r = recorderRef.current;
    if (r && r.state !== 'inactive') r.stop();
  }, []);

  useEffect(() => {
    return () => {
      cleanupTimers();
      const r = recorderRef.current;
      if (r && r.state !== 'inactive') r.stop();
    };
  }, [cleanupTimers]);

  return { start, stop, recording, elapsedMs, error };
}
