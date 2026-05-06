import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraTrackState = 'unknown' | 'live' | 'muted' | 'ended';

export function useCamera() {
  const streamRef = useRef<MediaStream | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  // Callback ref: tracks the currently-mounted <video> element across
  // mounts/unmounts (idle → preview → recording → review all involve different
  // <video> elements). State-backed so the attachment effect re-runs.
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const videoRef = useCallback((el: HTMLVideoElement | null) => {
    setVideoEl(el);
  }, []);

  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trackState, setTrackState] = useState<CameraTrackState>('unknown');

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStream(null);
    setActive(false);
    setTrackState('unknown');
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 15, max: 24 },
        },
        audio: false,
      });
      streamRef.current = s;
      setStream(s);
      setTrackState('live');
      s.getTracks().forEach((t) => {
        t.addEventListener('ended', () => {
          console.warn('[camera] track ended:', t.kind, t.label);
          setError('カメラが切断されました');
          setActive(false);
          setTrackState('ended');
        });
        t.addEventListener('mute', () => {
          console.warn('[camera] track muted:', t.kind, t.label);
          setTrackState('muted');
        });
        t.addEventListener('unmute', () => {
          setTrackState('live');
        });
      });
      setActive(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'カメラ起動に失敗しました';
      setError(msg);
    }
  }, []);

  // Reactively attach the stream to whichever <video> element is currently
  // mounted. Re-runs when either changes, so swapping between preview and
  // recording phases (different <video> elements with the same ref callback)
  // keeps the feed alive without manually re-attaching.
  useEffect(() => {
    if (!videoEl) return;
    if (stream) {
      // setting srcObject to the same stream is a no-op in modern browsers,
      // but be defensive.
      if (videoEl.srcObject !== stream) {
        videoEl.srcObject = stream;
      }
      // autoPlay handles this in most cases; explicit play() is a safety net
      // for browsers that pause when srcObject is reassigned.
      videoEl.play().catch(() => {});
    } else {
      videoEl.srcObject = null;
    }
  }, [videoEl, stream]);

  useEffect(() => {
    return () => stop();
  }, [stop]);

  return { videoRef, videoEl, start, stop, stream, active, error, trackState };
}
