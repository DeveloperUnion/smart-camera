import { useCallback, useEffect, useRef, useState } from 'react';

export type CameraTrackState = 'unknown' | 'live' | 'muted' | 'ended';

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trackState, setTrackState] = useState<CameraTrackState>('unknown');

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          // Lower than the model's 640 input — preprocess scales up anyway, and
          // iOS allocates a video frame pool sized to the requested resolution.
          width: { ideal: 480 },
          height: { ideal: 360 },
          frameRate: { ideal: 15, max: 24 },
        },
        audio: false,
      });
      streamRef.current = stream;
      setTrackState('live');
      stream.getTracks().forEach((t) => {
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
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        try {
          await video.play();
        } catch {
          // play() can reject if interrupted; ignore
        }
      }
      setActive(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'カメラ起動に失敗しました';
      setError(msg);
    }
  }, []);

  useEffect(() => {
    return () => stop();
  }, [stop]);

  return { videoRef, start, stop, active, error, trackState };
}
