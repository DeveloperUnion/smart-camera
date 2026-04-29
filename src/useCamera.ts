import { useCallback, useEffect, useRef, useState } from 'react';

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      stream.getTracks().forEach((t) => {
        t.addEventListener('ended', () => {
          console.warn('[camera] track ended:', t.kind, t.label);
          setError('カメラが切断されました');
          setActive(false);
        });
        t.addEventListener('mute', () => {
          console.warn('[camera] track muted:', t.kind, t.label);
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

  return { videoRef, start, stop, active, error };
}
