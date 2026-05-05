import { useCallback, useState } from 'react';
import type { Detection } from './types';

export type AnalyzeStatus = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';

async function blobToBase64(blob: Blob): Promise<string> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result as string);
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(blob);
  });
  const comma = dataUrl.indexOf(',');
  return dataUrl.slice(comma + 1);
}

export function useVideoDetector() {
  const [status, setStatus] = useState<AnalyzeStatus>('idle');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (blob: Blob, mimeType: string) => {
    setStatus('uploading');
    setError(null);
    setDetections([]);
    try {
      const data = await blobToBase64(blob);
      setStatus('analyzing');
      const res = await fetch('/api/detect-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video: data, mimeType }),
      });
      if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(`API ${res.status}: ${txt.slice(0, 120)}`);
      }
      const json = (await res.json()) as { detections: Detection[] };
      setDetections(json.detections);
      setStatus('done');
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setDetections([]);
    setError(null);
  }, []);

  return { analyze, reset, status, detections, error };
}
