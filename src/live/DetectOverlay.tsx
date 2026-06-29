import { useEffect, useRef } from 'react';
import type { PointerEvent, RefObject } from 'react';
import type { TrackedBox } from '../types';
import { cssToVideoPx } from '../coords';

const FLASH_MS = 700;

type Flash = { bbox: [number, number, number, number]; expiry: number };

// The video/overlay use object-fit: cover, so on phones the canvas extends past
// the visible viewport. Compute the visible canvas region (in canvas coords) so
// boxes can be clamped to what the user actually sees.
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

type Props = {
  video: HTMLVideoElement | null;
  // Live detection boxes, updated out-of-band by the detector hook; read every
  // frame so 3fps inference never re-renders the React tree.
  boxesRef: RefObject<TrackedBox[]>;
  // instance_ids currently in the cart, for the solid-blue highlight.
  selectedIds: Set<number>;
  // Called when a tap lands inside a box (the smallest one under the point).
  // Fired for every hit including re-taps/over-limit, matching the old flow —
  // the parent decides whether to actually add it.
  onPick: (box: TrackedBox) => void;
};

// Live-mode bbox overlay synced to the camera video. Tap-mode only — temporarily
// kept as the legacy detector overlay while talk mode is built alongside it.
export function DetectOverlay({ video, boxesRef, selectedIds, onPick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const flashesRef = useRef<Flash[]>([]);
  // Keep selection readable from the rAF loop without re-subscribing the effect.
  const selectedRef = useRef(selectedIds);
  useEffect(() => {
    selectedRef.current = selectedIds;
  });

  useEffect(() => {
    const canvas = canvasRef.current;
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

      // No per-box text overlay — the on-device detector's COCO class
      // predictions aren't trustworthy (posters get tagged "toothbrush", etc.).
      // Boxes alone are the signal; Gemini does identification post-stop.
      const visible = visibleCanvasRegion(canvas);
      for (const b of boxesRef.current) {
        const inCart = selectedRef.current.has(b.instance_id);
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
  }, [video, boxesRef]);

  const handlePointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !video) return;
    const rect = canvas.getBoundingClientRect();
    const { x, y } = cssToVideoPx(
      e.clientX,
      e.clientY,
      rect,
      canvas.width,
      canvas.height,
    );

    let hit: TrackedBox | null = null;
    let smallestArea = Infinity;
    for (const b of boxesRef.current) {
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
    onPick(hit);
  };

  return (
    <canvas
      ref={canvasRef}
      className="overlay"
      onPointerDown={handlePointerDown}
    />
  );
}
