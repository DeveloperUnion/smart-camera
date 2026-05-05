import type { Appearance, Detection } from './types';

// Window in seconds around currentTime to consider an appearance "active".
// Gemini samples ~1 fps so the gap between appearances can be ~1s; widen
// slightly so boxes don't blink between samples.
const ACTIVE_WINDOW_S = 0.6;

export type ActiveBox = {
  instance_id: number;
  label: string;
  // Pixel coords in canvas space.
  bbox: [number, number, number, number];
};

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpBbox(
  a: Appearance['bbox'],
  b: Appearance['bbox'],
  t: number,
): Appearance['bbox'] {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
    lerp(a[3], b[3], t),
  ];
}

// For a given detection, find the bbox closest to `t` (linearly interpolated
// between the surrounding appearances). Returns null if t is outside the
// detection's active window.
export function bboxAtTime(
  detection: Detection,
  t: number,
): Appearance['bbox'] | null {
  const apps = detection.appearances;
  if (apps.length === 0) return null;

  // Single appearance — show it within the window.
  if (apps.length === 1) {
    return Math.abs(apps[0].time_s - t) <= ACTIVE_WINDOW_S ? apps[0].bbox : null;
  }

  // Find surrounding pair.
  let prev: Appearance | null = null;
  let next: Appearance | null = null;
  for (const a of apps) {
    if (a.time_s <= t) {
      prev = a;
    } else {
      next = a;
      break;
    }
  }

  if (prev && next) {
    const span = next.time_s - prev.time_s;
    if (span <= 0) return prev.bbox;
    const ratio = (t - prev.time_s) / span;
    return lerpBbox(prev.bbox, next.bbox, ratio);
  }
  // Before first / after last — only show if within window of the edge.
  const edge = prev ?? next;
  if (edge && Math.abs(edge.time_s - t) <= ACTIVE_WINDOW_S) return edge.bbox;
  return null;
}

export function activeBoxesAt(
  detections: Detection[],
  t: number,
  canvasWidth: number,
  canvasHeight: number,
): ActiveBox[] {
  const out: ActiveBox[] = [];
  for (const d of detections) {
    const bb = bboxAtTime(d, t);
    if (!bb) continue;
    out.push({
      instance_id: d.instance_id,
      label: d.label,
      bbox: [
        bb[0] * canvasWidth,
        bb[1] * canvasHeight,
        bb[2] * canvasWidth,
        bb[3] * canvasHeight,
      ],
    });
  }
  return out;
}
