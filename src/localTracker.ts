import type { LiveBox, TrackedBox } from './types';

// Keep a tracker alive for this many ms after it was last matched, so brief
// detection misses (a flickered frame, a momentary occlusion) don't fragment
// the same physical object into two cart entries.
const TRACK_GC_MS = 1500;
// Minimum IoU between a tracker's last bbox and a fresh detection to consider
// them the same object. Tuned for ~1.5–3 fps inference where objects move a
// noticeable amount between frames.
const MATCH_IOU = 0.3;

type TrackerState = {
  label: string;
  classId: number;
  lastBbox: [number, number, number, number];
  lastSeenAt: number;
};

function iou(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const aArea = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const bArea = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = aArea + bArea - inter;
  return union <= 0 ? 0 : inter / union;
}

export class LocalTracker {
  private trackers = new Map<number, TrackerState>();
  private nextId = 1;

  update(rawBoxes: LiveBox[], nowMs: number): TrackedBox[] {
    for (const [id, t] of this.trackers) {
      if (nowMs - t.lastSeenAt > TRACK_GC_MS) this.trackers.delete(id);
    }

    type Pair = { iou: number; trackerId: number; detIdx: number };
    const pairs: Pair[] = [];
    for (let i = 0; i < rawBoxes.length; i++) {
      const det = rawBoxes[i];
      for (const [id, t] of this.trackers) {
        if (t.label !== det.label) continue;
        const v = iou(t.lastBbox, det.bbox);
        if (v >= MATCH_IOU) pairs.push({ iou: v, trackerId: id, detIdx: i });
      }
    }

    pairs.sort((a, b) => b.iou - a.iou);
    const usedTracker = new Set<number>();
    const usedDet = new Set<number>();
    const detToTracker = new Map<number, number>();
    for (const p of pairs) {
      if (usedTracker.has(p.trackerId) || usedDet.has(p.detIdx)) continue;
      usedTracker.add(p.trackerId);
      usedDet.add(p.detIdx);
      detToTracker.set(p.detIdx, p.trackerId);
    }

    const out: TrackedBox[] = [];
    for (let i = 0; i < rawBoxes.length; i++) {
      const det = rawBoxes[i];
      let id = detToTracker.get(i);
      if (id === undefined) {
        id = this.nextId++;
      }
      this.trackers.set(id, {
        label: det.label,
        classId: det.classId,
        lastBbox: det.bbox,
        lastSeenAt: nowMs,
      });
      out.push({ ...det, instance_id: id });
    }
    return out;
  }

  reset() {
    this.trackers.clear();
    this.nextId = 1;
  }
}
