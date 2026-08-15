import type { CartEntry } from '../types';
import type { RetainedFrame, ToolHandler } from './useLiveSession';
import { captureFrameJpeg } from '../captureSnapshot';

// Voice-mode cart tool HANDLERS (browser side). The matching tool declarations
// and the system prompt live server-side in api/live-token.ts, because they
// must be baked into the ephemeral token (client-supplied tools/prompt are
// ignored on a token connection). The client only executes the calls, matching
// on the tool name strings below.

function labelOf(e: CartEntry): string {
  return e.refined?.name ?? e.yolo_label;
}

// Loose match so "ペットボトル" hits an entry labeled "ペットボトル 500ml" and
// vice-versa, without matching unrelated items.
function matches(e: CartEntry, name: string): boolean {
  const l = labelOf(e);
  return l === name || l.includes(name) || name.includes(l);
}

type Deps = {
  // Reads the latest committed cart synchronously inside handlers. A getter
  // (not the ref itself) so callers don't touch `.current` during render.
  getCart: () => Map<number, CartEntry>;
  setCart: (updater: (prev: Map<number, CartEntry>) => Map<number, CartEntry>) => void;
  // Allocates a unique, negative instance_id so voice entries never collide with
  // the tracker's positive ids.
  nextVoiceId: () => number;
  max: number;
  // The last frame actually sent to the Live session (null before the first
  // send). add_to_cart crops the model-referenced object out of it.
  getLastFrame: () => RetainedFrame | null;
  // Fallback capture source for a voice add that lands before any frame was
  // sent (e.g. speaking within the first second of the session).
  getVideoEl: () => HTMLVideoElement | null;
};

export function createCartHandlers({
  getCart,
  setCart,
  nextVoiceId,
  max,
  getLastFrame,
  getVideoEl,
}: Deps): Record<string, ToolHandler> {
  return {
    add_to_cart: async (args) => {
      const name = String(args.name ?? '').trim() || '物体';
      const count = Math.max(1, Math.floor(Number(args.count) || 1));
      const note = args.note ? String(args.note) : undefined;
      const position = args.position ? String(args.position) : undefined;

      // The bbox is no longer determined here: capture a FULL frame of the
      // moment and let /api/refine-items locate the named object within it (it
      // gets a clean, non-stale frame + the spoken name, so its grounding beats
      // the Live model's box_2d). Prefer a fresh live capture over the last
      // frame SENT to the session, which can be ~1s stale.
      let full: string | undefined;
      const v = getVideoEl();
      if (v) {
        try {
          full = captureFrameJpeg(v, 1024);
        } catch {
          // camera not ready — fall through to the last sent frame
        }
      }
      if (!full) {
        // No live capture — best effort: reuse the last frame sent to Live.
        // Read it ONCE so a concurrent stop/teardown (which nulls the ref)
        // can't pull it out from under us.
        full = getLastFrame()?.data;
      }

      const cur = getCart();
      const room = Math.max(0, max - cur.size);
      const toAdd = Math.min(count, room);

      const newEntries: CartEntry[] = [];
      for (let i = 0; i < toAdd; i++) {
        // count > 1 shares one snapshot: same string reference, no extra memory.
        // The full frame stays uncropped ([0,0,1,1]) until refine-items returns
        // a box_2d for it; App then crops the thumbnail client-side.
        newEntries.push({
          instance_id: nextVoiceId(),
          yolo_label: name,
          snapshot_b64: full,
          snapshot_bbox: [0, 0, 1, 1],
          source: 'voice',
          note,
          position,
        });
      }
      if (newEntries.length) {
        setCart((prev) => {
          const next = new Map(prev);
          for (const e of newEntries) next.set(e.instance_id, e);
          return next;
        });
      }
      return {
        added: toAdd,
        name,
        capped: toAdd < count,
        total: cur.size + toAdd,
      };
    },

    remove_from_cart: (args) => {
      const name = String(args.name ?? '').trim();
      const limit =
        args.count != null ? Math.max(1, Math.floor(Number(args.count) || 1)) : Infinity;
      const cur = getCart();
      const ids = [...cur.values()]
        .filter((e) => matches(e, name))
        .slice(0, limit === Infinity ? undefined : limit)
        .map((e) => e.instance_id);

      if (ids.length) {
        setCart((prev) => {
          const next = new Map(prev);
          for (const id of ids) next.delete(id);
          return next;
        });
      }
      return { removed: ids.length, name, total: cur.size - ids.length };
    },

    clear_cart: () => {
      const had = getCart().size;
      setCart(() => new Map());
      return { cleared: had, total: 0 };
    },

    list_cart: () => {
      const groups = new Map<string, number>();
      for (const e of getCart().values()) {
        const l = labelOf(e);
        groups.set(l, (groups.get(l) ?? 0) + 1);
      }
      const items = [...groups.entries()].map(([name, count]) => ({ name, count }));
      return { items, total: getCart().size };
    },
  };
}
