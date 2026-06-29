import type { CartEntry } from '../types';
import type { ToolHandler } from './useLiveSession';

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
};

export function createCartHandlers({
  getCart,
  setCart,
  nextVoiceId,
  max,
}: Deps): Record<string, ToolHandler> {
  return {
    add_to_cart: (args) => {
      const name = String(args.name ?? '').trim() || '物体';
      const count = Math.max(1, Math.floor(Number(args.count) || 1));
      const note = args.note ? String(args.note) : undefined;
      const position = args.position ? String(args.position) : undefined;

      const cur = getCart();
      const room = Math.max(0, max - cur.size);
      const toAdd = Math.min(count, room);

      const newEntries: CartEntry[] = [];
      for (let i = 0; i < toAdd; i++) {
        newEntries.push({
          instance_id: nextVoiceId(),
          yolo_label: name,
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
