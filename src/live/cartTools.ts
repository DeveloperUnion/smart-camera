import { Type } from '@google/genai';
import type { FunctionDeclaration } from '@google/genai';
import type { CartEntry } from '../types';
import type { ToolHandler } from './useLiveSession';

// Voice-mode cart tools. The model calls these to add/remove/clear/list the
// "捨てる物カゴ" while the user talks. Declarations and handlers are kept here so
// App only wires up state access, not the per-tool argument parsing.

export const cartToolDeclarations: FunctionDeclaration[] = [
  {
    name: 'add_to_cart',
    description:
      'カゴ（捨てる物リスト）に物体を追加する。ユーザーが「○○を追加して」「これも入れて」などと言ったとき。',
    parameters: {
      type: Type.OBJECT,
      properties: {
        name: {
          type: Type.STRING,
          description: '物体の名称（例: ペットボトル, 冷蔵庫, 段ボール）',
        },
        count: {
          type: Type.INTEGER,
          description: '追加する個数。省略時は1。',
        },
        note: { type: Type.STRING, description: '補足メモ（任意）' },
        position: {
          type: Type.STRING,
          description: '置き場所など（任意。例: 棚の上）',
        },
      },
      required: ['name'],
    },
  },
  {
    name: 'remove_from_cart',
    description: 'カゴから指定した名称の物体を削除する。',
    parameters: {
      type: Type.OBJECT,
      properties: {
        name: { type: Type.STRING, description: '削除する物体の名称' },
        count: {
          type: Type.INTEGER,
          description: '削除する個数。省略時はその名称をすべて削除。',
        },
      },
      required: ['name'],
    },
  },
  {
    name: 'clear_cart',
    description: 'カゴを空にする。',
    parameters: { type: Type.OBJECT, properties: {} },
  },
  {
    name: 'list_cart',
    description: '現在のカゴの中身（名称と個数）を確認する。',
    parameters: { type: Type.OBJECT, properties: {} },
  },
];

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

export const TALK_SYSTEM_INSTRUCTION = [
  'あなたは廃棄物回収アプリ「SmartCamera」の音声アシスタントです。',
  'ユーザーはスマホのカメラで部屋を写しながら、捨てたい物をあなたに話しかけます。カメラ映像も1秒ごとに届きます。',
  'ユーザーの依頼に応じて add_to_cart / remove_from_cart / clear_cart / list_cart を呼び、カゴ（捨てる物リスト）を操作してください。',
  '応答は必ず日本語で、簡潔に。操作したら「ペットボトルを1つ追加しました」のように短く確認してください。',
  '映像に写っている物について聞かれたら答えてかまいませんが、雑談は最小限にしてください。',
  '「これ」「それ」と指される物は、今カメラに大きく写っている物体だと解釈してください。',
].join('\n');
