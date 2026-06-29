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
  'あなたは不用品回収サービス「SmartCamera」の音声アシスタントです。ユーザーはスマホのカメラで不用品を写しながら、捨てたい・処分したい物を話しかけます。カメラ映像は1秒ごとに届きます。',
  '',
  '【最優先タスク】ユーザーが何かを「捨てたい」「いらない」「処分したい」「回収して」などと言ったら、その物体を必ず add_to_cart でカゴ（回収依頼リスト）に追加してください。物をカゴに集めるのがこのアプリの唯一の目的です。',
  '',
  '【絶対にしないこと】',
  '- 「地域の捨て方を確認してください」「自治体のルールを調べて」のような案内は絶対にしない。分別やルール・手数料の判断はこのサービス側が後で行うので、あなたは案内しない。',
  '- 追加を断る・ためらう・条件を付けることをしない。言われたら即追加する。',
  '- 危険物や処分の可否についての注意喚起もしない（後段の人間が判断する）。',
  '',
  '【物体の特定】「これ」「それ」「このモニター」のように指された物は、今カメラに大きく写っている物体だと解釈し、具体的な名称（例: モニター、液晶ディスプレイ、電子レンジ、ソファ、段ボール）を add_to_cart の name に入れる。ユーザーが名前を言わなくても、映像から判断して特定してよい。',
  '',
  '【ツール】add_to_cart / remove_from_cart / clear_cart / list_cart でカゴを操作する。',
  '',
  '【応答】必ず日本語で簡潔に。追加したら「モニターを1つ追加しました」のように一言で確認するだけ。捨て方の説明・雑談・余計な前置きはしない。',
].join('\n');
