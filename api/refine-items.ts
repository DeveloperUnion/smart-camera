import type { VercelRequest, VercelResponse } from '@vercel/node';
import { GoogleGenAI, Type } from '@google/genai';

export const config = { maxDuration: 60 };

const apiKey = process.env.GEMINI_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

const MAX_ITEMS = 30;

// Primary/fallback pair mirrors the dustalk ImageModel service: when the
// primary returns 503 UNAVAILABLE we silently fail over to a smaller GA
// model that has spare capacity. Override either via env for experiments.
const PRIMARY_MODEL = process.env.GEMINI_MODEL || 'gemini-3.5-flash';
const FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL || 'gemini-3.1-flash-lite';

// Transient errors from Gemini (mostly 503/UNAVAILABLE, occasionally 429)
// retry-with-backoff first; if every retry on the primary still fails the
// caller below falls over to the fallback model.
function isTransient(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const m = err.message;
  return (
    m.includes('503') ||
    m.includes('UNAVAILABLE') ||
    m.includes('overloaded') ||
    m.includes('429') ||
    m.includes('RESOURCE_EXHAUSTED')
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

type IncomingItem = {
  id: number;
  yolo_label: string;
  image_b64: string;
  image_mime?: string;
  bbox: [number, number, number, number]; // 0-1 normalized
  // How the item was selected. 'tap' (default) → image is a pre-cropped object,
  // identify only. 'voice' → image is a FULL frame; locate the named object and
  // return its box_2d in addition to identifying it (the client then crops).
  source?: 'tap' | 'voice';
};

// Prompt deliberately mirrors the ImageModel waste-disposal pipeline so
// downstream consumers (dustalk-infra, DustalkChat) see the same field
// vocabulary regardless of whether items came from full-image detection
// or per-tap SmartCamera selection. Difference from ImageModel: that
// service detects all objects in one frame and returns box_2d per item;
// here the user has already pointed at one object per image, so we drop
// the box requirement and just identify what was selected.
const PROMPT_PREAMBLE = [
  '以下の画像はユーザーがライブカメラから選択した、捨てる対象になりうる物体です。家具 (机、椅子、棚など)・家電・オフィス機器・生活用品が主な対象。壁・床・天井など建物の構造部分は対象外。',
  '',
  '各画像には 2 種類あります。画像ごとの指示に必ず従ってください:',
  '- 「切り抜き済み」の画像: bbox 領域 (画像上の正規化 xyxy 座標、0-1 スケール) を中心に映る物体を 1 つ同定する。box_2d は返さなくてよい。',
  '- 「フレーム全体」の画像: 画面全体が写っており、ユーザーが指した対象名 (target) が併記されます。その物体を画像内で 1 つ特定し、その位置を box_2d=[ymin, xmin, ymax, xmax] (画像全体を 0〜1000 とする正規化整数) で返し、同時に同定する。',
  '',
  '同定方針:',
  '- 物体の一部 (ボタン配列、受話器、脚など) しか写っていなくても、形状や文脈から確信を持って種類を特定できる場合は同定してください。',
  '- 形状や文脈から自信を持って特定できない場合は、無理に推測せず name を「不明」としてください (誤同定を避ける)。',
  '- name はカテゴリ表記 (例:「複合機」「ビジネスフォン」「ソファ」「冷蔵庫」「ペットボトル」) にとどめ、型番やメーカー名は含めないでください。',
  '- description は物体の説明 (形状・用途の補足) のみ。型番・メーカー・年式・容量などの構造化フィールドに入る情報は description に含めないでください。',
  '- yolo_label は YOLO による粗い分類のヒント。参考にしてよいが、画像から自信を持って別カテゴリと判断できればそちらを優先。',
  '',
  '構造化フィールド (読み取れた場合のみ埋める。読み取れない・確信がない場合は必ず空文字 "" にする。推測禁止):',
  '- modelNumber: 銘板・ラベルから読み取れる製品型番 (例:「RICOH IM C4510」「NR-F500A」)。全カテゴリ対象。',
  '',
  '以下の属性は **無料引取候補カテゴリのみ** 対象。それ以外のカテゴリでは必ず空文字にしてください。',
  '対象カテゴリ: 冷蔵庫 / 冷凍庫 / 洗濯機 / 乾燥機 / エアコン / テレビ / 電子レンジ / オーブンレンジ / 自転車 / 電動アシスト / バイク / スクーター',
  '',
  '- manufacturer: 銘板・ロゴから確信できるメーカー名 (例:「Panasonic」「日立」「シャープ」)。',
  '- yearOfManufacture: 銘板の製造年月などから確信できる年式 (例:「2020」)。型番から年式が一意に決まる場合のみ型番由来でも可。',
  '- capacity: 確信できる容量。単位はカテゴリ別に必ず以下に従う:',
  '  - 冷蔵庫 / 冷凍庫 / 電子レンジ / オーブンレンジ → L (例:「300L」「500L」)',
  '  - 洗濯機 / 乾燥機 → kg (例:「7kg」「9kg」)',
  '  - エアコン → 畳数 (例:「6畳」「10畳」)',
  '  - テレビ → 型 または インチ (例:「32型」「55インチ」)',
  '  - 自転車 / 電動アシスト / バイク / スクーター → 容量は空文字 (メーカー・年式のみ対象)',
  '',
  '返却は items 配列で、items[i].id はリクエストで指定された id を必ず維持すること。',
  '',
  '画像一覧:',
].join('\n');

const RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    items: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.INTEGER },
          box_2d: {
            type: Type.ARRAY,
            items: { type: Type.INTEGER },
            description:
              '「フレーム全体」の画像のときのみ、対象物体の位置 [ymin, xmin, ymax, xmax]（画像全体を 0〜1000 とする正規化整数）。「切り抜き済み」の画像では省略する。',
          },
          refined: {
            type: Type.OBJECT,
            properties: {
              name: {
                type: Type.STRING,
                description:
                  '物体のカテゴリ名 (例:「複合機」「ビジネスフォン」「ソファ」「冷蔵庫」)。型番・メーカー名・容量は含めない。判別不能なら「不明」。',
              },
              description: {
                type: Type.STRING,
                description:
                  '物体の説明文。形状や用途の補足のみ。型番・メーカー・年式・容量はここに含めず、専用フィールドに入れる。',
              },
              modelNumber: {
                type: Type.STRING,
                description:
                  '銘板・ラベルから読み取れる製品型番 (例:「RICOH IM C4510」「NR-F500A」)。読み取れない場合は必ず空文字。',
              },
              manufacturer: {
                type: Type.STRING,
                description:
                  '無料引取候補カテゴリ (冷蔵庫/冷凍庫/洗濯機/乾燥機/エアコン/テレビ/電子レンジ/オーブンレンジ/自転車/電動アシスト/バイク/スクーター) に限り、銘板・ロゴから確信できるメーカー名。それ以外のカテゴリや読み取れない場合は必ず空文字。',
              },
              yearOfManufacture: {
                type: Type.STRING,
                description:
                  '上記対象カテゴリに限り、銘板の製造年月などから確信できる年式 (例:「2020」)。それ以外や読み取れない場合は必ず空文字。',
              },
              capacity: {
                type: Type.STRING,
                description:
                  '上記対象カテゴリに限り、確信できる容量。単位はカテゴリ別に冷蔵庫/冷凍庫/電子レンジ/オーブンレンジ=L、洗濯機/乾燥機=kg、エアコン=畳数、テレビ=型/インチ。自転車・バイク類および対象外カテゴリ、読み取れない場合は必ず空文字。',
              },
            },
            required: ['name'],
            propertyOrdering: [
              'name',
              'description',
              'modelNumber',
              'manufacturer',
              'yearOfManufacture',
              'capacity',
            ],
          },
        },
        required: ['id', 'refined'],
      },
    },
  },
  required: ['items'],
};

function fmtBbox(b: [number, number, number, number]): string {
  return `[${b.map((n) => n.toFixed(3)).join(', ')}]`;
}

// Type for the multimodal `parts` array we send Gemini.
type Part =
  | { text: string }
  | { inlineData: { data: string; mimeType: string } };

async function callGeminiWithFallback(
  parts: Part[],
): Promise<{ text: string; modelUsed: string }> {
  const models = [PRIMARY_MODEL, FALLBACK_MODEL].filter(
    (m, i, a) => m && a.indexOf(m) === i,
  );
  const backoffs = [500, 1500, 4000];
  let lastErr: unknown;

  for (const model of models) {
    for (let attempt = 0; attempt < backoffs.length; attempt++) {
      try {
        const response = await ai!.models.generateContent({
          model,
          contents: [{ role: 'user', parts }],
          config: {
            responseMimeType: 'application/json',
            responseSchema: RESPONSE_SCHEMA,
          },
        });
        if (model !== PRIMARY_MODEL) {
          console.info('refine-items: used fallback model', model);
        }
        return { text: response.text ?? '{"items":[]}', modelUsed: model };
      } catch (err) {
        lastErr = err;
        if (!isTransient(err)) throw err;
        if (attempt === backoffs.length - 1) {
          console.warn(
            `refine-items: ${model} exhausted retries on transient error, trying next model`,
            err instanceof Error ? err.message : err,
          );
          break;
        }
        console.warn(
          `refine-items: ${model} transient error (attempt ${attempt + 1}), retrying in ${backoffs[attempt]}ms`,
          err instanceof Error ? err.message : err,
        );
        await sleep(backoffs[attempt]);
      }
    }
  }
  throw lastErr ?? new Error('all models failed');
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'POST only' });
    return;
  }
  if (!ai) {
    res.status(500).json({ error: 'GEMINI_API_KEY not configured' });
    return;
  }

  const { items } = (req.body ?? {}) as { items?: IncomingItem[] };
  if (!Array.isArray(items) || items.length === 0) {
    res.status(400).json({ error: 'items[] required' });
    return;
  }
  if (items.length > MAX_ITEMS) {
    res.status(400).json({ error: `too many items (>${MAX_ITEMS})` });
    return;
  }
  for (const it of items) {
    if (
      typeof it.id !== 'number' ||
      typeof it.yolo_label !== 'string' ||
      typeof it.image_b64 !== 'string' ||
      !Array.isArray(it.bbox) ||
      it.bbox.length !== 4
    ) {
      res.status(400).json({ error: 'invalid item shape' });
      return;
    }
  }

  // Build a single multimodal turn: one preamble text part, then alternating
  // [text describing item], [inline image] pairs so the model can ground
  // each image to its id + yolo_label + bbox without relying on positional
  // order alone.
  const parts: Part[] = [{ text: PROMPT_PREAMBLE }];
  for (const it of items) {
    const desc =
      it.source === 'voice'
        ? `id=${it.id}, フレーム全体, target="${it.yolo_label}" — この物体を特定して box_2d を返し同定する`
        : `id=${it.id}, 切り抜き済み, yolo_label="${it.yolo_label}", bbox=${fmtBbox(it.bbox)}`;
    parts.push({ text: desc });
    parts.push({
      inlineData: {
        data: it.image_b64,
        mimeType: it.image_mime ?? 'image/jpeg',
      },
    });
  }

  try {
    const { text } = await callGeminiWithFallback(parts);

    const parsed = JSON.parse(text) as {
      items?: Array<{
        id: number;
        box_2d?: unknown;
        refined?: {
          name?: string;
          description?: string;
          modelNumber?: string;
          manufacturer?: string;
          yearOfManufacture?: string;
          capacity?: string;
        };
      }>;
    };

    const out = (parsed.items ?? [])
      .filter(
        (r) =>
          typeof r.id === 'number' &&
          r.refined &&
          typeof r.refined.name === 'string' &&
          r.refined.name.length > 0,
      )
      .map((r) => {
        const refined = r.refined!;
        // Strip empty strings so the client only renders non-empty fields,
        // matching the spec's "省略可能 if 空文字" intent.
        const trimmed: Record<string, string> = { name: refined.name! };
        for (const k of [
          'description',
          'modelNumber',
          'manufacturer',
          'yearOfManufacture',
          'capacity',
        ] as const) {
          const v = refined[k];
          if (typeof v === 'string' && v.length > 0) trimmed[k] = v;
        }
        // Pass through box_2d only when it's a clean 4-number array (voice
        // items). The client crops the full frame with it; tap items omit it.
        const box = r.box_2d;
        const box_2d =
          Array.isArray(box) &&
          box.length === 4 &&
          box.every((n) => typeof n === 'number' && Number.isFinite(n))
            ? (box as [number, number, number, number])
            : undefined;
        return box_2d
          ? { id: r.id, refined: trimmed, box_2d }
          : { id: r.id, refined: trimmed };
      });

    res.status(200).json({ items: out });
  } catch (e) {
    console.error('refine-items error', e);
    const msg = e instanceof Error ? e.message : 'inference failed';
    const status = isTransient(e) ? 503 : 500;
    res.status(status).json({ error: msg });
  }
}
