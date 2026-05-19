import type { VercelRequest, VercelResponse } from '@vercel/node';
import { GoogleGenAI, Type } from '@google/genai';

export const config = { maxDuration: 60 };

const apiKey = process.env.GEMINI_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

const MAX_ITEMS = 30;

type IncomingItem = {
  id: number;
  yolo_label: string;
  image_b64: string;
  image_mime?: string;
  bbox: [number, number, number, number]; // 0-1 normalized
};

const PROMPT_PREAMBLE = [
  '以下の画像はユーザーがライブカメラから選択した物体です。各画像で bbox 領域 (画像上の正規化座標) に映る物体を、可能な限り具体的に同定してください。',
  '',
  '判定ガイド:',
  '- bbox は物体のおおよその位置のヒント。画像全体を見て、最も自然な解釈を選ぶこと。',
  '- yolo_label は YOLO による粗い分類のヒント (例: 「缶」「ボトル」「マグカップ」)。参考にしてよいが、より具体的に特定できればそちらを優先。',
  '- specific_name は短い日本語の名詞句で、商品名・型番・材質などが分かれば含める (例: 「コカコーラ缶 350ml」「無印良品 アクリルマグ 白」「青色プラスチック製ボールペン」)。',
  '- brand はロゴ等から確信を持って読み取れる場合のみ。曖昧なら省略。',
  '- category は日常用途の大分類 (例: 飲料、文房具、食器、衣類、電化製品)。',
  '- color は最も支配的な色を簡潔に。',
  '- size_estimate は画像内の他物体や手のスケール等から推定 (例: 「直径6cm程度」「A4サイズ」)。確信が低ければ省略。',
  '- 確信が持てない属性は省略。specific_name だけは必ず埋める (どうしても判らなければ yolo_label をそのまま日本語名詞句として返す)。',
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
          refined: {
            type: Type.OBJECT,
            properties: {
              specific_name: { type: Type.STRING },
              brand: { type: Type.STRING },
              category: { type: Type.STRING },
              color: { type: Type.STRING },
              size_estimate: { type: Type.STRING },
            },
            required: ['specific_name'],
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
  // [text describing item], [inline image] pairs so the model can ground each
  // image to its id + yolo_label + bbox without relying on positional order
  // alone.
  const parts: Array<
    | { text: string }
    | { inlineData: { data: string; mimeType: string } }
  > = [{ text: PROMPT_PREAMBLE }];
  for (const it of items) {
    parts.push({
      text: `id=${it.id}, yolo_label="${it.yolo_label}", bbox=${fmtBbox(it.bbox)}`,
    });
    parts.push({
      inlineData: {
        data: it.image_b64,
        mimeType: it.image_mime ?? 'image/jpeg',
      },
    });
  }

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: [{ role: 'user', parts }],
      config: {
        responseMimeType: 'application/json',
        responseSchema: RESPONSE_SCHEMA,
      },
    });

    const text = response.text ?? '{"items":[]}';
    const parsed = JSON.parse(text) as {
      items?: Array<{
        id: number;
        refined?: {
          specific_name?: string;
          brand?: string;
          category?: string;
          color?: string;
          size_estimate?: string;
        };
      }>;
    };

    const out = (parsed.items ?? [])
      .filter(
        (r) =>
          typeof r.id === 'number' &&
          r.refined &&
          typeof r.refined.specific_name === 'string' &&
          r.refined.specific_name.length > 0,
      )
      .map((r) => ({
        id: r.id,
        refined: {
          specific_name: r.refined!.specific_name!,
          brand: r.refined!.brand,
          category: r.refined!.category,
          color: r.refined!.color,
          size_estimate: r.refined!.size_estimate,
        },
      }));

    res.status(200).json({ items: out });
  } catch (e) {
    console.error('refine-items error', e);
    const msg = e instanceof Error ? e.message : 'inference failed';
    res.status(500).json({ error: msg });
  }
}
