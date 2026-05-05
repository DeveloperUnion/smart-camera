import type { VercelRequest, VercelResponse } from '@vercel/node';
import { GoogleGenAI, Type } from '@google/genai';

export const config = { maxDuration: 60 };

const apiKey = process.env.GEMINI_API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

const PROMPT = [
  'この動画に登場する具体的な物体を **網羅的に** 検出してください。視認できる物体は積極的に列挙すること。',
  '出力ルール:',
  '- 同じ物体 (同一個体) は動画全体を通して同一の instance_id (整数、1 から連番) で識別する。',
  '- 別個体の同種物体 (例: 別のコーラ缶 2 本) は別の instance_id を割り当てる。',
  '- ラベルは自然で短い日本語の名詞句 (例: 「コーラ缶」「ノートパソコン」「リュック」「マグカップ」)。',
  '- appearances には物体が映っているフレームから複数件、box_2d ([ymin, xmin, ymax, xmax] を 0-1000 の整数で正規化) と time_s (動画先頭からの秒数) を含める。',
  '  movement のある物体は 3-5 件、静止物体でも 2 件以上。',
  '- 短時間 (1 秒未満) でも映ったものは出力対象。',
  '- 人物の顔・手・体の部位、および床・壁・天井などの構造物は除外。',
  '- 単純に "物体" と書かず、具体的な名詞 (商品名・道具名など) を可能な限り使う。',
].join('\n');

const RESPONSE_SCHEMA = {
  type: Type.ARRAY,
  items: {
    type: Type.OBJECT,
    properties: {
      instance_id: { type: Type.INTEGER },
      label: { type: Type.STRING },
      appearances: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            time_s: { type: Type.NUMBER },
            box_2d: {
              type: Type.ARRAY,
              items: { type: Type.INTEGER },
            },
          },
          required: ['time_s', 'box_2d'],
        },
      },
    },
    required: ['instance_id', 'label', 'appearances'],
  },
};

type RawAppearance = { time_s: number; box_2d: [number, number, number, number] };
type RawDetection = {
  instance_id: number;
  label: string;
  appearances: RawAppearance[];
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'POST only' });
    return;
  }
  if (!ai) {
    res.status(500).json({ error: 'GEMINI_API_KEY not configured' });
    return;
  }

  const { video, mimeType } = (req.body ?? {}) as {
    video?: string;
    mimeType?: string;
  };
  if (!video) {
    res.status(400).json({ error: 'video (base64) required' });
    return;
  }

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: [
        {
          role: 'user',
          parts: [
            {
              inlineData: { data: video, mimeType: mimeType ?? 'video/mp4' },
              // Default video sampling is 1 fps; bump for richer temporal
              // coverage (catches transient/quickly-moving objects).
              videoMetadata: { fps: 4 },
            },
            { text: PROMPT },
          ],
        },
      ],
      config: {
        responseMimeType: 'application/json',
        responseSchema: RESPONSE_SCHEMA,
        // No thinkingConfig — Gemini 3's agentic vision needs thinking budget
        // to do precise localization. Use the model's default (auto).
      },
    });

    const text = response.text ?? '[]';
    const items = JSON.parse(text) as RawDetection[];
    const detections = items
      .filter(
        (it) =>
          Array.isArray(it.appearances) &&
          it.appearances.length > 0 &&
          typeof it.label === 'string',
      )
      .map((it) => ({
        instance_id: it.instance_id,
        label: it.label,
        appearances: it.appearances
          .filter(
            (a) =>
              Array.isArray(a.box_2d) &&
              a.box_2d.length === 4 &&
              typeof a.time_s === 'number',
          )
          .map((a) => {
            const [ymin, xmin, ymax, xmax] = a.box_2d;
            return {
              time_s: a.time_s,
              bbox: [
                xmin / 1000,
                ymin / 1000,
                xmax / 1000,
                ymax / 1000,
              ] as [number, number, number, number],
            };
          })
          .filter((a) => a.bbox[2] > a.bbox[0] && a.bbox[3] > a.bbox[1]),
      }))
      .filter((d) => d.appearances.length > 0);

    res.status(200).json({ detections });
  } catch (e) {
    console.error('detect-video error', e);
    const msg = e instanceof Error ? e.message : 'inference failed';
    res.status(500).json({ error: msg });
  }
}
