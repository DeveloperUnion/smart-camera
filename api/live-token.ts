import type { VercelRequest, VercelResponse } from '@vercel/node';
import {
  GoogleGenAI,
  Modality,
  Type,
  type CreateAuthTokenConfig,
  type FunctionDeclaration,
} from '@google/genai';

export const config = { maxDuration: 15 };

// Talk-mode session config. It is INLINED here (not imported from src/) on
// purpose: Vercel serverless functions don't bundle cross-directory project
// imports, so importing ../src/* throws ERR_MODULE_NOT_FOUND at runtime. It
// also has to live server-side because client-supplied systemInstruction/tools
// are IGNORED when connecting with an ephemeral token — only what's baked into
// the token below takes effect. The browser only needs the matching tool-call
// handlers (src/live/cartTools.ts), not these declarations.
const TALK_TEMPERATURE = 0.2;

const CART_TOOLS: FunctionDeclaration[] = [
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
        count: { type: Type.INTEGER, description: '追加する個数。省略時は1。' },
        note: { type: Type.STRING, description: '補足メモ（任意）' },
        position: {
          type: Type.STRING,
          description: '置き場所など（任意。例: 棚の上）',
        },
        box_2d: {
          type: Type.ARRAY,
          items: { type: Type.INTEGER },
          description:
            '対象物体のバウンディングボックス [ymin, xmin, ymax, xmax]。' +
            '直近に届いたカメラフレーム全体を 0〜1000 とする正規化整数座標。' +
            '物体が映像に見えている場合は必ず指定する。count が 2 以上のときは代表 1 個の box でよい。',
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

const TALK_SYSTEM_INSTRUCTION = [
  'あなたは「dustalk」の動画アシスタントです。ユーザーはスマホのカメラで不用品を写しながら、捨てたい・処分したい物を話しかけます。カメラ映像は1秒ごとに届きます。',
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
  '【box_2d】add_to_cart を呼ぶときは、対象物体が直近のカメラフレームに写っている限り、必ず box_2d を [ymin, xmin, ymax, xmax]（フレーム全体を 0〜1000 とする正規化整数）で指定する。映像に見えない物についてのみ省略してよい。',
  '',
  '【ツール】add_to_cart / remove_from_cart / clear_cart / list_cart でカゴを操作する。',
  '',
  '【会話例】',
  'ユーザー:「このモニター捨てたい」→ ただちに add_to_cart(name:"モニター", box_2d:[180,220,760,820]) を呼び（box_2d は映像内のモニターの実際の位置）、「モニターを1つ追加しました」とだけ言う。',
  'ユーザー:「ペットボトル3本も」→ add_to_cart(name:"ペットボトル", count:3) を呼び、「ペットボトルを3つ追加しました」。',
  'ユーザー:「さっきのモニターやめる」→ remove_from_cart(name:"モニター") を呼ぶ。',
  'ユーザー:「今どれくらいある？」→ list_cart を呼んで結果を読み上げる。',
  '',
  '【応答】必ず日本語で簡潔に。追加したら「モニターを1つ追加しました」のように一言で確認するだけ。',
  '「モニターの捨て方ですね」「どう処分しますか」のような前置き・聞き返し・捨て方の説明は一切禁止。捨てたいと言われたら理由も方法も聞かず、即 add_to_cart を呼ぶこと。',
].join('\n');

// Ephemeral tokens are only supported on the v1alpha surface, so this client
// is pinned to it. The full API key never leaves the server: the browser only
// ever receives the short-lived token returned below.
const apiKey = process.env.GEMINI_API_KEY;
const ai = apiKey
  ? new GoogleGenAI({ apiKey, httpOptions: { apiVersion: 'v1alpha' } })
  : null;

// Primary/fallback Live models, mirroring the refine-items endpoint. The
// client tries the primary first and falls back on connect failure. Override
// either via env without a redeploy.
const LIVE_MODEL =
  process.env.GEMINI_LIVE_MODEL || 'gemini-3.1-flash-live-preview';
const LIVE_FALLBACK_MODEL =
  process.env.GEMINI_LIVE_FALLBACK_MODEL ||
  'gemini-2.5-flash-native-audio-preview-12-2025';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'POST only' });
    return;
  }
  if (!ai) {
    res.status(500).json({ error: 'GEMINI_API_KEY not configured' });
    return;
  }

  // expireTime: hard ceiling on how long sessions started with a token may run
  // (30 min). newSessionExpireTime: the token must be used to OPEN a session
  // within this window (1 min) and handed straight to the client. Each token is
  // single-use and LOCKED to one model — leaving the model unlocked makes the
  // service resolve the connect-time model as a project-scoped tuned model,
  // which API/token auth rejects. For fallback we therefore mint a separate
  // token per model and let the client try them in order.
  const now = Date.now();
  const expireTime = new Date(now + 30 * 60 * 1000).toISOString();
  const newSessionExpireTime = new Date(now + 60 * 1000).toISOString();

  // Bake the FULL session config into each token (see note above the config).
  const mint = async (model: string): Promise<string> => {
    const tokenConfig: CreateAuthTokenConfig = {
      uses: 1,
      expireTime,
      newSessionExpireTime,
      liveConnectConstraints: {
        model,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: TALK_SYSTEM_INSTRUCTION,
          tools: [{ functionDeclarations: CART_TOOLS }],
          outputAudioTranscription: {},
          temperature: TALK_TEMPERATURE,
        },
      },
    };
    const token = await ai.authTokens.create({ config: tokenConfig });
    if (!token.name) throw new Error('token creation returned no name');
    return token.name;
  };

  try {
    const primary = { token: await mint(LIVE_MODEL), model: LIVE_MODEL };

    // Fallback token is best-effort: if minting it fails (e.g. the fallback
    // model is unavailable) the primary still works on its own.
    let fallback: { token: string; model: string } | null = null;
    if (LIVE_FALLBACK_MODEL && LIVE_FALLBACK_MODEL !== LIVE_MODEL) {
      try {
        fallback = {
          token: await mint(LIVE_FALLBACK_MODEL),
          model: LIVE_FALLBACK_MODEL,
        };
      } catch (e) {
        console.warn('live-token: fallback mint failed', e);
      }
    }

    res.status(200).json({ primary, fallback, expireTime });
  } catch (e) {
    console.error('live-token error', e);
    const msg = e instanceof Error ? e.message : 'token creation failed';
    res.status(500).json({ error: msg });
  }
}
