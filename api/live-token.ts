import type { VercelRequest, VercelResponse } from '@vercel/node';
import {
  GoogleGenAI,
  Modality,
  type CreateAuthTokenConfig,
} from '@google/genai';

export const config = { maxDuration: 15 };

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

  const mint = async (model: string): Promise<string> => {
    const tokenConfig: CreateAuthTokenConfig = {
      uses: 1,
      expireTime,
      newSessionExpireTime,
      liveConnectConstraints: {
        model,
        config: { responseModalities: [Modality.AUDIO] },
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
