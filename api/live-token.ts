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

// Native-audio Live model. Override via env to try the fallback candidate
// (gemini-3.1-flash-live-preview) without a redeploy.
const LIVE_MODEL =
  process.env.GEMINI_LIVE_MODEL ||
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

  // expireTime: hard ceiling on how long sessions started with this token may
  // run (30 min). newSessionExpireTime: the token must be used to OPEN a
  // session within this window (1 min) — it is single-use (uses: 1) and meant
  // to be handed straight to the client to connect immediately.
  const now = Date.now();
  const expireTime = new Date(now + 30 * 60 * 1000).toISOString();
  const newSessionExpireTime = new Date(now + 60 * 1000).toISOString();

  const tokenConfig: CreateAuthTokenConfig = {
    uses: 1,
    expireTime,
    newSessionExpireTime,
    liveConnectConstraints: {
      model: LIVE_MODEL,
      config: {
        responseModalities: [Modality.AUDIO],
      },
    },
  };

  try {
    const token = await ai.authTokens.create({ config: tokenConfig });
    if (!token.name) {
      res.status(502).json({ error: 'token creation returned no name' });
      return;
    }
    res.status(200).json({ token: token.name, model: LIVE_MODEL, expireTime });
  } catch (e) {
    console.error('live-token error', e);
    const msg = e instanceof Error ? e.message : 'token creation failed';
    res.status(500).json({ error: msg });
  }
}
