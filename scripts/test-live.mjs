// Live API connect diagnostics. Run with:
//   node --env-file=.env scripts/test-live.mjs
//
// Isolates whether the "cannot use project-scoped features such as tuned
// models" disconnect is about (a) the model/config, (b) ephemeral tokens at
// all, or (c) leaving the model unlocked in the token. Tries several auth
// shapes against the same model and reports, for each, whether the session
// reached setupComplete or closed early (with code/reason).

import { GoogleGenAI, Modality } from '@google/genai';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('GEMINI_API_KEY not set. Create .env with GEMINI_API_KEY=...');
  process.exit(1);
}

const MODEL =
  process.env.GEMINI_LIVE_MODEL || 'gemini-3.1-flash-live-preview';
const FALLBACK =
  process.env.GEMINI_LIVE_FALLBACK_MODEL ||
  'gemini-2.5-flash-native-audio-preview-12-2025';

const v1alpha = { apiVersion: 'v1alpha' };

// Connect and resolve a verdict after either setupComplete, an early close, or
// a timeout. We never send audio — we only care if the session opens cleanly.
function tryConnect({ label, apiKey, model }) {
  return new Promise((resolve) => {
    const verdict = {
      label,
      model,
      opened: false,
      setupComplete: false,
      firstMessageKeys: null,
      closeCode: null,
      closeReason: null,
      error: null,
    };
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try {
        session?.then?.((s) => s.close());
      } catch {}
      resolve(verdict);
    };
    const timer = setTimeout(finish, 8000);

    const ai = new GoogleGenAI({ apiKey, httpOptions: v1alpha });
    let session;
    session = ai.live
      .connect({
        model,
        callbacks: {
          onopen: () => {
            verdict.opened = true;
          },
          onmessage: (msg) => {
            if (verdict.firstMessageKeys === null) {
              verdict.firstMessageKeys = Object.keys(msg ?? {}).filter(
                (k) => msg[k] !== undefined,
              );
            }
            if (msg?.setupComplete) {
              verdict.setupComplete = true;
              finish();
            }
          },
          onerror: (e) => {
            verdict.error = e?.message || String(e);
          },
          onclose: (e) => {
            verdict.closeCode = e?.code ?? null;
            verdict.closeReason = e?.reason ?? null;
            finish();
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: 'You are a test.',
          tools: [
            {
              functionDeclarations: [
                {
                  name: 'noop',
                  description: 'no-op',
                  parameters: { type: 'OBJECT', properties: {} },
                },
              ],
            },
          ],
          outputAudioTranscription: {},
        },
      })
      .catch((err) => {
        verdict.error = err?.message || String(err);
        finish();
      });
  });
}

async function mintToken({ model, lockModel }) {
  const ai = new GoogleGenAI({ apiKey, httpOptions: v1alpha });
  const now = Date.now();
  const config = {
    uses: 1,
    expireTime: new Date(now + 30 * 60 * 1000).toISOString(),
    newSessionExpireTime: new Date(now + 60 * 1000).toISOString(),
    liveConnectConstraints: {
      ...(lockModel ? { model } : {}),
      config: { responseModalities: [Modality.AUDIO] },
    },
  };
  const t = await ai.authTokens.create({ config });
  return t.name;
}

function report(v) {
  const status = v.setupComplete
    ? 'OK (setupComplete)'
    : v.error
      ? `ERROR: ${v.error}`
      : `CLOSED early code=${v.closeCode} reason=${v.closeReason}`;
  console.log(`\n[${v.label}] model=${v.model}`);
  console.log(`  result: ${status}`);
  console.log(
    `  opened=${v.opened} firstMessageKeys=${JSON.stringify(v.firstMessageKeys)}`,
  );
}

async function main() {
  console.log('genai live diagnostics');
  console.log('primary model :', MODEL);
  console.log('fallback model:', FALLBACK);

  // 1) Direct API key + primary model — isolates model/config validity.
  report(
    await tryConnect({ label: 'A: direct apiKey', apiKey, model: MODEL }),
  );

  // 2) Ephemeral token LOCKED to the model (our current endpoint).
  try {
    const locked = await mintToken({ model: MODEL, lockModel: true });
    report(
      await tryConnect({
        label: 'B: token (model locked)',
        apiKey: locked,
        model: MODEL,
      }),
    );
  } catch (e) {
    console.log(`\n[B: token (model locked)] mint failed: ${e?.message || e}`);
  }

  // 3) Ephemeral token UNLOCKED (reproduce the suspected bug).
  try {
    const unlocked = await mintToken({ model: MODEL, lockModel: false });
    report(
      await tryConnect({
        label: 'C: token (model UNLOCKED)',
        apiKey: unlocked,
        model: MODEL,
      }),
    );
  } catch (e) {
    console.log(`\n[C: token (model UNLOCKED)] mint failed: ${e?.message || e}`);
  }

  // 4) Direct API key + fallback model.
  report(
    await tryConnect({
      label: 'D: direct apiKey fallback',
      apiKey,
      model: FALLBACK,
    }),
  );

  process.exit(0);
}

main().catch((e) => {
  console.error('fatal', e);
  process.exit(1);
});
