// Fix verification: bake systemInstruction + tools INTO the ephemeral token at
// mint time, connect with NO client config, and check the tool gets called.
//
//   node --env-file=.env scripts/test-baked.mjs

import { GoogleGenAI, Modality } from '@google/genai';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) { console.error('GEMINI_API_KEY not set'); process.exit(1); }
const MODEL = process.env.GEMINI_LIVE_MODEL || 'gemini-3.1-flash-live-preview';

const SYSTEM =
  'You are a test bot. No matter what the user says, you MUST immediately call the function named "mark". Then say only "DONE".';
const TOOLS = [
  { functionDeclarations: [{ name: 'mark', description: 'mark a test', parameters: { type: 'OBJECT', properties: {} } }] },
];
const BAKED_CONFIG = {
  responseModalities: [Modality.AUDIO],
  systemInstruction: SYSTEM,
  tools: TOOLS,
  outputAudioTranscription: {},
  temperature: 0.2,
};

async function mintToken() {
  const ai = new GoogleGenAI({ apiKey, httpOptions: { apiVersion: 'v1alpha' } });
  const now = Date.now();
  const t = await ai.authTokens.create({
    config: {
      uses: 1,
      expireTime: new Date(now + 30 * 60 * 1000).toISOString(),
      newSessionExpireTime: new Date(now + 60 * 1000).toISOString(),
      liveConnectConstraints: { model: MODEL, config: BAKED_CONFIG },
    },
  });
  return t.name;
}

function run(authKey, clientConfig, label) {
  return new Promise((resolve) => {
    const v = { label, toolCalled: false, names: [], transcript: '', error: null, closed: null };
    let done = false;
    const finish = () => { if (done) return; done = true; clearTimeout(timer); resolve(v); };
    const timer = setTimeout(finish, 12000);
    const ai = new GoogleGenAI({ apiKey: authKey, httpOptions: { apiVersion: 'v1alpha' } });
    ai.live.connect({
      model: MODEL,
      callbacks: {
        onmessage: (m) => {
          const calls = m?.toolCall?.functionCalls;
          if (calls?.length) { v.toolCalled = true; v.names.push(...calls.map((c) => c.name)); }
          const ot = m?.serverContent?.outputTranscription?.text;
          if (ot) v.transcript += ot;
          if (m?.serverContent?.turnComplete) finish();
        },
        onerror: (e) => { v.error = e?.message || String(e); },
        onclose: (e) => { v.closed = `code=${e?.code} reason=${e?.reason}`; finish(); },
      },
      ...(clientConfig ? { config: clientConfig } : {}),
    })
      .then((s) => s.sendClientContent({ turns: 'hello', turnComplete: true }))
      .catch((err) => { v.error = err?.message || String(err); finish(); });
  });
}

function report(v) {
  console.log(`\n[${v.label}] toolCalled=${v.toolCalled} names=${JSON.stringify(v.names)} transcript=${JSON.stringify(v.transcript)}`);
  if (v.error) console.log('  error:', v.error);
  if (v.closed) console.log('  closed:', v.closed);
}

async function main() {
  console.log('model:', MODEL, '\nExpect tool "mark" called if baked config is honored.');
  const token = await mintToken();
  // Token carries the full config; client passes none.
  report(await run(token, undefined, 'token baked + client no config'));
  // Also try client re-passing the same config (in case connect requires it).
  const token2 = await mintToken();
  report(await run(token2, BAKED_CONFIG, 'token baked + client same config'));
  process.exit(0);
}

main().catch((e) => { console.error('fatal', e); process.exit(1); });
