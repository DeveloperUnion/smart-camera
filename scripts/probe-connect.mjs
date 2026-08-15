// Both 3.1-flash-live and 2.5-native-audio are visible to the key and support
// bidiGenerateContent, yet the WS to 3.1 closes before opening. So the config
// we send is likely the culprit. Try config variants against 3.1 and capture
// the raw error / close so we see EXACTLY what it rejects.
//
//   node --env-file=.env scripts/probe-connect.mjs

import { GoogleGenAI, Modality } from '@google/genai';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('GEMINI_API_KEY not set');
  process.exit(1);
}

const MODEL = process.env.PROBE_MODEL || 'gemini-3.1-flash-live-preview';

const noopTools = [
  {
    functionDeclarations: [
      { name: 'noop', description: 'no-op', parameters: { type: 'OBJECT', properties: {} } },
    ],
  },
];

const VARIANTS = [
  { label: '1 minimal AUDIO', config: { responseModalities: [Modality.AUDIO] } },
  { label: '2 AUDIO + systemInstruction', config: { responseModalities: [Modality.AUDIO], systemInstruction: 'test' } },
  { label: '3 AUDIO + tools', config: { responseModalities: [Modality.AUDIO], tools: noopTools } },
  { label: '4 AUDIO + outputAudioTranscription', config: { responseModalities: [Modality.AUDIO], outputAudioTranscription: {} } },
  { label: '5 AUDIO + all (our app config)', config: { responseModalities: [Modality.AUDIO], systemInstruction: 'test', tools: noopTools, outputAudioTranscription: {} } },
  { label: '6 TEXT only', config: { responseModalities: [Modality.TEXT] } },
  { label: '7 no config', config: undefined },
];

function attempt({ label, config }) {
  return new Promise((resolve) => {
    const v = { label, opened: false, setup: false, code: null, reason: null, error: null };
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(t);
      resolve(v);
    };
    const t = setTimeout(finish, 6000);

    const ai = new GoogleGenAI({ apiKey, httpOptions: { apiVersion: 'v1alpha' } });
    ai.live
      .connect({
        model: MODEL,
        callbacks: {
          onopen: () => { v.opened = true; },
          onmessage: (m) => { if (m?.setupComplete) { v.setup = true; finish(); } },
          onerror: (e) => { v.error = e?.message || String(e); },
          onclose: (e) => { v.code = e?.code ?? null; v.reason = e?.reason ?? null; finish(); },
        },
        ...(config ? { config } : {}),
      })
      .then((s) => { /* opened; keep waiting for setupComplete */ })
      .catch((err) => { v.error = err?.message || String(err); finish(); });
  });
}

async function main() {
  console.log('probe model:', MODEL, '\n');
  for (const variant of VARIANTS) {
    const v = await attempt(variant);
    const status = v.setup
      ? 'OK setupComplete'
      : v.error
        ? `ERROR: ${v.error}`
        : `CLOSED code=${v.code} reason=${JSON.stringify(v.reason)}`;
    console.log(`[${v.label}] opened=${v.opened} -> ${status}`);
  }
  process.exit(0);
}

main().catch((e) => { console.error('fatal', e); process.exit(1); });
