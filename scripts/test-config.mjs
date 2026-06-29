// Does the systemInstruction + tools we pass at connect actually take effect
// when using an EPHEMERAL TOKEN (which locks responseModalities)? Or does the
// token strip our client config?
//
// Test: give a system instruction that forces a tool call no matter what the
// user says, send "hello", and see if the `mark` tool gets called and what the
// model says. Run for both direct-apiKey and token auth and compare.
//
//   node --env-file=.env scripts/test-config.mjs

import { GoogleGenAI, Modality } from '@google/genai';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('GEMINI_API_KEY not set');
  process.exit(1);
}
const MODEL = process.env.GEMINI_LIVE_MODEL || 'gemini-3.1-flash-live-preview';

const SYSTEM =
  'You are a test bot. No matter what the user says, you MUST immediately call the function named "mark". Always call it. Then say only the word "DONE".';
const TOOLS = [
  {
    functionDeclarations: [
      { name: 'mark', description: 'mark a test', parameters: { type: 'OBJECT', properties: {} } },
    ],
  },
];

async function mintToken() {
  const ai = new GoogleGenAI({ apiKey, httpOptions: { apiVersion: 'v1alpha' } });
  const now = Date.now();
  const t = await ai.authTokens.create({
    config: {
      uses: 1,
      expireTime: new Date(now + 30 * 60 * 1000).toISOString(),
      newSessionExpireTime: new Date(now + 60 * 1000).toISOString(),
      liveConnectConstraints: {
        model: MODEL,
        config: { responseModalities: [Modality.AUDIO] },
      },
    },
  });
  return t.name;
}

function run({ label, authKey }) {
  return new Promise((resolve) => {
    const v = { label, toolCalled: false, toolNames: [], transcript: '', closed: null, error: null };
    let done = false;
    const finish = () => { if (done) return; done = true; clearTimeout(timer); resolve(v); };
    const timer = setTimeout(finish, 12000);

    const ai = new GoogleGenAI({ apiKey: authKey, httpOptions: { apiVersion: 'v1alpha' } });
    ai.live
      .connect({
        model: MODEL,
        callbacks: {
          onopen: () => {},
          onmessage: (m) => {
            const calls = m?.toolCall?.functionCalls;
            if (calls?.length) {
              v.toolCalled = true;
              v.toolNames.push(...calls.map((c) => c.name));
            }
            const ot = m?.serverContent?.outputTranscription?.text;
            if (ot) v.transcript += ot;
            if (m?.serverContent?.turnComplete) finish();
          },
          onerror: (e) => { v.error = e?.message || String(e); },
          onclose: (e) => { v.closed = `code=${e?.code} reason=${e?.reason}`; finish(); },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM,
          tools: TOOLS,
          outputAudioTranscription: {},
          temperature: 0.2,
        },
      })
      .then((session) => {
        // Send a user turn that, per the system prompt, must trigger the tool.
        session.sendClientContent({ turns: 'hello', turnComplete: true });
      })
      .catch((err) => { v.error = err?.message || String(err); finish(); });
  });
}

function report(v) {
  console.log(`\n[${v.label}]`);
  console.log(`  tool called : ${v.toolCalled}  names=${JSON.stringify(v.toolNames)}`);
  console.log(`  transcript  : ${JSON.stringify(v.transcript)}`);
  if (v.error) console.log(`  error       : ${v.error}`);
  if (v.closed) console.log(`  closed      : ${v.closed}`);
}

async function main() {
  console.log('model:', MODEL);
  console.log('Expect: if config is honored, tool "mark" is called and transcript ~ "DONE".');

  report(await run({ label: 'direct apiKey', authKey: apiKey }));

  try {
    const token = await mintToken();
    report(await run({ label: 'ephemeral token', authKey: token }));
  } catch (e) {
    console.log('\n[ephemeral token] mint failed:', e?.message || e);
  }
  process.exit(0);
}

main().catch((e) => { console.error('fatal', e); process.exit(1); });
