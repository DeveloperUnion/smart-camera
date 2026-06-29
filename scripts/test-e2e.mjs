// End-to-end: mint a token with the REAL prompt + add_to_cart tool baked in,
// say "このパソコン捨てたい", and confirm add_to_cart is called (not a lecture).
//   node --env-file=.env scripts/test-e2e.mjs

import { GoogleGenAI, Modality, Type } from '@google/genai';

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) { console.error('GEMINI_API_KEY not set'); process.exit(1); }
const MODEL = process.env.GEMINI_LIVE_MODEL || 'gemini-3.1-flash-live-preview';

const SYSTEM = [
  'あなたは不用品回収サービス「SmartCamera」の音声アシスタントです。捨てたい物を話しかけられます。',
  '【最優先】ユーザーが「捨てたい」「いらない」「処分したい」と言ったら、必ず add_to_cart でカゴに追加する。',
  '【禁止】「地域の捨て方を確認して」等の案内、聞き返し、「〜の捨て方ですね」の前置きは一切しない。即追加する。',
  '追加したら「○○を1つ追加しました」と一言だけ。',
].join('\n');

const TOOLS = [{
  functionDeclarations: [{
    name: 'add_to_cart',
    description: 'カゴに物体を追加する',
    parameters: { type: Type.OBJECT, properties: { name: { type: Type.STRING }, count: { type: Type.INTEGER } }, required: ['name'] },
  }],
}];

async function mint() {
  const ai = new GoogleGenAI({ apiKey, httpOptions: { apiVersion: 'v1alpha' } });
  const now = Date.now();
  const t = await ai.authTokens.create({
    config: {
      uses: 1,
      expireTime: new Date(now + 30 * 60 * 1000).toISOString(),
      newSessionExpireTime: new Date(now + 60 * 1000).toISOString(),
      liveConnectConstraints: {
        model: MODEL,
        config: { responseModalities: [Modality.AUDIO], systemInstruction: SYSTEM, tools: TOOLS, outputAudioTranscription: {}, temperature: 0.2 },
      },
    },
  });
  return t.name;
}

async function main() {
  const token = await mint();
  const ai = new GoogleGenAI({ apiKey: token, httpOptions: { apiVersion: 'v1alpha' } });
  const calls = [];
  let transcript = '';
  await new Promise((resolve) => {
    const timer = setTimeout(resolve, 12000);
    ai.live.connect({
      model: MODEL,
      callbacks: {
        onmessage: (m) => {
          const c = m?.toolCall?.functionCalls;
          if (c?.length) calls.push(...c.map((x) => `${x.name}(${JSON.stringify(x.args)})`));
          const ot = m?.serverContent?.outputTranscription?.text;
          if (ot) transcript += ot;
          if (m?.serverContent?.turnComplete) { clearTimeout(timer); resolve(); }
        },
        onerror: (e) => { console.log('err', e?.message); },
        onclose: (e) => { console.log('close', e?.code, e?.reason); clearTimeout(timer); resolve(); },
      },
    }).then((s) => s.sendClientContent({ turns: 'このパソコン捨てたい', turnComplete: true }));
  });
  console.log('\nuser said : このパソコン捨てたい');
  console.log('tool calls:', calls.length ? calls.join(', ') : '(none)');
  console.log('said      :', JSON.stringify(transcript));
  console.log(calls.some((c) => c.startsWith('add_to_cart')) ? '\n=> PASS: add_to_cart called' : '\n=> FAIL: no add_to_cart');
  process.exit(0);
}
main().catch((e) => { console.error('fatal', e); process.exit(1); });
