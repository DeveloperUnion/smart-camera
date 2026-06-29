// Why is gemini-3.1-flash-live-preview rejected while 2.5 native-audio works?
// Ask the REST surface directly (v1alpha), which returns a real HTTP status +
// error body, instead of the WebSocket path that just closes with a null code.
//
//   node --env-file=.env scripts/probe-models.mjs

const key = process.env.GEMINI_API_KEY;
if (!key) {
  console.error('GEMINI_API_KEY not set');
  process.exit(1);
}

const BASE = 'https://generativelanguage.googleapis.com/v1alpha';
const TARGETS = [
  'gemini-3.1-flash-live-preview',
  'gemini-2.5-flash-native-audio-preview-12-2025',
];

async function getModel(name) {
  const res = await fetch(`${BASE}/models/${name}?key=${key}`);
  const body = await res.json().catch(() => ({}));
  return { status: res.status, body };
}

async function listModels() {
  const out = [];
  let pageToken = '';
  do {
    const url = `${BASE}/models?key=${key}&pageSize=1000${
      pageToken ? `&pageToken=${pageToken}` : ''
    }`;
    const res = await fetch(url);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      console.error('list failed', res.status, JSON.stringify(body));
      break;
    }
    out.push(...(body.models ?? []));
    pageToken = body.nextPageToken ?? '';
  } while (pageToken);
  return out;
}

async function main() {
  for (const name of TARGETS) {
    const { status, body } = await getModel(name);
    console.log(`\n=== GET models/${name} -> HTTP ${status} ===`);
    if (status === 200) {
      console.log('  displayName :', body.displayName);
      console.log('  methods     :', (body.supportedGenerationMethods ?? []).join(', '));
      console.log('  description :', (body.description ?? '').slice(0, 120));
    } else {
      console.log('  error:', JSON.stringify(body.error ?? body));
    }
  }

  console.log('\n=== Live-capable models this key can access (bidiGenerateContent) ===');
  const models = await listModels();
  const live = models.filter((m) =>
    (m.supportedGenerationMethods ?? []).some((x) =>
      x.toLowerCase().includes('bidi'),
    ),
  );
  if (live.length === 0) {
    console.log('  (none found via supportedGenerationMethods; dumping names containing live/audio)');
    for (const m of models) {
      const n = m.name.replace('models/', '');
      if (/live|audio/i.test(n)) console.log('   -', n, '->', (m.supportedGenerationMethods ?? []).join(','));
    }
  } else {
    for (const m of live) {
      console.log('   -', m.name.replace('models/', ''), '->', (m.supportedGenerationMethods ?? []).join(','));
    }
  }
  console.log(`\n  total models visible to this key: ${models.length}`);
}

main().catch((e) => {
  console.error('fatal', e);
  process.exit(1);
});
