// Audio plumbing for the Gemini Live talk mode.
//
// Mic capture runs in a 16 kHz AudioContext — the rate the Live API expects for
// input — so no manual resampling is needed: a tiny AudioWorklet accumulates
// 2048-sample frames (~128 ms) and posts them to the main thread, which turns
// Float32 into little-endian Int16 PCM and base64-encodes it for the caller.
// Playback runs in a separate 24 kHz context (the rate of the model's audio
// output), scheduling each chunk back-to-back against a running cursor so
// speech plays gaplessly; barge-in clears the queue.

const RECORD_RATE = 16000;
const PLAYBACK_RATE = 24000;
const FRAME = 2048;

type ACtor = typeof AudioContext;
const AC: ACtor =
  window.AudioContext ||
  (window as unknown as { webkitAudioContext: ACtor }).webkitAudioContext;

function makeContext(sampleRate: number): AudioContext {
  // Most browsers honor the requested rate, but a few ignore it — we report the
  // actual rate back via `sampleRate` so the PCM header stays truthful.
  try {
    return new AC({ sampleRate });
  } catch {
    return new AC();
  }
}

// Accumulates mono input into FRAME-sized blocks and posts each to the main
// thread. Defined as a string so it can be loaded from a Blob URL without a
// separate bundler entry for the worklet.
const WORKLET_SRC = `
class PCMRecorder extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(${FRAME}); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) {
        this.port.postMessage(this.buf.slice(0));
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-recorder', PCMRecorder);
`;

function floatToB64(f: Float32Array): string {
  const i16 = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const s = Math.max(-1, Math.min(1, f[i]));
    i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  const bytes = new Uint8Array(i16.buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function b64ToFloat(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  // The byte length is always even (Int16 PCM), but guard the view length.
  const i16 = new Int16Array(bytes.buffer, 0, bytes.length >> 1);
  const f = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 0x8000;
  return f;
}

// Streaming linear resampler. iOS Safari commonly IGNORES the requested 16 kHz
// AudioContext rate and runs the mic at 48 kHz; sending that PCM labeled at its
// true rate still confuses Gemini Live (which expects 16 kHz), so the model
// hears nothing intelligible and never responds. Resampling to a fixed 16 kHz
// here makes input correct regardless of the context's actual rate. State (the
// fractional read position + previous buffer's last sample) carries across
// frames so back-to-back chunks join without clicks. Passthrough when rates
// already match (Android/desktop that honor the request).
class LinearResampler {
  private ratio: number;
  private pos = 0; // fractional read position in the current buffer's coords
  private last = 0; // previous buffer's final sample (virtual index -1)
  private passthrough: boolean;

  constructor(inRate: number, outRate: number) {
    this.ratio = inRate / outRate;
    this.passthrough = Math.abs(inRate - outRate) < 1;
  }

  process(input: Float32Array): Float32Array {
    if (this.passthrough || input.length === 0) return input;
    const n = input.length;
    const out: number[] = [];
    let pos = this.pos;
    while (pos < n - 1) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = i < 0 ? this.last : input[i];
      const b = input[i + 1];
      out.push(a + (b - a) * frac);
      pos += this.ratio;
    }
    this.pos = pos - n; // carry position into the next buffer's frame
    this.last = input[n - 1];
    return Float32Array.from(out);
  }
}

export class MicRecorder {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private inRate = RECORD_RATE;
  private resampler: LinearResampler | null = null;

  // Resolves once the mic is live and chunks are flowing to `onChunk`.
  async start(onChunk: (b64: string) => void): Promise<void> {
    const ctx = makeContext(RECORD_RATE);
    this.ctx = ctx;
    // The context may not honor RECORD_RATE (notably iOS) — resample its actual
    // rate down to the 16 kHz Gemini Live expects before sending.
    this.inRate = ctx.sampleRate;
    this.resampler = new LinearResampler(ctx.sampleRate, RECORD_RATE);

    const url = URL.createObjectURL(
      new Blob([WORKLET_SRC], { type: 'application/javascript' }),
    );
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    this.stream = stream;

    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'pcm-recorder');
    node.port.onmessage = (e) => {
      const resampled = this.resampler!.process(e.data as Float32Array);
      if (resampled.length) onChunk(floatToB64(resampled));
    };
    src.connect(node);
    // The worklet produces no audible output, but the graph must reach the
    // destination for `process()` to be pulled — route through a muted gain.
    const sink = ctx.createGain();
    sink.gain.value = 0;
    node.connect(sink).connect(ctx.destination);
    this.node = node;

    if (ctx.state === 'suspended') await ctx.resume();
  }

  // Output rate of the base64 chunks (post-resample) — always 16 kHz, so the
  // caller's `audio/pcm;rate=...` header stays truthful about what it sends.
  get sampleRate(): number {
    return RECORD_RATE;
  }

  // Actual mic/AudioContext rate before resampling — diagnostics only.
  get inputRate(): number {
    return this.inRate;
  }

  stop(): void {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close().catch(() => {});
    this.node = null;
    this.stream = null;
    this.ctx = null;
  }
}

export class AudioPlayer {
  private ctx: AudioContext | null = null;
  private cursor = 0;
  private sources = new Set<AudioBufferSourceNode>();

  private ensure(): AudioContext {
    if (!this.ctx) {
      this.ctx = makeContext(PLAYBACK_RATE);
      this.cursor = 0;
    }
    return this.ctx;
  }

  // Must be called from a user gesture so playback is allowed to start.
  async resume(): Promise<void> {
    const ctx = this.ensure();
    if (ctx.state === 'suspended') await ctx.resume();
  }

  enqueue(b64: string): void {
    const ctx = this.ensure();
    const f = b64ToFloat(b64);
    if (f.length === 0) return;
    const buf = ctx.createBuffer(1, f.length, PLAYBACK_RATE);
    // `.set` accepts ArrayLike<number>, sidestepping the Float32Array buffer-
    // generic mismatch that copyToChannel's stricter signature trips on.
    buf.getChannelData(0).set(f);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const start = Math.max(ctx.currentTime, this.cursor);
    src.start(start);
    this.cursor = start + buf.duration;
    this.sources.add(src);
    src.onended = () => this.sources.delete(src);
  }

  // Barge-in: stop everything queued and reset the cursor to "now".
  clear(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        // already stopped
      }
    }
    this.sources.clear();
    if (this.ctx) this.cursor = this.ctx.currentTime;
  }

  close(): void {
    this.clear();
    this.ctx?.close().catch(() => {});
    this.ctx = null;
  }
}
