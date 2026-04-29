import * as ort from 'onnxruntime-web/webgpu';
import type { Detection } from './types';

const INPUT_SIZE = 640;
const MAX_DETECTIONS = 300;
const SCORE_THRESHOLD = 0.3;

let session: ort.InferenceSession | null = null;
let activeBackend: 'webgpu' | 'wasm' | null = null;
let loggedSample = false;

ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';

export async function loadModel(): Promise<{ backend: 'webgpu' | 'wasm' }> {
  if (session && activeBackend) return { backend: activeBackend };

  const modelUrl = '/models/yolov10n_fp16.onnx';

  try {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
    activeBackend = 'webgpu';
  } catch (e) {
    console.warn('WebGPU unavailable, falling back to WASM:', e);
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    activeBackend = 'wasm';
  }

  return { backend: activeBackend };
}

type Meta = {
  scale: number;
  padX: number;
  padY: number;
  vw: number;
  vh: number;
};

function preprocess(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
): { input: Float32Array; meta: Meta } {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = INPUT_SIZE / Math.max(vw, vh);
  const newW = Math.round(vw * scale);
  const newH = Math.round(vh * scale);
  const padX = Math.floor((INPUT_SIZE - newW) / 2);
  const padY = Math.floor((INPUT_SIZE - newH) / 2);

  scratch.width = INPUT_SIZE;
  scratch.height = INPUT_SIZE;
  const ctx = scratch.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(video, 0, 0, vw, vh, padX, padY, newW, newH);

  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const stride = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    input[j] = data[i] / 255;
    input[j + stride] = data[i + 1] / 255;
    input[j + 2 * stride] = data[i + 2] / 255;
  }

  return { input, meta: { scale, padX, padY, vw, vh } };
}

function postprocess(output: Float32Array, meta: Meta): Detection[] {
  const detections: Detection[] = [];
  for (let i = 0; i < MAX_DETECTIONS; i++) {
    const offset = i * 6;
    const score = output[offset + 4];
    if (score < SCORE_THRESHOLD) continue;

    const x1 = (output[offset] - meta.padX) / meta.scale;
    const y1 = (output[offset + 1] - meta.padY) / meta.scale;
    const x2 = (output[offset + 2] - meta.padX) / meta.scale;
    const y2 = (output[offset + 3] - meta.padY) / meta.scale;

    detections.push({
      bbox: [
        Math.max(0, Math.min(meta.vw, x1)),
        Math.max(0, Math.min(meta.vh, y1)),
        Math.max(0, Math.min(meta.vw, x2)),
        Math.max(0, Math.min(meta.vh, y2)),
      ],
      score,
      classId: Math.round(output[offset + 5]),
    });
  }
  return detections;
}

export async function detect(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
): Promise<Detection[]> {
  if (!session) throw new Error('Model not loaded');
  if (!video.videoWidth) return [];

  const { input, meta } = preprocess(video, scratch);
  const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const inputName = session.inputNames[0];
  const results = await session.run({ [inputName]: tensor });
  const output = results[session.outputNames[0]].data as Float32Array;

  if (!loggedSample) {
    loggedSample = true;
    const top = [];
    for (let i = 0; i < 3; i++) {
      const o = i * 6;
      top.push({
        x1: output[o].toFixed(2),
        y1: output[o + 1].toFixed(2),
        x2: output[o + 2].toFixed(2),
        y2: output[o + 3].toFixed(2),
        score: output[o + 4].toFixed(3),
        class: Math.round(output[o + 5]),
      });
    }
    console.log('[yolo] sample raw output (top 3 of 300):', top);
    console.log('[yolo] meta:', meta);
  }

  return postprocess(output, meta);
}
