import * as ort from 'onnxruntime-web/wasm';
import { labelOf, OIV7_LABELS_JP } from './oiv7-labels';
import type { LiveBox } from './types';

// 512² input is the middle ground for OIV7: the 640² output activation
// (~20MB FP32) tripped iPhone WebKit's memory-pressure killer, but 416²
// hurt small-object recall noticeably. At 512² the head is ~13MB.
const INPUT_SIZE = 512;
const NUM_CLASSES = OIV7_LABELS_JP.length; // 601 (Open Images V7)
const NUM_ANCHORS = 5376; // 64*64 + 32*32 + 16*16
const SCORE_THRESHOLD = 0.3;
const NMS_IOU_THRESHOLD = 0.5;

let session: ort.InferenceSession | null = null;
let activeBackend: 'wasm' | null = null;

// Reused per-call buffers — allocated once at module init.
const inputBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

// Use the wasm-only build (no WebGPU asyncify). The bundled webgpu variant is
// ~23MB and pushes iPhone WebKit over its memory-pressure tab-kill threshold
// during model load. The wasm-only artifacts are ~10MB or less.
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
// Pin to a single thread — iPhone WebKit cannot use SharedArrayBuffer without
// COOP/COEP anyway, and the multi-thread loader path adds extra worker heaps
// that contribute to the memory-pressure kill.
ort.env.wasm.numThreads = 1;

export async function loadModel(): Promise<{ backend: 'wasm' }> {
  if (session && activeBackend) return { backend: activeBackend };

  const modelUrl = '/models/yolov8n_oiv7_512_uint8.onnx';

  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  activeBackend = 'wasm';
  return { backend: activeBackend };
}

type LetterboxMeta = {
  vw: number;
  vh: number;
  scale: number;
  padX: number;
  padY: number;
};

let scratchCtx: CanvasRenderingContext2D | null = null;

// YOLO11 expects letterbox-resized 640x640 with 114 gray padding,
// values normalized to [0, 1], RGB channels-first.
function preprocess(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
): LetterboxMeta {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if (!scratchCtx || scratchCtx.canvas !== scratch) {
    scratch.width = INPUT_SIZE;
    scratch.height = INPUT_SIZE;
    scratchCtx = scratch.getContext('2d', { willReadFrequently: true })!;
  }
  const ctx = scratchCtx;

  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const padX = (INPUT_SIZE - dw) / 2;
  const padY = (INPUT_SIZE - dh) / 2;

  // Fill with YOLO's gray padding color so border anchors don't fire.
  ctx.fillStyle = 'rgb(114, 114, 114)';
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(video, padX, padY, dw, dh);

  const data = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const stride = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    inputBuffer[j] = data[i] / 255;
    inputBuffer[j + stride] = data[i + 1] / 255;
    inputBuffer[j + 2 * stride] = data[i + 2] / 255;
  }

  return { vw, vh, scale, padX, padY };
}

type Candidate = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
};

function iou(a: Candidate, b: Candidate): number {
  const xi1 = Math.max(a.x1, b.x1);
  const yi1 = Math.max(a.y1, b.y1);
  const xi2 = Math.min(a.x2, b.x2);
  const yi2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
  if (inter <= 0) return 0;
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (aArea + bArea - inter);
}

// Output shape [1, 84, 8400]: 4 box (cx,cy,w,h in input pixel space) + 80 class
// scores per anchor. Class scores are already sigmoid'd by the export.
function postprocess(output: Float32Array, meta: LetterboxMeta): LiveBox[] {
  const candidates: Candidate[] = [];
  const stride = NUM_ANCHORS;

  for (let i = 0; i < NUM_ANCHORS; i++) {
    let bestClass = 0;
    let bestScore = output[4 * stride + i];
    for (let c = 1; c < NUM_CLASSES; c++) {
      const v = output[(4 + c) * stride + i];
      if (v > bestScore) {
        bestScore = v;
        bestClass = c;
      }
    }
    if (bestScore < SCORE_THRESHOLD) continue;

    const cx = output[i];
    const cy = output[stride + i];
    const w = output[2 * stride + i];
    const h = output[3 * stride + i];

    // Decode in 640x640 space, then undo letterbox to map back to video pixels.
    const lx1 = cx - w / 2;
    const ly1 = cy - h / 2;
    const lx2 = cx + w / 2;
    const ly2 = cy + h / 2;

    const x1 = (lx1 - meta.padX) / meta.scale;
    const y1 = (ly1 - meta.padY) / meta.scale;
    const x2 = (lx2 - meta.padX) / meta.scale;
    const y2 = (ly2 - meta.padY) / meta.scale;

    candidates.push({
      x1: Math.max(0, Math.min(meta.vw, x1)),
      y1: Math.max(0, Math.min(meta.vh, y1)),
      x2: Math.max(0, Math.min(meta.vw, x2)),
      y2: Math.max(0, Math.min(meta.vh, y2)),
      score: bestScore,
      classId: bestClass,
    });
  }

  // Class-agnostic NMS. OIV7 has heavily overlapping hierarchical classes
  // (e.g. Bus + Land vehicle + Vehicle all fire on one bus), so allowing
  // each class to keep its own bbox flooded the overlay with stacked
  // labels. Suppressing overlap regardless of class keeps only the most
  // confident label per location.
  candidates.sort((a, b) => b.score - a.score);
  const kept: Candidate[] = [];
  const suppressed = new Uint8Array(candidates.length);
  for (let i = 0; i < candidates.length; i++) {
    if (suppressed[i]) continue;
    kept.push(candidates[i]);
    for (let j = i + 1; j < candidates.length; j++) {
      if (suppressed[j]) continue;
      if (iou(candidates[i], candidates[j]) > NMS_IOU_THRESHOLD) {
        suppressed[j] = 1;
      }
    }
  }

  return kept.map((c) => ({
    bbox: [c.x1, c.y1, c.x2, c.y2],
    score: c.score,
    classId: c.classId,
    label: labelOf(c.classId),
  }));
}

export async function detect(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
): Promise<LiveBox[]> {
  if (!session) throw new Error('Model not loaded');
  if (!video.videoWidth) return [];

  const meta = preprocess(video, scratch);
  const tensor = new ort.Tensor('float32', inputBuffer, [
    1,
    3,
    INPUT_SIZE,
    INPUT_SIZE,
  ]);
  const inputName = session.inputNames[0];

  let results: ort.InferenceSession.OnnxValueMapType | null = null;
  try {
    results = await session.run({ [inputName]: tensor });
    const output = results[session.outputNames[0]].data as Float32Array;
    return postprocess(output, meta);
  } finally {
    tensor.dispose();
    if (results) {
      for (const name of session.outputNames) {
        const t = results[name];
        if (t && typeof t.dispose === 'function') t.dispose();
      }
    }
  }
}
