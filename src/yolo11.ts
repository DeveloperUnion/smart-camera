import * as ort from 'onnxruntime-web/wasm';
import type { LiveBox } from './types';

// FastSAM-s: YOLOv8s-seg architecture trained on SA-1B (Segment Anything's
// 1.1B-mask dataset). It's a single-class ("object") detector — every anchor
// just predicts "is there an object here?" — so it puts bboxes around things
// regardless of category. That's exactly what we want for the cart workflow:
// YOLO finds boxes, the user taps the interesting ones, and Gemini decides
// what each actually is via /api/refine-items. Coverage is no longer capped
// by COCO's 80 / OIV7's 601 trained classes; if it's a physical object,
// it'll get a box.
//
// File name `yolo11.ts` is kept for git/history continuity even though we're
// now running a YOLOv8-seg derivative — the preprocess pipeline (letterbox,
// 114-gray pad, RGB CHW, /255) is identical so the rename would just churn
// callers.
const INPUT_SIZE = 640;
// Single class ("object"). FastSAM-s' name table is just {0: "object"} —
// keeping NUM_CLASSES around for documentation, not used in postprocess
// since there's no per-class argmax anymore.
const NUM_CLASSES = 1;
const NUM_ANCHORS = 8400; // 80*80 + 40*40 + 20*20
// FastSAM is trained on SA-1B which deliberately includes "parts of objects",
// "regions", and "groups of things" as separate masks, so a raw forward pass
// produces 100+ candidates per frame and the overlay becomes a mess. The
// tunables below trim that down to a phone-screen-legible count while
// keeping high-confidence whole-object boxes.
//
// Every knob is URL-overridable so iPhone field-tuning doesn't need a
// redeploy: ?conf, ?iou, ?minarea, ?maxarea, ?contain, ?maxbox.
function urlNum(key: string, fallback: number, min = 0, max = 1): number {
  if (typeof window === 'undefined') return fallback;
  const p = new URLSearchParams(window.location.search).get(key);
  const n = p ? Number(p) : NaN;
  return Number.isFinite(n) && n > min && n <= max ? n : fallback;
}

// Score floor. 0.25 already cuts most of the "regions/parts" noise in
// SA-1B-trained models while keeping real-object boxes around.
const SCORE_THRESHOLD = urlNum('conf', 0.25);
// Stricter NMS than COCO YOLO (0.5) because SA-1B's "object + its parts"
// supervision means many candidates sit close on the same physical thing.
// 0.3 collapses these into a single bbox per spatial region.
const NMS_IOU_THRESHOLD = urlNum('iou', 0.3);
// Drop tiny boxes (texture noise) and huge boxes (the "whole scene" /
// background region that SA-1B sometimes labels as one giant mask).
// Fractions are of the original video frame area, not the 640² letterbox.
const MIN_AREA_FRAC = urlNum('minarea', 0.005);   // 0.5%
const MAX_AREA_FRAC = urlNum('maxarea', 0.7, 0, 1);  // 70%
// Containment filter: drop box B if ≥ CONTAIN_RATIO of B is inside a
// higher-score box A. Handles "mug handle inside mug" / "screen inside
// phone" cases that NMS misses because IoU is low (small box inside big
// box has high IoMin but low IoU).
const CONTAIN_RATIO = urlNum('contain', 0.8);
// Hard cap on the number of boxes shown per frame. Past ~15-20 the user
// can't reasonably tap anything anyway.
const MAX_BOXES_PER_FRAME = Math.round(urlNum('maxbox', 15, 0, 200));

// FastSAM output shape is `[1, 4+1+32, 8400]` (4 box + 1 objectness + 32 mask
// coefficients). We don't render masks so we slice off the coefficients and
// only read up to channel index 5. The proto-mask output (`output1`) is
// ignored entirely — onnxruntime-web allocates it but we just don't dereference
// it past the dispose() call.
const NUM_CHANNELS_PER_ANCHOR = 4 + NUM_CLASSES; // 5 — we read [0..4]
void NUM_CHANNELS_PER_ANCHOR; // silence unused-var lint, here for docs

let session: ort.InferenceSession | null = null;
let activeBackend: 'wasm' | null = null;

// Reused per-call buffers — allocated once at module init.
const inputBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

// Use the wasm-only build (no WebGPU asyncify). The bundled webgpu variant
// is ~23MB and pushes iPhone WebKit over its memory-pressure tab-kill
// threshold during model load.
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
// Pin to a single thread — iPhone WebKit cannot use SharedArrayBuffer without
// COOP/COEP anyway, and the multi-thread loader path adds extra worker heaps
// that contribute to the memory-pressure kill.
ort.env.wasm.numThreads = 1;

export async function loadModel(): Promise<{ backend: 'wasm' }> {
  if (session && activeBackend) return { backend: activeBackend };

  const modelUrl = '/models/fastsam_s_640_uint8.onnx';

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

// YOLO expects letterbox-resized 640x640 with 114 gray padding, RGB
// channels-first normalized to [0, 1]. FastSAM uses the same preprocess.
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

// Diagnostic counters from the last postprocess pass — exposed to the debug
// overlay so we can tell "model produced nothing" from "threshold rejected
// everything" when triaging recall issues.
export let lastMaxScore = 0;
export let lastRawCount = 0;
export let lastKeptCount = 0;

// Output shape [1, 37, NUM_ANCHORS]: 4 box (cx,cy,w,h in input pixel space)
// + 1 objectness score + 32 mask coefficients (ignored — we only need
// bboxes). The objectness is already sigmoid'd by the Ultralytics ONNX
// export.
function postprocess(output: Float32Array, meta: LetterboxMeta): LiveBox[] {
  const candidates: Candidate[] = [];
  const stride = NUM_ANCHORS;
  let frameMaxScore = 0;
  const frameArea = meta.vw * meta.vh;
  const minBoxArea = frameArea * MIN_AREA_FRAC;
  const maxBoxArea = frameArea * MAX_AREA_FRAC;

  // Channel 4 = objectness. No per-class argmax loop anymore — class-agnostic
  // detection collapses what used to be ~600 channel reads per anchor down
  // to one.
  for (let i = 0; i < NUM_ANCHORS; i++) {
    const score = output[4 * stride + i];
    if (score > frameMaxScore) frameMaxScore = score;
    if (score < SCORE_THRESHOLD) continue;

    const cx = output[i];
    const cy = output[stride + i];
    const w = output[2 * stride + i];
    const h = output[3 * stride + i];

    // Decode in 640x640 space, then undo letterbox to map back to video pixels.
    const lx1 = cx - w / 2;
    const ly1 = cy - h / 2;
    const lx2 = cx + w / 2;
    const ly2 = cy + h / 2;

    const x1 = Math.max(0, Math.min(meta.vw, (lx1 - meta.padX) / meta.scale));
    const y1 = Math.max(0, Math.min(meta.vh, (ly1 - meta.padY) / meta.scale));
    const x2 = Math.max(0, Math.min(meta.vw, (lx2 - meta.padX) / meta.scale));
    const y2 = Math.max(0, Math.min(meta.vh, (ly2 - meta.padY) / meta.scale));

    // Area filter — drop noise (texture / sub-pixel detections) and whole-
    // scene "background region" detections before they enter NMS, since
    // those are usually high-score and would dominate the kept list.
    const area = (x2 - x1) * (y2 - y1);
    if (area < minBoxArea || area > maxBoxArea) continue;

    candidates.push({ x1, y1, x2, y2, score });
  }

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

  // Containment filter — drop boxes that are mostly inside a higher-score
  // kept box. SA-1B intentionally labels both "the mug" and "the handle of
  // the mug" as objects; NMS keeps both because IoU(small, large) is low,
  // but for tap-to-select we want one box per physical thing. `kept` is
  // already sorted by score (descending), so for each box we only need to
  // check against the boxes that came before it.
  const final: Candidate[] = [];
  for (let i = 0; i < kept.length; i++) {
    const b = kept[i];
    const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
    let contained = false;
    for (let j = 0; j < final.length; j++) {
      const a = final[j];
      const xi1 = Math.max(a.x1, b.x1);
      const yi1 = Math.max(a.y1, b.y1);
      const xi2 = Math.min(a.x2, b.x2);
      const yi2 = Math.min(a.y2, b.y2);
      const inter = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
      if (inter / bArea >= CONTAIN_RATIO) {
        contained = true;
        break;
      }
    }
    if (!contained) final.push(b);
    if (final.length >= MAX_BOXES_PER_FRAME) break;
  }

  lastMaxScore = frameMaxScore;
  lastRawCount = candidates.length;
  lastKeptCount = final.length;

  // All boxes share a generic label — the model is class-agnostic and the
  // actual identification happens later in /api/refine-items.
  return final.map((c) => ({
    bbox: [c.x1, c.y1, c.x2, c.y2],
    score: c.score,
    classId: 0,
    label: '物体',
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
    // FastSAM emits two outputs (`output0` det head, `output1` proto masks).
    // We only consume the det head; the proto branch's allocation is paid
    // for either way but we don't read or hold a reference to it past the
    // dispose() call in the finally block.
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
