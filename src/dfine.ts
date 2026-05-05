import * as ort from 'onnxruntime-web/webgpu';
import type { Detection } from './types';

const INPUT_SIZE = 640;
const NUM_QUERIES = 300;
const NUM_CLASSES = 80;
const SCORE_THRESHOLD = 0.3;

let session: ort.InferenceSession | null = null;
let activeBackend: 'webgpu' | 'wasm' | null = null;
let loggedSample = false;

const inputBuffer = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';

export async function loadModel(): Promise<{ backend: 'webgpu' | 'wasm' }> {
  if (session && activeBackend) return { backend: activeBackend };

  const modelUrl = '/models/dfine_n_uint8.onnx';

  const params =
    typeof window !== 'undefined'
      ? new URLSearchParams(window.location.search)
      : new URLSearchParams();
  const backendParam = params.get('backend');
  // iOS Safari's WebGPU is still experimental and leaks GPU buffers under
  // sustained inference, killing the tab after ~10s. Default to WASM on iOS;
  // override with ?backend=webgpu.
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
  const isIOS =
    /iPad|iPhone|iPod/.test(ua) ||
    (ua.includes('Mac') && typeof document !== 'undefined' && 'ontouchend' in document);
  const forceWasm = backendParam === 'wasm' || (isIOS && backendParam !== 'webgpu');

  if (!forceWasm) {
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      activeBackend = 'webgpu';
      return { backend: activeBackend };
    } catch (e) {
      console.warn('WebGPU unavailable, falling back to WASM:', e);
    }
  }

  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  activeBackend = 'wasm';
  return { backend: activeBackend };
}

type Meta = { vw: number; vh: number };

let scratchCtx: CanvasRenderingContext2D | null = null;

// D-FINE / RT-DETR uses anisotropic resize to 640x640 (do_pad: false),
// rescale by 1/255, no mean/std normalization.
function preprocess(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
): Meta {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if (!scratchCtx || scratchCtx.canvas !== scratch) {
    scratch.width = INPUT_SIZE;
    scratch.height = INPUT_SIZE;
    scratchCtx = scratch.getContext('2d', { willReadFrequently: true })!;
  }
  const ctx = scratchCtx;
  ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);

  // getImageData allocates a fresh ~1.6MB Uint8ClampedArray each call. On
  // mobile Safari the GC can't keep up at 3fps and the tab gets killed under
  // memory pressure. Skip the ImageData wrapper entirely.
  const data = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const stride = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    inputBuffer[j] = data[i] / 255;
    inputBuffer[j + stride] = data[i + 1] / 255;
    inputBuffer[j + 2 * stride] = data[i + 2] / 255;
  }

  return { vw, vh };
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// logits: [300, 80] flat, pred_boxes: [300, 4] flat (cxcywh, normalized 0-1)
function postprocess(
  logits: Float32Array,
  boxes: Float32Array,
  meta: Meta,
): Detection[] {
  // sigmoid is monotonic so argmax(logits) == argmax(sigmoid(logits)).
  // Threshold once on sigmoid'd score; only call sigmoid for the winner.
  const logitThreshold = Math.log(SCORE_THRESHOLD / (1 - SCORE_THRESHOLD));

  const detections: Detection[] = [];
  for (let q = 0; q < NUM_QUERIES; q++) {
    const lo = q * NUM_CLASSES;
    let bestClass = 0;
    let bestLogit = logits[lo];
    for (let c = 1; c < NUM_CLASSES; c++) {
      const v = logits[lo + c];
      if (v > bestLogit) {
        bestLogit = v;
        bestClass = c;
      }
    }
    if (bestLogit < logitThreshold) continue;

    const score = sigmoid(bestLogit);
    const bo = q * 4;
    const cx = boxes[bo];
    const cy = boxes[bo + 1];
    const w = boxes[bo + 2];
    const h = boxes[bo + 3];

    // Normalized cxcywh → video-pixel xyxy. Resize was anisotropic, so x and y
    // map independently to video dims.
    const x1 = (cx - w / 2) * meta.vw;
    const y1 = (cy - h / 2) * meta.vh;
    const x2 = (cx + w / 2) * meta.vw;
    const y2 = (cy + h / 2) * meta.vh;

    detections.push({
      bbox: [
        Math.max(0, Math.min(meta.vw, x1)),
        Math.max(0, Math.min(meta.vh, y1)),
        Math.max(0, Math.min(meta.vw, x2)),
        Math.max(0, Math.min(meta.vh, y2)),
      ],
      score,
      classId: bestClass,
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

  const meta = preprocess(video, scratch);
  const tensor = new ort.Tensor('float32', inputBuffer, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const inputName = session.inputNames[0];

  let results: ort.InferenceSession.OnnxValueMapType | null = null;
  try {
    results = await session.run({ [inputName]: tensor });
    const logits = results['logits'].data as Float32Array;
    const boxes = results['pred_boxes'].data as Float32Array;

    if (!loggedSample) {
      loggedSample = true;
      const top = [];
      for (let q = 0; q < 3; q++) {
        const lo = q * NUM_CLASSES;
        let bestClass = 0;
        let bestLogit = logits[lo];
        for (let c = 1; c < NUM_CLASSES; c++) {
          if (logits[lo + c] > bestLogit) {
            bestLogit = logits[lo + c];
            bestClass = c;
          }
        }
        const bo = q * 4;
        top.push({
          cx: boxes[bo].toFixed(3),
          cy: boxes[bo + 1].toFixed(3),
          w: boxes[bo + 2].toFixed(3),
          h: boxes[bo + 3].toFixed(3),
          score: sigmoid(bestLogit).toFixed(3),
          class: bestClass,
        });
      }
      console.log('[dfine] sample raw output (top 3 of 300):', top);
      console.log('[dfine] meta:', meta);
    }

    return postprocess(logits, boxes, meta);
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
