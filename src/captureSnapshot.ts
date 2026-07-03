// Frame-capture helpers. Each returns just the base64 JPEG payload (no `data:`
// prefix, no mime) so it can be sent straight to a Gemini endpoint as
// `inlineData.data`.
//
// We downscale to a max long edge (default 720px) to keep request payloads well
// under Vercel's 4.5MB body limit even when 30 items are selected, and drop
// quality to 0.8 — Gemini's specific-name accuracy doesn't change meaningfully
// below that, but the byte count does.
//
// These are synchronous (toDataURL): for the 720px sizes here it's a few ms of
// main-thread work, fast enough to run inline on a tap without perceptible lag,
// and simpler than the async toBlob path for callers that need the string now.

function toJpegBase64(canvas: HTMLCanvasElement, quality: number): string {
  const url = canvas.toDataURL('image/jpeg', quality);
  const comma = url.indexOf(',');
  return comma >= 0 ? url.slice(comma + 1) : url;
}

// Whole current frame, downscaled so its long edge is at most maxLongEdge.
export function captureFrameJpeg(
  video: HTMLVideoElement,
  maxLongEdge = 720,
  quality = 0.8,
): string {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) throw new Error('video has no frame yet');

  const scale = Math.min(1, maxLongEdge / Math.max(vw, vh));
  const w = Math.round(vw * scale);
  const h = Math.round(vh * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2d context unavailable');
  ctx.drawImage(video, 0, 0, w, h);
  return toJpegBase64(canvas, quality);
}

// Normalized 0-1 xyxy box in some image's coordinate space.
export type NormBox = [number, number, number, number];

// A cropped snapshot: the JPEG payload plus where the original (pre-margin)
// box sits inside the crop, re-normalized to the crop's coordinate space —
// this is what refine-items expects as `bbox`.
export type CropResult = { data: string; innerBox: NormBox };

// Core box-crop: expand `box` by `margin` (fraction of the box's own width/
// height, per side), clamp to the frame, then crop + downscale (never upscale)
// so the crop's long edge is at most maxLongEdge. Throws if the clamped crop
// degenerates below 8px on either axis.
function cropRegionJpeg(
  source: CanvasImageSource,
  srcW: number,
  srcH: number,
  box: NormBox,
  margin = 0.18,
  maxLongEdge = 720,
  quality = 0.8,
): CropResult {
  const bx1 = box[0] * srcW;
  const by1 = box[1] * srcH;
  const bx2 = box[2] * srcW;
  const by2 = box[3] * srcH;
  const mx = (bx2 - bx1) * margin;
  const my = (by2 - by1) * margin;
  const sx = Math.max(0, bx1 - mx);
  const sy = Math.max(0, by1 - my);
  const ex = Math.min(srcW, bx2 + mx);
  const ey = Math.min(srcH, by2 + my);
  const cw = ex - sx;
  const ch = ey - sy;
  if (cw < 8 || ch < 8) throw new Error('crop region degenerate');

  const scale = Math.min(1, maxLongEdge / Math.max(cw, ch));
  const w = Math.round(cw * scale);
  const h = Math.round(ch * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2d context unavailable');
  ctx.drawImage(source, sx, sy, cw, ch, 0, 0, w, h);

  const innerBox: NormBox = [
    (bx1 - sx) / cw,
    (by1 - sy) / ch,
    (bx2 - sx) / cw,
    (by2 - sy) / ch,
  ];
  return { data: toJpegBase64(canvas, quality), innerBox };
}

// Tap path: crop a detection box (video pixel coords, xyxy) out of the live
// video with margin. Synchronous like captureFrameJpeg — runs inline on a tap.
export function captureBoxCropJpeg(
  video: HTMLVideoElement,
  bboxPx: [number, number, number, number],
  margin = 0.18,
  maxLongEdge = 720,
  quality = 0.8,
): CropResult {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) throw new Error('video has no frame yet');
  const box: NormBox = [
    bboxPx[0] / vw,
    bboxPx[1] / vh,
    bboxPx[2] / vw,
    bboxPx[3] / vh,
  ];
  return cropRegionJpeg(video, vw, vh, box, margin, maxLongEdge, quality);
}

// Voice path: crop a retained base64 JPEG (the last frame actually sent to the
// Live session) by the model-provided box_2d = [ymin, xmin, ymax, xmax] in
// Gemini's native 0-1000 normalized-integer convention. Returns null when the
// box is missing/invalid/degenerate — the caller falls back to the full frame.
// Only image-decode failures throw.
export async function cropJpegBase64(
  b64: string,
  box2d: unknown,
  margin = 0.18,
  maxLongEdge = 720,
  quality = 0.8,
): Promise<CropResult | null> {
  if (!Array.isArray(box2d) || box2d.length !== 4) return null;
  const nums = box2d.map(Number);
  if (nums.some((n) => !Number.isFinite(n))) return null;
  const clamped = nums.map((n) => Math.min(1000, Math.max(0, n)));
  // Salvage swapped coordinates (ymin > ymax etc.) via min/max.
  const y1 = Math.min(clamped[0], clamped[2]);
  const y2 = Math.max(clamped[0], clamped[2]);
  const x1 = Math.min(clamped[1], clamped[3]);
  const x2 = Math.max(clamped[1], clamped[3]);
  if (y2 - y1 < 5 || x2 - x1 < 5) return null;

  // <img> decode over createImageBitmap for iOS Safari compatibility.
  const img = new Image();
  img.src = 'data:image/jpeg;base64,' + b64;
  await img.decode();
  const box: NormBox = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000];
  return cropRegionJpeg(
    img,
    img.naturalWidth,
    img.naturalHeight,
    box,
    margin,
    maxLongEdge,
    quality,
  );
}

// Centered crop keeping the frame's aspect ratio: `ratio` is the fraction of
// the frame's width/height to keep (0.6 = central 60% box), then downscaled so
// the crop's long edge is at most maxLongEdge. Used by the upcoming 1fps Live
// API video feed to focus payload on the center of view.
export function captureCenterCropJpeg(
  video: HTMLVideoElement,
  ratio = 0.6,
  maxLongEdge = 720,
  quality = 0.8,
): string {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) throw new Error('video has no frame yet');

  const r = Math.min(1, Math.max(0.05, ratio));
  const cw = vw * r;
  const ch = vh * r;
  const sx = (vw - cw) / 2;
  const sy = (vh - ch) / 2;

  const scale = Math.min(1, maxLongEdge / Math.max(cw, ch));
  const w = Math.round(cw * scale);
  const h = Math.round(ch * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2d context unavailable');
  ctx.drawImage(video, sx, sy, cw, ch, 0, 0, w, h);
  return toJpegBase64(canvas, quality);
}
