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
