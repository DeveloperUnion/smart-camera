// Captures the current frame of a <video> element to a downsized JPEG and
// returns just the base64 payload (no `data:` prefix, no mime), so it can be
// sent straight to the refine-items endpoint as `inlineData.data`.
//
// We downscale to a max long edge (default 720px) to keep the request payload
// well under Vercel's 4.5MB body limit even when 30 items are selected, and we
// drop quality to 0.8 — Gemini's specific-name accuracy doesn't change
// meaningfully below that, but the byte count does.
export async function captureFrameJpeg(
  video: HTMLVideoElement,
  maxLongEdge = 720,
  quality = 0.8,
): Promise<string> {
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

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob((b) => resolve(b), 'image/jpeg', quality);
  });
  if (!blob) throw new Error('toBlob returned null');

  const buf = await blob.arrayBuffer();
  // btoa over the binary string is faster than FileReader for the size range
  // we deal with (~50-150KB per frame).
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}
