// Coordinate helpers for the live overlay.
//
// The <video> and its sibling <canvas> overlay both use `object-fit: cover`,
// so on phones the canvas's intrinsic pixel buffer (sized to videoWidth/
// videoHeight) extends past the visible CSS box: the browser scales the buffer
// up to fill the element and crops the overflow on the long axis. A raw
// pointer event gives us CSS px relative to the element; to hit-test against
// detection boxes (which live in video/canvas pixel space) we must undo that
// cover scaling + centering crop. cssToVideoPx does exactly that.
export function cssToVideoPx(
  clientX: number,
  clientY: number,
  rect: DOMRect,
  canvasW: number,
  canvasH: number,
): { x: number; y: number } {
  const canvasAspect = canvasW / canvasH;
  const rectAspect = rect.width / rect.height;
  let scale: number;
  let offsetX = 0;
  let offsetY = 0;
  if (canvasAspect > rectAspect) {
    // Canvas is wider than the element: cover scales by height, crops sides.
    scale = canvasH / rect.height;
    offsetX = (canvasW - rect.width * scale) / 2;
  } else {
    // Canvas is taller than the element: cover scales by width, crops top/bottom.
    scale = canvasW / rect.width;
    offsetY = (canvasH - rect.height * scale) / 2;
  }
  const x = (clientX - rect.left) * scale + offsetX;
  const y = (clientY - rect.top) * scale + offsetY;
  return { x, y };
}
