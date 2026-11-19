import { qualityFromGray, removeJpegApp1, type QualityReport } from "./quality";

export class PhotoPrepareError extends Error {
  constructor(message: string) { super(message); this.name = "PhotoPrepareError"; }
}

export async function preparePhoto(file: Blob, maxEdge = 1600): Promise<{ blob: Blob; width: number; height: number; quality: QualityReport }> {
  if (file.size > 20 * 1024 * 1024) throw new PhotoPrepareError("图片超过 20 MB，请先缩小后再导入。");
  const bytes = new Uint8Array(await file.arrayBuffer());
  const stripped = removeJpegApp1(bytes);
  const copy = new Uint8Array(stripped.byteLength);
  copy.set(stripped);
  const bitmap = await createImageBitmap(new Blob([copy], { type: file.type || "image/jpeg" }), { imageOrientation: "from-image" });
  try {
    const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) throw new PhotoPrepareError("这个浏览器不能处理图片。");
    context.drawImage(bitmap, 0, 0, width, height);
    const sample = downscale(context, canvas, 256);
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((value) => value ? resolve(value) : reject(new PhotoPrepareError("图片编码失败。")), "image/jpeg", 0.85);
    });
    return { blob, width, height, quality: qualityFromGray(sample.gray, sample.width, sample.height) };
  } finally {
    bitmap.close();
  }
}

function downscale(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement, maxEdge: number) {
  const scale = Math.min(1, maxEdge / Math.max(canvas.width, canvas.height));
  const width = Math.max(3, Math.round(canvas.width * scale));
  const height = Math.max(3, Math.round(canvas.height * scale));
  const sample = document.createElement("canvas");
  sample.width = width;
  sample.height = height;
  const sampleContext = sample.getContext("2d");
  if (!sampleContext) throw new PhotoPrepareError("这个浏览器不能读取像素。");
  sampleContext.drawImage(canvas, 0, 0, width, height);
  const data = sampleContext.getImageData(0, 0, width, height).data;
  const gray = new Uint8Array(width * height);
  for (let index = 0; index < gray.length; index += 1) {
    gray[index] = Math.round(data[index * 4] * 0.299 + data[index * 4 + 1] * 0.587 + data[index * 4 + 2] * 0.114);
  }
  return { gray, width, height };
}

export async function simulatedFrame(): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = 480;
  canvas.height = 640;
  const context = canvas.getContext("2d");
  if (!context) throw new PhotoPrepareError("无法生成模拟画面。");
  context.fillStyle = "#d7dde6";
  context.fillRect(0, 0, 480, 640);
  context.fillStyle = "#2458d6";
  context.fillRect(150, 180, 180, 220);
  context.fillStyle = "#102033";
  context.font = "28px sans-serif";
  context.fillText("模拟相机", 160, 80);
  context.font = "18px sans-serif";
  context.fillText("开发示例，不是实拍", 140, 560);
  const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new PhotoPrepareError("模拟画面编码失败。")), "image/jpeg", 0.9));
  return blob;
}
