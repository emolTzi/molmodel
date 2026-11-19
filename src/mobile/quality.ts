export const SHARPNESS_SOFT_BELOW = 80;
export const DUPLICATE_HAMMING_MAX = 5;

export interface QualityReport {
  sharpness: number;
  sharp: boolean;
  meanLuma: number;
  exposure: "dark" | "ok" | "bright";
  ahash: string;
}

export function laplacianVariance(gray: Uint8Array, width: number, height: number): number {
  if (width < 3 || height < 3) return 0;
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const value = 4 * gray[index] - gray[index - 1] - gray[index + 1] - gray[index - width] - gray[index + width];
      sum += value;
      sumSq += value * value;
      count += 1;
    }
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

export function exposureOf(meanLuma: number): QualityReport["exposure"] {
  if (meanLuma < 45) return "dark";
  if (meanLuma > 210) return "bright";
  return "ok";
}

export function qualityFromGray(gray: Uint8Array, width: number, height: number): QualityReport {
  const sharpness = laplacianVariance(gray, width, height);
  let total = 0;
  for (const value of gray) total += value;
  const meanLuma = gray.length ? total / gray.length : 0;
  return {
    sharpness,
    sharp: sharpness >= SHARPNESS_SOFT_BELOW,
    meanLuma,
    exposure: exposureOf(meanLuma),
    ahash: averageHash(gray, width, height),
  };
}

export function averageHash(gray: Uint8Array, width: number, height: number): string {
  const cells = 8;
  const samples: number[] = [];
  for (let gy = 0; gy < cells; gy += 1) {
    for (let gx = 0; gx < cells; gx += 1) {
      const x0 = Math.floor((gx * width) / cells);
      const x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * width) / cells));
      const y0 = Math.floor((gy * height) / cells);
      const y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * height) / cells));
      let sum = 0;
      let count = 0;
      for (let y = y0; y < y1 && y < height; y += 1) {
        for (let x = x0; x < x1 && x < width; x += 1) {
          sum += gray[y * width + x];
          count += 1;
        }
      }
      samples.push(count ? sum / count : 0);
    }
  }
  const average = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  let bits = 0n;
  samples.forEach((value, index) => {
    if (value >= average) bits |= 1n << BigInt(index);
  });
  return bits.toString(16).padStart(16, "0");
}

export function hammingHex(left: string, right: string): number {
  const value = BigInt(`0x${left}`) ^ BigInt(`0x${right}`);
  let count = 0;
  for (let bit = 0n; bit < 64n; bit += 1n) if ((value >> bit) & 1n) count += 1;
  return count;
}

export function removeJpegApp1(bytes: Uint8Array): Uint8Array {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return bytes;
  const parts: Uint8Array[] = [bytes.slice(0, 2)];
  let offset = 2;
  while (offset + 4 < bytes.length) {
    if (bytes[offset] !== 0xff) break;
    const marker = bytes[offset + 1];
    if (marker === 0xda) {
      parts.push(bytes.slice(offset));
      return concat(parts);
    }
    const length = (bytes[offset + 2] << 8) | bytes[offset + 3];
    if (length < 2 || offset + 2 + length > bytes.length) break;
    if (marker !== 0xe1) parts.push(bytes.slice(offset, offset + 2 + length));
    offset += 2 + length;
  }
  parts.push(bytes.slice(offset));
  return concat(parts);
}

function concat(parts: Uint8Array[]): Uint8Array {
  const size = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}
