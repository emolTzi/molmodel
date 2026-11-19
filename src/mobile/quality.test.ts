import { describe, expect, it } from "vitest";
import { averageHash, exposureOf, hammingHex, laplacianVariance, qualityFromGray, removeJpegApp1, SHARPNESS_SOFT_BELOW } from "./quality";
import { localSuggestions } from "./suggestions";

function solid(value: number, size = 16): Uint8Array {
  return new Uint8Array(size * size).fill(value);
}

function stripes(size = 16): Uint8Array {
  const gray = new Uint8Array(size * size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) gray[y * size + x] = x % 2 ? 255 : 0;
  }
  return gray;
}

describe("local photo measurements", () => {
  it("treats a flat frame as soft and a high-frequency frame as sharp", () => {
    const flat = laplacianVariance(solid(128), 16, 16);
    const busy = laplacianVariance(stripes(), 16, 16);
    expect(flat).toBe(0);
    expect(busy).toBeGreaterThan(SHARPNESS_SOFT_BELOW);
    expect(qualityFromGray(solid(128), 16, 16).sharp).toBe(false);
    expect(qualityFromGray(stripes(), 16, 16).sharp).toBe(true);
  });

  it("classifies exposure from mean luminance", () => {
    expect(exposureOf(10)).toBe("dark");
    expect(exposureOf(128)).toBe("ok");
    expect(exposureOf(240)).toBe("bright");
  });

  it("matches identical hashes and separates different patterns", () => {
    const split = solid(0);
    split.set(new Uint8Array(16 * 8).fill(255), 16 * 8);
    const left = averageHash(solid(20), 16, 16);
    const right = averageHash(solid(20), 16, 16);
    const other = averageHash(split, 16, 16);
    expect(hammingHex(left, right)).toBe(0);
    expect(hammingHex(left, other)).toBeGreaterThan(5);
  });

  it("removes JPEG APP1 metadata such as GPS", () => {
    const app1 = new TextEncoder().encode("GPS latitude secret");
    const bytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe1, 0x00, app1.length + 2, ...app1, 0xff, 0xd9]);
    const stripped = removeJpegApp1(bytes);
    expect(new TextDecoder().decode(stripped)).not.toContain("GPS");
    expect(stripped[0]).toBe(0xff);
    expect(stripped[1]).toBe(0xd8);
  });
});

describe("local reshoot suggestions", () => {
  it("cites the soft photo and does not invent coverage percentages", () => {
    const id = "11111111-1111-4111-8111-111111111111";
    const suggestions = localSuggestions([{
      id,
      quality: { sharpness: 1, sharp: false, meanLuma: 12, exposure: "dark", ahash: "0".repeat(16) },
    }], [{ id: "top", marked: false }, { id: "back", marked: false }]);
    const soft = suggestions.find((item) => item.kind === "reshoot-clearer");
    expect(soft?.photoIds).toEqual([id]);
    expect(soft?.certainty).toBe("observed");
    expect(soft?.evidence).toContain(String(SHARPNESS_SOFT_BELOW));
    expect(suggestions.find((item) => item.kind === "reshoot-top")?.certainty).toBe("unverified");
    expect(JSON.stringify(suggestions)).not.toMatch(/%|概率|热图/);
  });
});
