import { describe, expect, it } from "vitest";
import { zipStored } from "./zip";

describe("desktop export archive", () => {
  it("stores the manifest and photo bytes without compression", () => {
    const archive = zipStored([
      { name: "manifest.json", data: new TextEncoder().encode("{\"desktopSync\":\"manual\"}") },
      { name: "photos/a.jpg", data: new Uint8Array([0xff, 0xd8, 0xff, 0xd9]) },
    ]);
    const text = new TextDecoder().decode(archive);
    expect(text).toContain("manifest.json");
    expect(text).toContain("photos/a.jpg");
    expect(text).toContain("manual");
    expect(archive[0]).toBe(0x50);
    expect(archive[1]).toBe(0x4b);
  });
});
