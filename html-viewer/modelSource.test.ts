import { describe, expect, it } from "vitest";
import { offlineModelUrl } from "./modelSource";

describe("offline model location", () => {
  it("accepts the sibling ply and rejects remote or parent paths", () => {
    expect(offlineModelUrl(undefined, "http://127.0.0.1:4173/index.html")).toBeNull();
    expect(offlineModelUrl("model.ply", "http://127.0.0.1:4173/index.html")).toBe("http://127.0.0.1:4173/model.ply");
    for (const value of ["https://example.com/model.ply", "//cdn.example/model.ply", "/etc/model.ply", "../model.ply", "model.ply?x=1"]) {
      expect(() => offlineModelUrl(value, "http://127.0.0.1:4173/index.html")).toThrow(/beside index.html/);
    }
  });
});
