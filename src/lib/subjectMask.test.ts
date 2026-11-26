import { describe, expect, it } from "vitest";
import { restoredSubjectMask, segmentSubjectRequest, subjectMaskGate } from "./subjectMask";

const video = {
  kind: "generation" as const,
  projection: "perspective" as const,
  sourceKind: "video" as const,
  sourceCount: 1,
  hasAlpha: false,
  projectionConfirmed: true,
};

describe("subject mask eligibility", () => {
  it("allows one confirmed ordinary video", () => {
    expect(subjectMaskGate(video)).toEqual({ ok: true });
  });

  it("blocks images, panoramas, multiple inputs, alpha, and unconfirmed video", () => {
    expect(subjectMaskGate({ ...video, sourceKind: "images" }).ok).toBe(false);
    expect(subjectMaskGate({ ...video, projection: "equirectangular" })).toEqual({ ok: false, reason: "panorama" });
    expect(subjectMaskGate({ ...video, sourceCount: 2 })).toEqual({ ok: false, reason: "multi" });
    expect(subjectMaskGate({ ...video, hasAlpha: true })).toEqual({ ok: false, reason: "alpha" });
    expect(subjectMaskGate({ ...video, projection: "auto", projectionConfirmed: false })).toEqual({ ok: false, reason: "needConfirm" });
  });

  it("stays off for an old draft that has no subject-mask field", () => {
    expect(restoredSubjectMask(undefined)).toBeNull();
    expect(restoredSubjectMask({})).toBeNull();
  });

  it("sends an automatic first pass with no prompts and keeps manual prompts", () => {
    expect(segmentSubjectRequest("auto", [[0.2, 0.4]], [1], [0.1, 0.1, 0.4, 0.4])).toEqual({
      automatic: true,
      points: [],
      labels: [],
      boxXyxy: null,
    });
    expect(segmentSubjectRequest("manual", [[0.2, 0.4]], [1], null)).toEqual({
      automatic: false,
      points: [[0.2, 0.4]],
      labels: [1],
      boxXyxy: null,
    });
  });

  it("does not restore an accepted mask without a session", () => {
    expect(restoredSubjectMask({ enabled: true, accepted: true, sessionId: null })).toEqual({
      enabled: true,
      accepted: false,
      sessionId: null,
    });
    expect(restoredSubjectMask({ enabled: "yes" })).toBeNull();
  });
});
