export type SubjectMaskReason = "images" | "panorama" | "multi" | "alpha" | "needVideo" | "needConfirm";

export interface SubjectMaskGateInput {
  kind: "generation" | "reshoot";
  projection: "auto" | "perspective" | "equirectangular";
  sourceKind: "video" | "images" | "colmap" | "none";
  sourceCount: number;
  hasAlpha: boolean;
  projectionConfirmed: boolean;
}

export function subjectMaskGate(input: SubjectMaskGateInput): { ok: true } | { ok: false; reason: SubjectMaskReason } {
  if (input.kind !== "generation") return { ok: false, reason: "multi" };
  if (input.sourceKind === "images" || input.sourceKind === "colmap") return { ok: false, reason: "images" };
  if (input.sourceKind !== "video" || input.sourceCount !== 1) return { ok: false, reason: input.sourceCount > 1 ? "multi" : "needVideo" };
  if (input.projection === "equirectangular") return { ok: false, reason: "panorama" };
  if (input.projection !== "perspective" || !input.projectionConfirmed) return { ok: false, reason: "needConfirm" };
  if (input.hasAlpha) return { ok: false, reason: "alpha" };
  return { ok: true };
}

export interface SubjectMaskDraft {
  enabled: boolean;
  sessionId: string | null;
  accepted: boolean;
}

export type SubjectMaskSegmentMode = "auto" | "manual";

export function segmentSubjectRequest(
  mode: SubjectMaskSegmentMode,
  points: Array<[number, number]>,
  labels: number[],
  boxXyxy: number[] | null,
): { automatic: boolean; points: Array<[number, number]>; labels: number[]; boxXyxy: number[] | null } {
  if (mode === "auto") return { automatic: true, points: [], labels: [], boxXyxy: null };
  return { automatic: false, points, labels, boxXyxy };
}

export function restoredSubjectMask(value: unknown): SubjectMaskDraft | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  if (typeof record.enabled !== "boolean" || typeof record.accepted !== "boolean") return null;
  if (record.sessionId != null && typeof record.sessionId !== "string") return null;
  return {
    enabled: record.enabled,
    sessionId: typeof record.sessionId === "string" ? record.sessionId : null,
    accepted: record.accepted && typeof record.sessionId === "string",
  };
}
