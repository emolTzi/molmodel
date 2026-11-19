import type { Suggestion, StepId } from "./suggestions";
import type { QualityReport } from "./quality";

export interface AssistStatus {
  text: { configured: boolean; model: string; seesImages: boolean; label: string };
  vision: { configured: boolean; model: string | null; seesImages: boolean; reason?: string; note?: string; provider?: string };
  desktopReconstruction: string;
}

export interface AnalyzePhoto {
  id: string;
  source: "camera" | "file" | "simulated";
  width: number;
  height: number;
  thumbnailBase64?: string;
  quality: QualityReport;
}

export async function assistStatus(): Promise<AssistStatus | null> {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) return null;
    return await response.json() as AssistStatus;
  } catch {
    return null;
  }
}

export async function analyzeProject(input: {
  mode: "text" | "vision";
  projectName: string;
  steps: Array<{ id: StepId; marked: boolean }>;
  photos: AnalyzePhoto[];
}): Promise<{ channel: string; seesImages: boolean; notice: string; suggestions: Array<Omit<Suggestion, "source">> }> {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      mode: input.mode,
      consent: true,
      projectName: input.projectName,
      steps: input.steps,
      photos: input.photos.map((photo) => ({
        id: photo.id,
        source: photo.source,
        width: photo.width,
        height: photo.height,
        sharpness: photo.quality.sharpness,
        sharp: photo.quality.sharp,
        meanLuma: photo.quality.meanLuma,
        exposure: photo.quality.exposure,
        ahash: photo.quality.ahash,
        ...(input.mode === "vision" ? { thumbnailBase64: photo.thumbnailBase64 } : {}),
      })),
    }),
  });
  const body = await response.json().catch(() => ({ message: "分析服务没有返回 JSON" }));
  if (!response.ok) throw new Error(body.message || "模型分析失败");
  return body;
}
