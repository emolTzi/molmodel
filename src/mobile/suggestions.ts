import { DUPLICATE_HAMMING_MAX, hammingHex, type QualityReport, SHARPNESS_SOFT_BELOW } from "./quality";

export type SuggestionKind = "reshoot-clearer" | "reshoot-angle" | "reshoot-top" | "reshoot-back" | "note";
export type Certainty = "observed" | "unverified";
export type StepId = "front" | "right" | "back" | "left" | "top" | "detail";

export interface Suggestion {
  id: string;
  kind: SuggestionKind;
  title: string;
  detail: string;
  certainty: Certainty;
  photoIds: string[];
  evidence: string;
  source: "local" | "text-model" | "vision-model";
}

export function localSuggestions(
  photos: Array<{ id: string; quality: QualityReport | null }>,
  steps: Array<{ id: StepId; marked: boolean }>,
): Suggestion[] {
  const suggestions: Suggestion[] = [];
  for (const photo of photos) {
    if (!photo.quality) continue;
    if (!photo.quality.sharp) {
      suggestions.push({
        id: `soft-${photo.id}`,
        kind: "reshoot-clearer",
        title: "这张不够清楚，建议重拍",
        detail: "画面邻域对比很低。请拿稳后再拍同一位置，不要靠后期补细节。",
        certainty: "observed",
        photoIds: [photo.id],
        evidence: `四邻域 Laplacian 方差 ${photo.quality.sharpness.toFixed(1)}，低于 ${SHARPNESS_SOFT_BELOW}`,
        source: "local",
      });
    }
    if (photo.quality.exposure !== "ok") {
      suggestions.push({
        id: `exposure-${photo.id}`,
        kind: "reshoot-clearer",
        title: photo.quality.exposure === "dark" ? "这张偏暗" : "这张偏亮",
        detail: "曝光超出本地亮度阈值。调整光线或曝光后重拍。",
        certainty: "observed",
        photoIds: [photo.id],
        evidence: `平均亮度 ${photo.quality.meanLuma.toFixed(1)}，判定为 ${photo.quality.exposure}`,
        source: "local",
      });
    }
  }
  for (let index = 0; index < photos.length; index += 1) {
    for (let other = index + 1; other < photos.length; other += 1) {
      const left = photos[index].quality;
      const right = photos[other].quality;
      if (!left || !right) continue;
      const distance = hammingHex(left.ahash, right.ahash);
      const lumaGap = Math.abs(left.meanLuma - right.meanLuma);
      const sharpGap = Math.abs(left.sharpness - right.sharpness);
      const sharpScale = Math.max(left.sharpness, right.sharpness, 1);
      if (distance <= DUPLICATE_HAMMING_MAX && lumaGap <= 18 && sharpGap / sharpScale <= 0.35) {
        suggestions.push({
          id: `dup-${photos[index].id}-${photos[other].id}`,
          kind: "note",
          title: "这两张画面非常接近",
          detail: "感知哈希接近，可能是重复拍摄。这不能证明它们是同一三维角度。",
          certainty: "observed",
          photoIds: [photos[index].id, photos[other].id],
          evidence: `aHash 汉明距离 ${distance}，阈值 ${DUPLICATE_HAMMING_MAX}`,
          source: "local",
        });
      }
    }
  }
  for (const step of steps) {
    if (step.marked || (step.id !== "top" && step.id !== "back")) continue;
    suggestions.push({
      id: `missing-${step.id}`,
      kind: step.id === "top" ? "reshoot-top" : "reshoot-back",
      title: step.id === "top" ? "建议补拍顶部" : "建议补拍背面",
      detail: "目前没有照片被你标成这一面，也没有相机姿态。已有照片不能证明这一面没拍到，只能把它列为未验证的补拍建议。",
      certainty: "unverified",
      photoIds: [],
      evidence: `用户步骤 ${step.id} 未标记；首版没有 SfM 或 AR 姿态`,
      source: "local",
    });
  }
  suggestions.push({
    id: "coverage-limit",
    kind: "note",
    title: "覆盖范围未验证",
    detail: "手机没有重建相机轨迹。模糊、曝光和相似画面来自图像计算；缺的角度只是拍摄清单，不是三维完整度。",
    certainty: "unverified",
    photoIds: [],
    evidence: "没有 COLMAP 位姿",
    source: "local",
  });
  return suggestions;
}
