const KINDS = new Set(["reshoot-clearer", "reshoot-angle", "reshoot-top", "reshoot-back", "note"]);
const CERTAINTY = new Set(["observed", "unverified"]);
const STEPS = new Set(["front", "right", "back", "left", "top", "detail"]);

export function validateAnalyzeRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) fail(400, "bad-json", "请求必须是 JSON 对象");
  const allowed = new Set(["mode", "consent", "projectName", "steps", "photos"]);
  for (const key of Object.keys(body)) {
    if (!allowed.has(key)) fail(400, "unexpected-field", `不能接受字段 ${key}`);
  }
  if (body.mode !== "text" && body.mode !== "vision") fail(400, "bad-mode", "mode 只能是 text 或 vision");
  if (body.consent !== true) fail(400, "consent-required", "需要用户明确同意本次分析");
  if (typeof body.projectName !== "string" || body.projectName.length > 80) fail(400, "bad-project", "项目名无效");
  if (!Array.isArray(body.steps) || body.steps.length > STEPS.size) fail(400, "bad-steps", "拍摄步骤无效");
  const steps = [];
  for (const step of body.steps) {
    if (!step || typeof step !== "object" || !STEPS.has(step.id) || typeof step.marked !== "boolean") fail(400, "bad-steps", "拍摄步骤无效");
    steps.push({ id: step.id, marked: step.marked });
  }
  if (!Array.isArray(body.photos) || body.photos.length < 1 || body.photos.length > 8) fail(400, "too-many-photos", "每次分析需要 1 到 8 张缩略图信息");
  const photos = body.photos.map(validatePhoto);
  const ids = new Set(photos.map((photo) => photo.id));
  if (ids.size !== photos.length) fail(400, "duplicate-photo", "照片编号重复");
  if (body.mode === "vision") {
    if (photos.some((photo) => !photo.thumbnailBase64)) fail(400, "missing-thumbnail", "视觉分析必须带上已授权的缩略图");
  }
  return { mode: body.mode, projectName: body.projectName, steps, photos };
}

function validatePhoto(photo) {
  if (!photo || typeof photo !== "object") fail(400, "bad-photo", "照片记录无效");
  const allowed = new Set(["id", "source", "width", "height", "sharpness", "sharp", "meanLuma", "exposure", "ahash", "thumbnailBase64"]);
  for (const key of Object.keys(photo)) if (!allowed.has(key)) fail(400, "unexpected-field", `照片不能包含 ${key}`);
  if (!/^[0-9a-f-]{36}$/i.test(photo.id || "")) fail(400, "bad-photo", "照片编号必须是 UUID");
  if (!["camera", "file", "simulated"].includes(photo.source)) fail(400, "bad-photo", "照片来源无效");
  if (!integer(photo.width, 1, 8000) || !integer(photo.height, 1, 8000)) fail(400, "bad-photo", "照片尺寸无效");
  if (typeof photo.sharpness !== "number" || !Number.isFinite(photo.sharpness) || photo.sharpness < 0 || photo.sharpness > 1e12) fail(400, "bad-metric", "清晰度数值无效");
  if (typeof photo.sharp !== "boolean") fail(400, "bad-metric", "清晰度标记无效");
  if (typeof photo.meanLuma !== "number" || photo.meanLuma < 0 || photo.meanLuma > 255) fail(400, "bad-metric", "亮度无效");
  if (!["dark", "ok", "bright"].includes(photo.exposure)) fail(400, "bad-metric", "曝光标记无效");
  if (typeof photo.ahash !== "string" || !/^[0-9a-f]{16}$/i.test(photo.ahash)) fail(400, "bad-metric", "感知哈希无效");
  let thumbnailBase64 = null;
  if (photo.thumbnailBase64 != null) {
    if (typeof photo.thumbnailBase64 !== "string" || photo.thumbnailBase64.length > 180_000) fail(413, "thumbnail-too-large", "单张缩略图超过限制");
    const bytes = decodeBase64(photo.thumbnailBase64);
    if (!bytes || bytes.length > 120_000 || !isJpegOrPng(bytes)) fail(400, "bad-thumbnail", "缩略图必须是 JPEG 或 PNG");
    thumbnailBase64 = photo.thumbnailBase64;
  }
  return { id: photo.id, source: photo.source, width: photo.width, height: photo.height, sharpness: photo.sharpness, sharp: photo.sharp, meanLuma: photo.meanLuma, exposure: photo.exposure, ahash: photo.ahash, thumbnailBase64 };
}

export function validateModelPayload(payload, request) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) fail(502, "model-schema", "模型没有返回 JSON 对象");
  if (Object.keys(payload).some((key) => key !== "suggestions")) fail(502, "model-schema", "模型返回了建议以外的字段");
  if (!Array.isArray(payload.suggestions) || payload.suggestions.length > 8) fail(502, "model-schema", "建议数量无效");
  const known = new Set(request.photos.map((photo) => photo.id));
  const soft = new Set(request.photos.filter((photo) => !photo.sharp || photo.exposure !== "ok").map((photo) => photo.id));
  const duplicates = duplicateIds(request.photos);
  return payload.suggestions.map((item, index) => {
    if (!item || typeof item !== "object") fail(502, "model-schema", "建议格式无效");
    const allowed = new Set(["kind", "title", "detail", "certainty", "photoIds", "evidence"]);
    if (Object.keys(item).some((key) => !allowed.has(key))) fail(502, "model-schema", "建议包含未允许的字段");
    if (!KINDS.has(item.kind) || !CERTAINTY.has(item.certainty)) fail(502, "model-schema", "建议类型或确定性无效");
    if (typeof item.title !== "string" || typeof item.detail !== "string" || typeof item.evidence !== "string") fail(502, "model-schema", "建议文本无效");
    if (item.title.length < 1 || item.title.length > 80 || item.detail.length > 400 || item.evidence.length < 1 || item.evidence.length > 400) fail(502, "model-schema", "建议文本长度无效");
    if (!Array.isArray(item.photoIds) || item.photoIds.some((id) => !known.has(id))) fail(502, "model-schema", "建议引用了不存在的照片");
    let certainty = item.certainty;
    if (item.kind === "reshoot-angle" || item.kind === "reshoot-top" || item.kind === "reshoot-back") certainty = "unverified";
    if (item.kind === "reshoot-clearer" && certainty === "observed" && !item.photoIds.some((id) => soft.has(id))) certainty = "unverified";
    if (item.kind === "note" && certainty === "observed" && item.photoIds.length > 0 && !item.photoIds.every((id) => duplicates.has(id) || soft.has(id))) certainty = "unverified";
    if (item.photoIds.length === 0 && certainty !== "unverified") certainty = "unverified";
    return { id: `model-${index + 1}`, kind: item.kind, title: item.title, detail: item.detail, certainty, photoIds: item.photoIds, evidence: item.evidence };
  });
}

function duplicateIds(photos) {
  const found = new Set();
  for (let i = 0; i < photos.length; i += 1) {
    for (let j = i + 1; j < photos.length; j += 1) {
      if (hammingHex(photos[i].ahash, photos[j].ahash) <= 5) {
        found.add(photos[i].id);
        found.add(photos[j].id);
      }
    }
  }
  return found;
}

function hammingHex(left, right) {
  let n = 0;
  const value = BigInt(`0x${left}`) ^ BigInt(`0x${right}`);
  for (let bit = 0n; bit < 64n; bit += 1n) if ((value >> bit) & 1n) n += 1;
  return n;
}

function integer(value, min, max) {
  return Number.isInteger(value) && value >= min && value <= max;
}

function decodeBase64(value) {
  try {
    return Buffer.from(value, "base64");
  } catch {
    return null;
  }
}

function isJpegOrPng(bytes) {
  const jpeg = bytes[0] === 0xff && bytes[1] === 0xd8;
  const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
  return jpeg || png;
}

export function fail(status, code, message) {
  throw Object.assign(new Error(message), { status, code });
}
