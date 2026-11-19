/** Official DeepSeek facts checked against api-docs.deepseek.com on 2026-11-19.
 * Vision guide: deepseek-flash accepts images. deepseek-v4-pro does not.
 * deepseek-chat and deepseek-reasoner are not vision models here.
 * The retired alias deepseek-v4-flash-vision-exp is not used.
 */

export const OFFICIAL_ORIGIN = "https://api.deepseek.com";
export const OFFICIAL_CHAT_URL = `${OFFICIAL_ORIGIN}/chat/completions`;
export const OFFICIAL_VISION_MODEL = "deepseek-flash";
export const TEXT_ONLY_MODELS = new Set(["deepseek-flash", "deepseek-v4-pro"]);
export const NOT_VISION_MODELS = new Set([
  "deepseek-chat",
  "deepseek-reasoner",
  "deepseek-v4-pro",
  "deepseek-v4-flash",
  "deepseek-v4-flash-vision-exp",
]);

const BLOCKED_HOSTS = new Set(["metadata.google.internal"]);

export function readConfig(env = process.env) {
  const baseUrl = (env.DEEPSEEK_BASE_URL || OFFICIAL_ORIGIN).replace(/\/+$/, "");
  const textModel = env.DEEPSEEK_TEXT_MODEL || OFFICIAL_VISION_MODEL;
  const visionModel = env.DEEPSEEK_VISION_MODEL || "";
  const key = env.DEEPSEEK_API_KEY || "";
  const visionEnabled = env.MOBILE_VISION_ENABLED === "1";
  const selfHost = (env.MOBILE_VISION_BASE_URL || "").replace(/\/+$/, "");
  const selfModel = env.MOBILE_VISION_MODEL || "";
  const selfConfirmed = env.MOBILE_VISION_CONFIRMED === "true";
  return {
    baseUrl,
    textModel,
    visionModel,
    hasKey: key.length > 0,
    key,
    visionEnabled,
    selfHost,
    selfModel,
    selfConfirmed,
    token: env.MOBILE_ASSIST_TOKEN || "",
  };
}

export function publicStatus(config) {
  const officialVision = config.visionEnabled && config.visionModel === OFFICIAL_VISION_MODEL && config.baseUrl === OFFICIAL_ORIGIN && config.hasKey;
  const selfVision = Boolean(config.selfHost && config.selfModel && config.selfConfirmed && config.hasKey);
  return {
    text: {
      configured: config.hasKey && TEXT_ONLY_MODELS.has(config.textModel),
      model: config.textModel,
      seesImages: false,
      label: "仅文本建议，本次不识图",
    },
    vision: officialVision
      ? { configured: true, model: OFFICIAL_VISION_MODEL, provider: "deepseek-official", seesImages: true }
      : selfVision
        ? { configured: true, model: config.selfModel, provider: "self-hosted-confirmed", seesImages: true, note: "自托管模型是否真能看图，只来自 MOBILE_VISION_CONFIRMED，没有向官方目录复核。" }
        : { configured: false, model: null, seesImages: false, reason: visionBlockReason(config) },
    desktopReconstruction: "COLMAP、Brush 与 SAM 2 仍在桌面应用。手机不训练三维模型。",
  };
}

export function visionBlockReason(config) {
  if (!config.hasKey) return "没有配置 DEEPSEEK_API_KEY";
  if (!config.visionEnabled && !config.selfHost) return "视觉通道默认关闭（MOBILE_VISION_ENABLED 不是 1）";
  if (config.visionEnabled && NOT_VISION_MODELS.has(config.visionModel)) return `${config.visionModel} 不能当作视觉模型`;
  if (config.visionEnabled && config.visionModel !== OFFICIAL_VISION_MODEL) return "官方视觉模型只接受 deepseek-flash";
  if (config.selfHost && !config.selfConfirmed) return "自托管视觉端点未经 MOBILE_VISION_CONFIRMED=true 确认";
  return "视觉通道未就绪";
}

export function assertSafeBaseUrl(url) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw Object.assign(new Error("模型地址不是合法 URL"), { status: 500, code: "bad-base-url" });
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw Object.assign(new Error("模型地址必须是没有账号、查询参数的 https URL"), { status: 500, code: "bad-base-url" });
  }
  const host = parsed.hostname.toLowerCase();
  if (BLOCKED_HOSTS.has(host) || host === "0.0.0.0" || isLinkLocal(host)) {
    throw Object.assign(new Error("模型地址不能指向元数据或链路本地地址"), { status: 500, code: "bad-base-url" });
  }
  return parsed;
}

function isLinkLocal(host) {
  return /^169\.254\./.test(host.replace(/^\[|\]$/g, ""));
}

export function resolveCall(config, mode) {
  if (!config.hasKey) {
    throw Object.assign(new Error("没有配置密钥，不能调用模型。本地检查仍然可用。"), { status: 503, code: "no-credentials" });
  }
  if (mode === "text") {
    if (!TEXT_ONLY_MODELS.has(config.textModel)) {
      throw Object.assign(new Error("文本模型不在允许名单。deepseek-chat 与 deepseek-reasoner 不会被当作默认通道。"), { status: 400, code: "text-model-rejected" });
    }
    assertSafeBaseUrl(config.baseUrl);
    if (config.baseUrl !== OFFICIAL_ORIGIN) {
      throw Object.assign(new Error("文本通道只连接官方 https://api.deepseek.com"), { status: 400, code: "text-endpoint-rejected" });
    }
    return { url: OFFICIAL_CHAT_URL, model: config.textModel, channel: "text-only", sendImages: false };
  }
  if (mode !== "vision") {
    throw Object.assign(new Error("未知分析模式"), { status: 400, code: "bad-mode" });
  }
  const status = publicStatus(config).vision;
  if (!status.configured) {
    throw Object.assign(new Error(status.reason || "视觉通道不可用"), { status: 409, code: "vision-unavailable" });
  }
  if (status.provider === "deepseek-official") {
    return { url: OFFICIAL_CHAT_URL, model: OFFICIAL_VISION_MODEL, channel: "vision-official", sendImages: true };
  }
  const parsed = assertSafeBaseUrl(config.selfHost);
  return {
    url: `${parsed.origin}${parsed.pathname === "/" ? "" : parsed.pathname}/chat/completions`,
    model: config.selfModel,
    channel: "vision-self-hosted",
    sendImages: true,
  };
}
