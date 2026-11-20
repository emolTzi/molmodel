/** A site export may only name a file next to index.html. No remote host, no parent path. */
export function offlineModelUrl(modelUrl: string | undefined, baseHref: string): string | null {
  if (modelUrl == null) return null;
  if (modelUrl.length === 0 || modelUrl.length > 240) throw new Error("The exported model name is invalid.");
  if (/^[a-z][a-z0-9+.-]*:/i.test(modelUrl) || modelUrl.startsWith("//") || modelUrl.startsWith("/") || modelUrl.startsWith("\\") || modelUrl.includes("..") || modelUrl.includes("\\") || modelUrl.includes("?")) {
    throw new Error("The exported viewer only loads a model file beside index.html.");
  }
  const url = new URL(modelUrl, baseHref);
  const base = new URL(baseHref);
  if (url.origin !== base.origin || (url.protocol === "file:" && base.protocol !== "file:")) {
    throw new Error("The exported viewer only loads a model file beside index.html.");
  }
  return url.href;
}
