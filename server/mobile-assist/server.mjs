import { createServer } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { analyze, publicStatus } from "./analyze.mjs";
import { readConfig } from "./policy.mjs";

loadLocalEnv(fileURLToPath(new URL("./.env", import.meta.url)));
const config = readConfig();
const host = process.env.HOST || "127.0.0.1";
const port = Number(process.env.PORT || 8787);
const origins = new Set(["http://127.0.0.1:1421", "http://localhost:1421"]);
const hits = [];
let busy = false;

if (host !== "127.0.0.1" && host !== "localhost") {
  console.error("手机分析服务只监听 127.0.0.1");
  process.exit(1);
}

const server = createServer(async (request, response) => {
  const origin = request.headers.origin;
  if (origin && !origins.has(origin)) {
    send(response, 403, { error: "origin-rejected" });
    return;
  }
  if (origin) {
    response.setHeader("access-control-allow-origin", origin);
    response.setHeader("vary", "Origin");
    response.setHeader("access-control-allow-headers", "authorization, content-type");
    response.setHeader("access-control-allow-methods", "GET, POST, OPTIONS");
  }
  if (request.method === "OPTIONS") {
    response.writeHead(204);
    response.end();
    return;
  }
  if (config.token) {
    if (request.headers.authorization !== `Bearer ${config.token}`) {
      send(response, 401, { error: "unauthorized" });
      return;
    }
  }
  const url = new URL(request.url || "/", "http://127.0.0.1");
  if (request.method === "GET" && url.pathname === "/api/status") {
    send(response, 200, publicStatus(config));
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/analyze") {
    if (!allow(Date.now())) {
      send(response, 429, { error: "rate-limited" });
      return;
    }
    if (busy) {
      send(response, 429, { error: "busy" });
      return;
    }
    busy = true;
    try {
      const raw = await readBody(request, 1_500_000);
      const body = JSON.parse(raw || "{}");
      const result = await analyze(body, config, globalThis.fetch, AbortSignal.timeout(25_000));
      send(response, 200, result);
    } catch (error) {
      const status = Number(error.status) || (error instanceof SyntaxError ? 400 : 500);
      send(response, status, { error: error.code || "failed", message: error.message || "分析失败" });
    } finally {
      busy = false;
    }
    return;
  }
  send(response, 404, { error: "not-found" });
});

server.listen(port, host, () => {
  const status = publicStatus(config);
  console.log(`capture assist http://${host}:${port} text=${status.text.configured} vision=${status.vision.configured}`);
});

function allow(now) {
  while (hits.length && now - hits[0] > 60_000) hits.shift();
  if (hits.length >= 10) return false;
  hits.push(now);
  return true;
}

function readBody(request, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(Object.assign(new Error("请求过大"), { status: 413, code: "body-too-large" }));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    request.on("error", reject);
  });
}

function loadLocalEnv(path) {
  if (!existsSync(path)) return;
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 1) continue;
    const key = trimmed.slice(0, eq).trim();
    if (!/^[A-Z0-9_]+$/.test(key) || process.env[key] != null) continue;
    process.env[key] = trimmed.slice(eq + 1).trim();
  }
}

function send(response, status, body) {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
  response.end(JSON.stringify(body));
}
