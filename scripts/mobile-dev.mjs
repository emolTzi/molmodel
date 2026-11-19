import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const server = spawn(process.execPath, ["server/mobile-assist/server.mjs"], { cwd: root, stdio: "inherit", env: process.env });
const viteBin = resolve(root, "node_modules/vite/bin/vite.js");
const vite = spawn(process.execPath, [viteBin, "--config", "vite.mobile.config.ts", "--host", "127.0.0.1"], { cwd: root, stdio: "inherit", env: process.env });

function stop() {
  server.kill();
  vite.kill();
}
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
server.on("exit", (code) => { vite.kill(); process.exit(code ?? 0); });
vite.on("exit", (code) => { server.kill(); process.exit(code ?? 0); });
