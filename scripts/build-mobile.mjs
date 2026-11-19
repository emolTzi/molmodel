import { spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const viteBin = resolve(root, "node_modules/vite/bin/vite.js");
const built = spawnSync(process.execPath, [viteBin, "build", "--config", "vite.mobile.config.ts"], {
  cwd: root,
  stdio: "inherit",
});
if (built.status !== 0) process.exit(built.status ?? 1);

writeFileSync(resolve(root, "dist-mobile/index.html"), `<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=/mobile.html" />
    <title>拍摄助手</title>
  </head>
  <body>
    <p><a href="/mobile.html">打开拍摄助手</a></p>
  </body>
</html>
`);
