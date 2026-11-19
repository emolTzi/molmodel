# 手机拍摄助手

手机端负责拍照、本地质量检查和可选的模型建议。COLMAP、Brush、SAM 2 仍只在桌面 OOOSplat 里运行。导出的压缩包不会自动同步到桌面。

## 启动

```bash
npm run mobile:dev
```

浏览器打开 http://127.0.0.1:1421/mobile.html 。手机和电脑在同一网络时，仍只应通过本机反向代理使用分析服务；分析进程绑定 `127.0.0.1:8787`。

只开页面、不做模型调用：

```bash
npm run mobile
```

只开分析服务：

```bash
npm run mobile:server
```

没有 `DEEPSEEK_API_KEY` 时，页面上的本地检查可用，模型按钮会说明未配置密钥。不要把密钥写进前端环境变量、打包文件或浏览器存储。

## 模型界限

依据 2026-11-19 的官方文档：

- 对话接口：`POST https://api.deepseek.com/chat/completions`
- `deepseek-flash` 可以接收图像：https://api-docs.deepseek.com/guides/vision
- `deepseek-v4-pro` 的 Vision 为不支持：https://api-docs.deepseek.com/quick_start/pricing
- `deepseek-chat` 和 `deepseek-reasoner` 不会被当作视觉模型
- 文本建议不上传缩略图，界面标明「仅文本建议，本次不识图」
- 视觉建议必须由用户勾选将要上传的缩略图，且服务端 `MOBILE_VISION_ENABLED=1`
