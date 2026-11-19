import { useEffect, useRef, useState } from "react";
import { analyzeProject, assistStatus, type AssistStatus } from "./api";
import { preparePhoto, simulatedFrame } from "./image";
import { localSuggestions, type StepId, type Suggestion } from "./suggestions";
import { idbStore, type CapturePhoto, type CaptureProject, type CaptureStore } from "./store";
import { zipStored } from "./zip";

const stepLabels: Record<StepId, string> = {
  front: "正面", right: "右侧", back: "背面", left: "左侧", top: "顶部", detail: "局部细节",
};

const store: CaptureStore = idbStore();

export function App() {
  const [projects, setProjects] = useState<CaptureProject[]>([]);
  const [current, setCurrent] = useState<CaptureProject | null>(null);
  const [photos, setPhotos] = useState<CapturePhoto[]>([]);
  const [name, setName] = useState("新拍摄");
  const [error, setError] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [status, setStatus] = useState<AssistStatus | null>(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const urls = useRef(new Map<string, string>());

  useEffect(() => {
    void refresh();
    void assistStatus().then(setStatus);
  }, []);
  useEffect(() => {
    if (videoRef.current && stream) videoRef.current.srcObject = stream;
    return () => { stream?.getTracks().forEach((track) => track.stop()); };
  }, [stream]);

  async function refresh(project = current) {
    setProjects(await store.listProjects());
    if (project) {
      const next = (await store.listProjects()).find((item) => item.id === project.id) ?? null;
      setCurrent(next);
      setPhotos(next ? await store.listPhotos(next.id) : []);
    }
  }

  async function addBlob(blob: Blob, source: CapturePhoto["source"]) {
    if (!current) return;
    setError(null);
    try {
      const prepared = await preparePhoto(blob);
      const photo: CapturePhoto = {
        id: crypto.randomUUID(), projectId: current.id, createdAt: new Date().toISOString(), source,
        width: prepared.width, height: prepared.height, blob: prepared.blob, quality: prepared.quality,
      };
      await store.addPhoto(photo);
      await refresh(current);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "无法保存照片");
    }
  }

  async function startCamera() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("这个浏览器没有相机接口。请改用“选择照片”，那条路径不依赖相机权限。");
      return;
    }
    try {
      const next = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      setStream(next);
      if (videoRef.current) videoRef.current.srcObject = next;
    } catch {
      setError("相机权限被拒绝或没有可用相机。请改用“选择照片”。");
    }
  }

  async function capture() {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) { setError("拍照被取消或编码失败。"); return; }
    await addBlob(blob, "camera");
  }

  function thumb(photo: CapturePhoto) {
    const existing = urls.current.get(photo.id);
    if (existing) return existing;
    const url = URL.createObjectURL(photo.blob);
    urls.current.set(photo.id, url);
    return url;
  }

  async function runLocal() {
    if (!current) return;
    const next = localSuggestions(photos.map((photo) => ({ id: photo.id, quality: photo.quality })), current.steps);
    setSuggestions(next);
  }

  async function runModel(mode: "text" | "vision") {
    if (!current) return;
    if (!consent) { setError(mode === "vision" ? "请先勾选将要上传的缩略图。" : "请先确认用本地测量结果做文本建议。"); return; }
    setBusy(true);
    setError(null);
    try {
      const payloadPhotos = await Promise.all(photos.map(async (photo) => ({
        id: photo.id, source: photo.source, width: photo.width, height: photo.height, quality: photo.quality!,
        thumbnailBase64: mode === "vision" ? await blobBase64(photo.blob) : undefined,
      })));
      const result = await analyzeProject({ mode, projectName: current.name, steps: current.steps, photos: payloadPhotos });
      const source: Suggestion["source"] = mode === "vision" ? "vision-model" : "text-model";
      setSuggestions([
        ...localSuggestions(photos.map((photo) => ({ id: photo.id, quality: photo.quality })), current.steps),
        ...result.suggestions.map((item) => ({ ...item, source })),
      ]);
      setError(result.notice);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "模型分析失败");
    } finally {
      setBusy(false);
    }
  }

  async function exportProject() {
    if (!current) return;
    const manifest = {
      format: "ooosplat-capture-v1",
      desktopSync: "manual",
      desktopBoundary: "把 photos 文件夹作为普通图片序列导入桌面 OOOSplat。这个清单不会自动同步，也不包含 COLMAP 或 Brush 工程。手机没有训练三维模型。",
      project: current,
      photos: photos.map((photo) => ({ id: photo.id, file: `photos/${photo.id}.jpg`, source: photo.source, width: photo.width, height: photo.height, quality: photo.quality })),
      suggestions,
    };
    const archive = zipStored([
      { name: "manifest.json", data: new TextEncoder().encode(JSON.stringify(manifest, null, 2)) },
      ...await Promise.all(photos.map(async (photo) => ({ name: `photos/${photo.id}.jpg`, data: new Uint8Array(await photo.blob.arrayBuffer()) }))),
    ]);
    const copy = new Uint8Array(archive.byteLength);
    copy.set(archive);
    const url = URL.createObjectURL(new Blob([copy], { type: "application/zip" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${current.name}.zip`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="app">
      <header>
        <h1>拍摄助手</h1>
        <p>手机只负责采集和检查。三维重建仍在桌面 OOOSplat 完成，这里不会训练模型。</p>
      </header>
      {error && <p className="error" role="alert">{error}</p>}
      {!current && (
        <section className="card">
          <label>新建拍摄任务<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <div className="row"><button type="button" onClick={() => void store.createProject(name).then((project) => { setCurrent(project); void refresh(project); })}>创建</button></div>
          {projects.map((project) => (
            <button className="secondary" type="button" key={project.id} onClick={() => { setCurrent(project); void store.listPhotos(project.id).then(setPhotos); }}>{project.name}</button>
          ))}
        </section>
      )}
      {current && (
        <section>
          <div className="row">
            <button className="secondary" type="button" onClick={() => { stream?.getTracks().forEach((track) => track.stop()); setStream(null); setCurrent(null); setSuggestions([]); }}>返回</button>
            <strong>{current.name}</strong>
            <button className="secondary" type="button" onClick={() => void store.deleteProject(current.id).then(() => { setCurrent(null); void refresh(null); })}>删除项目</button>
          </div>
          <div className="card">
            {stream && <video ref={videoRef} autoPlay playsInline muted />}
            <div className="row">
              {stream ? <button type="button" onClick={() => void capture()}>拍照</button> : <button type="button" onClick={() => void startCamera()}>打开相机</button>}
              <label className="secondary" htmlFor="photo-file">选择照片</label>
              <input id="photo-file" hidden type="file" accept="image/*" multiple onChange={(event) => { const files = [...event.target.files ?? []]; event.target.value = ""; void Promise.all(files.map((file) => addBlob(file, "file"))); }} />
              <button className="secondary" type="button" onClick={() => void simulatedFrame().then((blob) => addBlob(blob, "simulated"))}>模拟相机</button>
            </div>
            <p className="note">模拟相机只生成一张标了“开发示例，不是实拍”的图片。没有相机权限时，用“选择照片”。</p>
          </div>
          <div className="thumbs">
            {photos.map((photo) => (
              <figure key={photo.id}>
                <img src={thumb(photo)} alt="" />
                {photo.source !== "camera" && <span className="badge simulated">{photo.source === "file" ? "文件" : "模拟"}</span>}
                <button type="button" onClick={() => void store.deletePhoto(photo.id).then(() => refresh(current))}>删除</button>
              </figure>
            ))}
          </div>
          <section className="card">
            <h2>你标过的拍摄面</h2>
            <p className="note">这些勾选只代表你打算拍过，不是已验证的三维覆盖。</p>
            {current.steps.map((step) => (
              <label className="check" key={step.id}><input type="checkbox" checked={step.marked} onChange={(event) => void store.setStep(current.id, step.id, event.target.checked).then(() => refresh(current))} />{stepLabels[step.id]}</label>
            ))}
            <button type="button" onClick={() => void runLocal()} disabled={!photos.length}>本地检查</button>
          </section>
          <section className="card">
            <h2>模型建议</h2>
            <p className="badge" role="status">{aiLabel(status)}</p>
            <p className="note">{status?.desktopReconstruction ?? "COLMAP、Brush 与 SAM 2 仍在桌面。手机不训练三维模型。"} 看图也只能评价已上传的缩略图，不能识别没拍到的区域。</p>
            <label className="check"><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />我同意本次分析。文本模式只上传清晰度、亮度和哈希；视觉模式额外上传当前 {photos.length} 张缩略图，不上传原始定位信息。</label>
            <div className="row">
              <button type="button" disabled={busy || !status?.text.configured || !photos.length} onClick={() => void runModel("text")}>文本建议，不识图</button>
              <button type="button" disabled={busy || !status?.vision.configured || !photos.length} onClick={() => void runModel("vision")}>上传缩略图看图</button>
            </div>
          </section>
          {suggestions.map((item) => (
            <article className="suggestion" key={item.id}>
              <span className={`badge ${item.certainty}`}>{item.certainty === "observed" ? "有依据" : "覆盖未验证"}</span>
              <span className="badge">{item.source === "local" ? "本地计算" : item.source === "text-model" ? "仅文本" : "视觉模型"}</span>
              <h2>{item.title}</h2>
              <p>{item.detail}</p>
              <p className="note">{item.evidence}{item.photoIds.length ? ` 照片 ${item.photoIds.join(", ")}` : ""}</p>
            </article>
          ))}
          <button type="button" disabled={!photos.length} onClick={() => void exportProject()}>导出给桌面</button>
          <p className="note">导出是一个 zip：manifest.json 和照片。请在桌面手动把照片文件夹当图片序列导入。没有自动同步。</p>
        </section>
      )}
    </main>
  );
}

function aiLabel(status: AssistStatus | null): string {
  if (!status) return "AI 未启用";
  if (status.vision.configured) return `视觉通道已配置：${status.vision.model ?? "未命名模型"}`;
  if (status.text.configured) return "仅文本通道已配置，本次不识图";
  return "AI 未启用";
}

function blobBase64(blob: Blob): Promise<string> {
  return blob.arrayBuffer().then((buffer) => {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    for (const value of bytes) binary += String.fromCharCode(value);
    return btoa(binary);
  });
}
