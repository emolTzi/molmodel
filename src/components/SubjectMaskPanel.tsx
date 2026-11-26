import { useEffect, useState, type MouseEvent } from "react";
import type { TaskDraft } from "../app/taskWorkspace";
import { resolvedProjection } from "../app/generationPreflight";
import { useI18n } from "../i18n";
import {
  cancelSubjectMask,
  getSubjectMaskSettings,
  prepareSubjectMask,
  saveSubjectMaskSettings,
  segmentSubjectMask,
  subjectMaskPreflight,
} from "../lib/backend";
import { segmentSubjectRequest, subjectMaskGate, type SubjectMaskDraft } from "../lib/subjectMask";

type Tool = "positive" | "negative" | "box";

function gateFor(draft: TaskDraft) {
  const materials = draft.inputs?.mode === "materials" ? draft.inputs.items : null;
  const single = materials?.length === 1 ? materials[0] : null;
  const sourceKind = draft.inputs?.mode === "colmap"
    ? "colmap"
    : single?.source.kind === "video" || (!materials && draft.inputType === "video" && draft.inputPath)
      ? "video"
      : single?.source.kind === "images" || single?.source.kind === "imageFolder" || draft.inputType === "images"
        ? "images"
        : "none";
  const projection = resolvedProjection(draft);
  return subjectMaskGate({
    kind: draft.kind,
    projection,
    sourceKind,
    sourceCount: materials?.length ?? (draft.inputPath ? 1 : 0),
    hasAlpha: Boolean(draft.video?.hasAlpha),
    projectionConfirmed: projection === "perspective",
  });
}

export function SubjectMaskPanel({
  draft,
  projectsRoot,
  disabled,
  onChange,
}: {
  draft: TaskDraft;
  projectsRoot: string | null;
  disabled: boolean;
  onChange: (patch: Partial<TaskDraft>) => void;
}) {
  const { t } = useI18n();
  const gate = gateFor(draft);
  const selection = draft.subjectMask ?? null;
  const [python, setPython] = useState("python");
  const [checkpoint, setCheckpoint] = useState("");
  const [modelCfg, setModelCfg] = useState("configs/sam2.1/sam2.1_hiera_t.yaml");
  const [birefnet, setBirefnet] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tool, setTool] = useState<Tool>("positive");
  const [preview, setPreview] = useState<string | null>(null);
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [points, setPoints] = useState<Array<{ x: number; y: number; label: 0 | 1 }>>([]);
  const [box, setBox] = useState<{ x: number; y: number } | null>(null);
  const [boxEnd, setBoxEnd] = useState<{ x: number; y: number } | null>(null);
  const [masks, setMasks] = useState<Array<{ index: number; frame: string; mask: string }>>([]);

  useEffect(() => {
    void getSubjectMaskSettings().then((settings) => {
      setPython(settings.python);
      setCheckpoint(settings.checkpoint);
      setModelCfg(settings.modelCfg);
      setBirefnet(settings.birefnetCheckpoint ?? "");
    }).catch(() => undefined);
  }, []);

  const setSelection = (next: SubjectMaskDraft | null) => onChange({ subjectMask: next });
  const reason = gate.ok ? null : t(`subjectMask.${gate.reason}`);

  const click = (event: MouseEvent<HTMLImageElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    const y = Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height));
    setSelection(selection ? { ...selection, accepted: false } : selection);
    if (tool === "box") {
      if (!box || boxEnd) {
        setBox({ x, y });
        setBoxEnd(null);
      } else {
        setBoxEnd({ x, y });
      }
      return;
    }
    const label: 0 | 1 = tool === "positive" ? 1 : 0;
    setPoints((current) => [...current, { x, y, label }].slice(-32));
  };

  const runSegment = async (mode: "auto" | "manual") => {
    if (!selection?.sessionId || !projectsRoot) return;
    setBusy(true);
    try {
      const request = segmentSubjectRequest(
        mode,
        points.map((point) => [point.x, point.y]),
        points.map((point) => point.label),
        box && boxEnd ? [box.x, box.y, boxEnd.x, boxEnd.y] : null,
      );
      const result = await segmentSubjectMask({
        projectsRoot,
        sessionId: selection.sessionId,
        frameIndex: 0,
        ...request,
      });
      setMasks(result.previews.map((item) => ({ index: item.index, frame: item.framePngBase64, mask: item.maskPngBase64 })));
      setMessage(result.trainingNote);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="form-section subject-mask">
      <div className="field-label-row">
        <label className="field-label">{t("subjectMask.title")}</label>
      </div>
      <p className="field-note">{t("subjectMask.hint")}</p>
      <button
        className="planner-switch"
        type="button"
        role="switch"
        aria-checked={Boolean(selection?.enabled)}
        disabled={disabled || !gate.ok}
        onClick={() => setSelection(selection?.enabled ? { enabled: false, sessionId: null, accepted: false } : { enabled: true, sessionId: null, accepted: false })}
      >
        <span><strong>{selection?.enabled ? t("subjectMask.on") : t("subjectMask.off")}</strong><small>{reason ?? t("subjectMask.install")}</small></span>
        <i aria-hidden="true"><span /></i>
      </button>
      {selection?.enabled && gate.ok && (
        <div className="subject-mask-body">
          <label>{t("subjectMask.python")}<input value={python} onChange={(event) => setPython(event.target.value)} /></label>
          <label>{t("subjectMask.checkpoint")}<input value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)} /></label>
          <label>{t("subjectMask.modelCfg")}<input value={modelCfg} onChange={(event) => setModelCfg(event.target.value)} /></label>
          <label>{t("subjectMask.birefnet")}<input value={birefnet} onChange={(event) => setBirefnet(event.target.value)} /></label>
          <p className="field-note">{t("subjectMask.autoNote")}</p>
          <div className="subject-mask-actions">
            <button type="button" disabled={busy} onClick={() => void saveSubjectMaskSettings({ python, checkpoint, modelCfg, birefnetCheckpoint: birefnet }).then(() => setMessage(t("common.saved"))).catch((error: unknown) => setMessage(error instanceof Error ? error.message : String(error)))}>{t("subjectMask.save")}</button>
            <button type="button" disabled={busy} onClick={() => void subjectMaskPreflight().then((result) => setMessage(String(result.message ?? ""))).catch((error: unknown) => setMessage(error instanceof Error ? error.message : String(error)))}>{t("subjectMask.preflight")}</button>
            <button type="button" disabled={busy || !projectsRoot || !draft.inputPath} onClick={() => void (async () => {
              setBusy(true);
              try {
                const prepared = await prepareSubjectMask({
                  videoPath: draft.inputPath!,
                  projectsRoot: projectsRoot!,
                  quality: draft.quality,
                  generationPolicy: { version: draft.generationPolicyVersion, autoRetryEnabled: draft.autoRetryEnabled, frameSamplingRatio: draft.frameSamplingRatio, frameSamplingFps: draft.frameSamplingFps ?? null },
                  projection: resolvedProjection(draft),
                });
                setPreview(prepared.previewPngBase64);
                setWidth(prepared.width);
                setHeight(prepared.height);
                setPoints([]);
                setBox(null);
                setBoxEnd(null);
                setMasks([]);
                setSelection({ enabled: true, sessionId: prepared.sessionId, accepted: false });
                setMessage(`${prepared.frameCount} · ${prepared.width}×${prepared.height}`);
              } catch (error) {
                setMessage(error instanceof Error ? error.message : String(error));
              } finally {
                setBusy(false);
              }
            })()}>{t("subjectMask.prepare")}</button>
          </div>
          {preview && (
            <>
              <div className="subject-mask-actions">
                <button type="button" aria-pressed={tool === "positive"} onClick={() => setTool("positive")}>{t("subjectMask.positive")}</button>
                <button type="button" aria-pressed={tool === "negative"} onClick={() => setTool("negative")}>{t("subjectMask.negative")}</button>
                <button type="button" aria-pressed={tool === "box"} onClick={() => setTool("box")}>{t("subjectMask.box")}</button>
              </div>
              <div className="subject-mask-frame">
                <img src={`data:image/png;base64,${preview}`} alt="" onClick={click} />
                {points.map((point, index) => <i key={index} className={point.label ? "fg" : "bg"} style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }} />)}
              </div>
              <p className="field-note">{width}×{height}</p>
              <div className="subject-mask-actions">
                <button type="button" disabled={busy || !selection.sessionId} onClick={() => void runSegment("auto")}>{t("subjectMask.auto")}</button>
                <button type="button" disabled={busy || !selection.sessionId || (!points.some((point) => point.label === 1) && !(box && boxEnd))} onClick={() => void runSegment("manual")}>{t("subjectMask.segment")}</button>
                <button type="button" disabled={busy} onClick={() => void cancelSubjectMask()}>{t("subjectMask.cancel")}</button>
                <button type="button" disabled={!masks.length || !selection.sessionId} onClick={() => setSelection({ enabled: true, sessionId: selection.sessionId, accepted: true })}>{t("subjectMask.accept")}</button>
                <button type="button" onClick={() => { setPoints([]); setBox(null); setBoxEnd(null); setMasks([]); setSelection({ enabled: true, sessionId: selection.sessionId, accepted: false }); }}>{t("subjectMask.rerun")}</button>
              </div>
            </>
          )}
          {masks.map((item) => (
            <div className="subject-mask-pair" key={item.index}>
              <img src={`data:image/png;base64,${item.frame}`} alt="" />
              <img src={`data:image/png;base64,${item.mask}`} alt="" />
            </div>
          ))}
          {selection.accepted && <p className="field-note">{t("subjectMask.accepted")}</p>}
          {message && <p className="field-note" role="status">{message}</p>}
        </div>
      )}
    </div>
  );
}
