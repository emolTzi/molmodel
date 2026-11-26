use std::{
    ffi::OsString,
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    commands::paths_for_app,
    engines::ffmpeg::extract_selected_frames,
    error::{Result, SplatError},
    pipeline::{
        generation_policy::GenerationPolicy,
        subject_mask::{
            self, SubjectInitializer, SubjectMaskCheckpoint, SubjectMaskStatus, SubjectPrompt,
            MAX_FRAMES,
        },
    },
    presets::Quality,
    process::{ProcessManager, ProcessSpec},
    project::catalog::{self, SubjectMaskSettings},
    video::panorama::{self, VideoProjection},
};

pub struct SubjectMaskRuntime {
    job: Mutex<Option<Arc<ProcessManager>>>,
}

impl Default for SubjectMaskRuntime {
    fn default() -> Self {
        Self {
            job: Mutex::new(None),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubjectMaskSelection {
    pub session_id: Uuid,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareSubjectMaskRequest {
    pub video_path: String,
    pub projects_root: String,
    pub quality: Quality,
    pub generation_policy: GenerationPolicy,
    pub projection: VideoProjection,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareSubjectMaskResponse {
    pub session_id: Uuid,
    pub frame_count: usize,
    pub width: u32,
    pub height: u32,
    pub preview_png_base64: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSubjectMaskRequest {
    pub projects_root: String,
    pub session_id: Uuid,
    pub frame_index: u32,
    pub points: Vec<(f64, f64)>,
    pub labels: Vec<u8>,
    pub box_xyxy: Option<[f64; 4]>,
    #[serde(default)]
    pub automatic: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskPreview {
    pub index: usize,
    pub frame_png_base64: String,
    pub mask_png_base64: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSubjectMaskResponse {
    pub session_id: Uuid,
    pub prompt_coverage: f64,
    pub dtype: String,
    pub previews: Vec<MaskPreview>,
    pub training_note: String,
}

#[tauri::command]
pub async fn get_subject_mask_settings() -> Result<SubjectMaskSettings> {
    Ok(catalog::load_settings().await?.subject_mask)
}

#[tauri::command]
pub async fn set_subject_mask_settings(
    settings: SubjectMaskSettings,
) -> Result<SubjectMaskSettings> {
    Ok(catalog::save_subject_mask_settings(settings)
        .await?
        .subject_mask)
}

#[tauri::command]
pub async fn subject_mask_preflight(app: AppHandle) -> Result<serde_json::Value> {
    let settings = catalog::load_settings().await?.subject_mask;
    run_preflight(&app, &settings).await
}

#[tauri::command]
pub async fn prepare_subject_mask(
    app: AppHandle,
    pipeline: State<'_, super::PipelineController>,
    request: PrepareSubjectMaskRequest,
) -> Result<PrepareSubjectMaskResponse> {
    ensure_idle(&pipeline).await?;
    if request.projection == VideoProjection::Equirectangular {
        return Err(SplatError::Process(
            "全景视频不能使用单物体去背景。请改回普通视频。".into(),
        ));
    }
    let video_path = PathBuf::from(&request.video_path);
    if !video_path.is_file() {
        return Err(SplatError::InvalidPath(video_path));
    }
    let projects_root = PathBuf::from(&request.projects_root);
    crate::project::ProjectManager::validate_root(&projects_root).await?;
    let engines = paths_for_app(&app);
    let manager = ProcessManager::new();
    let video =
        crate::engines::ffprobe::probe_video(&engines.ffprobe, &video_path, None, &manager).await?;
    if video.has_alpha {
        return Err(SplatError::Process(
            "带透明通道的视频不会按单物体掩码处理。请关闭此模式。".into(),
        ));
    }
    let detection = panorama::detect(
        &engines.ffprobe,
        &engines.ffmpeg,
        &video_path,
        &video,
        &manager,
        false,
    )
    .await?;
    if detection.confirmed_erp {
        return Err(SplatError::Process(
            "检测到全景视频，单物体去背景已禁用。".into(),
        ));
    }
    if detection.requires_choice && request.projection != VideoProjection::Perspective {
        return Err(SplatError::Process(
            "请先确认这是普通视频而不是全景，再使用单物体去背景。".into(),
        ));
    }
    let policy = request.generation_policy.validate()?;
    let plan = policy.frame_plan(&video, request.quality);
    if plan.selected_frames.is_empty() || plan.selected_frames.len() > MAX_FRAMES {
        return Err(SplatError::Process(format!(
            "当前抽帧为 {} 张，单物体模式只接受 1 到 {MAX_FRAMES} 张。请降低保留帧率。",
            plan.selected_frames.len()
        )));
    }
    let (source_width, source_height) = if video.rotation.rem_euclid(180) == 90 {
        (video.height, video.width)
    } else {
        (video.width, video.height)
    };
    let resolution = policy.resolution(request.quality, None, source_width, source_height, true);
    let session_id = Uuid::new_v4();
    let session = subject_mask::session_directory(&projects_root, session_id)?;
    tokio::fs::create_dir_all(session.join("frames")).await?;
    extract_selected_frames(
        &engines.ffmpeg,
        &video_path,
        &session.join("frames"),
        &session.join("colmap-unused"),
        &session.join("alpha-unused"),
        &plan.selected_frames,
        false,
        Some((resolution.working_width, resolution.working_height)),
        Some(session.join("extract.log")),
        &manager,
        None,
    )
    .await?;
    let frames = jpeg_frames(&session.join("frames"))?;
    if frames.len() != plan.selected_frames.len() {
        return Err(SplatError::Process(
            "抽帧数量与计划不一致，已停止分割。".into(),
        ));
    }
    let (width, height) = image::image_dimensions(&frames[0])
        .map_err(|error| SplatError::Process(error.to_string()))?;
    let preview = preview_png(&frames[0], 960)?;
    let prepared = serde_json::json!({
        "version": subject_mask::VERSION,
        "status": "prepared",
        "sessionId": session_id,
        "sourceVideoSha256": subject_mask::sha256_file(&video_path, &manager.child_token())?,
        "width": width,
        "height": height,
    });
    crate::project::manager::atomic_write_json(&session.join("prepared.json"), &prepared).await?;
    Ok(PrepareSubjectMaskResponse {
        session_id,
        frame_count: frames.len(),
        width,
        height,
        preview_png_base64: base64_encode(&preview),
    })
}

#[tauri::command]
pub async fn segment_subject_mask(
    app: AppHandle,
    pipeline: State<'_, super::PipelineController>,
    runtime: State<'_, SubjectMaskRuntime>,
    request: SegmentSubjectMaskRequest,
) -> Result<SegmentSubjectMaskResponse> {
    ensure_idle(&pipeline).await?;
    let settings = catalog::load_settings().await?.subject_mask;
    let initializer = subject_mask::plan_segment(
        request.automatic,
        request.frame_index,
        &request.points,
        &request.labels,
        request.box_xyxy,
        &settings.birefnet_checkpoint,
    )?;
    if settings.checkpoint.is_empty() {
        return Err(SplatError::Process(
            "还没有填写 SAM 2.1 权重路径，分割已停止。".into(),
        ));
    }
    let projects_root = PathBuf::from(&request.projects_root);
    crate::project::ProjectManager::validate_root(&projects_root).await?;
    let session = subject_mask::session_directory(&projects_root, request.session_id)?;
    let frames = jpeg_frames(&session.join("frames"))?;
    let frame_index = if initializer == SubjectInitializer::BiRefNet {
        0
    } else {
        request.frame_index as usize
    };
    let frame = frames
        .get(frame_index)
        .ok_or_else(|| SplatError::Process("提示帧不存在。".into()))?;
    let (width, height) =
        image::image_dimensions(frame).map_err(|error| SplatError::Process(error.to_string()))?;
    let mut points = Vec::new();
    let box_xyxy = if initializer == SubjectInitializer::SamPrompt {
        for (x, y) in &request.points {
            points.push(subject_mask::normalized_to_pixel(*x, *y, width, height)?);
        }
        request
            .box_xyxy
            .map(|[x0, y0, x1, y1]| -> Result<[f64; 4]> {
                let (ax, ay) = subject_mask::normalized_to_pixel(x0, y0, width, height)?;
                let (bx, by) = subject_mask::normalized_to_pixel(x1, y1, width, height)?;
                Ok([ax.min(bx), ay.min(by), ax.max(bx), ay.max(by)])
            })
            .transpose()?
    } else {
        None
    };
    let masks_dir = session.join("masks");
    if masks_dir.exists() {
        tokio::fs::remove_dir_all(&masks_dir).await?;
    }
    let init_mask = if initializer == SubjectInitializer::BiRefNet {
        let resolved =
            subject_mask::resolve_birefnet_snapshot(Path::new(&settings.birefnet_checkpoint))?;
        let output_mask = session.join("birefnet-init.png");
        let init_job_path = session.join("birefnet-job.json");
        let init_job = subject_mask::birefnet_job(
            &session,
            frame,
            &output_mask,
            &settings.birefnet_checkpoint,
        );
        crate::project::manager::atomic_write_json(&init_job_path, &init_job).await?;
        let init_script = adapter_script(&app, "birefnet_init.py")?;
        let init_output = run_subject_process(
            &runtime,
            PathBuf::from(&settings.python),
            subject_mask::birefnet_arguments(&init_script, &init_job_path),
            &session,
            "birefnet.log",
        )
        .await?;
        if !init_output.success {
            let _ = tokio::fs::remove_file(&output_mask).await;
            return Err(SplatError::Process(failure_message(
                &init_output.stdout,
                "BiRefNet 首帧分割失败。没有把背景涂黑，也没有开始跟踪或训练。",
            )));
        }
        let init_result =
            parse_result_json(&init_output.stdout, "BiRefNet 没有返回可解析的结果。")?;
        if init_result.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let _ = tokio::fs::remove_file(&output_mask).await;
            return Err(SplatError::Process(
                "BiRefNet 未报告成功，首帧结果已丢弃。".into(),
            ));
        }
        let mask = image::open(&output_mask)
            .map_err(|error| SplatError::Process(error.to_string()))?
            .into_luma8();
        if mask.width() != width || mask.height() != height {
            let _ = tokio::fs::remove_file(&output_mask).await;
            return Err(SplatError::Process(
                "BiRefNet 首帧掩码与抽帧尺寸不一致。".into(),
            ));
        }
        if let Err(error) = subject_mask::accept_automatic_coverage(subject_mask::coverage(&mask)) {
            let _ = tokio::fs::remove_file(&output_mask).await;
            return Err(error);
        }
        Some((output_mask, resolved.sha256))
    } else {
        None
    };
    let indexed = session.join("sam2-frames");
    tokio::fs::create_dir_all(&masks_dir).await?;
    tokio::fs::create_dir_all(&indexed).await?;
    let job_path = session.join("job.json");
    let sam_labels: Vec<u8> = if initializer == SubjectInitializer::SamPrompt {
        request.labels.clone()
    } else {
        Vec::new()
    };
    let job = subject_mask::sam2_job(subject_mask::Sam2Job {
        session: &session,
        checkpoint: &settings.checkpoint,
        model_cfg: &settings.model_cfg,
        frame_index,
        points: &points,
        labels: &sam_labels,
        box_xyxy,
        init_mask: init_mask.as_ref().map(|(path, _)| path.as_path()),
    });
    crate::project::manager::atomic_write_json(&job_path, &job).await?;
    let script = adapter_script(&app, "sam2_video.py")?;
    let manager = Arc::new(ProcessManager::new().with_complete_stdout_limit(2 * 1024 * 1024));
    *runtime.job.lock().await = Some(manager.clone());
    let output = manager
        .run(ProcessSpec {
            executable: PathBuf::from(&settings.python),
            args: subject_mask::sam2_arguments(&script, &job_path),
            working_directory: Some(session.clone()),
            log_path: Some(session.join("sam2.log")),
            observer: None,
        })
        .await;
    *runtime.job.lock().await = None;
    let output = output?;
    if !output.success {
        let _ = tokio::fs::remove_dir_all(&masks_dir).await;
        return Err(SplatError::Process(failure_message(
            &output.stdout,
            "SAM 2 分割失败。没有把背景涂黑，也没有开始训练。",
        )));
    }
    let result = parse_result_json(&output.stdout, "SAM 2 没有返回可解析的结果。")?;
    if result.get("ok") != Some(&serde_json::Value::Bool(true)) {
        return Err(SplatError::Process(
            "SAM 2 未报告成功，分割结果已丢弃。".into(),
        ));
    }
    let dtype = result
        .get("dtype")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_owned();
    let token = manager.child_token();
    let mut records = Vec::new();
    for (index, frame_path) in frames.iter().enumerate() {
        let name = frame_path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| SplatError::Process("抽帧文件名无效。".into()))?;
        let mask_name = subject_mask::brush_mask_file(name)?;
        let mask_path = masks_dir.join(&mask_name);
        let mask = image::open(&mask_path)
            .map_err(|error| SplatError::Process(error.to_string()))?
            .into_luma8();
        let (frame_w, frame_h) = image::image_dimensions(frame_path)
            .map_err(|error| SplatError::Process(error.to_string()))?;
        if mask.width() != frame_w || mask.height() != frame_h {
            return Err(SplatError::Process(format!(
                "{name} 的掩码与抽帧尺寸不一致。"
            )));
        }
        if index == frame_index {
            subject_mask::accept_prompt_coverage(subject_mask::coverage(&mask))?;
        }
        records.push(subject_mask::MaskFrame {
            name: name.to_owned(),
            sha256: subject_mask::sha256_file(frame_path, &token)?,
            width: frame_w,
            height: frame_h,
            mask: mask_name,
            mask_sha256: subject_mask::sha256_file(&mask_path, &token)?,
        });
    }
    let prompt_mask = image::open(masks_dir.join(&records[frame_index].mask))
        .map_err(|error| SplatError::Process(error.to_string()))?
        .into_luma8();
    let prompt_coverage = subject_mask::coverage(&prompt_mask);
    let checkpoint = SubjectMaskCheckpoint {
        version: subject_mask::VERSION,
        status: SubjectMaskStatus::Segmented,
        session_id: request.session_id,
        model_id: Path::new(&settings.checkpoint)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("sam2.1")
            .to_owned(),
        model_cfg: settings.model_cfg,
        checkpoint_sha256: subject_mask::sha256_file(Path::new(&settings.checkpoint), &token)?,
        source_video_sha256: prepared_video_hash(&session)?,
        dtype,
        prompt: SubjectPrompt {
            frame_index: u32::try_from(frame_index).unwrap_or(0),
            points: if initializer == SubjectInitializer::BiRefNet {
                Vec::new()
            } else {
                request.points
            },
            labels: if initializer == SubjectInitializer::BiRefNet {
                Vec::new()
            } else {
                request.labels
            },
            box_xyxy: if initializer == SubjectInitializer::BiRefNet {
                None
            } else {
                request.box_xyxy
            },
            initializer,
        },
        frames: records,
        prompt_coverage,
        init_model_id: init_mask.as_ref().map(|_| "BiRefNet".to_owned()),
        init_checkpoint_sha256: init_mask.as_ref().map(|(_, hash)| hash.clone()),
    };
    checkpoint.validate()?;
    crate::project::manager::atomic_write_json(&session.join("manifest.json"), &checkpoint).await?;
    let mut previews = Vec::new();
    for index in preview_indexes(frames.len(), frame_index) {
        previews.push(MaskPreview {
            index,
            frame_png_base64: base64_encode(&preview_png(&frames[index], 480)?),
            mask_png_base64: base64_encode(&preview_png(
                &masks_dir.join(&checkpoint.frames[index].mask),
                480,
            )?),
        });
    }
    Ok(SegmentSubjectMaskResponse {
        session_id: request.session_id,
        prompt_coverage,
        dtype: checkpoint.dtype,
        previews,
        training_note: "掩码已可在生成时交给支持 --alpha-mode masked 的 Brush。它只忽略背景损失，不恢复主体没拍到的细节。若当前引擎不支持，训练会在启动前停止。".into(),
    })
}

#[tauri::command]
pub async fn cancel_subject_mask(runtime: State<'_, SubjectMaskRuntime>) -> Result<()> {
    if let Some(job) = runtime.job.lock().await.as_ref() {
        job.cancel();
    }
    Ok(())
}

async fn ensure_idle(pipeline: &State<'_, super::PipelineController>) -> Result<()> {
    if pipeline
        .running()
        .await
        .map_err(|error| SplatError::Process(error.to_string()))?
        .is_some()
    {
        return Err(SplatError::Process(
            "已有生成任务正在使用显卡。请等它结束后再做单物体分割。".into(),
        ));
    }
    Ok(())
}

fn adapter_script(app: &AppHandle, file_name: &str) -> Result<PathBuf> {
    let relative = Path::new("scripts/subject_mask").join(file_name);
    if let Ok(resources) = app.path().resource_dir() {
        let bundled = resources.join(&relative);
        if bundled.is_file() {
            return Ok(bundled);
        }
    }
    let dev = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(&relative);
    if dev.is_file() {
        return Ok(dev);
    }
    Err(SplatError::Process(format!(
        "找不到本机分割脚本 scripts/subject_mask/{file_name}。"
    )))
}

async fn run_subject_process(
    runtime: &SubjectMaskRuntime,
    python: PathBuf,
    args: Vec<OsString>,
    session: &Path,
    log_name: &str,
) -> Result<crate::process::ProcessOutput> {
    let manager = Arc::new(ProcessManager::new().with_complete_stdout_limit(2 * 1024 * 1024));
    *runtime.job.lock().await = Some(manager.clone());
    let output = manager
        .run(ProcessSpec {
            executable: python,
            args,
            working_directory: Some(session.to_path_buf()),
            log_path: Some(session.join(log_name)),
            observer: None,
        })
        .await;
    *runtime.job.lock().await = None;
    output
}

async fn run_preflight(
    app: &AppHandle,
    settings: &SubjectMaskSettings,
) -> Result<serde_json::Value> {
    let mut value = run_json_script(
        &settings.python,
        preflight_arguments(&adapter_script(app, "sam2_video.py")?, &settings.checkpoint),
        "分割预检没有返回结果。",
    )
    .await?;
    let birefnet = run_json_script(
        &settings.python,
        preflight_arguments(
            &adapter_script(app, "birefnet_init.py")?,
            &settings.birefnet_checkpoint,
        ),
        "BiRefNet 预检没有返回结果。",
    )
    .await?;
    let sam_message = value
        .get("message")
        .and_then(|item| item.as_str())
        .unwrap_or("")
        .to_owned();
    let birefnet_message = birefnet
        .get("message")
        .and_then(|item| item.as_str())
        .unwrap_or("")
        .to_owned();
    value["birefnetOk"] = birefnet
        .get("ok")
        .cloned()
        .unwrap_or(serde_json::Value::Bool(false));
    value["birefnet"] = birefnet;
    value["message"] = serde_json::Value::String(
        format!("{sam_message} {birefnet_message}")
            .trim()
            .to_owned(),
    );
    Ok(value)
}

fn preflight_arguments(script: &Path, checkpoint: &str) -> Vec<OsString> {
    let mut args = vec![script.as_os_str().to_owned(), "--preflight".into()];
    if !checkpoint.is_empty() {
        args.push("--checkpoint".into());
        args.push(PathBuf::from(checkpoint).into());
    }
    args
}

async fn run_json_script(
    python: &str,
    args: Vec<OsString>,
    empty_message: &str,
) -> Result<serde_json::Value> {
    let output = ProcessManager::new()
        .with_complete_stdout_limit(256 * 1024)
        .run(ProcessSpec {
            executable: PathBuf::from(python),
            args,
            working_directory: None,
            log_path: None,
            observer: None,
        })
        .await?;
    parse_result_json(&output.stdout, empty_message).map_err(|_| {
        SplatError::Process(format!(
            "{empty_message}{}",
            output.stderr.chars().take(400).collect::<String>()
        ))
    })
}

fn parse_result_json(stdout: &str, empty_message: &str) -> Result<serde_json::Value> {
    serde_json::from_str(last_json_line(stdout).unwrap_or("{}"))
        .map_err(|_| SplatError::Process(empty_message.to_owned()))
}

fn failure_message(stdout: &str, fallback: &str) -> String {
    last_json_line(stdout)
        .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .and_then(|value| {
            value
                .get("message")
                .and_then(|item| item.as_str())
                .map(str::to_owned)
        })
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| fallback.to_owned())
}

fn last_json_line(stdout: &str) -> Option<&str> {
    stdout.lines().rev().find(|line| line.starts_with('{'))
}

fn jpeg_frames(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut frames = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|value| value.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
        })
        .collect::<Vec<_>>();
    frames.sort();
    if frames.is_empty() {
        return Err(SplatError::Process(
            "没有找到可用于分割的 JPEG 抽帧。".into(),
        ));
    }
    Ok(frames)
}

fn prepared_video_hash(session: &Path) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(session.join("prepared.json"))?)
            .map_err(|error| SplatError::Process(error.to_string()))?;
    value
        .get("sourceVideoSha256")
        .and_then(|item| item.as_str())
        .map(str::to_owned)
        .ok_or_else(|| SplatError::Process("分割会话缺少原视频校验。".into()))
}

fn preview_indexes(count: usize, prompt: usize) -> Vec<usize> {
    let mut indexes = vec![prompt];
    if count > 1 {
        indexes.push(0);
        indexes.push(count - 1);
    }
    indexes.sort_unstable();
    indexes.dedup();
    indexes
}

fn preview_png(path: &Path, max_edge: u32) -> Result<Vec<u8>> {
    let image = image::open(path).map_err(|error| SplatError::Process(error.to_string()))?;
    let (width, height) = (image.width().max(1), image.height().max(1));
    let scale = f64::from(max_edge) / f64::from(width.max(height));
    let resized = if scale < 1.0 {
        image.resize(
            ((f64::from(width) * scale).round() as u32).max(1),
            ((f64::from(height) * scale).round() as u32).max(1),
            image::imageops::FilterType::Triangle,
        )
    } else {
        image
    };
    let mut bytes = Cursor::new(Vec::new());
    resized
        .write_to(&mut bytes, image::ImageFormat::Png)
        .map_err(|error| SplatError::Process(error.to_string()))?;
    Ok(bytes.into_inner())
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let a = chunk[0] as u32;
        let b = chunk.get(1).copied().unwrap_or(0) as u32;
        let c = chunk.get(2).copied().unwrap_or(0) as u32;
        let value = (a << 16) | (b << 8) | c;
        out.push(TABLE[((value >> 18) & 63) as usize] as char);
        out.push(TABLE[((value >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((value >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(value & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
