//! Optional ordinary-video subject masks. Absence means the historical full scene.
//! Masks are a separate Brush input. They never replace RGB and never feed COLMAP.

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

use image::{GrayImage, Luma};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    error::{Result, SplatError},
    reconstruction::direct_input::{Model, Point},
};

pub const VERSION: u32 = 1;
pub const MAX_FRAMES: usize = 400;
pub const MAX_PROMPTS: usize = 32;
pub const MIN_PROMPT_COVERAGE: f64 = 0.005;
pub const MAX_COVERAGE: f64 = 0.95;
pub const SESSION_DIRECTORY: &str = ".ooosplat-subject-masks";

pub const TRAINING_BLOCKED: &str = "\
当前 Brush 不能确认会按掩码计算损失，单物体模式已停止在训练之前。\
锁定的 OOOBrush ooo-v1.0.0（866a7d65b4d6a591ba0d52a88f8b20ea3465c1e5）\
需要同时提供 --alpha-mode masked 和 masks 目录；白色保留、黑色忽略，并且不修改 RGB。\
本次没有把背景涂黑，也没有启动训练。请更新到含 --alpha-mode 与 --invert-masks 的同一支 OOOBrush 后再生成。";

/// Git LFS size and SHA256 of ZhengPeng7/BiRefNet `model.safetensors`
/// at revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4. Not an estimate.
pub const BIREFNET_BYTES: u64 = 444_473_596;
pub const BIREFNET_SHA256: &str =
    "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154";
const BIREFNET_CODE_FILES: [&str; 3] = ["config.json", "birefnet.py", "BiRefNet_config.py"];
const FORBIDDEN_WEIGHT_TOKENS: [&str; 6] = ["bria", "rmbg", "u2net", "isnet", "silueta", "rembg"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SubjectMaskStatus {
    Prepared,
    Segmented,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskFrame {
    pub name: String,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
    pub mask: String,
    pub mask_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubjectPrompt {
    pub frame_index: u32,
    pub points: Vec<(f64, f64)>,
    pub labels: Vec<u8>,
    pub box_xyxy: Option<[f64; 4]>,
    /// Missing on masks created before automatic first-pass: those stay manual SAM prompts.
    #[serde(default)]
    pub initializer: SubjectInitializer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum SubjectInitializer {
    #[default]
    SamPrompt,
    BiRefNet,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubjectMaskCheckpoint {
    pub version: u32,
    pub status: SubjectMaskStatus,
    pub session_id: Uuid,
    pub model_id: String,
    pub model_cfg: String,
    pub checkpoint_sha256: String,
    pub source_video_sha256: String,
    pub dtype: String,
    pub prompt: SubjectPrompt,
    pub frames: Vec<MaskFrame>,
    pub prompt_coverage: f64,
    /// Set only when BiRefNet produced the first-frame mask. Older manifests omit both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_checkpoint_sha256: Option<String>,
}

impl SubjectMaskCheckpoint {
    pub fn validate(&self) -> Result<()> {
        let prompted = !self.prompt.points.is_empty() || self.prompt.box_xyxy.is_some();
        let automatic = self.prompt.initializer == SubjectInitializer::BiRefNet;
        let init_hash_ok = self
            .init_checkpoint_sha256
            .as_deref()
            .is_some_and(|hash| hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()));
        if self.version != VERSION
            || self.frames.is_empty()
            || self.frames.len() > MAX_FRAMES
            || self.prompt.points.len() != self.prompt.labels.len()
            || self.prompt.points.len() > MAX_PROMPTS
            || (automatic && (prompted || self.prompt.frame_index != 0))
            || (!automatic && !prompted)
            || (automatic && (!init_hash_ok || self.init_model_id.as_deref() != Some("BiRefNet")))
            || (!automatic
                && (self.init_model_id.is_some() || self.init_checkpoint_sha256.is_some()))
            || self.prompt.frame_index as usize >= self.frames.len()
            || !(MIN_PROMPT_COVERAGE..=MAX_COVERAGE).contains(&self.prompt_coverage)
            || self.model_cfg.contains(['\0', '\n', '\r'])
            || self.dtype != "bf16" && self.dtype != "fp32"
        {
            return Err(invalid("subject mask checkpoint is incomplete"));
        }
        if self.status != SubjectMaskStatus::Segmented {
            return Err(invalid("subject mask has not been segmented"));
        }
        for frame in &self.frames {
            if !safe_file_name(&frame.name) || !safe_file_name(&frame.mask) {
                return Err(invalid("subject mask frame name is not a single file name"));
            }
            if frame.width == 0 || frame.height == 0 || frame.sha256.len() != 64 {
                return Err(invalid("subject mask frame record is invalid"));
            }
        }
        Ok(())
    }
}

pub fn session_directory(projects_root: &Path, session_id: Uuid) -> Result<PathBuf> {
    let root = projects_root
        .canonicalize()
        .map_err(|_| invalid("projects root is unavailable"))?;
    let path = root.join(SESSION_DIRECTORY).join(session_id.to_string());
    if !path.starts_with(&root) {
        return Err(invalid("subject mask session escaped the projects root"));
    }
    Ok(path)
}

pub fn load_segmented(projects_root: &Path, session_id: Uuid) -> Result<SubjectMaskCheckpoint> {
    let path = session_directory(projects_root, session_id)?.join("manifest.json");
    let checkpoint: SubjectMaskCheckpoint = serde_json::from_slice(
        &std::fs::read(&path).map_err(|_| invalid("subject mask session is missing"))?,
    )
    .map_err(|_| invalid("subject mask session cannot be read"))?;
    if checkpoint.session_id != session_id {
        return Err(invalid(
            "subject mask session id does not match its manifest",
        ));
    }
    checkpoint.validate()?;
    Ok(checkpoint)
}

pub fn sha256_file(path: &Path, cancel: &CancellationToken) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if cancel.is_cancelled() {
            return Err(SplatError::Cancelled);
        }
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn frames_match(
    checkpoint: &SubjectMaskCheckpoint,
    frames_dir: &Path,
    cancel: &CancellationToken,
) -> Result<()> {
    checkpoint.validate()?;
    let mut seen = HashMap::new();
    for entry in std::fs::read_dir(frames_dir)? {
        let path = entry?.path();
        if path.is_file() {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| invalid("training frame name is not valid UTF-8"))?
                .to_owned();
            seen.insert(name, path);
        }
    }
    if seen.len() != checkpoint.frames.len() {
        return Err(mismatch());
    }
    for frame in &checkpoint.frames {
        let path = seen.get(&frame.name).ok_or_else(mismatch)?;
        let (width, height) = image::image_dimensions(path)
            .map_err(|error| SplatError::Process(error.to_string()))?;
        if width != frame.width
            || height != frame.height
            || sha256_file(path, cancel)? != frame.sha256
        {
            return Err(mismatch());
        }
    }
    Ok(())
}

pub fn brush_mask_file(image_name: &str) -> Result<String> {
    if !safe_file_name(image_name) {
        return Err(invalid("training image name cannot be used as a mask name"));
    }
    let stem = Path::new(image_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("training image name cannot be used as a mask name"))?;
    Ok(format!("{stem}.png"))
}

pub fn copy_aligned_mask(source: &Path, destination: &Path, width: u32, height: u32) -> Result<()> {
    let mask = open_mask(source)?;
    if mask.width() != width || mask.height() != height {
        return Err(invalid(
            "subject mask size does not match the training image",
        ));
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)?;
    }
    mask.save_with_format(destination, image::ImageFormat::Png)
        .map_err(|error| SplatError::Process(error.to_string()))?;
    Ok(())
}

pub fn resample_mask(
    source: &GrayImage,
    out_width: u32,
    out_height: u32,
    cancel: &CancellationToken,
    map_output_to_source: impl Fn(u32, u32) -> Result<(f64, f64)>,
) -> Result<GrayImage> {
    if out_width == 0 || out_height == 0 || out_width > 16_384 || out_height > 16_384 {
        return Err(invalid(
            "undistorted subject mask size is outside the supported range",
        ));
    }
    let mut output = GrayImage::new(out_width, out_height);
    for y in 0..out_height {
        if cancel.is_cancelled() {
            return Err(SplatError::Cancelled);
        }
        for x in 0..out_width {
            let (px, py) = map_output_to_source(x, y)?;
            let value = if px.is_finite() && py.is_finite() {
                let sx = px.round() as i64;
                let sy = py.round() as i64;
                if sx >= 0 && sy >= 0 && sx < source.width() as i64 && sy < source.height() as i64 {
                    source.get_pixel(sx as u32, sy as u32)[0]
                } else {
                    0
                }
            } else {
                0
            };
            output.put_pixel(x, y, Luma([value]));
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointFilter {
    pub kept: usize,
    pub removed: usize,
}

pub fn filter_background_points(
    model: &mut Model,
    masks: &HashMap<u32, GrayImage>,
) -> Result<PointFilter> {
    let images = model
        .images
        .iter()
        .map(|image| (image.pose.id, image))
        .collect::<HashMap<_, _>>();
    let before = model.points.len();
    model
        .points
        .retain(|point| keep_point(point, &images, masks));
    Ok(PointFilter {
        kept: model.points.len(),
        removed: before.saturating_sub(model.points.len()),
    })
}

fn keep_point(
    point: &Point,
    images: &HashMap<u32, &crate::reconstruction::direct_input::ModelImage>,
    masks: &HashMap<u32, GrayImage>,
) -> bool {
    let mut seen = 0_u32;
    let mut foreground = 0_u32;
    for &(image_id, index) in &point.track {
        let Some(image) = images.get(&image_id) else {
            continue;
        };
        let Some(mask) = masks.get(&image_id) else {
            continue;
        };
        let Some(observation) = image.observations.get(index as usize) else {
            continue;
        };
        if !observation.x.is_finite() || !observation.y.is_finite() {
            continue;
        }
        seen += 1;
        if mask_value(mask, observation.x, observation.y) >= 128 {
            foreground += 1;
        }
    }
    foreground >= 2 || (seen > 0 && foreground == seen)
}

fn mask_value(mask: &GrayImage, x: f64, y: f64) -> u8 {
    let sx = x.round() as i64;
    let sy = y.round() as i64;
    if sx < 0 || sy < 0 || sx >= mask.width() as i64 || sy >= mask.height() as i64 {
        0
    } else {
        mask.get_pixel(sx as u32, sy as u32)[0]
    }
}

pub fn coverage(mask: &GrayImage) -> f64 {
    let total = mask.width() as f64 * mask.height() as f64;
    if total == 0.0 {
        return 0.0;
    }
    let foreground = mask.pixels().filter(|pixel| pixel[0] >= 128).count() as f64;
    foreground / total
}

pub fn accept_automatic_coverage(value: f64) -> Result<()> {
    if (MIN_PROMPT_COVERAGE..=MAX_COVERAGE).contains(&value) {
        Ok(())
    } else if value > MAX_COVERAGE {
        Err(invalid(
            "BiRefNet 首帧掩码几乎覆盖整帧。不能当作单物体分割成功。请改用点或框，不要继续跟踪。",
        ))
    } else {
        Err(invalid(
            "BiRefNet 首帧里没有足够的主体像素。请改用点或框，不要用空白掩码继续。",
        ))
    }
}

pub fn accept_prompt_coverage(value: f64) -> Result<()> {
    if (MIN_PROMPT_COVERAGE..=MAX_COVERAGE).contains(&value) {
        Ok(())
    } else if value > MAX_COVERAGE {
        Err(invalid(
            "掩码几乎覆盖整帧。这不能当作单物体分割成功，请改点或框后重跑。",
        ))
    } else {
        Err(invalid(
            "提示帧里没有足够的主体像素。请改点或框后重跑，不要用空白掩码继续。",
        ))
    }
}

pub fn read_brush_help(executable: &Path) -> Result<String> {
    if !executable.is_file() {
        return Err(SplatError::EngineMissing(executable.display().to_string()));
    }
    let output = std::process::Command::new(executable)
        .arg("--help")
        .output()
        .map_err(|error| SplatError::Process(format!("无法读取 Brush 参数：{error}")))?;
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text.truncate(1024 * 1024);
    Ok(text)
}

pub fn brush_help_supports_masked_loss(help: &str) -> bool {
    help.lines().any(|line| line.contains("--alpha-mode"))
        && help.lines().any(|line| line.contains("--invert-masks"))
}

pub fn sam2_arguments(script: &Path, job: &Path) -> Vec<std::ffi::OsString> {
    vec![
        script.as_os_str().to_owned(),
        "--job".into(),
        job.as_os_str().to_owned(),
    ]
}

pub fn birefnet_arguments(script: &Path, job: &Path) -> Vec<std::ffi::OsString> {
    sam2_arguments(script, job)
}

pub fn official_birefnet_weight(bytes: u64, sha256: &str) -> bool {
    bytes == BIREFNET_BYTES && sha256.eq_ignore_ascii_case(BIREFNET_SHA256)
}

pub fn plan_segment(
    automatic: bool,
    frame_index: u32,
    points: &[(f64, f64)],
    labels: &[u8],
    box_xyxy: Option<[f64; 4]>,
    birefnet_checkpoint: &str,
) -> Result<SubjectInitializer> {
    if automatic {
        if frame_index != 0 || !points.is_empty() || box_xyxy.is_some() {
            return Err(invalid(
                "自动首帧只使用第 0 帧，并且不接受点或框。若要修正，请改用手动 SAM 提示。",
            ));
        }
        if birefnet_checkpoint.trim().is_empty() {
            return Err(invalid(
                "还没有填写 BiRefNet 本地快照。自动首帧已停止，不会自动下载权重。",
            ));
        }
        if birefnet_checkpoint.chars().any(|c| c.is_control())
            || forbidden_weight_path(birefnet_checkpoint)
        {
            return Err(invalid(
                "BiRefNet 路径无效。自动首帧只接受官方 MIT 权重，不会改用 Bria。",
            ));
        }
        return Ok(SubjectInitializer::BiRefNet);
    }
    if points.len() != labels.len()
        || points.len() > MAX_PROMPTS
        || (points.is_empty() && box_xyxy.is_none())
        || labels.iter().any(|label| *label > 1)
    {
        return Err(invalid("主体提示无效。需要前景点或主体框。"));
    }
    Ok(SubjectInitializer::SamPrompt)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedBiRefNet {
    pub snapshot: PathBuf,
    pub weight: PathBuf,
    pub sha256: String,
}

pub fn resolve_birefnet_snapshot(checkpoint: &Path) -> Result<ResolvedBiRefNet> {
    if forbidden_weight_path(&checkpoint.to_string_lossy()) {
        return Err(invalid(
            "这个路径像是 Bria RMBG、rembg 或其他未授权权重。自动首帧只接受官方 MIT BiRefNet，不会改用 Bria。",
        ));
    }
    let snapshot = if checkpoint.is_dir() {
        checkpoint.to_path_buf()
    } else {
        checkpoint
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };
    if !checkpoint.is_dir()
        && checkpoint.file_name().and_then(|name| name.to_str()) != Some("model.safetensors")
    {
        return Err(invalid(
            "只接受官方文件名 model.safetensors。请指向 ZhengPeng7/BiRefNet 的本地快照。",
        ));
    }
    let weight = snapshot.join("model.safetensors");
    if !weight.is_file() {
        return Err(invalid(
            "找不到官方 BiRefNet 权重 model.safetensors。本程序不会自动下载。",
        ));
    }
    for name in BIREFNET_CODE_FILES {
        if !snapshot.join(name).is_file() {
            return Err(invalid(&format!(
                "本地快照缺少官方代码 {name}。请下载完整的 ZhengPeng7/BiRefNet 目录，不要只放权重。本程序不会自动下载。"
            )));
        }
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)
            .map_err(|error| invalid(&format!("无法读取官方 BiRefNet 配置：{error}")))?;
    if config.get("bb_pretrained") != Some(&serde_json::Value::Bool(false)) {
        return Err(invalid(
            "官方配置必须是 bb_pretrained=false，避免下载骨干网络。",
        ));
    }
    if config
        .get("auto_map")
        .and_then(|value| value.get("AutoModelForImageSegmentation"))
        .and_then(|value| value.as_str())
        != Some("birefnet.BiRefNet")
    {
        return Err(invalid("本地代码不是官方 BiRefNet。"));
    }
    if config.get("architectures") != Some(&serde_json::json!(["BiRefNet"])) {
        return Err(invalid("本地配置的模型结构不是 BiRefNet。"));
    }
    let bytes = weight.metadata()?.len();
    if bytes != BIREFNET_BYTES {
        return Err(invalid(&format!(
            "model.safetensors 字节数为 {bytes}，官方文件是 {BIREFNET_BYTES}。请重新下载，不要继续。"
        )));
    }
    let digest = sha256_file(&weight, &CancellationToken::new())?;
    if !official_birefnet_weight(bytes, &digest) {
        return Err(invalid(
            "model.safetensors 的 SHA256 与官方 Git LFS 校验不一致。请重新下载，不要继续。",
        ));
    }
    Ok(ResolvedBiRefNet {
        snapshot,
        weight,
        sha256: digest,
    })
}

pub fn birefnet_job(
    session: &Path,
    frame: &Path,
    output: &Path,
    checkpoint: &str,
) -> serde_json::Value {
    serde_json::json!({
        "sessionDir": session,
        "frame": frame,
        "output": output,
        "checkpoint": checkpoint,
    })
}

pub struct Sam2Job<'a> {
    pub session: &'a Path,
    pub checkpoint: &'a str,
    pub model_cfg: &'a str,
    pub frame_index: usize,
    pub points: &'a [(f64, f64)],
    pub labels: &'a [u8],
    pub box_xyxy: Option<[f64; 4]>,
    pub init_mask: Option<&'a Path>,
}

pub fn sam2_job(job: Sam2Job<'_>) -> serde_json::Value {
    let mut value = serde_json::json!({
        "sessionDir": job.session,
        "framesDir": job.session.join("frames"),
        "masksDir": job.session.join("masks"),
        "indexedDir": job.session.join("sam2-frames"),
        "checkpoint": job.checkpoint,
        "modelCfg": job.model_cfg,
        "frameIndex": job.frame_index,
        "points": job.points,
        "labels": job.labels,
        "box": job.box_xyxy,
    });
    if let Some(mask) = job.init_mask {
        value["initMask"] = serde_json::Value::String(mask.display().to_string());
    }
    value
}

fn forbidden_weight_path(path: &str) -> bool {
    let lowered = path.to_ascii_lowercase();
    FORBIDDEN_WEIGHT_TOKENS
        .iter()
        .any(|token| lowered.contains(token))
}

pub fn normalized_to_pixel(x: f64, y: f64, width: u32, height: u32) -> Result<(f64, f64)> {
    if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) || width == 0 || height == 0 {
        return Err(invalid("subject prompt is outside the frame"));
    }
    Ok((x * f64::from(width), y * f64::from(height)))
}

fn open_mask(path: &Path) -> Result<GrayImage> {
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(crate::video::panorama::MAX_DECODE_BYTES);
    reader.limits(limits);
    reader
        .decode()
        .map(|image| image.into_luma8())
        .map_err(|error| SplatError::Process(error.to_string()))
}

fn safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 240
        && Path::new(name)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && !name.contains(['\0', '/', '\\', ':'])
}

fn invalid(message: &str) -> SplatError {
    SplatError::Process(message.to_owned())
}

fn mismatch() -> SplatError {
    invalid(
        "抽帧与已确认的单物体掩码不一致。请重新分割；不会使用未对齐的掩码，也不会退回整场景训练。",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruction::direct_input::{ModelImage, Observation};
    use crate::video::panorama::PosedImage;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn image(id: u32, observations: Vec<(f64, f64)>) -> ModelImage {
        ModelImage {
            pose: PosedImage {
                id,
                camera_id: 1,
                name: format!("frame_{id:010}.jpg"),
                qvec: [1.0, 0.0, 0.0, 0.0],
                tvec: [0.0, 0.0, 0.0],
            },
            observations: observations
                .into_iter()
                .map(|(x, y)| Observation { x, y, point: None })
                .collect(),
        }
    }

    fn point(track: Vec<(u32, u32)>) -> Point {
        Point {
            id: 1,
            xyz: [0.0, 0.0, 1.0],
            rgb: [0, 0, 0],
            error: 0.1,
            track,
        }
    }

    fn half_mask() -> GrayImage {
        let mut mask = GrayImage::from_pixel(4, 2, Luma([0]));
        for x in 0..2 {
            mask.put_pixel(x, 0, Luma([255]));
            mask.put_pixel(x, 1, Luma([255]));
        }
        mask
    }

    #[test]
    fn old_project_state_stays_a_full_scene() {
        let state = crate::project::PipelineStateFile::created(crate::presets::Quality::Balanced);
        let mut value = serde_json::to_value(&state).unwrap();
        assert!(value.get("subjectMask").is_none());
        value.as_object_mut().unwrap().remove("subjectMask");
        let restored: crate::project::PipelineStateFile = serde_json::from_value(value).unwrap();
        assert!(restored.subject_mask.is_none());
    }

    #[test]
    fn multi_view_subject_points_stay_and_background_points_go() {
        let mask = half_mask();
        let mut model = Model {
            cameras: vec![],
            images: vec![
                image(1, vec![(0.0, 0.0), (3.0, 0.0)]),
                image(2, vec![(1.0, 1.0), (3.0, 1.0)]),
                image(3, vec![(3.0, 0.0)]),
            ],
            points: vec![
                point(vec![(1, 0), (2, 0), (3, 0)]),
                point(vec![(1, 1), (2, 1)]),
                point(vec![(1, 0)]),
                point(vec![(1, 1), (2, 0)]),
            ],
        };
        let masks = HashMap::from([(1, mask.clone()), (2, mask.clone()), (3, mask)]);
        let report = filter_background_points(&mut model, &masks).unwrap();
        assert_eq!(report.kept, 2);
        assert_eq!(report.removed, 2);
        assert_eq!(model.points[0].track.len(), 3);
        assert_eq!(model.points[1].track, vec![(1, 0)]);
    }

    #[test]
    fn full_foreground_and_empty_prompt_masks_are_rejected() {
        let full = GrayImage::from_pixel(4, 4, Luma([255]));
        let empty = GrayImage::from_pixel(4, 4, Luma([0]));
        assert!(accept_prompt_coverage(coverage(&full)).is_err());
        assert!(accept_prompt_coverage(coverage(&empty)).is_err());
        assert!(accept_prompt_coverage(coverage(&half_mask())).is_ok());
    }

    #[test]
    fn normalized_prompts_map_into_the_extracted_frame() {
        assert_eq!(
            normalized_to_pixel(0.25, 0.5, 640, 480).unwrap(),
            (160.0, 240.0)
        );
        assert!(normalized_to_pixel(1.1, 0.0, 10, 10).is_err());
    }

    #[test]
    fn resample_uses_the_source_pixel_and_does_not_fill_outside() {
        let mut source = GrayImage::from_pixel(2, 2, Luma([0]));
        source.put_pixel(0, 0, Luma([255]));
        let output = resample_mask(&source, 2, 2, &CancellationToken::new(), |x, y| {
            Ok((x as f64, y as f64))
        })
        .unwrap();
        assert_eq!(output.get_pixel(0, 0)[0], 255);
        assert_eq!(output.get_pixel(1, 0)[0], 0);
        let outside = resample_mask(&source, 1, 1, &CancellationToken::new(), |_, _| {
            Ok((-1.0, 0.0))
        })
        .unwrap();
        assert_eq!(outside.get_pixel(0, 0)[0], 0);
    }

    #[test]
    fn changed_frame_bytes_do_not_match_an_accepted_mask() {
        let root = tempfile::tempdir().unwrap();
        let name = "frame_0000000001.jpg";
        let path = root.path().join(name);
        image::RgbImage::from_pixel(1, 1, image::Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();
        let hash = sha256_file(&path, &CancellationToken::new()).unwrap();
        let checkpoint = SubjectMaskCheckpoint {
            version: VERSION,
            status: SubjectMaskStatus::Segmented,
            session_id: Uuid::nil(),
            model_id: "sam2.1_hiera_tiny".into(),
            model_cfg: "configs/sam2.1/sam2.1_hiera_t.yaml".into(),
            checkpoint_sha256: "ab".repeat(32),
            source_video_sha256: "cd".repeat(32),
            dtype: "fp32".into(),
            prompt: SubjectPrompt {
                frame_index: 0,
                points: vec![(0.2, 0.2)],
                labels: vec![1],
                box_xyxy: None,
                initializer: SubjectInitializer::SamPrompt,
            },
            init_model_id: None,
            init_checkpoint_sha256: None,
            frames: vec![MaskFrame {
                name: name.into(),
                sha256: hash,
                width: 1,
                height: 1,
                mask: "frame_0000000001.png".into(),
                mask_sha256: "ef".repeat(32),
            }],
            prompt_coverage: 0.25,
        };
        image::RgbImage::from_pixel(1, 1, image::Rgb([4, 5, 6]))
            .save(&path)
            .unwrap();
        assert!(frames_match(&checkpoint, root.path(), &CancellationToken::new()).is_err());
    }

    #[test]
    fn cancellation_interrupts_hashing_and_arguments_are_not_a_shell_string() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("frame.jpg");
        std::fs::write(&path, vec![7_u8; 1024]).unwrap();
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(matches!(
            sha256_file(&path, &cancel),
            Err(SplatError::Cancelled)
        ));
        let args = sam2_arguments(
            Path::new("scripts/subject_mask/sam2_video.py"),
            Path::new("job.json"),
        );
        assert_eq!(args.len(), 3);
        assert!(args.iter().all(|arg| !arg.to_string_lossy().contains("&&")));
        let _ = SystemTime::now().duration_since(UNIX_EPOCH);
    }

    #[test]
    fn session_ids_cannot_escape_the_projects_root() {
        let root = tempfile::tempdir().unwrap();
        let session = session_directory(root.path(), Uuid::nil()).unwrap();
        assert!(session.starts_with(root.path().canonicalize().unwrap()));
        assert!(session.ends_with("00000000-0000-0000-0000-000000000000"));
    }

    #[test]
    fn brush_help_must_advertise_both_mask_flags() {
        assert!(!brush_help_supports_masked_loss("--alpha-mode\n"));
        assert!(brush_help_supports_masked_loss(
            "--alpha-mode <MODE>\n--invert-masks\n"
        ));
    }

    fn manual_checkpoint(name: &str, hash: String) -> SubjectMaskCheckpoint {
        SubjectMaskCheckpoint {
            version: VERSION,
            status: SubjectMaskStatus::Segmented,
            session_id: Uuid::nil(),
            model_id: "sam2.1_hiera_tiny.pt".into(),
            model_cfg: "configs/sam2.1/sam2.1_hiera_t.yaml".into(),
            checkpoint_sha256: "ab".repeat(32),
            source_video_sha256: "cd".repeat(32),
            dtype: "fp32".into(),
            prompt: SubjectPrompt {
                frame_index: 0,
                points: vec![(0.2, 0.2)],
                labels: vec![1],
                box_xyxy: None,
                initializer: SubjectInitializer::SamPrompt,
            },
            frames: vec![MaskFrame {
                name: name.into(),
                sha256: hash,
                width: 1,
                height: 1,
                mask: "frame.png".into(),
                mask_sha256: "ef".repeat(32),
            }],
            prompt_coverage: 0.25,
            init_model_id: None,
            init_checkpoint_sha256: None,
        }
    }

    #[test]
    fn old_manual_checkpoint_without_initializer_still_validates() {
        let checkpoint = manual_checkpoint("frame.jpg", "ab".repeat(32));
        let mut value = serde_json::to_value(&checkpoint).unwrap();
        value["prompt"]
            .as_object_mut()
            .unwrap()
            .remove("initializer");
        value.as_object_mut().unwrap().remove("initModelId");
        value
            .as_object_mut()
            .unwrap()
            .remove("initCheckpointSha256");
        let restored: SubjectMaskCheckpoint = serde_json::from_value(value).unwrap();
        assert_eq!(restored.prompt.initializer, SubjectInitializer::SamPrompt);
        restored.validate().unwrap();
    }

    #[test]
    fn automatic_checkpoint_keeps_hash_rejection_without_point_prompts() {
        let root = tempfile::tempdir().unwrap();
        let name = "frame_0000000001.jpg";
        let path = root.path().join(name);
        image::RgbImage::from_pixel(1, 1, image::Rgb([1, 2, 3]))
            .save(&path)
            .unwrap();
        let hash = sha256_file(&path, &CancellationToken::new()).unwrap();
        let mut checkpoint = manual_checkpoint(name, hash);
        checkpoint.prompt.points.clear();
        checkpoint.prompt.labels.clear();
        checkpoint.prompt.initializer = SubjectInitializer::BiRefNet;
        checkpoint.init_model_id = Some("BiRefNet".into());
        checkpoint.init_checkpoint_sha256 = Some(BIREFNET_SHA256.into());
        checkpoint.validate().unwrap();
        image::RgbImage::from_pixel(1, 1, image::Rgb([4, 5, 6]))
            .save(&path)
            .unwrap();
        assert!(frames_match(&checkpoint, root.path(), &CancellationToken::new()).is_err());
    }

    #[test]
    fn automatic_plan_needs_birefnet_and_manual_plan_still_needs_prompts() {
        let missing = plan_segment(true, 0, &[], &[], None, "");
        assert!(missing.unwrap_err().to_string().contains("不会自动下载"));
        assert_eq!(
            plan_segment(true, 0, &[], &[], None, "/models/BiRefNet").unwrap(),
            SubjectInitializer::BiRefNet
        );
        assert!(plan_segment(true, 1, &[], &[], None, "/models/BiRefNet").is_err());
        assert!(plan_segment(true, 0, &[(0.2, 0.2)], &[1], None, "/models/BiRefNet").is_err());
        assert!(plan_segment(false, 0, &[], &[], None, "/models/BiRefNet").is_err());
        assert_eq!(
            plan_segment(false, 0, &[(0.2, 0.2)], &[1], None, "").unwrap(),
            SubjectInitializer::SamPrompt
        );
        assert!(plan_segment(
            true,
            0,
            &[],
            &[],
            None,
            "/weights/bria-rmbg/model.safetensors"
        )
        .is_err());
    }

    #[test]
    fn automatic_sam_job_carries_the_init_mask_and_manual_job_keeps_points() {
        let session = Path::new("/projects/.ooosplat-subject-masks/session");
        let init = session.join("birefnet-init.png");
        let automatic = sam2_job(Sam2Job {
            session,
            checkpoint: "sam2.1_hiera_tiny.pt",
            model_cfg: "configs/sam2.1/sam2.1_hiera_t.yaml",
            frame_index: 0,
            points: &[],
            labels: &[],
            box_xyxy: None,
            init_mask: Some(&init),
        });
        assert!(automatic["points"].as_array().unwrap().is_empty());
        assert!(automatic.get("box").unwrap().is_null());
        assert!(automatic["initMask"]
            .as_str()
            .unwrap()
            .ends_with("birefnet-init.png"));
        let manual = sam2_job(Sam2Job {
            session,
            checkpoint: "sam2.1_hiera_tiny.pt",
            model_cfg: "configs/sam2.1/sam2.1_hiera_t.yaml",
            frame_index: 0,
            points: &[(12.0, 8.0)],
            labels: &[1],
            box_xyxy: None,
            init_mask: None,
        });
        assert!(manual.get("initMask").is_none());
        assert_eq!(manual["points"][0][0], 12.0);
        assert_eq!(manual["labels"][0], 1);
    }

    #[test]
    fn copying_a_subject_mask_does_not_change_the_colmap_frame() {
        let root = tempfile::tempdir().unwrap();
        let colmap = root.path().join("colmap_frames");
        std::fs::create_dir_all(&colmap).unwrap();
        let frame = colmap.join("frame_0000000001.jpg");
        image::RgbImage::from_pixel(2, 2, image::Rgb([9, 8, 7]))
            .save(&frame)
            .unwrap();
        let before = std::fs::read(&frame).unwrap();
        let source = root.path().join("init.png");
        let mut mask = GrayImage::from_pixel(2, 2, Luma([0]));
        mask.put_pixel(0, 0, Luma([255]));
        mask.save(&source).unwrap();
        let destination = root
            .path()
            .join("dataset")
            .join("masks")
            .join(brush_mask_file("frame_0000000001.jpg").unwrap());
        copy_aligned_mask(&source, &destination, 2, 2).unwrap();
        assert_eq!(std::fs::read(&frame).unwrap(), before);
        assert!(destination.is_file());
        assert!(!colmap.join("frame_0000000001.png").exists());
        assert_ne!(destination, frame);
    }

    #[test]
    fn birefnet_snapshot_must_be_the_official_local_file() {
        assert!(official_birefnet_weight(BIREFNET_BYTES, BIREFNET_SHA256));
        assert!(!official_birefnet_weight(
            BIREFNET_BYTES - 1,
            BIREFNET_SHA256
        ));
        assert!(!official_birefnet_weight(
            BIREFNET_BYTES,
            "ab".repeat(32).as_str()
        ));
        let missing = resolve_birefnet_snapshot(Path::new(
            "/tmp/ooosplat-missing-birefnet/model.safetensors",
        ));
        assert!(missing.unwrap_err().to_string().contains("不会自动下载"));
        let forbidden = resolve_birefnet_snapshot(Path::new("/tmp/bria-rmbg/model.safetensors"));
        assert!(forbidden.unwrap_err().to_string().contains("Bria"));
        let root = tempfile::tempdir().unwrap();
        let weight = root.path().join("model.safetensors");
        std::fs::write(&weight, b"not-the-official-weight").unwrap();
        std::fs::write(root.path().join("birefnet.py"), b"class BiRefNet: pass\n").unwrap();
        std::fs::write(
            root.path().join("BiRefNet_config.py"),
            b"class BiRefNetConfig: pass\n",
        )
        .unwrap();
        std::fs::write(
            root.path().join("config.json"),
            r#"{"bb_pretrained":true,"architectures":["BiRefNet"],"auto_map":{"AutoModelForImageSegmentation":"birefnet.BiRefNet"}}"#,
        )
        .unwrap();
        assert!(resolve_birefnet_snapshot(&weight)
            .unwrap_err()
            .to_string()
            .contains("bb_pretrained"));
        std::fs::write(
            root.path().join("config.json"),
            r#"{"bb_pretrained":false,"architectures":["BiRefNet"],"auto_map":{"AutoModelForImageSegmentation":"birefnet.BiRefNet"}}"#,
        )
        .unwrap();
        assert!(resolve_birefnet_snapshot(&weight)
            .unwrap_err()
            .to_string()
            .contains("字节数"));
    }

    fn subject_mask_script(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../scripts/subject_mask")
            .join(name)
    }

    fn python_output(script: &Path, args: &[&std::ffi::OsStr]) -> (bool, String) {
        let python = ["python3", "python"]
            .into_iter()
            .find(|name| {
                std::process::Command::new(name)
                    .arg("--version")
                    .output()
                    .is_ok_and(|output| output.status.success())
            })
            .expect("python3 or python is required to prove subject-mask preflight fails closed");
        let output = std::process::Command::new(python)
            .arg(script)
            .args(args)
            .output()
            .expect("python started");
        (
            output.status.success(),
            String::from_utf8_lossy(&output.stdout).into_owned(),
        )
    }

    fn last_python_json(stdout: &str) -> serde_json::Value {
        let line = stdout
            .lines()
            .rev()
            .find(|line| line.starts_with('{'))
            .unwrap_or("{}");
        serde_json::from_str(line).unwrap()
    }

    #[test]
    fn birefnet_preflight_fails_when_weights_or_runtime_are_missing() {
        let script = subject_mask_script("birefnet_init.py");
        let (ok, stdout) = python_output(
            &script,
            &[
                "--preflight".as_ref(),
                "--checkpoint".as_ref(),
                "/tmp/ooosplat-missing-birefnet/model.safetensors".as_ref(),
            ],
        );
        assert!(!ok);
        let payload = last_python_json(&stdout);
        assert_eq!(payload["ok"], false);
        assert!(payload["message"]
            .as_str()
            .unwrap()
            .contains("不会自动下载"));
        let (runtime_ok, runtime_stdout) = python_output(&script, &["--preflight".as_ref()]);
        assert!(!runtime_ok);
        let runtime = last_python_json(&runtime_stdout);
        assert_eq!(runtime["ok"], false);
        let message = runtime["message"].as_str().unwrap();
        assert!(
            message.contains("PyTorch")
                || message.contains("CUDA")
                || message.contains("transformers")
                || message.contains("未填写"),
            "{message}"
        );
        let (forbidden_ok, forbidden_stdout) = python_output(
            &script,
            &[
                "--preflight".as_ref(),
                "--checkpoint".as_ref(),
                "/tmp/bria-rmbg/model.safetensors".as_ref(),
            ],
        );
        assert!(!forbidden_ok);
        assert!(last_python_json(&forbidden_stdout)["message"]
            .as_str()
            .unwrap()
            .contains("Bria"));
    }

    #[test]
    fn sam2_manual_path_still_requires_prompts_and_init_mask_cannot_mix() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("frames")).unwrap();
        let job_path = root.path().join("job.json");
        let manual = serde_json::json!({
            "sessionDir": root.path(),
            "framesDir": root.path().join("frames"),
            "masksDir": root.path().join("masks"),
            "indexedDir": root.path().join("sam2-frames"),
            "checkpoint": root.path().join("missing.pt"),
            "modelCfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "frameIndex": 0,
            "points": [],
            "labels": [],
            "box": null,
        });
        std::fs::write(&job_path, serde_json::to_vec(&manual).unwrap()).unwrap();
        let script = subject_mask_script("sam2_video.py");
        let (ok, stdout) = python_output(&script, &["--job".as_ref(), job_path.as_os_str()]);
        assert!(!ok);
        let payload = last_python_json(&stdout);
        assert_eq!(payload["ok"], false);
        assert_eq!(payload["code"], "bad_prompt");
        let mut mixed = manual.clone();
        mixed["initMask"] =
            serde_json::Value::String(root.path().join("birefnet-init.png").display().to_string());
        mixed["points"] = serde_json::json!([[0.2, 0.2]]);
        mixed["labels"] = serde_json::json!([1]);
        std::fs::write(&job_path, serde_json::to_vec(&mixed).unwrap()).unwrap();
        let (mixed_ok, mixed_stdout) =
            python_output(&script, &["--job".as_ref(), job_path.as_os_str()]);
        assert!(!mixed_ok);
        assert_eq!(last_python_json(&mixed_stdout)["code"], "bad_prompt");
        let mut automatic = manual;
        automatic["initMask"] =
            serde_json::Value::String(root.path().join("birefnet-init.png").display().to_string());
        std::fs::write(&job_path, serde_json::to_vec(&automatic).unwrap()).unwrap();
        let (auto_ok, auto_stdout) =
            python_output(&script, &["--job".as_ref(), job_path.as_os_str()]);
        assert!(!auto_ok);
        assert_eq!(last_python_json(&auto_stdout)["code"], "missing_init_mask");
        assert_ne!(last_python_json(&auto_stdout)["ok"], true);
    }

    #[test]
    fn adapter_source_does_not_download_or_fall_back_to_bria() {
        let birefnet = std::fs::read_to_string(subject_mask_script("birefnet_init.py")).unwrap();
        assert!(birefnet.contains("local_files_only=True"));
        assert!(birefnet.contains("HF_HUB_OFFLINE"));
        assert!(!birefnet.lines().any(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("import rembg") || trimmed.starts_with("from rembg")
        }));
        assert!(!birefnet.contains("briaai"));
        assert!(!birefnet.contains("from_pretrained('ZhengPeng7"));
        assert!(!birefnet.contains("from_pretrained(\"ZhengPeng7"));
        let sam = std::fs::read_to_string(subject_mask_script("sam2_video.py")).unwrap();
        assert!(sam.contains("add_new_mask"));
        assert!(sam.contains("add_new_points_or_box"));
    }

    #[test]
    fn automatic_coverage_rejects_empty_and_full_first_frames() {
        assert!(accept_automatic_coverage(0.0).is_err());
        assert!(accept_automatic_coverage(1.0).is_err());
        assert!(accept_automatic_coverage(0.25).is_ok());
    }
}
