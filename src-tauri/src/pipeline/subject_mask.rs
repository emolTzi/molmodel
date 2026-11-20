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
}

impl SubjectMaskCheckpoint {
    pub fn validate(&self) -> Result<()> {
        if self.version != VERSION
            || self.frames.is_empty()
            || self.frames.len() > MAX_FRAMES
            || self.prompt.points.len() != self.prompt.labels.len()
            || self.prompt.points.len() > MAX_PROMPTS
            || (self.prompt.points.is_empty() && self.prompt.box_xyxy.is_none())
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
            },
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
}
