use crate::app_settings::load_settings;
use image::{DynamicImage, GenericImageView, GrayImage, imageops};
use image_hasher::{HashAlg, HasherConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter, State};

use crate::{PicPortalState, face_processing, image_loader, subject_inference};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct CullingSettings {
    /// Product-facing amount presets, intentionally not raw algorithm thresholds.
    pub selection_amount: String,
    /// Lenient/moderate/strict maps to the measured sharpness threshold.
    pub blur_severity: String,
    pub detect_duplicates: bool,
    pub detect_blurry: bool,
    pub detect_closed_eyes: bool,
    pub detect_highlights: bool,
    pub detect_subject: bool,
    pub subject_profile: String,
    pub auto_assign_stars: bool,
}

impl Default for CullingSettings {
    fn default() -> Self {
        Self {
            selection_amount: "standard".to_owned(),
            blur_severity: "moderate".to_owned(),
            detect_duplicates: true,
            // Off by default for v1: both detectors were measured unreliable on
            // real photos (MAX-21); they stay available as opt-in experiments.
            detect_blurry: false,
            detect_closed_eyes: false,
            detect_highlights: true,
            detect_subject: true,
            subject_profile: "general".to_owned(),
            auto_assign_stars: true,
        }
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AttributedFace {
    pub role: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
    pub eye_state: String,
    pub eye_confidence: f32,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageAnalysisResult {
    pub path: String,
    pub quality_score: f64,
    pub sharpness_metric: f64,
    pub center_focus_metric: f64,
    pub exposure_metric: f64,
    pub width: u32,
    pub height: u32,
    /// `open`, `closed`, `notApplicable`, or `unknown`.
    pub eye_state: String,
    pub eye_confidence: f32,
    /// Explicitly identifies the deterministic local heuristic when available.
    pub eye_method: String,
    pub face_count: usize,
    pub face_thumbnails: Vec<String>,
    /// This is a local technical signal, not an artistic or subject score.
    pub quality_method: String,
    pub reasons: Vec<String>,
    pub category: String,
    pub suggested_rating: u8,
    pub subject_status: String,
    pub subject_method: String,
    pub subject_boxes: Vec<subject_inference::SubjectBox>,
    pub attributed_faces: Vec<AttributedFace>,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullGroup {
    pub representative: ImageAnalysisResult,
    pub duplicates: Vec<ImageAnalysisResult>,
}

#[derive(Serialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CullingSuggestions {
    pub similar_groups: Vec<CullGroup>,
    pub selected_images: Vec<ImageAnalysisResult>,
    pub highlight_images: Vec<ImageAnalysisResult>,
    pub duplicate_images: Vec<ImageAnalysisResult>,
    pub blurry_images: Vec<ImageAnalysisResult>,
    pub closed_eye_images: Vec<ImageAnalysisResult>,
    pub unrated_images: Vec<ImageAnalysisResult>,
    pub results: Vec<ImageAnalysisResult>,
    pub failed_paths: Vec<String>,
    pub star_assignments: HashMap<String, u8>,
    pub color_assignments: HashMap<String, Option<String>>,
    pub eye_analysis_status: String,
    pub subject_analysis_status: String,
    /// Folder the analysis was started for, so the UI can reopen it after a reload.
    pub folder_path: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
struct CullingProgress {
    current: usize,
    total: usize,
    /// Legacy English label, kept for compatibility; the UI translates `stage_code`.
    stage: String,
    /// `preparing`, `analyzing`, `subject` or `grouping`.
    stage_code: &'static str,
}

impl CullingProgress {
    fn new(current: usize, total: usize, stage_code: &'static str) -> Self {
        let stage = match stage_code {
            "preparing" => "Preparing local analysis...",
            "analyzing" => "Analyzing images locally...",
            "subject" => "Reviewing local subject, pose and eye signals...",
            _ => "Grouping similar images and assigning review buckets...",
        };
        Self {
            current,
            total,
            stage: stage.to_owned(),
            stage_code,
        }
    }
}

pub const CULLING_CANCELLED: &str = "CULLING_CANCELLED";
pub const CULLING_ALREADY_RUNNING: &str = "CULLING_ALREADY_RUNNING";

/// Run lifecycle in one atomic: running and outcome cannot disagree, so a cancel
/// is either accepted before the result is committed or refused afterwards,
/// including while the finished run releases its guard.
static CULLING_STATE: AtomicU8 = AtomicU8::new(STATE_IDLE);
const STATE_IDLE: u8 = 0;
const STATE_RUNNING: u8 = 1;
const STATE_CANCELLED: u8 = 2;
const STATE_COMMITTED: u8 = 3;

fn culling_running() -> bool {
    CULLING_STATE.load(Ordering::Acquire) != STATE_IDLE
}

/// What the UI needs to rebuild itself after a webview reload: events sent
/// before the reload are lost, this snapshot is not.
#[derive(Serialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CullingSession {
    pub running: bool,
    pub folder_path: Option<String>,
    progress: Option<CullingProgress>,
    /// Finished analysis the user has not closed yet.
    pub result: Option<CullingSuggestions>,
}

static CULLING_SESSION: Mutex<Option<CullingSession>> = Mutex::new(None);

fn update_session(update: impl FnOnce(&mut CullingSession)) {
    if let Ok(mut session) = CULLING_SESSION.lock() {
        update(session.get_or_insert_with(CullingSession::default));
    }
}

fn record_progress(progress: &CullingProgress) {
    update_session(|session| session.progress = Some(progress.clone()));
}

fn emit_progress(app_handle: &AppHandle, progress: CullingProgress) {
    record_progress(&progress);
    let _ = app_handle.emit("culling-progress", progress);
}

/// Single-run guard: only one analysis at a time, released even on early return.
struct CullingRunGuard;

impl CullingRunGuard {
    fn acquire() -> Result<Self, String> {
        CULLING_STATE
            .compare_exchange(
                STATE_IDLE,
                STATE_RUNNING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| CULLING_ALREADY_RUNNING.to_owned())?;
        Ok(Self)
    }

    fn is_cancelled(&self) -> bool {
        CULLING_STATE.load(Ordering::Acquire) == STATE_CANCELLED
    }

    /// Claims the result for delivery. Fails when a cancel was accepted first;
    /// once it succeeds, `cancel_culling` reports that it is too late.
    fn try_commit(&self) -> bool {
        CULLING_STATE
            .compare_exchange(
                STATE_RUNNING,
                STATE_COMMITTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

impl Drop for CullingRunGuard {
    fn drop(&mut self) {
        update_session(|session| {
            session.running = false;
            session.progress = None;
        });
        CULLING_STATE.store(STATE_IDLE, Ordering::Release);
    }
}

/// Requests cancellation of the running analysis. The flag is checked between
/// images and phases; nothing is written by an analysis, so cancelling never
/// loses data. Returns whether the cancel was accepted: false when nothing
/// runs, or when the results are already being delivered.
#[tauri::command]
pub fn cancel_culling() -> bool {
    match CULLING_STATE.compare_exchange(
        STATE_RUNNING,
        STATE_CANCELLED,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => true,
        Err(current) => current == STATE_CANCELLED,
    }
}

/// Current analysis, read by the UI when it (re)starts.
#[tauri::command]
pub fn culling_session() -> CullingSession {
    CULLING_SESSION
        .lock()
        .ok()
        .and_then(|session| session.clone())
        .unwrap_or_default()
}

/// The user closed the results: a later reload must not reopen them.
#[tauri::command]
pub fn dismiss_culling_result() {
    update_session(|session| session.result = None);
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullingCapabilities {
    /// Sharpness, exposure and similarity are pure Rust and always available.
    pub sharpness: &'static str,
    /// `ready` or `unavailable` (face and eye models).
    pub faces: &'static str,
    /// `ready` or `unavailable` (optional subject/pose worker).
    pub subject: &'static str,
    /// First reason code explaining an unavailable capability, faces first.
    pub reason_code: Option<String>,
    pub faces_reason_code: Option<String>,
    pub subject_reason_code: Option<String>,
    /// True when an analysis started before (e.g. before a webview reload) is still running.
    pub running: bool,
}

fn reason_code(error: &str) -> String {
    error.split(':').next().unwrap_or(error).trim().to_owned()
}

/// Pre-flight check shown before launching the analysis (spec MAX-19 §1.9).
#[tauri::command]
pub async fn culling_capabilities(
    app_handle: AppHandle,
    face_state: State<'_, PicPortalState>,
) -> Result<CullingCapabilities, String> {
    let faces_result = match face_state.face_runtime.lock() {
        Ok(mut guard) => face_processing::ensure_runtime(&mut guard, &app_handle).map(|_| ()),
        Err(_) => Err("FACE_RUNTIME_LOCK_POISONED".to_owned()),
    };
    let subject_app = app_handle.clone();
    let subject_result =
        tauri::async_runtime::spawn_blocking(move || subject_inference::capability(&subject_app))
            .await
            .unwrap_or_else(|_| Err("LOCAL_CULLING_WORKER_START_FAILED".to_owned()));
    let faces_reason_code = faces_result.err().map(|error| reason_code(&error));
    let subject_reason_code = subject_result.err().map(|error| reason_code(&error));
    Ok(CullingCapabilities {
        sharpness: "ready",
        faces: if faces_reason_code.is_none() {
            "ready"
        } else {
            "unavailable"
        },
        subject: if subject_reason_code.is_none() {
            "ready"
        } else {
            "unavailable"
        },
        reason_code: faces_reason_code
            .clone()
            .or_else(|| subject_reason_code.clone()),
        faces_reason_code,
        subject_reason_code,
        running: culling_running(),
    })
}

struct ImageAnalysisData {
    hash: image_hasher::ImageHash,
    result: ImageAnalysisResult,
}

const SUBJECT_OVERLAP_MIN: f32 = 0.5;
const NOSE_DISTANCE_FACTOR: f32 = 1.5;
const TORSO_INSIDE_MIN: usize = 2;
const YUNET_PRIMARY_THRESHOLD: f32 = 0.9;
// Match the verified f049 YuNet contract: only retry at 0.7 when 0.9 found no face.
const YUNET_FALLBACK_THRESHOLD: f32 = 0.7;

fn should_try_yunet_fallback(primary_face_count: usize) -> bool {
    primary_face_count == 0
}

#[derive(Clone)]
struct AssociationFace {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

#[derive(Clone)]
struct AssociationPose {
    nose_x: f32,
    nose_y: f32,
    torso: Vec<(f32, f32)>,
}

fn subject_phrases(profile: &str) -> &'static [&'static str] {
    match profile {
        "dance" => &["a couple dancing together.", "a person dancing."],
        "portrait" => &["a person being photographed."],
        "sports" => &["a person participating in a sport."],
        "wedding" => &["a couple or group of people at a wedding."],
        _ => &["a person or group of people."],
    }
}

fn point_in_subject(x: f32, y: f32, subject: &subject_inference::SubjectBox) -> bool {
    x >= subject.x
        && x <= subject.x + subject.width
        && y >= subject.y
        && y <= subject.y + subject.height
}

fn face_overlap(face: &AssociationFace, boxes: &[subject_inference::SubjectBox]) -> f32 {
    let area = face.width * face.height;
    if area <= 0.0 {
        return 0.0;
    }
    boxes
        .iter()
        .map(|subject| {
            let left = face.x.max(subject.x);
            let top = face.y.max(subject.y);
            let right = (face.x + face.width).min(subject.x + subject.width);
            let bottom = (face.y + face.height).min(subject.y + subject.height);
            ((right - left).max(0.0) * (bottom - top).max(0.0)) / area
        })
        .fold(0.0, f32::max)
}

fn pose_matches_subject(pose: &AssociationPose, boxes: &[subject_inference::SubjectBox]) -> bool {
    pose.torso
        .iter()
        .filter(|(x, y)| {
            boxes
                .iter()
                .any(|subject| point_in_subject(*x, *y, subject))
        })
        .count()
        >= TORSO_INSIDE_MIN
}

fn face_is_linked(
    face: &AssociationFace,
    boxes: &[subject_inference::SubjectBox],
    poses: &[AssociationPose],
) -> bool {
    if face_overlap(face, boxes) < SUBJECT_OVERLAP_MIN {
        return false;
    }
    let center_x = face.x + face.width / 2.0;
    let center_y = face.y + face.height / 2.0;
    let limit = NOSE_DISTANCE_FACTOR * face.width.max(face.height);
    poses
        .iter()
        .filter(|pose| pose_matches_subject(pose, boxes))
        .any(|pose| {
            ((center_x - pose.nose_x).powi(2) + (center_y - pose.nose_y).powi(2)).sqrt() <= limit
        })
}

fn reviewed_eye_state(eyes: &[face_processing::EyeAssessment]) -> &'static str {
    if eyes
        .iter()
        .any(|eye| eye.state == face_processing::EyeState::Closed)
    {
        "closed"
    } else if eyes.is_empty()
        || eyes
            .iter()
            .any(|eye| eye.state != face_processing::EyeState::Open)
    {
        "unknown"
    } else {
        "open"
    }
}

fn scale_pose_frame(
    frame: subject_inference::PoseFrame,
    origin_x: f32,
    origin_y: f32,
    target_width: f32,
    target_height: f32,
) -> Vec<AssociationPose> {
    let scale_x = target_width / frame.width.max(1.0);
    let scale_y = target_height / frame.height.max(1.0);
    frame
        .bodies
        .into_iter()
        .map(|body| AssociationPose {
            nose_x: origin_x + body.nose_x * scale_x,
            nose_y: origin_y + body.nose_y * scale_y,
            torso: body
                .torso
                .into_iter()
                .map(|(x, y)| (origin_x + x * scale_x, origin_y + y * scale_y))
                .collect(),
        })
        .collect()
}

fn exact_subject_crop(
    image: &DynamicImage,
    subject: &subject_inference::SubjectBox,
) -> Option<(DynamicImage, f32, f32, f32, f32)> {
    let x0 = subject.x.max(0.0);
    let y0 = subject.y.max(0.0);
    let x1 = (subject.x + subject.width).min(image.width() as f32);
    let y1 = (subject.y + subject.height).min(image.height() as f32);
    let x = x0.floor() as u32;
    let y = y0.floor() as u32;
    if x >= image.width() || y >= image.height() {
        return None;
    }
    let width = (x1.ceil() as u32).saturating_sub(x).min(image.width() - x);
    let height = (y1.ceil() as u32).saturating_sub(y).min(image.height() - y);
    if width < 8 || height < 8 {
        return None;
    }
    let crop = DynamicImage::ImageRgba8(imageops::crop_imm(image, x, y, width, height).to_image());
    Some((crop, x as f32, y as f32, width as f32, height as f32))
}

fn collect_association_poses(
    worker: &mut subject_inference::LocalWorker,
    image: &DynamicImage,
    frame: &DynamicImage,
    boxes: &[subject_inference::SubjectBox],
) -> Result<Vec<AssociationPose>, String> {
    let mut poses = scale_pose_frame(
        worker.pose(frame)?,
        0.0,
        0.0,
        image.width() as f32,
        image.height() as f32,
    );
    for subject in boxes {
        if let Some((crop, x, y, width, height)) = exact_subject_crop(image, subject) {
            if let Ok(crop_pose) = worker.pose(&crop) {
                poses.extend(scale_pose_frame(crop_pose, x, y, width, height));
            }
        }
    }
    Ok(poses)
}

fn load_culling_image(
    path: &str,
    settings: &crate::app_settings::AppSettings,
) -> Result<DynamicImage, String> {
    let source_path = crate::file_management::parse_virtual_path(path).0;
    if crate::file_management::is_cloud_placeholder(&source_path) {
        return Err(format!("'{path}' is stored in iCloud and not downloaded"));
    }
    let bytes = std::fs::read(&source_path).map_err(|error| error.to_string())?;
    image_loader::load_base_image_from_bytes(
        &bytes,
        &source_path.to_string_lossy(),
        true,
        settings,
        None,
    )
    .map_err(|error| error.to_string())
}

fn apply_assisted_signals(
    result: &mut ImageAnalysisResult,
    image: &DynamicImage,
    settings: &CullingSettings,
    worker: &mut Option<subject_inference::LocalWorker>,
    face_runtime: Option<&mut face_processing::FaceRuntime>,
) {
    let frame = image.thumbnail(1920, 1920);
    let sx = image.width().max(1) as f32 / frame.width().max(1) as f32;
    let sy = image.height().max(1) as f32 / frame.height().max(1) as f32;
    let mut boxes = Vec::new();
    let mut subject_method = if settings.detect_subject {
        "grounding-dino-unavailable"
    } else {
        "disabled"
    }
    .to_owned();
    if settings.detect_subject {
        if let Some(local_worker) = worker.as_mut().filter(|item| item.subject_is_ready()) {
            let mut failed = false;
            for phrase in subject_phrases(&settings.subject_profile) {
                match local_worker.detect(&frame, phrase) {
                    Ok(found) => {
                        boxes = found
                            .into_iter()
                            .filter(|item| item.width > 0.0 && item.height > 0.0)
                            .map(|item| subject_inference::SubjectBox {
                                x: item.x * sx,
                                y: item.y * sy,
                                width: item.width * sx,
                                height: item.height * sy,
                                score: item.score,
                                label: item.label,
                            })
                            .collect();
                        if !boxes.is_empty() {
                            break;
                        }
                    }
                    Err(_) => {
                        failed = true;
                        break;
                    }
                }
            }
            subject_method = if failed {
                "grounding-dino-error".to_owned()
            } else {
                "grounding-dino-local".to_owned()
            };
        }
    }

    let mut faces = Vec::new();
    let mut face_method = "yunet-unavailable".to_owned();
    if let Some(runtime) = face_runtime {
        match runtime.analyze_for_culling(image, YUNET_PRIMARY_THRESHOLD) {
            Ok(primary) => {
                faces = primary;
                face_method = "yunet-0.9".to_owned();
                if should_try_yunet_fallback(faces.len()) {
                    match runtime.analyze_for_culling(image, YUNET_FALLBACK_THRESHOLD) {
                        Ok(fallback) if !fallback.is_empty() => {
                            faces = fallback;
                            face_method = "yunet-fallback-0.7".to_owned();
                        }
                        Ok(_) => face_method = "yunet-no-face".to_owned(),
                        Err(_) => face_method.push_str(";fallback-error"),
                    }
                }
            }
            Err(_) => face_method = "yunet-error".to_owned(),
        }
    }

    let mut pose_method = "pose-not-requested".to_owned();
    let roles = if !settings.detect_subject {
        vec!["primary"; faces.len()]
    } else if boxes.is_empty() {
        vec!["unknown"; faces.len()]
    } else if let Some(local_worker) = worker.as_mut().filter(|item| item.pose_is_ready()) {
        match collect_association_poses(local_worker, image, &frame, &boxes) {
            Ok(poses) => {
                pose_method = "pose-lite".to_owned();
                faces
                    .iter()
                    .map(|face| {
                        let association = AssociationFace {
                            x: face.bbox.x,
                            y: face.bbox.y,
                            width: face.bbox.width,
                            height: face.bbox.height,
                        };
                        if face_is_linked(&association, &boxes, &poses) {
                            "primary"
                        } else {
                            "secondary"
                        }
                    })
                    .collect()
            }
            Err(_) => {
                pose_method = "pose-error".to_owned();
                vec!["unknown"; faces.len()]
            }
        }
    } else {
        pose_method = "pose-unavailable".to_owned();
        vec!["unknown"; faces.len()]
    };

    let primary_count = roles.iter().filter(|role| **role == "primary").count();
    result.subject_status = (if !settings.detect_subject {
        "not-evaluated"
    } else if boxes.is_empty() || primary_count == 0 {
        "unknown"
    } else if primary_count > 1 {
        "multiple"
    } else {
        "primary"
    })
    .to_owned();
    result.subject_method = format!("{subject_method};{pose_method};{face_method}");
    let source_width = image.width().max(1) as f32;
    let source_height = image.height().max(1) as f32;
    result.subject_boxes = boxes
        .iter()
        .map(|item| subject_inference::SubjectBox {
            x: item.x / source_width,
            y: item.y / source_height,
            width: item.width / source_width,
            height: item.height / source_height,
            score: item.score,
            label: item.label.clone(),
        })
        .collect();
    result.face_count = faces.len();
    result.face_thumbnails.clear();
    result.attributed_faces = faces
        .iter()
        .zip(&roles)
        .map(|(face, role)| AttributedFace {
            role: (*role).to_owned(),
            x: face.bbox.x / source_width,
            y: face.bbox.y / source_height,
            width: face.bbox.width / source_width,
            height: face.bbox.height / source_height,
            confidence: face.confidence,
            eye_state: if *role == "primary" && settings.detect_closed_eyes {
                match face.eye.state {
                    face_processing::EyeState::Open => "open",
                    face_processing::EyeState::Closed => "closed",
                    _ => "unknown",
                }
                .to_owned()
            } else if settings.detect_closed_eyes {
                "not-attributed".to_owned()
            } else {
                "disabled".to_owned()
            },
            eye_confidence: if *role == "primary" {
                face.eye.confidence
            } else {
                0.0
            },
        })
        .collect();

    if !settings.detect_closed_eyes {
        result.eye_state = "disabled".to_owned();
        result.eye_method = "disabled".to_owned();
        result.eye_confidence = 0.0;
        return;
    }
    let primary_eyes: Vec<_> = faces
        .iter()
        .zip(&roles)
        .filter(|(_, role)| **role == "primary")
        .map(|(face, _)| face.eye)
        .collect();
    result.eye_confidence = primary_eyes
        .iter()
        .map(|eye| eye.confidence)
        .fold(0.0, f32::max);
    result.eye_state = if faces.is_empty() && face_method == "yunet-no-face" {
        "notApplicable".to_owned()
    } else {
        reviewed_eye_state(&primary_eyes).to_owned()
    };
    result.eye_method = if faces.is_empty() && face_method == "yunet-no-face" {
        "yunet-no-face".to_owned()
    } else if primary_eyes.is_empty() && !faces.is_empty() {
        "subject-not-attributed".to_owned()
    } else if result.eye_state == "unknown" && face_method == "yunet-unavailable" {
        "unavailable".to_owned()
    } else {
        format!("{face_method};local-heuristic")
    };
}

const WEIGHT_SHARPNESS: f64 = 0.40;
const WEIGHT_CENTER_FOCUS: f64 = 0.35;
const WEIGHT_EXPOSURE: f64 = 0.25;

fn calculate_laplacian_variance(image: &GrayImage) -> f64 {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return 0.0;
    }
    let mut laplacian_values = Vec::with_capacity(((width - 2) * (height - 2)) as usize);
    let mut sum = 0.0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let p_center = image.get_pixel(x, y)[0] as i32;
            let p_north = image.get_pixel(x, y - 1)[0] as i32;
            let p_south = image.get_pixel(x, y + 1)[0] as i32;
            let p_west = image.get_pixel(x - 1, y)[0] as i32;
            let p_east = image.get_pixel(x + 1, y)[0] as i32;
            let value = (p_north + p_south + p_west + p_east - 4 * p_center) as f64;
            laplacian_values.push(value);
            sum += value;
        }
    }
    if laplacian_values.is_empty() {
        return 0.0;
    }
    let mean = sum / laplacian_values.len() as f64;
    laplacian_values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / laplacian_values.len() as f64
}

fn calculate_exposure_metric(image: &GrayImage) -> f64 {
    let histogram = imageproc::stats::histogram(image);
    let total_pixels = (image.width() * image.height()) as f64;
    if total_pixels == 0.0 {
        return 0.0;
    }
    let dark_pixels = histogram.channels[0][0..5].iter().sum::<u32>() as f64;
    let bright_pixels = histogram.channels[0][250..256].iter().sum::<u32>() as f64;
    (1.0 - ((dark_pixels / total_pixels) * 5.0 + (bright_pixels / total_pixels) * 5.0)).max(0.0)
}

fn analyze_image(
    path: &str,
    hasher: &image_hasher::Hasher,
    settings: &crate::app_settings::AppSettings,
) -> Result<ImageAnalysisData, String> {
    const ANALYSIS_DIM: u32 = 720;
    let source_path = crate::file_management::parse_virtual_path(path).0;
    if crate::file_management::is_cloud_placeholder(&source_path) {
        return Err(format!("'{path}' is stored in iCloud and not downloaded"));
    }
    let file_bytes = std::fs::read(&source_path).map_err(|error| error.to_string())?;
    let source_path_string = source_path.to_string_lossy();
    let image = image_loader::load_base_image_from_bytes(
        &file_bytes,
        &source_path_string,
        true,
        settings,
        None,
    )
    .map_err(|error| error.to_string())?;
    let (width, height) = image.dimensions();
    let thumbnail = image.thumbnail(ANALYSIS_DIM, ANALYSIS_DIM);
    let gray_thumbnail = thumbnail.to_luma8();
    let sharpness_metric = calculate_laplacian_variance(&gray_thumbnail);
    let exposure_metric = calculate_exposure_metric(&gray_thumbnail);
    let (thumb_w, thumb_h) = gray_thumbnail.dimensions();
    let center_crop = imageops::crop_imm(
        &gray_thumbnail,
        thumb_w / 4,
        thumb_h / 4,
        thumb_w / 2,
        thumb_h / 2,
    )
    .to_image();
    let center_focus_metric = calculate_laplacian_variance(&center_crop);
    let normalized_sharpness = ((sharpness_metric + 1.0).log10() / 3.5).clamp(0.0, 1.0);
    let normalized_center_focus = ((center_focus_metric + 1.0).log10() / 3.5).clamp(0.0, 1.0);
    let quality_score = normalized_sharpness * WEIGHT_SHARPNESS
        + normalized_center_focus * WEIGHT_CENTER_FOCUS
        + exposure_metric * WEIGHT_EXPOSURE;
    Ok(ImageAnalysisData {
        hash: hasher.hash_image(&thumbnail),
        result: ImageAnalysisResult {
            path: path.to_owned(),
            quality_score,
            sharpness_metric,
            center_focus_metric,
            exposure_metric,
            width,
            height,
            eye_state: "unknown".to_owned(),
            eye_confidence: 0.0,
            eye_method: "unavailable".to_owned(),
            face_count: 0,
            face_thumbnails: Vec::new(),
            quality_method: "local-technical".to_owned(),
            reasons: Vec::new(),
            category: "unrated".to_owned(),
            suggested_rating: 0,
            subject_status: "not-evaluated".to_owned(),
            subject_method: "not-requested".to_owned(),
            subject_boxes: Vec::new(),
            attributed_faces: Vec::new(),
        },
    })
}

fn connected_components(
    item_count: usize,
    mut is_similar: impl FnMut(usize, usize) -> bool,
) -> Vec<Vec<usize>> {
    let mut components = Vec::new();
    let mut processed_indices = vec![false; item_count];
    for i in 0..item_count {
        if processed_indices[i] {
            continue;
        }
        let mut component = vec![i];
        let mut queue = VecDeque::from([i]);
        processed_indices[i] = true;
        while let Some(current) = queue.pop_front() {
            for j in 0..item_count {
                if processed_indices[j] || !is_similar(current, j) {
                    continue;
                }
                processed_indices[j] = true;
                component.push(j);
                queue.push_back(j);
            }
        }
        components.push(component);
    }
    components
}

fn has_evaluated_open_eyes(result: &ImageAnalysisResult, settings: &CullingSettings) -> bool {
    settings.detect_closed_eyes
        && result.eye_state == "open"
        && !matches!(result.eye_method.as_str(), "unavailable" | "disabled")
}

fn compare_group_candidates(
    left: &ImageAnalysisResult,
    right: &ImageAnalysisResult,
    settings: &CullingSettings,
) -> std::cmp::Ordering {
    is_selection_candidate(right, settings, false)
        .cmp(&is_selection_candidate(left, settings, false))
        .then_with(|| {
            has_evaluated_open_eyes(right, settings).cmp(&has_evaluated_open_eyes(left, settings))
        })
        .then_with(|| right.quality_score.total_cmp(&left.quality_score))
        .then_with(|| left.path.cmp(&right.path))
}

fn is_selection_candidate(
    result: &ImageAnalysisResult,
    settings: &CullingSettings,
    is_duplicate: bool,
) -> bool {
    !is_duplicate
        && !is_blurry(result, settings)
        && !(settings.detect_closed_eyes && result.eye_state == "closed")
}

fn category_rating(category: &str) -> u8 {
    match category {
        "selected" => 5,
        "highlights" => 4,
        "duplicate" => 3,
        "blurred" => 2,
        "closedEyes" => 1,
        _ => 0,
    }
}

fn category_color(category: &str) -> Option<&'static str> {
    match category {
        "selected" => Some("green"),
        "highlights" => Some("blue"),
        "duplicate" => Some("yellow"),
        "blurred" => Some("red"),
        "closedEyes" => Some("purple"),
        _ => None,
    }
}

fn similarity_threshold() -> u32 {
    // Keep this implementation detail stable while exposing only the
    // understandable "Duplicate photos" switch in the UI.
    28
}

fn blur_threshold(severity: &str) -> f64 {
    match severity {
        "lenient" => 65.0,
        "strict" => 160.0,
        _ => 100.0,
    }
}

fn selected_limit(total: usize, amount: &str) -> usize {
    if total == 0 {
        return 0;
    }
    let fraction = match amount {
        "extreme" => 0.05,
        "few" => 0.10,
        "more" => 0.35,
        _ => 0.20,
    };
    ((total as f64 * fraction).ceil() as usize).clamp(1, total)
}

fn is_blurry(result: &ImageAnalysisResult, settings: &CullingSettings) -> bool {
    settings.detect_blurry && result.sharpness_metric < blur_threshold(&settings.blur_severity)
}

fn has_unknown_eye_signal(result: &ImageAnalysisResult, settings: &CullingSettings) -> bool {
    settings.detect_closed_eyes
        && (result.eye_method == "unavailable" || result.eye_state == "unknown")
}

fn culling_category(
    result: &ImageAnalysisResult,
    settings: &CullingSettings,
    is_duplicate: bool,
    is_selected: bool,
    is_highlight: bool,
) -> &'static str {
    // A lower rating means a stronger review warning. The order is explicit
    // so an image with multiple findings keeps one deterministic primary label
    // while `reasons` retains every finding.
    if settings.detect_closed_eyes && result.eye_state == "closed" {
        "closedEyes"
    } else if is_blurry(result, settings) {
        "blurred"
    } else if settings.detect_duplicates && is_duplicate {
        "duplicate"
    } else if is_selected {
        "selected"
    } else if is_highlight {
        "highlights"
    } else {
        "unrated"
    }
}

fn detector_reasons(
    result: &ImageAnalysisResult,
    settings: &CullingSettings,
    is_duplicate: bool,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if settings.detect_duplicates && is_duplicate {
        reasons.push("duplicate".to_owned());
    }
    if is_blurry(result, settings) {
        reasons.push("blurred".to_owned());
    }
    if settings.detect_closed_eyes && result.eye_state == "closed" {
        reasons.push("closedEyes".to_owned());
    }
    if has_unknown_eye_signal(result, settings) {
        reasons.push("eyesUnknown".to_owned());
    }
    if result.subject_status == "unknown" {
        reasons.push("subjectUnknown".to_owned());
    }
    reasons
}

#[tauri::command]
pub async fn cull_images(
    paths: Vec<String>,
    settings: CullingSettings,
    folder_path: Option<String>,
    app_handle: AppHandle,
    face_state: State<'_, PicPortalState>,
) -> Result<CullingSuggestions, String> {
    if paths.is_empty() {
        return Ok(CullingSuggestions::default());
    }
    let run = CullingRunGuard::acquire()?;
    // A new run replaces the previous session, including an undismissed result.
    update_session(|session| {
        *session = CullingSession {
            running: true,
            folder_path: folder_path.clone(),
            ..Default::default()
        }
    });
    let app_settings = load_settings(app_handle.clone()).unwrap_or_default();
    let total_count = paths.len();
    let completed_count = Arc::new(AtomicUsize::new(0));
    let _ = app_handle.emit("culling-start", total_count);
    emit_progress(
        &app_handle,
        CullingProgress::new(0, total_count, "preparing"),
    );
    let cancelled = |app_handle: &AppHandle| -> Result<CullingSuggestions, String> {
        log::info!("Assisted culling cancelled by the user; nothing was written");
        let _ = app_handle.emit("culling-cancelled", ());
        Err(CULLING_CANCELLED.to_owned())
    };
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(16, 16)
        .to_hasher();
    let analysis_results: Vec<Result<ImageAnalysisData, (String, String)>> = paths
        .par_iter()
        .map(|path| {
            if run.is_cancelled() {
                return Err((path.to_owned(), CULLING_CANCELLED.to_owned()));
            }
            let result = analyze_image(path, &hasher, &app_settings)
                .map_err(|error| (path.to_owned(), error));
            // Count after the work so the bar reflects finished images, not started ones.
            let completed = completed_count.fetch_add(1, Ordering::Relaxed) + 1;
            emit_progress(
                &app_handle,
                CullingProgress::new(completed, total_count, "analyzing"),
            );
            result
        })
        .collect();
    if run.is_cancelled() {
        return cancelled(&app_handle);
    }
    let mut successful = Vec::new();
    let mut failed_paths = Vec::new();
    for result in analysis_results {
        match result {
            Ok(data) => successful.push(data),
            Err((path, error)) => {
                log::warn!("Failed to analyze image {path}: {error}");
                failed_paths.push(path);
            }
        }
    }

    let mut local_worker = None;
    let mut subject_status = (if settings.detect_subject {
        if successful.is_empty() {
            "no-images"
        } else {
            "pending"
        }
    } else {
        "disabled"
    })
    .to_owned();
    if settings.detect_subject && !successful.is_empty() {
        match subject_inference::LocalWorker::start(&app_handle) {
            Ok(worker) => {
                subject_status = if !worker.subject_is_ready() {
                    "subject-unavailable".to_owned()
                } else if !worker.pose_is_ready() {
                    "subject-ready-pose-unavailable".to_owned()
                } else {
                    "ready".to_owned()
                };
                local_worker = Some(worker);
            }
            Err(_) => subject_status = "unavailable".to_owned(),
        }
    }

    // Photos that could not be re-read for the eye/subject pass are listed as failed
    // and never receive an automatic rating or label (ported from PR #2, 205c84e4).
    let mut unprocessed_paths: HashSet<String> = HashSet::new();
    let needs_face_runtime =
        (settings.detect_subject || settings.detect_closed_eyes) && !successful.is_empty();
    let mut runtime_guard = if needs_face_runtime {
        Some(
            face_state
                .face_runtime
                .lock()
                .map_err(|_| "face runtime lock is poisoned".to_owned())?,
        )
    } else {
        None
    };
    let face_runtime_available = runtime_guard
        .as_mut()
        .is_some_and(|guard| face_processing::ensure_runtime(guard, &app_handle).is_ok());
    let eye_status = if !settings.detect_closed_eyes {
        "disabled".to_owned()
    } else if face_runtime_available {
        "available-local-heuristic".to_owned()
    } else {
        "unavailable".to_owned()
    };
    if settings.detect_subject && !face_runtime_available && subject_status == "ready" {
        subject_status = "face-unavailable".to_owned();
    }

    if settings.detect_subject || settings.detect_closed_eyes {
        emit_progress(&app_handle, CullingProgress::new(0, total_count, "subject"));
        let subject_total = successful.len();
        for (index, data) in successful.iter_mut().enumerate() {
            if run.is_cancelled() {
                return cancelled(&app_handle);
            }
            match load_culling_image(&data.result.path, &app_settings) {
                Ok(image) => {
                    let face_runtime = runtime_guard.as_mut().and_then(|guard| guard.as_mut());
                    apply_assisted_signals(
                        &mut data.result,
                        &image,
                        &settings,
                        &mut local_worker,
                        face_runtime,
                    );
                }
                Err(error) => {
                    failed_paths.push(data.result.path.clone());
                    unprocessed_paths.insert(data.result.path.clone());
                    data.result.subject_status = if settings.detect_subject {
                        "unknown"
                    } else {
                        "not-evaluated"
                    }
                    .to_owned();
                    data.result.subject_method = "image-unavailable".to_owned();
                    if settings.detect_closed_eyes {
                        data.result.eye_state = "unknown".to_owned();
                        data.result.eye_method = "unavailable".to_owned();
                    }
                    log::debug!(
                        "Assisted culling review unavailable for {}: {error}",
                        data.result.path
                    );
                }
            }
            emit_progress(
                &app_handle,
                CullingProgress::new(index + 1, subject_total, "subject"),
            );
        }
    }
    if run.is_cancelled() {
        return cancelled(&app_handle);
    }

    emit_progress(
        &app_handle,
        CullingProgress::new(total_count, total_count, "grouping"),
    );
    let mut suggestions = build_suggestions(&mut successful, &settings, &unprocessed_paths);
    suggestions.failed_paths = failed_paths;
    suggestions.eye_analysis_status = eye_status;
    suggestions.subject_analysis_status = subject_status;
    suggestions.folder_path = folder_path;
    // A cancel accepted while grouping wins over the result; after the commit,
    // cancel reports that it is too late and the results are delivered.
    if !run.try_commit() {
        return cancelled(&app_handle);
    }
    // Committed result and "not running" land in one snapshot: a reload never
    // sees a finished analysis as still in progress.
    update_session(|session| {
        session.running = false;
        session.progress = None;
        session.result = Some(suggestions.clone());
    });
    let _ = app_handle.emit("culling-complete", &suggestions);
    Ok(suggestions)
}

/// Groups similar photos, picks the selection and assigns review buckets and
/// proposals. Pure: shared by `cull_images` and the corpus measurement harness.
fn build_suggestions(
    successful: &mut [ImageAnalysisData],
    settings: &CullingSettings,
    unprocessed_paths: &HashSet<String>,
) -> CullingSuggestions {
    let mut suggestions = CullingSuggestions::default();
    let mut duplicate_paths = HashSet::new();
    let mut similar_group_indices = Vec::new();
    if settings.detect_duplicates {
        for mut group_indices in connected_components(successful.len(), |left, right| {
            !unprocessed_paths.contains(&successful[left].result.path)
                && !unprocessed_paths.contains(&successful[right].result.path)
                && successful[left].hash.dist(&successful[right].hash) <= similarity_threshold()
        }) {
            if group_indices.len() > 1 {
                group_indices.sort_by(|left, right| {
                    compare_group_candidates(
                        &successful[*left].result,
                        &successful[*right].result,
                        settings,
                    )
                });
                for index in group_indices.iter().skip(1) {
                    duplicate_paths.insert(successful[*index].result.path.clone());
                }
                similar_group_indices.push(group_indices);
            }
        }
    }

    let mut ranked_clean_indices: Vec<usize> = successful
        .iter()
        .enumerate()
        .filter(|(_, data)| {
            !unprocessed_paths.contains(&data.result.path)
                && is_selection_candidate(
                    &data.result,
                    settings,
                    duplicate_paths.contains(&data.result.path),
                )
        })
        .map(|(index, _)| index)
        .collect();
    ranked_clean_indices.sort_by(|left, right| {
        compare_group_candidates(
            &successful[*left].result,
            &successful[*right].result,
            settings,
        )
    });
    let selected_count = selected_limit(ranked_clean_indices.len(), &settings.selection_amount);
    let selected_paths: HashSet<String> = ranked_clean_indices
        .iter()
        .take(selected_count)
        .map(|index| successful[*index].result.path.clone())
        .collect();
    let highlight_paths: HashSet<String> = if settings.detect_highlights {
        ranked_clean_indices
            .iter()
            .skip(selected_count)
            .take(selected_count)
            .map(|index| successful[*index].result.path.clone())
            .collect()
    } else {
        HashSet::new()
    };

    for data in successful.iter_mut() {
        if unprocessed_paths.contains(&data.result.path) {
            data.result.reasons = vec!["unprocessed".to_owned()];
            data.result.category = "unrated".to_owned();
            data.result.suggested_rating = 0;
            suggestions.unrated_images.push(data.result.clone());
            continue;
        }
        let is_duplicate = duplicate_paths.contains(&data.result.path);
        let category = culling_category(
            &data.result,
            settings,
            is_duplicate,
            selected_paths.contains(&data.result.path),
            highlight_paths.contains(&data.result.path),
        );
        let mut reasons = detector_reasons(&data.result, settings, is_duplicate);
        if reasons.is_empty() {
            reasons.push(category.to_owned());
        }
        data.result.reasons = reasons;
        data.result.category = category.to_owned();
        data.result.suggested_rating = category_rating(category);
        if settings.auto_assign_stars {
            suggestions
                .star_assignments
                .insert(data.result.path.clone(), data.result.suggested_rating);
            suggestions.color_assignments.insert(
                data.result.path.clone(),
                category_color(category).map(str::to_owned),
            );
        }
        match category {
            "selected" => suggestions.selected_images.push(data.result.clone()),
            "highlights" => suggestions.highlight_images.push(data.result.clone()),
            "duplicate" => suggestions.duplicate_images.push(data.result.clone()),
            "blurred" => suggestions.blurry_images.push(data.result.clone()),
            "closedEyes" => suggestions.closed_eye_images.push(data.result.clone()),
            _ => suggestions.unrated_images.push(data.result.clone()),
        }
    }

    for group_indices in similar_group_indices {
        let representative = group_indices[0];
        suggestions.similar_groups.push(CullGroup {
            representative: successful[representative].result.clone(),
            duplicates: group_indices
                .iter()
                .skip(1)
                .map(|index| successful[*index].result.clone())
                .collect(),
        });
    }
    suggestions.results = successful.iter().map(|data| data.result.clone()).collect();
    suggestions
        .results
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .selected_images
        .sort_by(|left, right| right.quality_score.total_cmp(&left.quality_score));
    suggestions
        .highlight_images
        .sort_by(|left, right| right.quality_score.total_cmp(&left.quality_score));
    suggestions
        .duplicate_images
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .closed_eye_images
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .unrated_images
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .blurry_images
        .sort_by(|left, right| left.sharpness_metric.total_cmp(&right.sharpness_metric));
    suggestions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_guard_is_exclusive_and_cancellation_is_scoped_to_the_run() {
        // Global state: keep every assertion about it in this single test.
        assert!(
            !cancel_culling(),
            "no run: cancel reports nothing to cancel"
        );
        let run = CullingRunGuard::acquire().expect("first run starts");
        assert!(!run.is_cancelled());
        assert_eq!(
            CullingRunGuard::acquire().err().as_deref(),
            Some(CULLING_ALREADY_RUNNING)
        );
        assert!(cancel_culling(), "running: cancel is accepted");
        assert!(run.is_cancelled());
        drop(run);
        let next = CullingRunGuard::acquire().expect("guard released on drop");
        assert!(
            !next.is_cancelled(),
            "a new run never inherits a stale cancel"
        );

        // Cancel accepted while grouping: the result must not be delivered.
        assert!(cancel_culling());
        assert!(
            !next.try_commit(),
            "an accepted cancel wins over completion"
        );
        drop(next);

        // Completion first: a late cancel is refused, so the UI expects results.
        let run = CullingRunGuard::acquire().expect("third run starts");
        assert!(run.try_commit());
        assert!(!cancel_culling(), "too late: results are being delivered");
        assert!(!run.is_cancelled());
        drop(run);

        // Session snapshot outlives the run until the user dismisses the result.
        *CULLING_SESSION.lock().unwrap() = Some(CullingSession {
            running: true,
            folder_path: Some("/shoot".to_owned()),
            progress: None,
            result: None,
        });
        let run = CullingRunGuard::acquire().expect("fourth run starts");
        record_progress(&CullingProgress::new(4, 10, "analyzing"));
        assert_eq!(culling_session().progress.map(|p| p.current), Some(4));
        update_session(|session| session.result = Some(CullingSuggestions::default()));
        drop(run);
        let session = culling_session();
        assert!(!session.running && session.progress.is_none());
        assert_eq!(session.folder_path.as_deref(), Some("/shoot"));
        assert!(
            session.result.is_some(),
            "finished result survives for a reload"
        );
        dismiss_culling_result();
        assert!(culling_session().result.is_none());
        *CULLING_SESSION.lock().unwrap() = None;

        // Cancel racing the commit and the guard release on another thread:
        // an accepted cancel and a delivered result are mutually exclusive.
        for _ in 0..2_000 {
            let run = CullingRunGuard::acquire().expect("run starts");
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
            let canceller = {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    (0..64).fold(false, |accepted, _| cancel_culling() || accepted)
                })
            };
            barrier.wait();
            let committed = run.try_commit();
            drop(run);
            let accepted = canceller.join().unwrap();
            assert_ne!(
                committed, accepted,
                "exactly one of commit and cancel wins (committed={committed})"
            );
            assert!(!culling_running());
        }
        *CULLING_SESSION.lock().unwrap() = None;
    }

    #[test]
    fn progress_carries_a_translatable_stage_code() {
        let progress = serde_json::to_value(CullingProgress::new(3, 10, "analyzing")).unwrap();
        assert_eq!(progress["stageCode"], "analyzing");
        assert_eq!(progress["stage"], "Analyzing images locally...");
        assert_eq!(progress["current"], 3);
        assert_eq!(
            reason_code("LOCAL_CULLING_ARTIFACT_MISSING: dino.onnx"),
            "LOCAL_CULLING_ARTIFACT_MISSING"
        );
    }

    fn analysis_result(path: &str, quality_score: f64) -> ImageAnalysisResult {
        ImageAnalysisResult {
            path: path.to_owned(),
            quality_score,
            sharpness_metric: 200.0,
            center_focus_metric: 0.0,
            exposure_metric: 0.0,
            width: 1,
            height: 1,
            eye_state: "notApplicable".to_owned(),
            eye_confidence: 0.0,
            eye_method: "disabled".to_owned(),
            face_count: 0,
            face_thumbnails: Vec::new(),
            quality_method: "local-technical".to_owned(),
            reasons: Vec::new(),
            category: "unrated".to_owned(),
            suggested_rating: 0,
            subject_status: "not-evaluated".to_owned(),
            subject_method: "not-requested".to_owned(),
            subject_boxes: Vec::new(),
            attributed_faces: Vec::new(),
        }
    }

    fn analysis_data(path: &str, quality_score: f64, hash_byte: u8) -> ImageAnalysisData {
        ImageAnalysisData {
            hash: image_hasher::ImageHash::from_bytes(&[hash_byte; 32]).expect("hash"),
            result: analysis_result(path, quality_score),
        }
    }

    #[test]
    fn unprocessed_photos_get_no_proposal_and_stay_out_of_groups() {
        // Identical hashes: without the guard, the three photos form one similar group.
        let mut data = vec![
            analysis_data("a.jpg", 0.9, 7),
            analysis_data("b.jpg", 0.8, 7),
            analysis_data("c.jpg", 0.7, 7),
        ];
        let unprocessed: HashSet<String> = ["a.jpg".to_owned()].into_iter().collect();
        let settings = CullingSettings::default();
        let suggestions = build_suggestions(&mut data, &settings, &unprocessed);

        let a = suggestions
            .results
            .iter()
            .find(|r| r.path == "a.jpg")
            .unwrap();
        assert_eq!(a.category, "unrated");
        assert_eq!(a.reasons, vec!["unprocessed".to_owned()]);
        assert!(!suggestions.star_assignments.contains_key("a.jpg"));
        assert!(!suggestions.color_assignments.contains_key("a.jpg"));
        assert_eq!(suggestions.similar_groups.len(), 1);
        let group = &suggestions.similar_groups[0];
        assert_eq!(group.representative.path, "b.jpg");
        assert_eq!(group.duplicates.len(), 1);
        assert_eq!(group.duplicates[0].path, "c.jpg");
        assert!(suggestions.star_assignments.contains_key("b.jpg"));
    }

    /// Corpus measurement harness (not a unit test). Runs the production local
    /// pipeline (analysis, YuNet/eyes, grouping, buckets) without the UI:
    /// PICPORTAL_CULLING_CORPUS=<dir> PICPORTAL_CULLING_REPORT=<file.jsonl>
    /// cargo test --lib culling::tests::corpus_measurement -- --ignored --nocapture
    #[test]
    #[ignore]
    fn corpus_measurement() {
        let corpus = std::path::PathBuf::from(
            std::env::var("PICPORTAL_CULLING_CORPUS").expect("PICPORTAL_CULLING_CORPUS"),
        );
        let report = std::env::var("PICPORTAL_CULLING_REPORT").expect("PICPORTAL_CULLING_REPORT");
        let mut paths: Vec<String> = std::fs::read_dir(&corpus)
            .expect("corpus dir")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| crate::formats::is_supported_image_file(path))
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
        paths.sort();
        let app_settings = crate::app_settings::AppSettings::default();
        let settings = CullingSettings {
            detect_subject: false,
            // The harness measures the opt-in detectors, so both stay on unless disabled.
            detect_closed_eyes: std::env::var("PICPORTAL_CULLING_EYES")
                .map_or(true, |value| value != "0"),
            detect_blurry: std::env::var("PICPORTAL_CULLING_BLUR")
                .map_or(true, |value| value != "0"),
            ..Default::default()
        };
        let hasher = HasherConfig::new()
            .hash_alg(HashAlg::DoubleGradient)
            .hash_size(16, 16)
            .to_hasher();
        let started = std::time::Instant::now();
        let analysed: Vec<_> = paths
            .par_iter()
            .map(|path| analyze_image(path, &hasher, &app_settings))
            .collect();
        let analysis_seconds = started.elapsed().as_secs_f64();
        let mut failed = Vec::new();
        let mut successful = Vec::new();
        for (path, result) in paths.iter().zip(analysed) {
            match result {
                Ok(data) => successful.push(data),
                Err(error) => failed.push(format!("{path}: {error}")),
            }
        }
        let face_directory =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/face");
        let mut face_runtime = face_processing::FaceRuntime::load(&face_directory).ok();
        println!("face runtime available: {}", face_runtime.is_some());
        let mut unprocessed = HashSet::new();
        let started = std::time::Instant::now();
        let mut worker = None;
        for data in successful.iter_mut() {
            match load_culling_image(&data.result.path, &app_settings) {
                Ok(image) => apply_assisted_signals(
                    &mut data.result,
                    &image,
                    &settings,
                    &mut worker,
                    face_runtime.as_mut(),
                ),
                Err(_) => {
                    unprocessed.insert(data.result.path.clone());
                }
            }
        }
        let eye_seconds = started.elapsed().as_secs_f64();
        let suggestions = build_suggestions(&mut successful, &settings, &unprocessed);
        let mut group_of = HashMap::new();
        for (index, group) in suggestions.similar_groups.iter().enumerate() {
            group_of.insert(group.representative.path.clone(), index);
            for duplicate in &group.duplicates {
                group_of.insert(duplicate.path.clone(), index);
            }
        }
        let mut lines = String::new();
        for result in &suggestions.results {
            let line = serde_json::json!({
                "path": result.path,
                "category": result.category,
                "reasons": result.reasons,
                "suggestedRating": result.suggested_rating,
                "sharpness": result.sharpness_metric,
                "quality": result.quality_score,
                "blurry": is_blurry(result, &settings),
                "eyeState": result.eye_state,
                "eyeConfidence": result.eye_confidence,
                "eyeMethod": result.eye_method,
                "faceCount": result.face_count,
                "faceWidthsPx": result.attributed_faces.iter().map(|face| (face.width * result.width as f32).round()).collect::<Vec<_>>(),
                "faceEyes": result.attributed_faces.iter().map(|face| face.eye_state.clone()).collect::<Vec<_>>(),
                "imageWidth": result.width,
                "faces": result.attributed_faces.iter().map(|face| [face.x, face.y, face.width, face.height]).collect::<Vec<_>>(),
                "group": group_of.get(&result.path),
            });
            lines.push_str(&line.to_string());
            lines.push('\n');
        }
        std::fs::write(&report, lines).expect("write report");
        println!(
            "images={} ok={} failed={} unprocessed={} groups={} analysis_s={analysis_seconds:.1} eyes_s={eye_seconds:.1}",
            paths.len(),
            suggestions.results.len(),
            failed.len(),
            unprocessed.len(),
            suggestions.similar_groups.len()
        );
        for failure in failed {
            println!("failed: {failure}");
        }
    }

    fn eye_settings(detect_closed_eyes: bool) -> CullingSettings {
        CullingSettings {
            detect_closed_eyes,
            detect_blurry: false,
            detect_duplicates: false,
            ..Default::default()
        }
    }

    fn category_and_rating(
        result: &ImageAnalysisResult,
        settings: &CullingSettings,
        is_duplicate: bool,
        is_selected: bool,
        is_highlight: bool,
    ) -> (&'static str, u8) {
        let category = culling_category(result, settings, is_duplicate, is_selected, is_highlight);
        (category, category_rating(category))
    }

    fn subject_box(x: f32, y: f32, width: f32, height: f32) -> subject_inference::SubjectBox {
        subject_inference::SubjectBox {
            x,
            y,
            width,
            height,
            score: 0.8,
            label: "person".to_owned(),
        }
    }

    #[test]
    fn yunet_fallback_is_only_used_when_primary_pass_is_empty() {
        assert!(should_try_yunet_fallback(0));
        assert!(!should_try_yunet_fallback(1));
        assert!(!should_try_yunet_fallback(2));
        assert_eq!(YUNET_PRIMARY_THRESHOLD, 0.9);
        assert_eq!(YUNET_FALLBACK_THRESHOLD, 0.7);
    }

    #[test]
    fn subject_link_requires_box_overlap_and_a_pose_bust_nose() {
        let subjects = [subject_box(100.0, 100.0, 200.0, 300.0)];
        let face = AssociationFace {
            x: 130.0,
            y: 130.0,
            width: 40.0,
            height: 50.0,
        };
        let linked_pose = AssociationPose {
            nose_x: 150.0,
            nose_y: 150.0,
            torso: vec![(120.0, 220.0), (180.0, 230.0)],
        };
        assert!(face_is_linked(&face, &subjects, &[linked_pose.clone()]));

        let no_bust_pose = AssociationPose {
            nose_x: 150.0,
            nose_y: 150.0,
            torso: vec![(20.0, 20.0), (30.0, 30.0)],
        };
        assert!(!face_is_linked(&face, &subjects, &[no_bust_pose]));

        let outside_face = AssociationFace { x: 400.0, ..face };
        assert!(!face_is_linked(&outside_face, &subjects, &[linked_pose]));
    }

    #[test]
    fn unknown_subject_is_visible_but_does_not_change_category_score() {
        let settings = CullingSettings {
            detect_closed_eyes: false,
            detect_blurry: false,
            detect_duplicates: false,
            ..Default::default()
        };
        let mut result = analysis_result("unknown-subject.jpg", 0.8);
        result.subject_status = "unknown".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings, false, true, false),
            ("selected", 5)
        );
        assert_eq!(
            detector_reasons(&result, &settings, false),
            vec!["subjectUnknown"]
        );
    }

    #[test]
    fn referenced_category_mapping_is_fixed() {
        assert_eq!(category_rating("selected"), 5);
        assert_eq!(category_rating("highlights"), 4);
        assert_eq!(category_rating("duplicate"), 3);
        assert_eq!(category_rating("blurred"), 2);
        assert_eq!(category_rating("closedEyes"), 1);
        assert_eq!(category_rating("unrated"), 0);
        assert_eq!(category_color("selected"), Some("green"));
        assert_eq!(category_color("highlights"), Some("blue"));
        assert_eq!(category_color("duplicate"), Some("yellow"));
        assert_eq!(category_color("blurred"), Some("red"));
        assert_eq!(category_color("closedEyes"), Some("purple"));
        assert_eq!(category_color("unrated"), None);
    }

    #[test]
    fn similarity_groups_include_lower_index_neighbors_discovered_later() {
        let edges = [(0, 2), (2, 0), (2, 1), (1, 2)];
        let groups = connected_components(3, |left, right| edges.contains(&(left, right)));
        assert_eq!(groups, vec![vec![0, 2, 1]]);
    }

    #[test]
    fn equal_quality_representatives_are_independent_of_input_order() {
        let first = analysis_result("a.jpg", 0.75);
        let second = analysis_result("b.jpg", 0.75);
        let mut forward = [&first, &second];
        let mut reversed = [&second, &first];

        let settings = CullingSettings::default();
        forward.sort_by(|left, right| compare_group_candidates(left, right, &settings));
        reversed.sort_by(|left, right| compare_group_candidates(left, right, &settings));

        assert_eq!(forward[0].path, "a.jpg");
        assert_eq!(reversed[0].path, "a.jpg");
    }

    #[test]
    fn any_closed_primary_eye_remains_a_review_signal() {
        let settings = eye_settings(true);
        let eyes = [
            face_processing::EyeAssessment {
                state: face_processing::EyeState::Closed,
                confidence: 0.5,
                method: "local-heuristic",
            },
            face_processing::EyeAssessment {
                state: face_processing::EyeState::Open,
                confidence: 0.8,
                method: "local-heuristic",
            },
        ];
        let mut result = analysis_result("faces.jpg", 0.75);
        result.eye_state = reviewed_eye_state(&eyes).to_owned();
        result.eye_method = "local-heuristic".to_owned();

        assert_eq!(result.eye_state, "closed");
        assert_eq!(
            category_and_rating(&result, &settings, false, true, false),
            ("closedEyes", 1)
        );
    }

    #[test]
    fn open_primary_eyes_leave_selection_to_technical_quality() {
        let settings = eye_settings(true);
        let eyes = [
            face_processing::EyeAssessment {
                state: face_processing::EyeState::Open,
                confidence: 0.5,
                method: "local-heuristic",
            },
            face_processing::EyeAssessment {
                state: face_processing::EyeState::Open,
                confidence: 0.7,
                method: "local-heuristic",
            },
        ];
        let mut result = analysis_result("open.jpg", 0.75);
        result.eye_state = reviewed_eye_state(&eyes).to_owned();
        result.eye_method = "local-heuristic".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings, false, true, false),
            ("selected", 5)
        );
        assert_eq!(reviewed_eye_state(&[]), "unknown");
        assert_eq!(
            reviewed_eye_state(&[face_processing::EyeAssessment {
                state: face_processing::EyeState::Unknown,
                confidence: 0.0,
                method: "local-heuristic"
            }]),
            "unknown"
        );
    }

    #[test]
    fn unknown_eye_signal_keeps_the_unrated_category_and_reason() {
        let settings = eye_settings(true);
        let mut result = analysis_result("unknown.jpg", 0.75);
        result.eye_state = "unknown".to_owned();
        result.eye_method = "unavailable".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings, false, false, false),
            ("unrated", 0)
        );
        assert_eq!(
            detector_reasons(&result, &settings, false),
            vec!["eyesUnknown"]
        );
    }

    #[test]
    fn eye_outcome_analysis_failure_is_unknown() {
        let settings = eye_settings(true);
        let mut result = analysis_result("failed.jpg", 0.75);
        result.eye_state = "unknown".to_owned();
        result.eye_method = "unavailable".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings, false, true, false),
            ("selected", 5)
        );
        assert!(is_selection_candidate(&result, &settings, false));
        assert_eq!(
            detector_reasons(&result, &settings, false),
            vec!["eyesUnknown"]
        );
    }

    #[test]
    fn open_eyes_do_not_displace_an_eligible_unknown_eye_representative() {
        let settings = CullingSettings {
            detect_blurry: true,
            blur_severity: "moderate".to_owned(),
            ..eye_settings(true)
        };
        let open_but_blurry = ImageAnalysisResult {
            eye_state: "open".to_owned(),
            eye_method: "local-heuristic".to_owned(),
            sharpness_metric: 40.0,
            ..analysis_result("open.jpg", 0.99)
        };
        let unknown_but_sharp = ImageAnalysisResult {
            eye_state: "unknown".to_owned(),
            eye_method: "unavailable".to_owned(),
            sharpness_metric: 150.0,
            ..analysis_result("unknown.jpg", 0.3)
        };
        let mut group = [&open_but_blurry, &unknown_but_sharp];

        group.sort_by(|left, right| compare_group_candidates(left, right, &settings));

        assert_eq!(group[0].path, "unknown.jpg");
        assert!(is_selection_candidate(group[0], &settings, false));
        assert!(!is_selection_candidate(group[1], &settings, false));
    }

    #[test]
    fn open_eye_evaluation_precedes_quality_without_rejecting_unknown_eyes() {
        let settings = eye_settings(true);
        let open = ImageAnalysisResult {
            eye_state: "open".to_owned(),
            eye_method: "local-heuristic".to_owned(),
            ..analysis_result("open.jpg", 0.4)
        };
        let unknown = ImageAnalysisResult {
            eye_state: "unknown".to_owned(),
            eye_method: "unavailable".to_owned(),
            ..analysis_result("unknown.jpg", 0.99)
        };

        assert_eq!(
            compare_group_candidates(&open, &unknown, &settings),
            std::cmp::Ordering::Less
        );
        assert!(is_selection_candidate(&unknown, &settings, false));
        assert!(!is_selection_candidate(
            &ImageAnalysisResult {
                eye_state: "closed".to_owned(),
                ..analysis_result("closed.jpg", 1.0)
            },
            &settings,
            false,
        ));
    }

    #[test]
    fn eye_outcome_disabled_analysis_uses_quality() {
        let settings = eye_settings(false);
        let mut result = analysis_result("disabled.jpg", 0.75);
        result.eye_state = "unknown".to_owned();
        result.eye_method = "unavailable".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings, false, true, false),
            ("selected", 5)
        );
        assert!(detector_reasons(&result, &settings, false).is_empty());
    }

    #[test]
    fn disabled_blur_detector_does_not_assign_blurred_category() {
        let settings = CullingSettings {
            detect_blurry: false,
            detect_closed_eyes: false,
            ..Default::default()
        };
        let mut result = analysis_result("disabled-blur.jpg", 0.9);
        result.sharpness_metric = 0.0;

        assert_eq!(
            category_and_rating(&result, &settings, false, true, false),
            ("selected", 5)
        );
        assert!(detector_reasons(&result, &settings, false).is_empty());
    }

    #[test]
    fn overlap_priority_is_deterministic_and_reasons_are_retained() {
        // Opt-in detectors: this checks their priority when the user enables them.
        let settings = CullingSettings {
            detect_blurry: true,
            detect_closed_eyes: true,
            ..Default::default()
        };
        let mut result = analysis_result("overlap.jpg", 0.9);
        result.sharpness_metric = 0.0;
        result.eye_state = "closed".to_owned();
        result.eye_method = "local-heuristic".to_owned();

        assert_eq!(
            category_and_rating(&result, &settings, true, true, true),
            ("closedEyes", 1)
        );
        assert_eq!(
            detector_reasons(&result, &settings, true),
            vec!["duplicate", "blurred", "closedEyes"]
        );
    }

    #[test]
    fn amount_presets_change_selected_count() {
        assert_eq!(selected_limit(20, "extreme"), 1);
        assert_eq!(selected_limit(20, "few"), 2);
        assert_eq!(selected_limit(20, "standard"), 4);
        assert_eq!(selected_limit(20, "more"), 7);
    }

    #[test]
    fn detectors_measured_unreliable_are_off_by_default() {
        // MAX-21 corpus: 88/120 "closed eyes" and 59/120 "blurry" on sharp, open-eyed photos.
        let settings = CullingSettings::default();
        assert!(!settings.detect_blurry);
        assert!(!settings.detect_closed_eyes);
        let from_partial: CullingSettings = serde_json::from_str("{}").unwrap();
        assert!(!from_partial.detect_blurry && !from_partial.detect_closed_eyes);
    }

    #[test]
    fn blur_severity_changes_the_real_threshold() {
        assert!(blur_threshold("lenient") < blur_threshold("moderate"));
        assert!(blur_threshold("moderate") < blur_threshold("strict"));
    }
}
