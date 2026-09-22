use crate::app_settings::load_settings;
use image::{DynamicImage, GenericImageView, GrayImage, imageops};
use image_hasher::{HashAlg, HasherConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use tauri::{AppHandle, Emitter, WebviewWindow};

use crate::{face_processing, image_loader, subject_inference};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct CullingSettings {
    pub similarity_threshold: u32,
    pub blur_threshold: f64,
    pub group_similar: bool,
    pub filter_blurry: bool,
    pub selection_amount: String,
    pub blur_severity: String,
    pub detect_subject: bool,
    pub subject_profile: String,
    pub detect_closed_eyes: bool,
    pub review_focus: bool,
}

impl Default for CullingSettings {
    fn default() -> Self {
        Self {
            similarity_threshold: 28,
            blur_threshold: 100.0,
            group_similar: true,
            filter_blurry: true,
            selection_amount: "standard".to_owned(),
            blur_severity: "moderate".to_owned(),
            detect_subject: true,
            subject_profile: "general".to_owned(),
            detect_closed_eyes: false,
            review_focus: true,
        }
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SubjectBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
    pub label: String,
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
    pub focus_signal: Option<f64>,
    pub focus_status: String,
    pub focus_method: String,
    pub focus_crop_width: Option<u32>,
    pub focus_crop_height: Option<u32>,
    pub focus_input_width: Option<u32>,
    pub focus_input_height: Option<u32>,
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
    pub subject_status: String,
    pub subject_method: String,
    pub subject_boxes: Vec<SubjectBox>,
    pub attributed_faces: Vec<AttributedFace>,
    pub focus_signal: Option<f64>,
    pub focus_status: String,
    pub focus_method: String,
    pub focus_crop_width: Option<u32>,
    pub focus_crop_height: Option<u32>,
    pub focus_input_width: Option<u32>,
    pub focus_input_height: Option<u32>,
    pub eye_state: String,
    pub eye_confidence: f32,
    pub eye_method: String,
    pub review_alerts: Vec<String>,
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
    pub blurry_images: Vec<ImageAnalysisResult>,
    pub review_alerts: Vec<ImageAnalysisResult>,
    pub unknown_images: Vec<ImageAnalysisResult>,
    pub failed_paths: Vec<String>,
    pub subject_analysis_status: String,
}

#[derive(Serialize, Clone)]
struct CullingProgress {
    current: usize,
    total: usize,
    stage: String,
}

struct ImageAnalysisData {
    hash: image_hasher::ImageHash,
    result: ImageAnalysisResult,
}

const SUBJECT_OVERLAP_MIN: f32 = 0.5;
const NOSE_DISTANCE_FACTOR: f32 = 1.5;
const TORSO_INSIDE_MIN: usize = 2;
const WEIGHT_SHARPNESS: f64 = 0.40;
const WEIGHT_CENTER_FOCUS: f64 = 0.35;
const WEIGHT_EXPOSURE: f64 = 0.25;

struct CullingCancellation {
    owner: String,
    paths_to_cull: Vec<String>,
    progress: Mutex<CullingProgress>,
    requested: AtomicBool,
    worker: Mutex<Option<subject_inference::LocalWorkerCancellation>>,
}

impl CullingCancellation {
    fn new(owner: &str, paths_to_cull: &[String]) -> Self {
        Self {
            owner: owner.to_owned(),
            paths_to_cull: paths_to_cull.to_vec(),
            progress: Mutex::new(CullingProgress {
                current: 0,
                total: paths_to_cull.len(),
                stage: "Initializing...".to_owned(),
            }),
            requested: AtomicBool::new(false),
            worker: Mutex::new(None),
        }
    }

    fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn update_progress(&self, progress: &CullingProgress) {
        let mut current = self
            .progress
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if progress.stage != current.stage || progress.current >= current.current {
            *current = progress.clone();
        }
    }

    fn request(&self) {
        self.requested.store(true, Ordering::Release);
        let worker = self
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(worker) = worker.as_ref() {
            worker.cancel();
        }
    }

    fn attach_worker(&self, worker: subject_inference::LocalWorkerCancellation) {
        let mut attached = self
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *attached = Some(worker);
        if self.is_requested()
            && let Some(worker) = attached.as_ref()
        {
            worker.cancel();
        }
    }

    fn detach_worker(&self) {
        let mut attached = self
            .worker
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *attached = None;
    }
}

static ACTIVE_CULLING_INVOCATIONS: LazyLock<Mutex<HashMap<String, Arc<CullingCancellation>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct CullingInvocationGuard {
    invocation_id: String,
    cancellation: Arc<CullingCancellation>,
    active: bool,
}

impl CullingInvocationGuard {
    fn register(invocation_id: &str, owner: &str, paths_to_cull: &[String]) -> Result<Self, String> {
        let cancellation = Arc::new(CullingCancellation::new(owner, paths_to_cull));
        let mut active = ACTIVE_CULLING_INVOCATIONS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if active.contains_key(invocation_id) {
            return Err("CULLING_INVOCATION_ALREADY_ACTIVE".to_owned());
        }
        if active.values().any(|value| value.owner == owner) {
            return Err("CULLING_OWNER_ALREADY_ACTIVE".to_owned());
        }
        active.insert(invocation_id.to_owned(), Arc::clone(&cancellation));
        Ok(Self {
            invocation_id: invocation_id.to_owned(),
            cancellation,
            active: true,
        })
    }

    fn conclude(&mut self) -> bool {
        let mut active = ACTIVE_CULLING_INVOCATIONS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.active
            && active
                .get(&self.invocation_id)
                .is_some_and(|value| Arc::ptr_eq(value, &self.cancellation))
        {
            active.remove(&self.invocation_id);
        }
        self.active = false;
        self.cancellation.is_requested()
    }
}

impl Drop for CullingInvocationGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = self.conclude();
        }
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ActiveCullingInvocation {
    invocation_id: String,
    progress: CullingProgress,
    paths_to_cull: Vec<String>,
    is_cancelling: bool,
}

fn active_culling_for_owner(owner: &str) -> Option<ActiveCullingInvocation> {
    let active = ACTIVE_CULLING_INVOCATIONS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    active.iter().find_map(|(invocation_id, cancellation)| {
        if cancellation.owner != owner {
            return None;
        }
        let progress = cancellation
            .progress
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        Some(ActiveCullingInvocation {
            invocation_id: invocation_id.clone(),
            progress,
            paths_to_cull: cancellation.paths_to_cull.clone(),
            is_cancelling: cancellation.is_requested(),
        })
    })
}

fn request_culling_cancel(owner: &str, invocation_id: &str) -> bool {
    let active = ACTIVE_CULLING_INVOCATIONS
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let Some(cancellation) = active.get(invocation_id) else {
        return false;
    };
    if cancellation.owner != owner {
        return false;
    }
    cancellation.request();
    true
}

fn effective_blur_threshold(settings: &CullingSettings) -> f64 {
    match settings.blur_severity.as_str() {
        "lenient" => 75.0,
        "strict" => 150.0,
        _ => 100.0,
    }
}

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
            let conv_val = (p_north + p_south + p_west + p_east - 4 * p_center) as f64;
            laplacian_values.push(conv_val);
            sum += conv_val;
        }
    }

    if laplacian_values.is_empty() {
        return 0.0;
    }
    let mean = sum / laplacian_values.len() as f64;

    laplacian_values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / laplacian_values.len() as f64
}

fn calculate_exposure_metric(image: &GrayImage) -> f64 {
    let histogram = imageproc::stats::histogram(image);
    let total_pixels = (image.width() * image.height()) as f64;
    if total_pixels == 0.0 {
        return 0.0;
    }

    let clip_threshold_dark = 5;
    let clip_threshold_bright = 250;

    let dark_pixels = histogram.channels[0][0..clip_threshold_dark]
        .iter()
        .sum::<u32>() as f64;
    let bright_pixels = histogram.channels[0][clip_threshold_bright..256]
        .iter()
        .sum::<u32>() as f64;

    let dark_clip_ratio = dark_pixels / total_pixels;
    let bright_clip_ratio = bright_pixels / total_pixels;

    let penalty = (dark_clip_ratio * 5.0) + (bright_clip_ratio * 5.0);

    (1.0f64 - penalty).max(0.0)
}

fn load_analysis_image(
    path: &str,
    settings: &crate::app_settings::AppSettings,
) -> Result<DynamicImage, String> {
    if crate::file_management::is_cloud_placeholder(Path::new(path)) {
        return Err(format!("'{}' is stored in iCloud and not downloaded", path));
    }
    let file_bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    image_loader::load_base_image_from_bytes(&file_bytes, path, true, settings, None)
        .map_err(|e| e.to_string())
}

fn analyze_image(
    path: &str,
    hasher: &image_hasher::Hasher,
    settings: &crate::app_settings::AppSettings,
) -> Result<ImageAnalysisData, String> {
    const ANALYSIS_DIM: u32 = 720; // FIXME: How should we calculate good focus if it's downscaled?!?

    let img = load_analysis_image(path, settings)?;

    let (width, height) = img.dimensions();
    let thumbnail = img.thumbnail(ANALYSIS_DIM, ANALYSIS_DIM);
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

    let normalized_sharpness = ((sharpness_metric + 1.0).log10() / 3.5).min(1.0);
    let normalized_center_focus = ((center_focus_metric + 1.0).log10() / 3.5).min(1.0);

    let quality_score = (normalized_sharpness * WEIGHT_SHARPNESS)
        + (normalized_center_focus * WEIGHT_CENTER_FOCUS)
        + (exposure_metric * WEIGHT_EXPOSURE);

    let hash = hasher.hash_image(&thumbnail);

    Ok(ImageAnalysisData {
        hash,
        result: ImageAnalysisResult {
            path: path.to_string(),
            quality_score,
            sharpness_metric,
            center_focus_metric,
            exposure_metric,
            width,
            height,
            subject_status: "not-evaluated".to_owned(),
            subject_method: "not-requested".to_owned(),
            subject_boxes: Vec::new(),
            attributed_faces: Vec::new(),
            focus_signal: None,
            focus_status: "not-evaluated".to_owned(),
            focus_method: "not-requested".to_owned(),
            focus_crop_width: None,
            focus_crop_height: None,
            focus_input_width: None,
            focus_input_height: None,
            eye_state: "not-evaluated".to_owned(),
            eye_confidence: 0.0,
            eye_method: "not-requested".to_owned(),
            review_alerts: Vec::new(),
        },
    })
}

#[derive(Clone, Copy)]
struct SubjectPhrase {
    phrase: &'static str,
}

fn subject_phrases(profile: &str) -> Vec<SubjectPhrase> {
    match profile {
        "dance" => vec![
            SubjectPhrase {
                phrase: "a couple dancing together.",
            },
            SubjectPhrase {
                phrase: "a person dancing.",
            },
        ],
        "portrait" => vec![SubjectPhrase {
            phrase: "a person being photographed.",
        }],
        "sports" => vec![SubjectPhrase {
            phrase: "a person participating in a sport.",
        }],
        "wedding" => vec![SubjectPhrase {
            phrase: "a couple or group of people at a wedding.",
        }],
        _ => vec![SubjectPhrase {
            phrase: "a person or group of people.",
        }],
    }
}

fn choose_subject_phrase<'a>(
    passes: &'a [(SubjectPhrase, Vec<SubjectBox>)],
) -> Option<&'a (SubjectPhrase, Vec<SubjectBox>)> {
    passes.iter().find(|(_, boxes)| !boxes.is_empty())
}

fn focus_review_alerts(score: f64) -> Vec<&'static str> {
    let _ = score;
    Vec::new()
}

fn focus_calibration_status() -> &'static str {
    "calibration-unavailable"
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttributedRole {
    Primary,
    Unknown,
    Secondary,
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

struct SubjectBust {
    nose_x: f32,
    nose_y: f32,
}

fn point_in_box(x: f32, y: f32, subject: &SubjectBox) -> bool {
    x >= subject.x
        && x <= subject.x + subject.width
        && y >= subject.y
        && y <= subject.y + subject.height
}

fn face_overlap(face: &AssociationFace, boxes: &[SubjectBox]) -> f32 {
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

fn subject_busts(poses: &[AssociationPose], boxes: &[SubjectBox]) -> Vec<SubjectBust> {
    let mut busts = Vec::new();
    for pose in poses {
        let inside = pose
            .torso
            .iter()
            .filter(|(x, y)| boxes.iter().any(|subject| point_in_box(*x, *y, subject)))
            .count();
        if inside >= TORSO_INSIDE_MIN {
            busts.push(SubjectBust {
                nose_x: pose.nose_x,
                nose_y: pose.nose_y,
            });
        }
    }
    busts
}

fn face_linked_to_subject_bust(
    face: &AssociationFace,
    boxes: &[SubjectBox],
    poses: &[AssociationPose],
) -> bool {
    if face_overlap(face, boxes) < SUBJECT_OVERLAP_MIN {
        return false;
    }
    let center_x = face.x + face.width / 2.0;
    let center_y = face.y + face.height / 2.0;
    let limit = NOSE_DISTANCE_FACTOR * face.width.max(face.height);
    subject_busts(poses, boxes).into_iter().any(|bust| {
        let distance =
            ((center_x - bust.nose_x).powi(2) + (center_y - bust.nose_y).powi(2)).sqrt();
        distance <= limit
    })
}

fn attribute_faces(
    faces: &[AssociationFace],
    boxes: &[SubjectBox],
    poses: &[AssociationPose],
) -> Vec<AttributedRole> {
    faces
        .iter()
        .map(|face| {
            if face_linked_to_subject_bust(face, boxes, poses) {
                AttributedRole::Primary
            } else {
                AttributedRole::Secondary
            }
        })
        .collect()
}

fn status_when_face_model_missing(status: &str) -> &'static str {
    if matches!(
        status,
        "ready"
            | "subject-ready-pose-unavailable"
            | "subject-ready-focus-unavailable"
            | "subject-ready-focus-calibration-unavailable"
            | "focus-calibration-unavailable"
            | "focus-unavailable"
            | "disabled"
    ) {
        "face-unavailable"
    } else {
        "unavailable"
    }
}

fn display_subject_boxes(boxes: Vec<SubjectBox>, width: f32, height: f32) -> Vec<SubjectBox> {
    let width = width.max(1.0);
    let height = height.max(1.0);
    boxes
        .into_iter()
        .map(|subject_box| SubjectBox {
            x: subject_box.x / width,
            y: subject_box.y / height,
            width: subject_box.width / width,
            height: subject_box.height / height,
            score: subject_box.score,
            label: subject_box.label,
        })
        .collect()
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
    subject: &SubjectBox,
) -> Option<(DynamicImage, f32, f32, f32, f32)> {
    let width = image.width() as f32;
    let height = image.height() as f32;
    let x0 = subject.x.max(0.0);
    let y0 = subject.y.max(0.0);
    let x1 = (subject.x + subject.width).min(width);
    let y1 = (subject.y + subject.height).min(height);
    let x = x0.floor() as u32;
    let y = y0.floor() as u32;
    if x >= image.width() || y >= image.height() {
        return None;
    }
    let crop_width = (x1.ceil() as u32)
        .saturating_sub(x)
        .min(image.width() - x);
    let crop_height = (y1.ceil() as u32)
        .saturating_sub(y)
        .min(image.height() - y);
    if crop_width < 8 || crop_height < 8 {
        return None;
    }
    let cropped = DynamicImage::ImageRgba8(
        image::imageops::crop_imm(image, x, y, crop_width, crop_height).to_image(),
    );
    Some((
        cropped,
        x as f32,
        y as f32,
        crop_width as f32,
        crop_height as f32,
    ))
}

fn collect_association_poses(
    worker: &mut subject_inference::LocalWorker,
    image: &DynamicImage,
    frame: &DynamicImage,
    boxes: &[SubjectBox],
) -> Result<Vec<AssociationPose>, String> {
    let mut poses = scale_pose_frame(
        worker.pose(frame)?,
        0.0,
        0.0,
        image.width() as f32,
        image.height() as f32,
    );
    for subject in boxes {
        if let Some((crop, origin_x, origin_y, crop_width, crop_height)) =
            exact_subject_crop(image, subject)
            && let Ok(crop_frame) = worker.pose(&crop)
        {
            poses.extend(scale_pose_frame(
                crop_frame,
                origin_x,
                origin_y,
                crop_width,
                crop_height,
            ));
        }
    }
    Ok(poses)
}

fn eye_state_label(state: face_processing::EyeState) -> &'static str {
    match state {
        face_processing::EyeState::Open => "open",
        face_processing::EyeState::Closed => "closed",
        face_processing::EyeState::Unknown => "unknown",
    }
}

fn reviewed_primary_eye_state(states: &[face_processing::EyeState]) -> &'static str {
    if states
        .iter()
        .any(|state| *state == face_processing::EyeState::Closed)
    {
        "closed"
    } else if states.is_empty() {
        "not-evaluated"
    } else if states
        .iter()
        .any(|state| *state == face_processing::EyeState::Unknown)
    {
        "unknown"
    } else {
        "open"
    }
}

fn primary_eye_review_alert(states: &[face_processing::EyeState]) -> Option<&'static str> {
    if states
        .iter()
        .any(|state| *state == face_processing::EyeState::Closed)
    {
        Some("eyesClosed")
    } else if !states.is_empty()
        && states
            .iter()
            .any(|state| *state == face_processing::EyeState::Unknown)
    {
        Some("eyesUnknown")
    } else {
        None
    }
}

fn review_coverage_is_unknown(
    settings: &CullingSettings,
    subject_status: &str,
    focus_status: &str,
    eye_state: &str,
) -> bool {
    subject_status == "unknown"
        || (settings.review_focus && matches!(focus_status, "unknown" | "unavailable"))
        || (settings.detect_closed_eyes && eye_state == "not-evaluated")
}

fn subject_status_label(
    detect_subject: bool,
    has_subject_boxes: bool,
    primary_count: usize,
) -> &'static str {
    if !detect_subject {
        "not-evaluated"
    } else if !has_subject_boxes || primary_count == 0 {
        "unknown"
    } else if primary_count > 1 {
        "multiple"
    } else {
        "primary"
    }
}

fn update_assisted_result(
    data: &mut ImageAnalysisData,
    image: &DynamicImage,
    settings: &CullingSettings,
    worker: &mut Option<subject_inference::LocalWorker>,
    face_runtime: &mut Option<face_processing::FaceRuntime>,
) {
    let frame = image.thumbnail(1920, 1920);
    let frame_width = frame.width().max(1) as f32;
    let frame_height = frame.height().max(1) as f32;
    let source_width = image.width().max(1) as f32;
    let source_height = image.height().max(1) as f32;
    let mut subject_boxes = Vec::new();
    let mut subject_method = "grounding-dino-unavailable".to_owned();

    if settings.detect_subject {
        if let Some(local_worker) = worker.as_mut() {
            let mut passes = Vec::new();
            let mut failed = false;
            for phrase in subject_phrases(&settings.subject_profile) {
                match local_worker.detect(&frame, phrase.phrase) {
                    Ok(boxes) => {
                        let mapped = boxes
                            .into_iter()
                            .filter(|item| item.width > 0.0 && item.height > 0.0)
                            .map(|item| SubjectBox {
                                x: item.x * source_width / frame_width,
                                y: item.y * source_height / frame_height,
                                width: item.width * source_width / frame_width,
                                height: item.height * source_height / frame_height,
                                score: item.score,
                                label: item.label,
                            })
                            .collect::<Vec<_>>();
                        let found = !mapped.is_empty();
                        passes.push((phrase, mapped));
                        if found {
                            break;
                        }
                    }
                    Err(_) => {
                        failed = true;
                        break;
                    }
                }
            }
            if failed {
                subject_method = "grounding-dino-error".to_owned();
            } else if let Some((_, boxes)) = choose_subject_phrase(&passes) {
                subject_boxes = boxes.clone();
                subject_method = "grounding-dino-local".to_owned();
            } else if !passes.is_empty() {
                subject_method = "grounding-dino-local".to_owned();
            }
        }
    } else {
        subject_method = "disabled".to_owned();
    }

    let mut faces = Vec::new();
    let mut face_method = "yunet-unavailable".to_owned();
    if let Some(runtime) = face_runtime.as_mut() {
        let rgb = image.to_rgb8();
        match runtime.analyze(&rgb, runtime.primary_threshold()) {
            Ok(primary) => {
                faces = primary;
                face_method = "yunet-0.9".to_owned();
            }
            Err(_) => {
                face_method = "yunet-error".to_owned();
            }
        }
        if let Some(fallback_threshold) =
            face_processing::choose_fallback_threshold(runtime, &faces)
        {
            match runtime.analyze(&rgb, fallback_threshold) {
                Ok(fallback) => {
                    if !fallback.is_empty() {
                        faces = fallback;
                        face_method = "yunet-fallback-0.7".to_owned();
                    }
                }
                Err(_) => {
                    face_method = format!("{face_method};fallback-error");
                }
            }
        }
    }

    let mut pose_method = "pose-not-requested".to_owned();
    let roles = if !settings.detect_subject {
        vec![AttributedRole::Primary; faces.len()]
    } else if subject_boxes.is_empty() {
        vec![AttributedRole::Secondary; faces.len()]
    } else if let Some(local_worker) = worker.as_mut() {
        if local_worker.pose_is_ready() {
            match collect_association_poses(local_worker, image, &frame, &subject_boxes) {
                Ok(poses) => {
                    pose_method = "pose-lite".to_owned();
                    attribute_faces(
                        &faces
                            .iter()
                            .map(|face| AssociationFace {
                                x: face.bbox.x,
                                y: face.bbox.y,
                                width: face.bbox.width,
                                height: face.bbox.height,
                            })
                            .collect::<Vec<_>>(),
                        &subject_boxes,
                        &poses,
                    )
                }
                Err(_) => {
                    pose_method = "pose-error".to_owned();
                    vec![AttributedRole::Unknown; faces.len()]
                }
            }
        } else {
            pose_method = "pose-unavailable".to_owned();
            vec![AttributedRole::Unknown; faces.len()]
        }
    } else {
        pose_method = "pose-unavailable".to_owned();
        vec![AttributedRole::Unknown; faces.len()]
    };
    let primary_indices: Vec<usize> = roles
        .iter()
        .enumerate()
        .filter(|(_, role)| **role == AttributedRole::Primary)
        .map(|(index, _)| index)
        .collect();

    let mut attributed_faces = Vec::new();
    let mut focus_values = Vec::new();
    let mut focus_error = false;
    let mut focus_dimensions = None;
    let mut eye_states = Vec::new();
    let mut review_alerts = Vec::new();
    let focus_available = settings.review_focus
        && worker
            .as_ref()
            .map(subject_inference::LocalWorker::focus_is_ready)
            .unwrap_or(false);

    for (index, face) in faces.iter().enumerate() {
        let is_primary = roles.get(index) == Some(&AttributedRole::Primary);
        let role = match roles.get(index).copied().unwrap_or(AttributedRole::Secondary) {
            AttributedRole::Primary => "primary",
            AttributedRole::Unknown => "unknown",
            AttributedRole::Secondary => "secondary",
        };
        let mut focus_crop_dimensions = None;
        let mut focus_model_dimensions = None;
        let eye_state = if is_primary && settings.detect_closed_eyes {
            eye_states.push(face.eye.state);
            eye_state_label(face.eye.state).to_owned()
        } else if settings.detect_closed_eyes {
            "not-attributed".to_owned()
        } else {
            "disabled".to_owned()
        };
        let mut focus_signal = None;
        let mut focus_status = if !settings.review_focus {
            "disabled".to_owned()
        } else if !is_primary {
            "not-attributed".to_owned()
        } else if !focus_available {
            "unavailable".to_owned()
        } else {
            "pending".to_owned()
        };
        let mut focus_method = if focus_available {
            "vgg-native-local".to_owned()
        } else {
            "vgg-native-unavailable".to_owned()
        };
        if is_primary && focus_available {
            if let Some(crop) = face_processing::crop_face(image, &face.bbox, 1.6) {
                if let Some(local_worker) = worker.as_mut() {
                    match local_worker.focus(&crop) {
                        Ok(measurement) => {
                            focus_signal = Some(measurement.score);
                            focus_status = focus_calibration_status().to_owned();
                            if focus_dimensions.is_none() {
                                focus_dimensions = Some((measurement.width, measurement.height));
                            }
                            focus_crop_dimensions = Some((measurement.width, measurement.height));
                            focus_model_dimensions =
                                Some((measurement.input_width, measurement.input_height));
                            focus_values.push(measurement.score);
                            review_alerts.extend(
                                focus_review_alerts(measurement.score)
                                    .into_iter()
                                    .map(str::to_owned),
                            );
                        }
                        Err(_) => {
                            focus_status = "unknown".to_owned();
                            focus_method = "vgg-native-error".to_owned();
                            focus_error = true;
                        }
                    }
                }
            } else {
                focus_status = "unknown".to_owned();
                focus_method = "vgg-native-crop-failed".to_owned();
                focus_error = true;
            }
        }
        attributed_faces.push(AttributedFace {
            role: role.to_owned(),
            x: face.bbox.x / source_width,
            y: face.bbox.y / source_height,
            width: face.bbox.width / source_width,
            height: face.bbox.height / source_height,
            confidence: face.confidence,
            eye_state,
            eye_confidence: face.eye.confidence,
            focus_signal,
            focus_status,
            focus_method,
            focus_crop_width: focus_crop_dimensions.map(|value| value.0),
            focus_crop_height: focus_crop_dimensions.map(|value| value.1),
            focus_input_width: focus_model_dimensions.map(|value| value.0),
            focus_input_height: focus_model_dimensions.map(|value| value.1),
        });
    }

    let subject_status = subject_status_label(
        settings.detect_subject,
        !subject_boxes.is_empty(),
        primary_indices.len(),
    );
    let eye_state = if !settings.detect_closed_eyes {
        "disabled"
    } else {
        reviewed_primary_eye_state(&eye_states)
    };
    if settings.detect_closed_eyes
        && let Some(alert) = primary_eye_review_alert(&eye_states)
    {
        review_alerts.push(alert.to_owned());
    }
    if subject_status == "unknown" {
        review_alerts.push("subjectUnknown".to_owned());
    }

    data.result.subject_status = subject_status.to_owned();
    data.result.subject_method = if subject_method == "grounding-dino-error" {
        subject_method
    } else if settings.detect_subject {
        format!("{subject_method};{pose_method};{face_method}")
    } else if subject_boxes.is_empty() && !faces.is_empty() {
        format!("{subject_method};{face_method}")
    } else {
        subject_method
    };
    data.result.subject_boxes = display_subject_boxes(subject_boxes, source_width, source_height);
    data.result.attributed_faces = attributed_faces;
    data.result.eye_state = eye_state.to_owned();
    data.result.eye_confidence = faces
        .iter()
        .enumerate()
        .filter(|(index, _)| primary_indices.contains(index))
        .map(|(_, face)| face.eye.confidence)
        .fold(0.0_f32, f32::max);
    data.result.eye_method = if settings.detect_closed_eyes {
        face_method.clone()
    } else {
        "disabled".to_owned()
    };
    data.result.focus_signal = if focus_values.is_empty() {
        None
    } else {
        Some(focus_values.iter().sum::<f64>() / focus_values.len() as f64)
    };
    data.result.focus_status = if !settings.review_focus {
        "disabled".to_owned()
    } else if data.result.focus_signal.is_some() {
        focus_calibration_status().to_owned()
    } else if !focus_available {
        "unavailable".to_owned()
    } else if primary_indices.is_empty() {
        "not-attributed".to_owned()
    } else {
        "unknown".to_owned()
    };
    data.result.focus_method = if focus_error {
        "vgg-native-error".to_owned()
    } else if focus_available {
        "vgg-native-local".to_owned()
    } else {
        "vgg-native-unavailable".to_owned()
    };
    data.result.focus_crop_width = focus_dimensions.map(|value| value.0);
    data.result.focus_crop_height = focus_dimensions.map(|value| value.1);
    data.result.focus_input_width = data
        .result
        .attributed_faces
        .iter()
        .find_map(|face| face.focus_input_width);
    data.result.focus_input_height = data
        .result
        .attributed_faces
        .iter()
        .find_map(|face| face.focus_input_height);
    data.result.review_alerts = review_alerts;
}

fn apply_assisted_image_load(
    data: &mut ImageAnalysisData,
    loaded: Result<DynamicImage, String>,
    settings: &CullingSettings,
    worker: &mut Option<subject_inference::LocalWorker>,
    face_runtime: &mut Option<face_processing::FaceRuntime>,
    failed_paths: &mut Vec<String>,
) {
    match loaded {
        Ok(image) => update_assisted_result(data, &image, settings, worker, face_runtime),
        Err(error) => {
            failed_paths.push(data.result.path.clone());
            data.result.subject_status = "unknown".to_owned();
            data.result.subject_method = format!("image-unavailable:{error}");
            data.result.focus_status = "unknown".to_owned();
            data.result.eye_state = "not-evaluated".to_owned();
            data.result.review_alerts.push("subjectUnknown".to_owned());
        }
    }
}

fn subject_analysis_status_for(
    worker: &subject_inference::LocalWorker,
    subject_requested: bool,
    focus_requested: bool,
) -> String {
    if subject_requested {
        if !worker.subject_is_ready() {
            "subject-unavailable".to_owned()
        } else if !worker.pose_is_ready() {
            "subject-ready-pose-unavailable".to_owned()
        } else if focus_requested && !worker.focus_is_ready() {
            "subject-ready-focus-unavailable".to_owned()
        } else if focus_requested {
            "subject-ready-focus-calibration-unavailable".to_owned()
        } else {
            "ready".to_owned()
        }
    } else if !focus_requested {
        "disabled".to_owned()
    } else if worker.focus_is_ready() {
        "focus-calibration-unavailable".to_owned()
    } else {
        "focus-unavailable".to_owned()
    }
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct CullingBoundEvent<T> {
    invocation_id: String,
    body: T,
}

fn emit_culling<T: Serialize + Clone>(
    app_handle: &AppHandle,
    event: &str,
    invocation_id: &str,
    body: T,
) {
    let _ = app_handle.emit(
        event,
        CullingBoundEvent {
            invocation_id: invocation_id.to_owned(),
            body,
        },
    );
}

fn emit_culling_progress(
    app_handle: &AppHandle,
    invocation_id: &str,
    cancellation: &CullingCancellation,
    progress: CullingProgress,
) {
    cancellation.update_progress(&progress);
    emit_culling(app_handle, "culling-progress", invocation_id, progress);
}

#[tauri::command]
pub fn get_active_culling(window: WebviewWindow) -> Option<ActiveCullingInvocation> {
    active_culling_for_owner(window.label())
}

#[tauri::command]
pub fn cancel_culling(
    invocation_id: String,
    app_handle: AppHandle,
    window: WebviewWindow,
) -> bool {
    let accepted = request_culling_cancel(window.label(), &invocation_id);
    if accepted {
        emit_culling(&app_handle, "culling-cancelling", &invocation_id, ());
    }
    accepted
}

fn finish_cancelled_culling(
    guard: &mut CullingInvocationGuard,
    app_handle: &AppHandle,
    invocation_id: &str,
) -> Result<CullingSuggestions, String> {
    guard.conclude();
    emit_culling(app_handle, "culling-cancelled", invocation_id, ());
    Ok(CullingSuggestions::default())
}

#[tauri::command]
pub async fn cull_images(
    paths: Vec<String>,
    settings: CullingSettings,
    invocation_id: String,
    app_handle: AppHandle,
    window: WebviewWindow,
) -> Result<CullingSuggestions, String> {
    let mut invocation = CullingInvocationGuard::register(&invocation_id, window.label(), &paths)?;
    let cancellation = Arc::clone(&invocation.cancellation);
    let total_count = paths.len();
    emit_culling(&app_handle, "culling-start", &invocation_id, total_count);

    if paths.is_empty() {
        let suggestions = CullingSuggestions::default();
        if invocation.conclude() {
            emit_culling(&app_handle, "culling-cancelled", &invocation_id, ());
        } else {
            emit_culling(
                &app_handle,
                "culling-complete",
                &invocation_id,
                &suggestions,
            );
        }
        return Ok(suggestions);
    }

    let app_settings = load_settings(app_handle.clone()).unwrap_or_default();
    let completed_count = Arc::new(AtomicUsize::new(0));

    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(16, 16)
        .to_hasher();

    let mut analysis_results: Vec<Result<ImageAnalysisData, (String, String)>> =
        Vec::with_capacity(total_count);
    let batch_size = rayon::current_num_threads().max(1);
    for batch in paths.chunks(batch_size) {
        if cancellation.is_requested() {
            return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
        }
        let batch_results = batch
            .par_iter()
            .filter_map(|path| {
                if cancellation.is_requested() {
                    return None;
                }
                let result =
                    analyze_image(path, &hasher, &app_settings).map_err(|e| (path.to_string(), e));
                let completed = completed_count.fetch_add(1, Ordering::Relaxed) + 1;
                emit_culling_progress(
                    &app_handle,
                    &invocation_id,
                    &cancellation,
                    CullingProgress {
                        current: completed,
                        total: total_count,
                        stage: "Analyzing images...".to_string(),
                    },
                );
                Some(result)
            })
            .collect::<Vec<_>>();
        analysis_results.extend(batch_results);
    }
    if cancellation.is_requested() {
        return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
    }

    let mut successful_analyses = Vec::new();
    let mut failed_paths = Vec::new();
    for res in analysis_results {
        match res {
            Ok(data) => successful_analyses.push(data),
            Err((path, error)) => {
                eprintln!("Failed to analyze image {}: {}", path, error);
                failed_paths.push(path);
            }
        }
    }

    let mut worker = None;
    let needs_local_worker = settings.detect_subject || settings.review_focus;
    let mut subject_analysis_status = if !needs_local_worker {
        "disabled".to_owned()
    } else {
        match subject_inference::LocalWorker::start(&app_handle) {
            Ok(local_worker) => {
                let status = subject_analysis_status_for(
                    &local_worker,
                    settings.detect_subject,
                    settings.review_focus,
                );
                cancellation.attach_worker(local_worker.cancellation_handle());
                worker = Some(local_worker);
                status
            }
            Err(_) => "unavailable".to_owned(),
        }
    };
    if cancellation.is_requested() {
        drop(worker.take());
        cancellation.detach_worker();
        return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
    }

    let mut face_runtime =
        if settings.detect_subject || settings.detect_closed_eyes || settings.review_focus {
            match face_processing::load_for_app(&app_handle) {
                Ok(runtime) => Some(runtime),
                Err(_) => {
                    subject_analysis_status =
                        status_when_face_model_missing(&subject_analysis_status).to_owned();
                    None
                }
            }
        } else {
            None
        };

    if settings.detect_subject || settings.detect_closed_eyes || settings.review_focus {
        emit_culling_progress(
            &app_handle,
            &invocation_id,
            &cancellation,
            CullingProgress {
                current: total_count,
                total: total_count,
                stage: "Reviewing local subject, eyes and focus signals...".to_owned(),
            },
        );
        for data in &mut successful_analyses {
            if cancellation.is_requested() {
                drop(worker.take());
                cancellation.detach_worker();
                drop(face_runtime.take());
                return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
            }
            let loaded = load_analysis_image(&data.result.path, &app_settings);
            apply_assisted_image_load(
                data,
                loaded,
                &settings,
                &mut worker,
                &mut face_runtime,
                &mut failed_paths,
            );
            if cancellation.is_requested() {
                drop(worker.take());
                cancellation.detach_worker();
                drop(face_runtime.take());
                return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
            }
        }
        if (settings.detect_subject
            && successful_analyses
                .iter()
                .any(|data| data.result.subject_method == "grounding-dino-error"))
            || (settings.review_focus
                && successful_analyses
                    .iter()
                    .any(|data| data.result.focus_method == "vgg-native-error"))
        {
            subject_analysis_status = "error".to_owned();
        }
    }

    drop(worker.take());
    cancellation.detach_worker();
    drop(face_runtime.take());
    if cancellation.is_requested() {
        return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
    }

    emit_culling_progress(
        &app_handle,
        &invocation_id,
        &cancellation,
        CullingProgress {
            current: total_count,
            total: total_count,
            stage: "Grouping similar images...".to_string(),
        },
    );

    let mut suggestions = CullingSuggestions {
        failed_paths,
        subject_analysis_status,
        ..Default::default()
    };
    let mut processed_indices = vec![false; successful_analyses.len()];

    if settings.group_similar {
        for i in 0..successful_analyses.len() {
            if cancellation.is_requested() {
                return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
            }
            if processed_indices[i] {
                continue;
            }

            let mut current_group_indices = vec![];
            let mut queue = VecDeque::new();

            processed_indices[i] = true;
            current_group_indices.push(i);
            queue.push_back(i);

            while let Some(current_idx) = queue.pop_front() {
                if cancellation.is_requested() {
                    return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
                }
                for j in (current_idx + 1)..successful_analyses.len() {
                    if processed_indices[j] {
                        continue;
                    }

                    let dist = successful_analyses[current_idx]
                        .hash
                        .dist(&successful_analyses[j].hash);
                    if dist <= settings.similarity_threshold {
                        processed_indices[j] = true;
                        current_group_indices.push(j);
                        queue.push_back(j);
                    }
                }
            }

            if current_group_indices.len() > 1 {
                current_group_indices.sort_by(|&a, &b| {
                    successful_analyses[b]
                        .result
                        .quality_score
                        .partial_cmp(&successful_analyses[a].result.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let representative_idx = current_group_indices[0];
                let duplicate_indices = &current_group_indices[1..];

                suggestions.similar_groups.push(CullGroup {
                    representative: successful_analyses[representative_idx].result.clone(),
                    duplicates: duplicate_indices
                        .iter()
                        .map(|&idx| successful_analyses[idx].result.clone())
                        .collect(),
                });
            }
        }
    }

    if settings.filter_blurry {
        for i in 0..successful_analyses.len() {
            if cancellation.is_requested() {
                return finish_cancelled_culling(&mut invocation, &app_handle, &invocation_id);
            }
            if !processed_indices[i] {
                let item = &successful_analyses[i];
                if item.result.sharpness_metric < effective_blur_threshold(&settings) {
                    suggestions.blurry_images.push(item.result.clone());
                }
            }
        }
        suggestions.blurry_images.sort_by(|a, b| {
            a.sharpness_metric
                .partial_cmp(&b.sharpness_metric)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    for data in &successful_analyses {
        if !data.result.review_alerts.is_empty() {
            suggestions.review_alerts.push(data.result.clone());
        }
        if review_coverage_is_unknown(
            &settings,
            &data.result.subject_status,
            &data.result.focus_status,
            &data.result.eye_state,
        ) {
            suggestions.unknown_images.push(data.result.clone());
        }
    }
    suggestions
        .review_alerts
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .unknown_images
        .sort_by(|left, right| left.path.cmp(&right.path));

    if invocation.conclude() {
        emit_culling(&app_handle, "culling-cancelled", &invocation_id, ());
        return Ok(CullingSuggestions::default());
    }
    emit_culling(
        &app_handle,
        "culling-complete",
        &invocation_id,
        &suggestions,
    );
    Ok(suggestions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_is_scoped_and_allows_restart_after_settlement() {
        let first_id = "culling-cancel-test-first";
        let second_id = "culling-cancel-test-second";
        let owner = "culling-owner-main";
        let other_owner = "culling-owner-other";
        let paths = vec!["/photos/one.raw".to_owned(), "/photos/two.raw".to_owned()];
        let (mut first, mut second) = (
            CullingInvocationGuard::register(first_id, owner, &paths).unwrap(),
            CullingInvocationGuard::register(second_id, other_owner, &paths).unwrap(),
        );

        assert!(CullingInvocationGuard::register("duplicate-owner", owner, &paths).is_err());
        first.cancellation.update_progress(&CullingProgress {
            current: 1,
            total: 2,
            stage: "Analyzing images...".to_owned(),
        });
        let recovered = active_culling_for_owner(owner).unwrap();
        assert_eq!(recovered.invocation_id, first_id);
        assert_eq!(recovered.paths_to_cull, paths);
        assert_eq!(recovered.progress.current, 1);
        assert!(!request_culling_cancel(other_owner, first_id));
        assert!(request_culling_cancel(owner, first_id));
        assert!(first.cancellation.is_requested());
        assert!(!second.cancellation.is_requested());
        assert!(first.conclude());
        assert!(active_culling_for_owner(owner).is_none());
        assert!(!request_culling_cancel(owner, first_id));
        assert!(!second.conclude());

        let mut restarted = CullingInvocationGuard::register(first_id, owner, &paths).unwrap();
        assert!(!restarted.cancellation.is_requested());
        assert!(!restarted.conclude());
    }

    #[test]
    fn native_focus_score_does_not_raise_review_without_calibration() {
        for score in [0.02, 0.0249, 0.0595, 0.060, 0.2981, 0.40] {
            assert!(
                focus_review_alerts(score).is_empty(),
                "score {score} must not raise focusReview"
            );
        }
        assert_eq!(focus_calibration_status(), "calibration-unavailable");
    }

    #[test]
    fn mixed_primary_eyes_report_closed_when_any_face_is_closed() {
        use face_processing::EyeState;

        assert_eq!(
            reviewed_primary_eye_state(&[EyeState::Closed, EyeState::Open]),
            "closed"
        );
        assert_eq!(
            primary_eye_review_alert(&[EyeState::Closed, EyeState::Open]),
            Some("eyesClosed")
        );
        assert_eq!(
            reviewed_primary_eye_state(&[EyeState::Closed, EyeState::Unknown]),
            "closed"
        );
        assert_eq!(
            primary_eye_review_alert(&[EyeState::Open, EyeState::Unknown]),
            Some("eyesUnknown")
        );
        assert_eq!(
            reviewed_primary_eye_state(&[EyeState::Unknown, EyeState::Unknown]),
            "unknown"
        );
        assert_eq!(
            reviewed_primary_eye_state(&[EyeState::Open, EyeState::Open]),
            "open"
        );
        assert_eq!(primary_eye_review_alert(&[EyeState::Open]), None);
        assert_eq!(reviewed_primary_eye_state(&[]), "not-evaluated");
        assert_eq!(primary_eye_review_alert(&[]), None);
        assert_eq!(
            primary_eye_review_alert(&[EyeState::Closed]),
            Some("eyesClosed")
        );
    }

    #[test]
    fn eyes_only_without_a_detected_face_stays_explicitly_unknown() {
        let settings = CullingSettings {
            detect_subject: false,
            detect_closed_eyes: true,
            review_focus: false,
            ..Default::default()
        };

        assert!(review_coverage_is_unknown(
            &settings,
            "not-evaluated",
            "disabled",
            "not-evaluated"
        ));
        assert!(!review_coverage_is_unknown(
            &settings,
            "not-evaluated",
            "disabled",
            "open"
        ));
        assert_eq!(primary_eye_review_alert(&[]), None);
    }

    #[test]
    fn assisted_reread_failure_is_unprocessed_and_not_evaluated() {
        let path = std::env::temp_dir().join(format!(
            "picportal-culling-reread-{}-{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        image::RgbImage::from_pixel(16, 16, image::Rgb([96, 128, 160]))
            .save(&path)
            .unwrap();

        let app_settings = crate::app_settings::AppSettings::default();
        let hasher = HasherConfig::new()
            .hash_alg(HashAlg::DoubleGradient)
            .hash_size(16, 16)
            .to_hasher();
        let path_string = path.to_string_lossy().into_owned();
        let mut data = analyze_image(&path_string, &hasher, &app_settings).unwrap();
        std::fs::remove_file(&path).unwrap();
        let reread = load_analysis_image(&path_string, &app_settings);
        assert!(reread.is_err());

        let settings = CullingSettings {
            detect_closed_eyes: true,
            ..Default::default()
        };
        let mut worker = None;
        let mut face_runtime = None;
        let mut failed_paths = Vec::new();
        apply_assisted_image_load(
            &mut data,
            reread,
            &settings,
            &mut worker,
            &mut face_runtime,
            &mut failed_paths,
        );

        assert_eq!(failed_paths, vec![path_string]);
        assert_eq!(data.result.eye_state, "not-evaluated");
        assert_eq!(data.result.subject_status, "unknown");
        assert!(review_coverage_is_unknown(
            &settings,
            &data.result.subject_status,
            &data.result.focus_status,
            &data.result.eye_state,
        ));
    }

    #[test]
    fn blur_severity_is_stricter_than_lenient_without_the_numeric_slider() {
        let mut strict_at_low_slider = CullingSettings::default();
        strict_at_low_slider.blur_threshold = 25.0;
        strict_at_low_slider.blur_severity = "strict".to_owned();

        let mut lenient_at_high_slider = CullingSettings::default();
        lenient_at_high_slider.blur_threshold = 100.0;
        lenient_at_high_slider.blur_severity = "lenient".to_owned();

        let strict = effective_blur_threshold(&strict_at_low_slider);
        let lenient = effective_blur_threshold(&lenient_at_high_slider);
        assert!(
            strict > lenient,
            "strict threshold {strict} must flag more than lenient {lenient}"
        );
        assert_eq!(lenient, 75.0);
        assert_eq!(strict, 150.0);

        let mut moderate = CullingSettings::default();
        moderate.blur_threshold = 25.0;
        moderate.blur_severity = "moderate".to_owned();
        assert_eq!(effective_blur_threshold(&moderate), 100.0);
    }

    fn box_at(x: f32, y: f32, width: f32, height: f32) -> SubjectBox {
        SubjectBox {
            x,
            y,
            width,
            height,
            score: 0.5,
            label: "subject".to_owned(),
        }
    }

    fn face_at(x: f32, y: f32, width: f32, height: f32) -> AssociationFace {
        AssociationFace {
            x,
            y,
            width,
            height,
        }
    }

    fn bust_at(nose_x: f32, nose_y: f32, subject: &SubjectBox) -> AssociationPose {
        AssociationPose {
            nose_x,
            nose_y,
            torso: vec![
                (subject.x + 10.0, subject.y + subject.height * 0.4),
                (subject.x + 40.0, subject.y + subject.height * 0.4),
                (subject.x + 12.0, subject.y + subject.height * 0.7),
                (subject.x + 42.0, subject.y + subject.height * 0.7),
            ],
        }
    }

    #[test]
    fn dance_detection_uses_separate_passes_and_keeps_the_first_hit() {
        let phrases = subject_phrases("dance");
        assert_eq!(phrases.len(), 2);
        assert_eq!(phrases[0].phrase, "a couple dancing together.");
        assert_eq!(phrases[1].phrase, "a person dancing.");
        assert!(phrases.iter().all(|phrase| phrase.phrase != "a couple or group of people dancing together."));

        let couple = box_at(0.0, 0.0, 10.0, 10.0);
        let person = box_at(20.0, 20.0, 5.0, 5.0);
        let passes = [
            (phrases[0], vec![couple.clone()]),
            (phrases[1], vec![person.clone()]),
        ];
        let selected = choose_subject_phrase(&passes).unwrap();
        assert_eq!(selected.0.phrase, "a couple dancing together.");
        assert_eq!(selected.1[0].x, couple.x);

        let empty_first = [
            (phrases[0], Vec::new()),
            (phrases[1], vec![person.clone()]),
        ];
        let selected = choose_subject_phrase(&empty_first).unwrap();
        assert_eq!(selected.0.phrase, "a person dancing.");
        assert_eq!(selected.1[0].x, person.x);
    }

    #[test]
    fn measured_link_keeps_every_face_inside_the_nose_limit() {
        let subject = box_at(0.0, 0.0, 1000.0, 1600.0);
        let couple = face_at(100.0, 100.0, 80.0, 100.0);
        let extra = face_at(200.0, 100.0, 100.0, 120.0);
        let bust = bust_at(140.0, 150.0, &subject);
        let extra_limit = NOSE_DISTANCE_FACTOR * extra.width.max(extra.height);
        let extra_distance = ((250.0 - 140.0_f32).powi(2) + (160.0 - 150.0_f32).powi(2)).sqrt();
        assert!(extra_distance <= extra_limit);
        assert!(face_overlap(&extra, &[subject.clone()]) >= SUBJECT_OVERLAP_MIN);

        let roles = attribute_faces(&[couple, extra.clone()], &[subject.clone()], &[bust.clone()]);
        assert_eq!(roles, vec![AttributedRole::Primary, AttributedRole::Primary]);

        let beyond = face_at(500.0, 100.0, 80.0, 80.0);
        let beyond_limit = NOSE_DISTANCE_FACTOR * beyond.width.max(beyond.height);
        let beyond_distance = ((540.0 - 140.0_f32).powi(2) + (140.0 - 150.0_f32).powi(2)).sqrt();
        assert!(beyond_distance > beyond_limit);
        let roles = attribute_faces(&[beyond], &[subject], &[bust]);
        assert_eq!(roles, vec![AttributedRole::Secondary]);
    }

    #[test]
    fn center_or_small_overlap_does_not_make_a_face_primary() {
        let subject = box_at(0.0, 0.0, 400.0, 400.0);
        let mostly_outside = face_at(380.0, 180.0, 80.0, 80.0);
        let bust = bust_at(200.0, 200.0, &subject);
        let roles = attribute_faces(&[mostly_outside], &[subject.clone()], &[bust]);
        assert_eq!(roles[0], AttributedRole::Secondary);

        let half_short = face_at(360.0, 100.0, 100.0, 100.0);
        let close_bust = bust_at(390.0, 150.0, &subject);
        let roles = attribute_faces(&[half_short], &[subject], &[close_bust]);
        assert_ne!(roles[0], AttributedRole::Primary);
    }

    #[test]
    fn distinct_busts_link_independently_including_a_third_face() {
        let subject = box_at(0.0, 0.0, 1000.0, 1600.0);
        let left = face_at(80.0, 80.0, 90.0, 110.0);
        let right = face_at(700.0, 90.0, 90.0, 110.0);
        let extra = face_at(400.0, 80.0, 80.0, 100.0);
        let roles = attribute_faces(
            &[left, right, extra],
            &[subject.clone()],
            &[
                bust_at(125.0, 135.0, &subject),
                bust_at(745.0, 145.0, &subject),
                bust_at(440.0, 130.0, &subject),
            ],
        );
        assert_eq!(
            roles,
            vec![
                AttributedRole::Primary,
                AttributedRole::Primary,
                AttributedRole::Primary
            ]
        );
    }

    #[test]
    fn half_overlap_and_bust_nose_links_without_a_center_shortcut() {
        let subject = box_at(0.0, 0.0, 200.0, 200.0);
        let face = face_at(150.0, 40.0, 100.0, 80.0);
        assert!((face_overlap(&face, &[subject.clone()]) - 0.5).abs() < 0.001);
        let bust = bust_at(180.0, 80.0, &subject);
        let roles = attribute_faces(&[face], &[subject], &[bust]);
        assert_eq!(roles[0], AttributedRole::Primary);
    }

    #[test]
    fn face_model_failure_stays_face_unavailable_for_live_pipeline_statuses() {
        for status in [
            "ready",
            "subject-ready-pose-unavailable",
            "subject-ready-focus-unavailable",
            "subject-ready-focus-calibration-unavailable",
            "focus-calibration-unavailable",
            "focus-unavailable",
        ] {
            assert_eq!(status_when_face_model_missing(status), "face-unavailable");
        }
        assert_eq!(status_when_face_model_missing("disabled"), "face-unavailable");
        assert_eq!(status_when_face_model_missing("unavailable"), "unavailable");
        assert_eq!(
            status_when_face_model_missing("subject-unavailable"),
            "unavailable"
        );
        assert_eq!(status_when_face_model_missing("focus-ready"), "unavailable");
    }

    #[test]
    fn overlap_is_the_largest_single_box_not_the_sum() {
        let face = face_at(0.0, 0.0, 100.0, 100.0);
        let repeated = box_at(0.0, 0.0, 40.0, 100.0);
        let adjacent = box_at(40.0, 0.0, 30.0, 100.0);
        let repeated_overlap = face_overlap(&face, &[repeated.clone(), repeated.clone()]);
        let split_overlap = face_overlap(&face, &[repeated.clone(), adjacent]);
        assert!((repeated_overlap - 0.4).abs() < 0.001);
        assert!((split_overlap - 0.4).abs() < 0.001);
        assert!(repeated_overlap < SUBJECT_OVERLAP_MIN);

        let bust = bust_at(20.0, 50.0, &repeated);
        let roles = attribute_faces(&[face.clone()], &[repeated.clone(), repeated], &[bust]);
        assert_eq!(roles, vec![AttributedRole::Secondary]);

        let covering = box_at(0.0, 0.0, 60.0, 100.0);
        let covering_bust = bust_at(30.0, 50.0, &covering);
        let roles = attribute_faces(
            &[face],
            &[covering, box_at(70.0, 0.0, 20.0, 100.0)],
            &[covering_bust],
        );
        assert_eq!(roles, vec![AttributedRole::Primary]);
    }

    #[test]
    fn subject_pose_crop_uses_only_the_exact_subject_box() {
        let image = DynamicImage::new_rgba8(100, 100);
        let subject = box_at(20.0, 30.0, 40.0, 50.0);
        let (crop, origin_x, origin_y, crop_width, crop_height) =
            exact_subject_crop(&image, &subject).unwrap();

        assert_eq!((origin_x, origin_y), (20.0, 30.0));
        assert_eq!((crop_width, crop_height), (40.0, 50.0));
        assert_eq!((crop.width(), crop.height()), (40, 50));
    }

    #[test]
    fn subject_boxes_are_normalized_for_display_without_changing_pixels() {
        let pixel = box_at(2205.0, 700.0, 800.0, 400.0);
        let shown = display_subject_boxes(vec![pixel.clone()], 6048.0, 4024.0);
        assert!((shown[0].x - 2205.0 / 6048.0).abs() < 1e-6);
        assert!((shown[0].y - 700.0 / 4024.0).abs() < 1e-6);
        assert!(shown[0].x < 1.0);
        assert!(shown[0].width < 1.0);
        assert_eq!(pixel.x, 2205.0);
    }

    #[test]
    fn subject_status_reports_linked_groups_and_leaves_empty_sets_unknown() {
        assert_eq!(subject_status_label(true, true, 2), "multiple");
        assert_eq!(subject_status_label(true, true, 0), "unknown");
        assert_eq!(subject_status_label(true, true, 1), "primary");
        assert_eq!(subject_status_label(false, false, 0), "not-evaluated");
    }
}
