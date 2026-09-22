use crate::app_settings::load_settings;
use image::{DynamicImage, GenericImageView, GrayImage, imageops};
use image_hasher::{HashAlg, HasherConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tauri::{AppHandle, Emitter};

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

const NATIVE_IN_FOCUS_REVIEW_FLOOR: f64 = 0.060;

fn native_focus_needs_review(in_focus_mean: f64) -> bool {
    in_focus_mean < NATIVE_IN_FOCUS_REVIEW_FLOOR
}
const WEIGHT_SHARPNESS: f64 = 0.40;
const WEIGHT_CENTER_FOCUS: f64 = 0.35;
const WEIGHT_EXPOSURE: f64 = 0.25;

fn effective_blur_threshold(settings: &CullingSettings) -> f64 {
    let multiplier = match settings.blur_severity.as_str() {
        "lenient" => 0.75,
        "strict" => 1.5,
        _ => 1.0,
    };
    settings.blur_threshold * multiplier
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

fn profile_allows_multiple_subjects(profile: &str) -> bool {
    matches!(profile, "dance" | "wedding" | "general")
}

fn point_in_subject(face: &face_processing::FaceBox, subject: &SubjectBox) -> bool {
    let center_x = face.x + face.width / 2.0;
    let center_y = face.y + face.height / 2.0;
    center_x >= subject.x
        && center_x <= subject.x + subject.width
        && center_y >= subject.y
        && center_y <= subject.y + subject.height
}

fn subject_box_iou(face: &face_processing::FaceBox, subject: &SubjectBox) -> f32 {
    let left = face.x.max(subject.x);
    let top = face.y.max(subject.y);
    let right = (face.x + face.width).min(subject.x + subject.width);
    let bottom = (face.y + face.height).min(subject.y + subject.height);
    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let face_area = face.width * face.height;
    if face_area > 0.0 {
        intersection / face_area
    } else {
        0.0
    }
}

fn eye_state_label(state: face_processing::EyeState) -> &'static str {
    match state {
        face_processing::EyeState::Open => "open",
        face_processing::EyeState::Closed => "closed",
        face_processing::EyeState::Unknown => "unknown",
    }
}

fn subject_status_label(
    detect_subject: bool,
    has_subject_boxes: bool,
    attribution_is_ambiguous: bool,
    primary_count: usize,
    multiple_is_valid: bool,
) -> &'static str {
    if !detect_subject {
        "not-evaluated"
    } else if !has_subject_boxes {
        "unknown"
    } else if attribution_is_ambiguous || primary_count == 0 {
        "unknown"
    } else if primary_count > 1 && multiple_is_valid {
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
            match local_worker.detect(&frame, &settings.subject_profile) {
                Ok(boxes) => {
                    subject_method = "grounding-dino-local".to_owned();
                    subject_boxes = boxes
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
                        .collect();
                }
                Err(_) => {
                    subject_method = "grounding-dino-error".to_owned();
                }
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

    let mut subject_candidates = Vec::new();
    for (index, face) in faces.iter().enumerate() {
        let inside = subject_boxes.iter().any(|subject| {
            point_in_subject(&face.bbox, subject) || subject_box_iou(&face.bbox, subject) >= 0.05
        });
        if inside {
            subject_candidates.push(index);
        }
    }
    let multiple_is_valid = profile_allows_multiple_subjects(&settings.subject_profile);
    let attribution_is_ambiguous =
        subject_candidates.len() > 1 && !multiple_is_valid && !subject_boxes.is_empty();
    let primary_indices = if !settings.detect_subject {
        (0..faces.len()).collect()
    } else if !subject_boxes.is_empty() && !attribution_is_ambiguous {
        subject_candidates.clone()
    } else {
        Vec::new()
    };

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
        let is_primary = primary_indices.contains(&index);
        let role = if is_primary {
            "primary"
        } else if attribution_is_ambiguous && subject_candidates.contains(&index) {
            "unknown"
        } else {
            "secondary"
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
                            focus_status = "available".to_owned();
                            if focus_dimensions.is_none() {
                                focus_dimensions = Some((measurement.width, measurement.height));
                            }
                            focus_crop_dimensions = Some((measurement.width, measurement.height));
                            focus_model_dimensions =
                                Some((measurement.input_width, measurement.input_height));
                            focus_values.push(measurement.score);
                            if native_focus_needs_review(measurement.score) {
                                review_alerts.push("focusReview".to_owned());
                            }
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
        attribution_is_ambiguous,
        primary_indices.len(),
        multiple_is_valid,
    );
    let eye_state = if !settings.detect_closed_eyes {
        "disabled"
    } else if eye_states.is_empty() {
        "unknown"
    } else if eye_states
        .iter()
        .all(|state| *state == face_processing::EyeState::Open)
    {
        "open"
    } else if eye_states
        .iter()
        .all(|state| *state == face_processing::EyeState::Closed)
    {
        "closed"
    } else {
        "unknown"
    };
    if settings.detect_closed_eyes && eye_state == "closed" {
        review_alerts.push("eyesClosed".to_owned());
    } else if settings.detect_closed_eyes && eye_state == "unknown" {
        review_alerts.push("eyesUnknown".to_owned());
    }
    if subject_status == "unknown" {
        review_alerts.push("subjectUnknown".to_owned());
    }

    data.result.subject_status = subject_status.to_owned();
    data.result.subject_method = if subject_boxes.is_empty() && !faces.is_empty() {
        format!("{subject_method};{face_method}")
    } else {
        subject_method
    };
    data.result.subject_boxes = subject_boxes;
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
        "available".to_owned()
    } else if !focus_available {
        "unavailable".to_owned()
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

fn subject_analysis_status_for(
    worker: &subject_inference::LocalWorker,
    subject_requested: bool,
    focus_requested: bool,
) -> String {
    if subject_requested {
        if !worker.subject_is_ready() {
            "subject-unavailable".to_owned()
        } else if focus_requested && !worker.focus_is_ready() {
            "subject-ready-focus-unavailable".to_owned()
        } else {
            "ready".to_owned()
        }
    } else if worker.focus_is_ready() {
        "focus-ready".to_owned()
    } else {
        "focus-unavailable".to_owned()
    }
}

#[tauri::command]
pub async fn cull_images(
    paths: Vec<String>,
    settings: CullingSettings,
    app_handle: AppHandle,
) -> Result<CullingSuggestions, String> {
    if paths.is_empty() {
        return Ok(CullingSuggestions::default());
    }

    let app_settings = load_settings(app_handle.clone()).unwrap_or_default();

    let total_count = paths.len();
    let completed_count = Arc::new(AtomicUsize::new(0));
    let _ = app_handle.emit("culling-start", total_count);

    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::DoubleGradient)
        .hash_size(16, 16)
        .to_hasher();

    let analysis_results: Vec<Result<ImageAnalysisData, (String, String)>> = paths
        .par_iter()
        .map(|path| {
            let completed = completed_count.fetch_add(1, Ordering::Relaxed) + 1;
            let _ = app_handle.emit(
                "culling-progress",
                CullingProgress {
                    current: completed,
                    total: total_count,
                    stage: "Analyzing images...".to_string(),
                },
            );

            analyze_image(path, &hasher, &app_settings).map_err(|e| (path.to_string(), e))
        })
        .collect();

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
                worker = Some(local_worker);
                status
            }
            Err(_) => "unavailable".to_owned(),
        }
    };
    let mut face_runtime =
        if settings.detect_subject || settings.detect_closed_eyes || settings.review_focus {
            match face_processing::load_for_app(&app_handle) {
                Ok(runtime) => Some(runtime),
                Err(_) => {
                    subject_analysis_status = if matches!(
                        subject_analysis_status.as_str(),
                        "ready" | "focus-ready" | "subject-ready-focus-unavailable"
                    ) {
                        "face-unavailable".to_owned()
                    } else {
                        "unavailable".to_owned()
                    };
                    None
                }
            }
        } else {
            None
        };

    if settings.detect_subject || settings.detect_closed_eyes || settings.review_focus {
        let _ = app_handle.emit(
            "culling-progress",
            CullingProgress {
                current: total_count,
                total: total_count,
                stage: "Reviewing local subject, eyes and focus signals...".to_owned(),
            },
        );
        for data in &mut successful_analyses {
            match load_analysis_image(&data.result.path, &app_settings) {
                Ok(image) => {
                    update_assisted_result(data, &image, &settings, &mut worker, &mut face_runtime);
                }
                Err(error) => {
                    data.result.subject_status = "unknown".to_owned();
                    data.result.subject_method = format!("image-unavailable:{error}");
                    data.result.focus_status = "unknown".to_owned();
                    data.result.eye_state = if settings.detect_closed_eyes {
                        "unknown".to_owned()
                    } else {
                        "disabled".to_owned()
                    };
                    data.result.review_alerts.push("subjectUnknown".to_owned());
                }
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

    let _ = app_handle.emit(
        "culling-progress",
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
            if processed_indices[i] {
                continue;
            }

            let mut current_group_indices = vec![];
            let mut queue = VecDeque::new();

            processed_indices[i] = true;
            current_group_indices.push(i);
            queue.push_back(i);

            while let Some(current_idx) = queue.pop_front() {
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
        if data.result.subject_status == "unknown"
            || (settings.review_focus
                && matches!(data.result.focus_status.as_str(), "unknown" | "unavailable"))
        {
            suggestions.unknown_images.push(data.result.clone());
        }
    }
    suggestions
        .review_alerts
        .sort_by(|left, right| left.path.cmp(&right.path));
    suggestions
        .unknown_images
        .sort_by(|left, right| left.path.cmp(&right.path));

    let _ = app_handle.emit("culling-complete", &suggestions);
    Ok(suggestions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_focus_review_uses_the_measured_in_focus_floor() {
        assert!(native_focus_needs_review(0.02));
        assert!(!native_focus_needs_review(0.40));
        assert!(!native_focus_needs_review(NATIVE_IN_FOCUS_REVIEW_FLOOR));

        let benchmark = [
            ("img_015_0", "net", 0.2290, false),
            ("img_021_0", "net", 0.4612, false),
            ("img_040_0", "net", 0.1176, false),
            ("img_053_0", "mou", 0.0232, true),
            ("img_053_1", "mou", 0.0392, true),
            ("img_055_0", "net", 0.1042, false),
            ("img_056_0", "net", 0.0595, true),
            ("img_061_0", "net", 0.9170, false),
            ("img_065_1", "mou", 0.0283, true),
            ("img_066_0", "mou", 0.0350, true),
            ("img_067_0", "net", 0.0780, false),
            ("img_073_0", "net", 0.2005, false),
            ("img_085_0", "net", 0.3986, false),
            ("img_106_0", "net", 0.2222, false),
            ("img_111_0", "net", 0.4552, false),
            ("img_117_0", "net", 0.3259, false),
            ("img_123_0", "net", 0.3590, false),
            ("img_123_1", "net", 0.1365, false),
            ("img_125_0", "net", 0.3309, false),
            ("img_126_0", "net", 0.4361, false),
            ("img_130_0", "net", 0.3161, false),
            ("img_134_0", "mou", 0.0410, true),
            ("img_135_0", "net", 0.4679, false),
            ("img_135_1", "mou", 0.1297, false),
            ("img_136_0", "net", 0.4212, false),
            ("img_136_1", "mou", 0.1674, false),
            ("img_137_0", "net", 0.3301, false),
            ("img_137_1", "mou", 0.1401, false),
        ];
        let mut true_positive = 0;
        let mut false_negative = 0;
        let mut false_positive = 0;
        let mut true_negative = 0;
        for (name, label, score, expected) in benchmark {
            let alert = native_focus_needs_review(score);
            assert_eq!(alert, expected, "{name}");
            match (label, alert) {
                ("mou", true) => true_positive += 1,
                ("mou", false) => false_negative += 1,
                (_, true) => false_positive += 1,
                (_, false) => true_negative += 1,
            }
        }
        assert_eq!(
            (true_positive, false_negative, false_positive, true_negative),
            (5, 3, 1, 19)
        );
    }

    #[test]
    fn blur_severity_scales_only_the_existing_technical_threshold() {
        let mut settings = CullingSettings::default();
        settings.blur_threshold = 100.0;

        settings.blur_severity = "lenient".to_owned();
        assert_eq!(effective_blur_threshold(&settings), 75.0);

        settings.blur_severity = "moderate".to_owned();
        assert_eq!(effective_blur_threshold(&settings), 100.0);

        settings.blur_severity = "strict".to_owned();
        assert_eq!(effective_blur_threshold(&settings), 150.0);
    }

    #[test]
    fn subject_profiles_allow_groups_only_when_the_profile_supports_them() {
        assert!(profile_allows_multiple_subjects("general"));
        assert!(profile_allows_multiple_subjects("wedding"));
        assert!(profile_allows_multiple_subjects("dance"));
        assert!(!profile_allows_multiple_subjects("portrait"));
        assert!(!profile_allows_multiple_subjects("sports"));
    }

    #[test]
    fn subject_status_distinguishes_valid_groups_from_ambiguous_single_subjects() {
        assert_eq!(subject_status_label(true, true, false, 2, true), "multiple");
        assert_eq!(subject_status_label(true, true, true, 2, false), "unknown");
        assert_eq!(subject_status_label(true, true, false, 1, true), "primary");
        assert_eq!(
            subject_status_label(false, false, false, 0, false),
            "not-evaluated"
        );
    }
}
